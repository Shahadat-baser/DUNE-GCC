"""
STEP 3 - daily dynamic predictors at every point in inputs/stations.csv, 2019-2025.

For each UTC day and point: the value of the 0.01 deg cell containing the point (<band>) and the
mean of the 3x3 cells around it (<band>_3x3, masked cells ignored).  Day-level properties travel
with each row: number of S5P orbits that day and whether the ERA5 / ERA5-Land / precipitation
images existed.

Fetched month by month straight to disk (tables/station_daily/), resumable: rerun and finished
months are skipped.  A month that times out is split automatically.  Finally all months are joined
into tables/<STATION_DAILY_CSV> (Station_Daily_2019_2026.csv after the extension).

Points added to stations.csv later (e.g. a monitor from the Abu Dhabi SDI feed) are fetched for every
already-finished month into separate files station_daily_add-<n>pts_<month>.csv; finished files are
never rewritten.  Rows fetched twice must be identical - the join checks this and reports it.
"""
import os, sys, glob
# --- locate the code folder: works with "python script.py", "%run script.py" AND a pasted notebook cell
CODE_DIR = os.getcwd()                      # fallback when __file__ is undefined (notebook)
try:
    CODE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:                       # pasted into a notebook cell: __file__ does not exist
    if os.path.exists(os.path.join(os.getcwd(), "dune_config.py")):
        CODE_DIR = os.getcwd()
if not os.path.exists(os.path.join(CODE_DIR, "dune_config.py")):
    raise SystemExit(f"dune_config.py not found in {CODE_DIR} - unzip DUNE_GEE_v3 into that folder")
sys.path.insert(0, CODE_DIR)
for _m in ("dune_config", "dune_env", "dune_gee"):   # always load the CURRENT files, never a copy
    sys.modules.pop(_m, None)                         # cached in the notebook kernel from an earlier run
import dune_env
dune_env.ensure("station_daily")      # installs / checks every package this step needs
from concurrent.futures import ThreadPoolExecutor
import hashlib
import numpy as np, pandas as pd
import ee
import dune_config as C
import dune_gee as G
if getattr(G, "PIPELINE_VERSION", None) != "2026-10-08a":
    raise SystemExit(f"{G.__file__} is an older version than this script. Unzip the WHOLE DUNE_GEE_v3.zip "
                     "again into the code folder and choose 'Replace the files in the destination'.")

OUT = os.path.join(C.DIRS["tables"], "station_daily")
BANDS = G.ALL_DAILY_BANDS
B3 = [b + "_3x3" for b in BANDS]


def build(stations):
    def fc_for(start, end):
        def per_day(img):
            gridded = G.to_grid(img)
            m3 = gridded.select(BANDS).reduceNeighborhood(ee.Reducer.mean(), ee.Kernel.square(1, "pixels")).rename(B3)
            both = gridded.addBands(m3)
            fc = both.reduceRegions(collection=stations, reducer=ee.Reducer.first(),
                                    crs=C.CRS, crsTransform=C.CRS_TRANSFORM)
            props = img.toDictionary(G.DAY_PROPS)
            return fc.map(lambda f: f.set(props))
        return G.daily_collection(start, end).map(per_day).flatten()
    return fc_for


def main():
    G.init()
    os.makedirs(OUT, exist_ok=True)
    st = pd.read_csv(C.STATIONS_CSV)
    fc_for = build(G.stations_fc(st))
    sel = ["Station", "Network"] + G.DAY_PROPS + BANDS + B3
    chunks = G.month_chunks(C.DATE_START, C.DATE_END)
    G.log(f"station daily: {len(st)} points, {len(chunks)} months, {C.N_THREADS} threads", "station_daily")

    # finished months that lack points added to stations.csv since they were fetched
    jobs = []
    for s, e in chunks:
        f = os.path.join(OUT, f"station_daily_{s}_{e}.csv")
        if not os.path.exists(f):
            jobs.append((s, e, f, fc_for))
            continue
        have = set(pd.read_csv(f, usecols=["Station"]).Station)
        for a in glob.glob(os.path.join(OUT, f"station_daily_add-*_{s}_{e}.csv")):
            have |= set(pd.read_csv(a, usecols=["Station"]).Station)
        miss = st[~st.Station.isin(have)]
        if len(miss):
            tag = hashlib.sha1("|".join(sorted(miss.Station)).encode()).hexdigest()[:6]
            jobs.append((s, e, os.path.join(OUT, f"station_daily_add-{len(miss)}pts-{tag}_{s}_{e}.csv"),
                         build(G.stations_fc(miss))))
    n_add = sum("_add-" in j[2] for j in jobs)
    G.log(f"months to fetch: {len(jobs) - n_add} new + {n_add} add-on files for points added later", "station_daily")

    def job(j):
        s, e, f, fn = j
        r = G.fetch_range(fn, s, e, f, sel, "station_daily")
        G.log(f"{os.path.basename(f)}: {r}", "station_daily")
        return r

    with ThreadPoolExecutor(C.N_THREADS) as ex:
        res = list(ex.map(job, jobs))

    files = sorted(f for f in glob.glob(os.path.join(OUT, "station_daily_*.csv")) if "__" not in f)
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    dup = df[df.duplicated(["Station", "date"], keep=False)]
    if len(dup):        # the same point-day fetched twice must give the same numbers
        num = [c for c in BANDS + B3 if c in dup]
        g = dup.groupby(["Station", "date"])[num].agg(lambda x: np.nanmax(x) - np.nanmin(x) if x.notna().any() else 0)
        G.log(f"rows fetched twice: {len(dup) // 2:,} | max difference between fetches {float(np.nanmax(g.values)):.3g}",
              "station_daily")
    df = df.sort_values(["Station", "date"], kind="stable").drop_duplicates(["Station", "date"])   # earliest fetch kept
    out = os.path.join(C.DIRS["tables"], C.STATION_DAILY_CSV)
    df.to_csv(out, index=False)
    failed = sum(r == "FAILED" or "FAILED" in r for r in res)
    G.log(f"wrote {out}: {len(df):,} rows, {df.Station.nunique()} points, {df.date.nunique()} days | "
          f"months with failures: {failed}", "station_daily")
    print("\nNO2 valid-day fraction by network (cell value):")
    print(df.assign(v=df.NO2_trop.notna()).groupby("Network").v.mean().round(3).to_string())
    G.write_manifest("station_daily", {"rows": len(df), "points": int(df.Station.nunique()), "failed_months": failed})


if __name__ == "__main__":
    main()
