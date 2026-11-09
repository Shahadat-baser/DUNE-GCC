"""
STEP 4 - daily dynamic predictors averaged over each of the 744 districts, 2019-2025.

Area mean of the 0.01 deg cells whose centre lies inside the district (Earth Engine reduceRegions
on the fixed grid).  NO2_valid is unmasked 0/1, so its mean is the FRACTION of the district observed
by Sentinel-5P that day; keep it, the weekly/monthly aggregation weights by it.  Districts with no
cell centre inside them return empty rows here and are covered by their point sample from STEP 3
(Kind = district_point).

Fetched week by week to tables/district_daily/, resumable, auto-splitting; then joined into
tables/<DISTRICT_DAILY_CSV> (District_Daily_2019_2026.csv after the extension, ~2.1 million rows).

Weekly and monthly series are built LOCALLY from this daily table in the next step, with the same
code for stations and districts, so one aggregation rule governs the whole paper.
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
dune_env.ensure("district_daily")      # installs / checks every package this step needs
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import ee
import dune_config as C
import dune_gee as G
if getattr(G, "PIPELINE_VERSION", None) != "2026-10-08a":
    raise SystemExit(f"{G.__file__} is an older version than this script. Unzip the WHOLE DUNE_GEE_v3.zip "
                     "again into the code folder and choose 'Replace the files in the destination'.")

OUT = os.path.join(C.DIRS["tables"], "district_daily")
BANDS = G.ALL_DAILY_BANDS


def build(districts):
    def fc_for(start, end):
        def per_day(img):
            fc = img.reduceRegions(collection=districts, reducer=ee.Reducer.mean(),
                                   crs=C.CRS, crsTransform=C.CRS_TRANSFORM, tileScale=4)
            props = img.toDictionary(G.DAY_PROPS)
            return fc.map(lambda f: f.set(props))
        return G.daily_collection(start, end).map(per_day).flatten()
    return fc_for


def main():
    G.init()
    os.makedirs(OUT, exist_ok=True)
    fc_for = build(G.districts_fc())
    sel = [C.DISTRICT_ID] + G.DAY_PROPS + BANDS
    chunks = G.date_chunks(C.DATE_START, C.DATE_END, C.DISTRICT_CHUNK_DAYS)
    G.log(f"district daily: {len(chunks)} chunks of {C.DISTRICT_CHUNK_DAYS} days, {C.N_THREADS} threads", "district_daily")

    def job(ch):
        s, e = ch
        r = G.fetch_range(fc_for, s, e, os.path.join(OUT, f"district_daily_{s}_{e}.csv"), sel, "district_daily")
        G.log(f"{s}..{e}: {r}", "district_daily")
        return r

    with ThreadPoolExecutor(C.N_THREADS) as ex:
        res = list(ex.map(job, chunks))

    files = sorted(f for f in glob.glob(os.path.join(OUT, "district_daily_*.csv")) if "__" not in f)
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    dup = df[df.duplicated([C.DISTRICT_ID, "date"], keep=False)]
    if len(dup):        # e.g. the last 2025 chunk, re-fetched with the extended period: must be identical
        import numpy as np
        num = [c for c in BANDS if c in dup]
        g = dup.groupby([C.DISTRICT_ID, "date"])[num].agg(lambda x: np.nanmax(x) - np.nanmin(x) if x.notna().any() else 0)
        G.log(f"rows fetched twice: {len(dup) // 2:,} | max difference between fetches {float(np.nanmax(g.values)):.3g}",
              "district_daily")
    df = df.sort_values([C.DISTRICT_ID, "date"], kind="stable").drop_duplicates([C.DISTRICT_ID, "date"])   # earliest fetch kept
    out = os.path.join(C.DIRS["tables"], C.DISTRICT_DAILY_CSV)
    df.to_csv(out, index=False)
    failed = sum("FAILED" in r for r in res)
    G.log(f"wrote {out}: {len(df):,} rows, {df[C.DISTRICT_ID].nunique()} districts, {df.date.nunique()} days | "
          f"chunks with failures: {failed}", "district_daily")
    G.write_manifest("district_daily", {"rows": len(df), "failed_chunks": failed})


if __name__ == "__main__":
    main()
