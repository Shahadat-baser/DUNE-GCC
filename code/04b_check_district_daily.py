"""
STEP 4b - quality report on the district and station daily tables (local, about 1 minute).

Checks
  * completeness: rows = districts x days, no duplicate (district, date)
  * per band: share missing, min / median / max, values outside physical limits
  * districts that are always empty (no cell centre inside -> covered by their point sample)
  * NO2 coverage (mean valid fraction) by year and by month - shows the summer / dust gaps
  * S5P processor versions by year
  * EXTENSION INTEGRITY: every row of the frozen 2019-2025 tables (written before the period was
    extended to Sep 2026) must be present, bit-identical, in the extended tables (row hashes compared)
  * weekly rasters on disk per ISO year
  * agreement between district means and station cells in the same district is NOT tested here;
    that is part of the ground-truth step.
"""
import os, sys
CODE_DIR = os.getcwd()                      # fallback when __file__ is undefined (notebook)
try:
    CODE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    if os.path.exists(os.path.join(os.getcwd(), "dune_config.py")):
        CODE_DIR = os.getcwd()
sys.path.insert(0, CODE_DIR)
for _m in ("dune_config", "dune_env", "dune_gee"):
    sys.modules.pop(_m, None)
import dune_env
dune_env.ensure("station_daily")
import numpy as np, pandas as pd
import dune_config as C

LIMITS = {"NO2_trop": (-1e-4, 3e-3), "S5P_cloud": (0, 1), "S5P_AAI": (-10, 20), "T2m": (250, 330),
          "Td2m": (200, 310), "U10": (-40, 40), "V10": (-40, 40), "SP": (70000, 106000),
          "PBLH": (5, 6000), "PRECIP": (0, 500), "NO2_valid": (0, 1), "MET_fill": (0, 1)}
OUT = []


def say(s=""):
    print(s)
    OUT.append(str(s))


def band_table(df, bands):
    rows = []
    for b in bands:
        if b not in df:
            rows.append((b, "column missing", "", "", "", ""))
            continue
        v = pd.to_numeric(df[b], errors="coerce")
        lo, hi = LIMITS.get(b, (-np.inf, np.inf))
        out = ((v < lo) | (v > hi)).sum()
        rows.append((b, f"{v.isna().mean():.1%}", f"{v.min():.4g}", f"{v.median():.4g}", f"{v.max():.4g}", f"{out:,}"))
    t = pd.DataFrame(rows, columns=["band", "missing", "min", "median", "max", "outside limits"])
    for line in t.to_string(index=False).splitlines():
        say("   " + line)


def row_hashes(df, keys, cols, numeric):
    """one uint64 per row from the value columns; numbers compared as float64 (so 3 and 3.0 are equal,
    NaN equals NaN), text as text"""
    v = pd.DataFrame({c: (pd.to_numeric(df[c], errors="coerce").astype("float64")
                          if c in numeric else df[c].astype(str)) for c in cols})
    h = pd.util.hash_pandas_object(v, index=False)
    return pd.DataFrame({**{k: df[k].astype(str).values for k in keys}, "h": h.values})


def frozen_check(new, frozen_name, keys):
    """every row of the frozen pre-extension table must exist, with identical values, in the new table"""
    p = os.path.join(C.DIRS["tables"], frozen_name)
    if not os.path.exists(p) or frozen_name in (C.STATION_DAILY_CSV, C.DISTRICT_DAILY_CSV):
        say(f"   integrity: no separate frozen table {frozen_name} - nothing to compare")
        return
    old = pd.read_csv(p, low_memory=False)
    cols = sorted((set(old.columns) & set(new.columns)) - set(keys))
    numeric = {c for c in cols if pd.api.types.is_numeric_dtype(old[c])}
    ho = row_hashes(old, keys, cols, numeric)
    n_old = len(old)
    del old
    hn = row_hashes(new[new.date.astype(str) < C.FROZEN_END], keys, cols, numeric)
    m = ho.merge(hn, on=keys, how="left", suffixes=("_old", "_new"))
    missing = int(m.h_new.isna().sum())
    changed = m[m.h_new.notna() & (m.h_old != m.h_new)]
    say(f"   integrity vs {frozen_name}: {n_old:,} earlier rows | missing {missing:,} | changed {len(changed):,} | "
        f"{'IDENTICAL' if missing == 0 and len(changed) == 0 else 'DIFFERENT - check this report'}")
    if len(changed):
        say("      first changed keys: " + "; ".join(" ".join(r) for r in changed[keys].head(5).astype(str).values.tolist()))


def main():
    T = C.DIRS["tables"]
    bands = list(LIMITS)

    say("=== District daily")
    d = pd.read_csv(os.path.join(T, C.DISTRICT_DAILY_CSV), low_memory=False)
    say(f"   file {C.DISTRICT_DAILY_CSV}")
    frozen_check(d, C.FROZEN_DAILY["district"], [C.DISTRICT_ID, "date"])
    d["date"] = pd.to_datetime(d["date"])
    nd, ndays = d[C.DISTRICT_ID].nunique(), d["date"].nunique()
    say(f"   rows {len(d):,} = {nd} districts x {ndays} days -> {'complete' if len(d) == nd * ndays else 'INCOMPLETE'} | "
        f"duplicates {d.duplicated([C.DISTRICT_ID, 'date']).sum():,}")
    band_table(d, bands)
    empty = d.groupby(C.DISTRICT_ID)["T2m"].apply(lambda s: s.notna().sum() == 0)
    say(f"   districts with no value on any day (sub-cell districts, use their point sample): {int(empty.sum())}")
    d["year"], d["month"] = d.date.dt.year, d.date.dt.month
    say("   mean NO2 valid fraction of district area, by year:")
    say("      " + " | ".join(f"{y}: {v:.2f}" for y, v in d.groupby("year").NO2_valid.mean().items()))
    say("   ... by calendar month (all years):")
    say("      " + " | ".join(f"{m:02d}: {v:.2f}" for m, v in d.groupby("month").NO2_valid.mean().items()))
    if "S5P_proc" in d:
        say("   S5P processor versions by year:")
        for y, g in d.groupby("year"):
            say(f"      {y}: {({k: int(v) for k, v in g.drop_duplicates('date').S5P_proc.value_counts().items()})}")
    days = d.drop_duplicates("date")
    say(f"   days with no S5P orbit: {(days.S5P_n_orbits == 0).sum()} | no ERA5-Land: {(days.ERA5L_n == 0).sum()} | "
        f"no ERA5: {(days.ERA5_n == 0).sum()} | no precipitation image: {(days.PRECIP_n == 0).sum()}")
    del d

    say("\n=== Station daily")
    s = pd.read_csv(os.path.join(T, C.STATION_DAILY_CSV), low_memory=False)
    say(f"   file {C.STATION_DAILY_CSV}")
    frozen_check(s, C.FROZEN_DAILY["station"], ["Station", "date"])
    pts = s.groupby("date").Station.nunique()
    say(f"   points per day: min {pts.min()} max {pts.max()} (every day must hold every point of stations.csv)")
    say(f"   rows {len(s):,} | points {s.Station.nunique()} | days {s.date.nunique()}")
    band_table(s, bands)
    s3 = [b + "_3x3" for b in bands if b + "_3x3" in s]
    say("   3x3 means:")
    band_table(s, s3)
    m = s[s.Network != "district_point"]
    say("   NO2 cell value missing, by network: " +
        " | ".join(f"{k} {v:.1%}" for k, v in m.groupby("Network").NO2_trop.apply(lambda x: x.isna().mean()).items()))
    say("   MET_fill share (ERA5 used instead of ERA5-Land), by network: " +
        " | ".join(f"{k} {v:.1%}" for k, v in m.groupby("Network").MET_fill.mean().items()))

    say("\n=== Weekly rasters on disk (ISO years)")
    wdir = os.path.join(C.DIRS["rasters"], "weekly")
    for y in C.RASTER_YEARS:
        f = os.path.join(wdir, str(y))
        say(f"   {y}: {len([x for x in os.listdir(f) if x.endswith('.tif')]) if os.path.isdir(f) else 0} files")

    p = os.path.join(C.DIRS["logs"], "daily_tables_report.txt")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("\n".join(OUT))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
