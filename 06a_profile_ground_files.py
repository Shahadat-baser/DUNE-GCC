"""
STEP 6a - profile every ground-observation file (local only, a few seconds).

Nothing is changed or combined here.  The ground-truth step (06b) will convert units, aggregate
hourly/daily/monthly values and apply quality control, and every one of those rules must come from
what the files actually contain - not from assumptions such as "Riyadh values are in ppb".  For each
CSV / Excel file this prints: columns and types, every column that looks like a unit, pollutant,
frequency or indicator label with its distinct values, the date range, stations with coordinates,
and NO2 value statistics.

Files examined: every .csv / .xlsx directly in BASE_DIR (D:\\) and in OLD_DIR's
ground files, plus OLD_DIR\\OpenAQ. 
"""
import os, sys, glob
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
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import dune_config as C

OUT = []
SKIP = {"stations.csv", "pip_constraints.txt"}
OLD_GROUND = ["Oman_Ground_Observations_Master_2023_2024.csv", "UAE_Ground_Observations_Final.csv",
              "Riyadh_Ground_Observations_NO2_SO2.csv", "Bahrain_Ground_NO2_Monthly_2021_2025.csv",
              os.path.join("OpenAQ", "OpenAQ_NO2_Ground_Truth_Master.csv"),
              "OpenAQ_v3_GCC_NO2_daily.csv", "OpenAQ_v3_GCC_NO2_stations.csv"]
LABEL_HINTS = ["unit", "component", "pollutant", "parameter", "indicator", "periodicity", "freq", "type",
               "country", "network", "city", "averag"]


def say(s=""):
    print(s)
    OUT.append(str(s))


def files():
    fs = [f for f in glob.glob(os.path.join(C.BASE_DIR, "*")) if f.lower().endswith((".csv", ".xlsx", ".xls"))]
    for f in OLD_GROUND:
        p = os.path.join(C.OLD_DIR, f)
        hits = [p] if os.path.exists(p) else glob.glob(os.path.join(C.OLD_DIR, "**", os.path.basename(f)), recursive=True)
        fs += hits[:1]
    return [f for f in dict.fromkeys(fs) if os.path.basename(f) not in SKIP]


def read(f):
    if f.lower().endswith(".csv"):
        for enc in ("utf-8-sig", "utf-8", "cp1256", "latin-1"):
            try:
                return pd.read_csv(f, encoding=enc, low_memory=False), enc
            except UnicodeDecodeError:
                continue
    return pd.read_excel(f), "excel"


def profile(f):
    say("=" * 100)
    say(f"FILE {f}  ({os.path.getsize(f) / 1e6:.1f} MB)")
    try:
        df, enc = read(f)
    except Exception as exc:
        say(f"   cannot read: {exc}")
        return
    say(f"   rows {len(df):,} | encoding {enc}")
    say(f"   columns: " + ", ".join(f"{c} [{t}]" for c, t in df.dtypes.astype(str).items()))
    for c in df.columns:
        lc = str(c).lower()
        if not pd.api.types.is_numeric_dtype(df[c]) and (any(h in lc for h in LABEL_HINTS) or df[c].nunique() <= 12):
            vc = df[c].value_counts(dropna=False)
            if len(vc) <= 40:
                say(f"   {c}: " + " | ".join(f"{k} ({v:,})" for k, v in vc.items()))
            else:
                say(f"   {c}: {len(vc)} distinct, e.g. {list(vc.index[:8])}")
    for c in df.columns:
        if any(k in str(c).lower() for k in ["date", "time", "timestamp", "year", "month", "day"]):
            s = pd.to_datetime(df[c], errors="coerce") if not pd.api.types.is_numeric_dtype(df[c]) else None
            if s is not None and s.notna().mean() > 0.5:
                say(f"   {c}: {s.min()} -> {s.max()} | parsed {s.notna().mean():.0%} | "
                    f"distinct days {s.dt.floor('D').nunique():,} | example raw values {list(df[c].dropna().astype(str).unique()[:3])}")
            elif pd.api.types.is_numeric_dtype(df[c]):
                say(f"   {c}: {df[c].min()} -> {df[c].max()}")
    st = next((c for c in df.columns if str(c).lower() in ("station", "station name", "station_name", "site", "location")), None)
    lat = next((c for c in df.columns if str(c).lower() in ("lat", "latitude")), None)
    lon = next((c for c in df.columns if str(c).lower() in ("lon", "lng", "longitude")), None)
    if st:
        g = df.groupby(st)
        info = g.size().rename("rows").to_frame()
        if lat and lon:
            info["Lat"], info["Lon"] = g[lat].first(), g[lon].first()
        say(f"   stations ({df[st].nunique()}):")
        for line in info.to_string().splitlines():
            say("      " + line)
    for c in df.columns:
        lc = str(c).lower()
        if ("no2" in lc or lc in ("value", "concentration", "nox", "conc")) and pd.api.types.is_numeric_dtype(df[c]):
            q = df[c].describe(percentiles=[.01, .5, .99])
            say(f"   {c}: n {int(q['count']):,} | min {q['min']:.2f} | p1 {q['1%']:.2f} | median {q['50%']:.2f} | "
                f"p99 {q['99%']:.2f} | max {q['max']:.2f} | negative {(df[c] < 0).sum():,} | zero {(df[c] == 0).sum():,}")
    say("   first 3 rows:")
    for line in df.head(3).to_string().splitlines():
        say("      " + line)


def main():
    os.makedirs(C.DIRS["logs"], exist_ok=True)
    fs = files()
    say(f"{len(fs)} files")
    for f in fs:
        profile(f)
    p = os.path.join(C.DIRS["logs"], "ground_profile.txt")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("\n".join(OUT))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
