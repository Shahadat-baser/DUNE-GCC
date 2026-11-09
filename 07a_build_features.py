"""
STEP 7a - modelling feature tables (local only, ~10-30 min: reads the 247 weekly rasters once).

Principle: the model is TRAINED on exactly the numbers it is later APPLIED to.  The 1 km product is
made from the weekly 1 km rasters (step 05), so the station features are sampled from those same
rasters (cell value, 3x3 and 5x5 means), not rebuilt from the daily tables.  Every derived feature
(humidity, wind speed, column / boundary-layer ratio ...) is computed from the weekly means, which is
also all a raster holds.  Monthly features are day-weighted averages of the weekly ones, built the
same way for stations now and for the 1 km grid later.

Features (no latitude / longitude: the model must not memorise places)
    satellite   NO2_trop (cell, 3x3, 5x5), NO2_PBLH = NO2_trop / PBLH, NO2_valid_frac, S5P_cloud, S5P_AAI
    weather     T2m, RH (from T2m / Td2m), WS = |(U10, V10)|, U10, V10, SP, PBLH, PRECIP (mm, >= 0), MET_fill
    static      DEM, LandFrac, CoastDist_km, BUILT_frac, POP, SMOD, VIIRS, NDVI (cell and 3x3), by year
    time        season (sin / cos of day of year), S5P processor version (number, and >= 2.4 flag)

Outputs
    features/station_weekly_features.csv    every point of stations.csv x every ISO week 2022-W01 -> 2026-W39
    features/station_monthly_features.csv   every point x every month with >= 90 % of its days covered
    model/weekly_table.csv                  ground weekly targets (train / test / other) + features
    model/monthly_table.csv                 ground monthly targets + features
    logs/features_report.txt                coverage, missing values, and feature-target correlation
                                            on TRAINING rows only (2022-2023) - test rows are not looked at
The sealed forward targets (tables/forward/) are NOT read here.
"""
import os, sys, glob, re
CODE_DIR = os.getcwd()                      # fallback when __file__ is undefined (notebook)
try:
    CODE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    if os.path.exists(os.path.join(os.getcwd(), "dune_config.py")):
        CODE_DIR = os.getcwd()
sys.path.insert(0, CODE_DIR)
for _m in ("dune_config", "dune_env", "dune_gee", "dune_ground"):
    sys.modules.pop(_m, None)
import dune_env
dune_env.ensure("features")
import datetime as dt
import numpy as np, pandas as pd, rasterio
from rasterio.windows import Window
import dune_config as C

OUT = []
FEAT_DIR = os.path.join(C.BASE_DIR, "features")
MODEL_DIR = os.path.join(C.BASE_DIR, "model")
STATIC = ["DEM", "LandFrac", "CoastDist_km", "BUILT_frac", "POP", "SMOD", "VIIRS", "NDVI"]


def say(s=""):
    print(s)
    OUT.append(str(s))


def rh_from_t_td(t, td):
    """relative humidity (%) from temperature and dew point (K), Magnus formula over water"""
    a, b = 17.625, 243.04
    tc, tdc = t - 273.15, td - 273.15
    return 100 * np.exp(a * tdc / (b + tdc) - a * tc / (b + tc))


def proc_number(v):
    """'2.4.0' -> 2.4 ; '02.09.01' -> 2.9 ; 'none' -> NaN"""
    m = re.match(r"^\s*0*(\d+)\.0*(\d+)", str(v))
    return float(f"{int(m.group(1))}.{int(m.group(2))}") if m else np.nan


# ============================================================================ sampling the rasters
def weekly_files(folder=None):
    fs = sorted(glob.glob(os.path.join(folder or os.path.join(C.DIRS["rasters"], "weekly"), "*", "DUNE_W*.tif")))
    rows = []
    for f in fs:
        m = re.search(r"DUNE_W(\d{4})-(\d{2})_(\d{4}-\d{2}-\d{2})\.tif$", f)
        if m:
            rows.append((int(m.group(1)), int(m.group(2)), pd.Timestamp(m.group(3)), f))
    return rows


def sample_week(f, rows, cols):
    """cell value, 3x3 and 5x5 nan-means of every band at the given cells"""
    out = {}
    with rasterio.open(f) as src:
        names = list(src.descriptions)
        r0, r1 = max(int(rows.min()) - 2, 0), min(int(rows.max()) + 3, src.height)
        c0, c1 = max(int(cols.min()) - 2, 0), min(int(cols.max()) + 3, src.width)
        # points are spread over the domain: read per point (5x5 windows), cheap with tiled GeoTIFFs
        data = {n: np.full((len(rows), 5, 5), np.nan, dtype="float32") for n in names}
        for i, (r, c) in enumerate(zip(rows, cols)):
            rr0, cc0 = max(r - 2, 0), max(c - 2, 0)
            rr1, cc1 = min(r + 3, src.height), min(c + 3, src.width)
            blk = src.read(window=Window(cc0, rr0, cc1 - cc0, rr1 - rr0)).astype("float32")
            for k, n in enumerate(names):
                data[n][i, rr0 - (r - 2):rr0 - (r - 2) + blk.shape[1], cc0 - (c - 2):cc0 - (c - 2) + blk.shape[2]] = blk[k]
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for n, a in data.items():
                out[n] = a[:, 2, 2]
                out[n + "_3x3"] = np.nanmean(a[:, 1:4, 1:4].reshape(len(rows), -1), axis=1)
                out[n + "_5x5"] = np.nanmean(a.reshape(len(rows), -1), axis=1)
    return out


def derive(df):
    """features computed from weekly (or monthly) means - identical for stations and the 1 km grid"""
    df = df.copy()
    df["PRECIP"] = df.pop("PRECIP_wk").clip(lower=0) if "PRECIP_wk" in df else df["PRECIP"].clip(lower=0)
    df["RH"] = rh_from_t_td(df.T2m, df.Td2m).clip(0, 100)
    df["WS"] = np.hypot(df.U10, df.V10)
    df["NO2_PBLH"] = df.NO2_trop / df.PBLH.where(df.PBLH > 10)
    df["NO2_PBLH_3x3"] = df.NO2_trop_3x3 / df.PBLH.where(df.PBLH > 10)
    return df


# ============================================================================ main
def main():
    os.makedirs(FEAT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    st = pd.read_csv(C.STATIONS_CSV)
    wk = weekly_files()
    if not wk:
        raise SystemExit("no weekly rasters found - run 05_weekly_rasters.py")
    say(f"=== Feature tables (step 07a)\n   points {len(st)} | weekly rasters {len(wk)} "
        f"({wk[0][0]}-W{wk[0][1]:02d} -> {wk[-1][0]}-W{wk[-1][1]:02d})")
    # transfer tests (box rasters from step 05b) are sampled only for the monitors of each test
    jobs = [(y, w, m, f, st, "main") for y, w, m, f in wk]
    for key, t in getattr(C, "TRANSFER_TESTS", {}).items():
        tf = weekly_files(t["raster_dir"])
        pts = st[st.Network.isin(t["networks"])]
        jobs += [(y, w, m, f, pts, key) for y, w, m, f in tf]
        say(f"   {key} ({t['title']}) rasters: {len(tf)} weeks for {len(pts)} monitors"
            + ("" if tf else "  [none - run 05b_weekly_rasters_tests.py]"))

    # ---------------- 1. weekly features at every point, sampled from the rasters
    keep = ["NO2_trop", "S5P_cloud", "S5P_AAI", "T2m", "Td2m", "U10", "V10", "SP", "PBLH", "MET_fill", "PRECIP_wk", "NO2_ndays"]
    frames = []
    for k, (y, w, monday, f, pts, rset) in enumerate(jobs):
        s = sample_week(f, pts.Row.values.astype(int), pts.Col.values.astype(int))
        d = pd.DataFrame({"Station": pts.Station.values, "iso_year": y, "iso_week": w, "week_start": monday,
                          "raster_set": rset})
        for b in keep:
            d[b] = s[b]
        for b in ("NO2_trop", "S5P_cloud", "S5P_AAI"):
            d[b + "_3x3"], d[b + "_5x5"] = s[b + "_3x3"], s[b + "_5x5"]
        frames.append(d)
        if (k + 1) % 25 == 0 or k + 1 == len(jobs):
            print(f"   sampled {k + 1}/{len(jobs)} raster weeks")
    W = pd.concat(frames, ignore_index=True)
    dup = W.duplicated(["Station", "week_start"]).sum()
    if dup:
        raise SystemExit(f"{dup} station-weeks sampled twice (test and main rasters overlap?) - check RASTER_YEARS / TRANSFER_TESTS years")
    W["NO2_valid_frac"] = W.NO2_ndays / 7.0
    W = derive(W)
    mid = W.week_start + pd.Timedelta(days=3)
    W["doy_sin"], W["doy_cos"] = np.sin(2 * np.pi * mid.dt.dayofyear / 365.25), np.cos(2 * np.pi * mid.dt.dayofyear / 365.25)

    # S5P processor version per week (majority of the days; a day property, the same for every point)
    days = pd.read_csv(os.path.join(C.DIRS["tables"], C.STATION_DAILY_CSV), usecols=["date", "S5P_proc"]).drop_duplicates("date")
    days["date"] = pd.to_datetime(days.date)
    days["proc"] = days.S5P_proc.map(proc_number)
    days["week_start"] = days.date - pd.to_timedelta(days.date.dt.weekday, unit="D")
    pw = days.groupby("week_start").proc.agg(lambda x: x.mode().iloc[0] if x.notna().any() else np.nan)
    W["S5P_proc"] = W.week_start.map(pw)
    W["S5P_proc_ge24"] = (W.S5P_proc >= 2.4).astype(float)

    # static layers by year (2026 rows were carried forward from 2025 in step 02)
    S = pd.read_csv(os.path.join(C.DIRS["tables"], "Station_Static_by_Year.csv"))
    scol = [c for c in S.columns if any(c.startswith(p + "_") for p in STATIC) and c.endswith(("_px", "_3x3"))]
    W = W.merge(S[["Station", "Year"] + scol].rename(columns={"Year": "iso_year"}), on=["Station", "iso_year"], how="left")
    W.to_csv(os.path.join(FEAT_DIR, "station_weekly_features.csv"), index=False)
    say(f"   weekly features: {len(W):,} rows ({W.Station.nunique()} points x {W.week_start.nunique()} weeks)")

    # ---------------- 2. monthly features = day-weighted averages of the weekly ones
    expl = []
    for wsd in W.week_start.unique():
        ds = pd.date_range(wsd, periods=7, freq="D")
        for mth, n in pd.Series(ds.to_period("M")).value_counts().items():
            expl.append((wsd, mth.to_timestamp(), n))
    E = pd.DataFrame(expl, columns=["week_start", "month", "days"])
    M = W.merge(E, on="week_start")
    meanc = [c for c in W.columns if c not in ("Station", "iso_year", "iso_week", "week_start", "NO2_ndays", "PRECIP",
                                               "NO2_valid_frac", "S5P_proc", "S5P_proc_ge24", "doy_sin", "doy_cos")
             and pd.api.types.is_numeric_dtype(W[c]) and not c.startswith(tuple(STATIC))]
    no2c = [c for c in meanc if c.startswith(("NO2_trop", "NO2_PBLH", "S5P_cloud", "S5P_AAI"))]
    metc = [c for c in meanc if c not in no2c]
    M["w_day"] = M.days                                     # weather: weight = days of the week in the month
    M["w_no2"] = M.days * M.NO2_ndays / 7.0                 # satellite: weight = valid days in the month (approx.)
    M["PRECIP_part"] = M.PRECIP * M.days / 7.0
    for c in metc + no2c:                                   # weighted sums, counting weights only where c exists
        wcol = "w_day" if c in metc else "w_no2"
        ok = M[c].notna()
        M[c + "__ws"] = (M[c] * M[wcol]).where(ok, 0.0)
        M[c + "__w"] = M[wcol].where(ok, 0.0)
    g = M.groupby(["Station", "month"])
    sums = g[[c + "__ws" for c in metc + no2c] + [c + "__w" for c in metc + no2c] + ["days", "w_no2", "PRECIP_part"]].sum()
    agg = pd.DataFrame({"days_covered": sums.days, "NO2_ndays": sums.w_no2, "PRECIP": sums.PRECIP_part,
                        "raster_set": g.raster_set.first()})
    for c in metc + no2c:
        agg[c] = (sums[c + "__ws"] / sums[c + "__w"]).where(sums[c + "__w"] > 0)
    agg = agg.reset_index()
    agg["days_in_month"] = agg.month.dt.days_in_month
    agg = agg[agg.days_covered >= 0.9 * agg.days_in_month].copy()
    agg["NO2_valid_frac"] = agg.NO2_ndays / agg.days_covered
    agg["RH"] = rh_from_t_td(agg.T2m, agg.Td2m).clip(0, 100)        # recomputed from monthly means
    agg["WS"] = np.hypot(agg.U10, agg.V10)
    agg["NO2_PBLH"] = agg.NO2_trop / agg.PBLH.where(agg.PBLH > 10)          # ratio of the monthly means, as weekly
    agg["NO2_PBLH_3x3"] = agg.NO2_trop_3x3 / agg.PBLH.where(agg.PBLH > 10)
    mid = agg.month + pd.Timedelta(days=14)
    agg["doy_sin"], agg["doy_cos"] = np.sin(2 * np.pi * mid.dt.dayofyear / 365.25), np.cos(2 * np.pi * mid.dt.dayofyear / 365.25)
    days["month"] = days.date.dt.to_period("M").dt.to_timestamp()
    agg["S5P_proc"] = agg.month.map(days.groupby("month").proc.agg(lambda x: x.mode().iloc[0] if x.notna().any() else np.nan))
    agg["S5P_proc_ge24"] = (agg.S5P_proc >= 2.4).astype(float)
    agg["Year"] = agg.month.dt.year
    agg = agg.merge(S[["Station", "Year"] + scol], on=["Station", "Year"], how="left")
    agg.to_csv(os.path.join(FEAT_DIR, "station_monthly_features.csv"), index=False)
    say(f"   monthly features: {len(agg):,} rows ({agg.month.nunique()} months with >= 90 % of days covered)")

    # ---------------- 3. join the ground targets (forward targets are sealed and NOT read)
    sites = pd.read_csv(os.path.join(C.DIRS["tables"], "ground_sites.csv"))
    anchor = sites.set_index("Target_ID")[["Anchor_Station", "Country", "Network", "Use"]]
    gw = pd.read_csv(os.path.join(C.DIRS["tables"], "ground_weekly.csv"), parse_dates=["week_start"])
    gm = pd.read_csv(os.path.join(C.DIRS["tables"], "ground_monthly.csv"), parse_dates=["month"])
    assert not (gw.period == "forward").any() and not (gm.period == "forward").any(), "forward rows in main tables"
    gw = gw.join(anchor, on="Target_ID", rsuffix="_site")
    gm = gm.join(anchor, on="Target_ID", rsuffix="_site")
    TW = gw.rename(columns={"NO2": "y"}).merge(
        W.drop(columns=["iso_year", "iso_week"]).rename(columns={"Station": "Anchor_Station"}),
        on=["Anchor_Station", "week_start"], how="inner")
    TM = gm.rename(columns={"NO2": "y"}).merge(
        agg.rename(columns={"Station": "Anchor_Station"}), on=["Anchor_Station", "month"], how="inner")
    TW.to_csv(os.path.join(MODEL_DIR, "weekly_table.csv"), index=False)
    TM.to_csv(os.path.join(MODEL_DIR, "monthly_table.csv"), index=False)
    say(f"   weekly table {len(TW):,} rows | monthly table {len(TM):,} rows")
    for name, g_, T, tcol in (("weekly", gw, TW, "week_start"), ("monthly", gm, TM, "month")):
        keyc = ["Target_ID", tcol]
        miss = g_.merge(T[keyc], on=keyc, how="left", indicator=True)
        miss = miss[(miss["_merge"] == "left_only") & (miss[tcol].dt.year >= 2019)]
        if len(miss):
            say(f"   {name} targets since 2019 WITHOUT features (no raster covers them), by country and year:")
            say("      " + miss.pivot_table(index="Country", columns=miss[tcol].dt.year, values="period",
                                            aggfunc="size", fill_value=0).to_string().replace("\n", "\n      "))
        else:
            say(f"   {name}: every target since 2019 has features")

    # ---------------- 4. report
    feats = [c for c in TW.columns if c not in gw.columns and c not in ("y", "Anchor_Station", "NO2_ndays", "raster_set")]
    for name, T, tcol in (("WEEKLY", TW, "week_start"), ("MONTHLY", TM, "month")):
        say(f"\n=== {name} table: rows by period and country (NO2 targets; Bahrain NOx listed separately)")
        x = T.assign(Country=np.where(T.pollutant == "NOx_reported", T.Country + " (NOx)", T.Country))
        say(x.pivot_table(index="Country", columns="period", values="y", aggfunc="size", fill_value=0).to_string())
        say(f"   sites per period: " + str(T.groupby("period").Site_ID.nunique().to_dict()))
        tr = T[(T.period == "train") & (T.pollutant == "NO2")]
        if len(tr) > 10:
            fc = [c for c in feats if c in tr.columns]
            miss = tr[fc].isna().mean().sort_values(ascending=False)
            say(f"   missing share on TRAIN rows (top 6): " + ", ".join(f"{k} {v:.0%}" for k, v in miss.head(6).items()))
            sp = tr[fc + ["y"]].corr(method="spearman")["y"].drop("y").dropna()
            sp = sp.reindex(sp.abs().sort_values(ascending=False).index)
            say("   Spearman correlation with ground NO2, TRAIN rows only (top 15):")
            for k, v in sp.head(15).items():
                say(f"      {k:<22} {v:+.2f}")
            say(f"   train target: n {len(tr)} | mean {tr.y.mean():.1f} | sd {tr.y.std():.1f} | range {tr.y.min():.1f}-{tr.y.max():.1f} ug/m3")
    for key, t in getattr(C, "TRANSFER_TESTS", {}).items():
        for name, T, tcol in (("weekly", TW, "week_start"), ("monthly", TM, "month")):
            tr_ = T[(T.raster_set == key) & (T.pollutant == "NO2")]
            lock = tr_[tcol].between(pd.Timestamp(t["lockdown"][0]), pd.Timestamp(t["lockdown"][1]))
            say(f"   test {key} ({t['title']}) {name} rows: {len(tr_)} at {tr_.Site_ID.nunique()} sites "
                f"({int(lock.sum())} inside the COVID restrictions)")
    for name, T in (("weekly", TW), ("monthly", TM)):
        say(f"\n   S5P processor by raster set and period ({name} table): " +
            str(T.groupby(["raster_set", "period"]).S5P_proc.agg(lambda x: sorted(x.dropna().unique().tolist())).to_dict()))
    p = os.path.join(C.DIRS["logs"], "features_report.txt")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("\n".join(OUT))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
