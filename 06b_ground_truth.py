"""
STEP 6b - ground-truth assembly and quality control (local only, about 1 minute).

Reads the RAW national files, converts units, applies documented QC,
aggregates to daily / weekly (ISO, Monday start, same weeks as the step-05 rasters) / monthly values,
removes duplicate reporting of the same physical site, and writes ONE value per site per period.

Inputs (names in dune_config.GROUND_FILES, searched in BASE_DIR, OLD_DIR and sub-folders)
    Oman         hourly ug/m3, timestamps hour-ENDING local time (UTC+4)  -> hourly QC -> daily
    UAE_EAD      OpenAQ v3 daily aggregates, ug/m3, with observed hour count
    UAE_monthly  monthly (and annual) ug/m3 - annual rows ignored
    KSA_Riyadh   ppb; 'Avg / Hourly' daily and monthly rows                -> x NO2_PPB_TO_UGM3
    Bahrain_SCE  monthly, labelled "Nitrogen Oxides" -> handled by BAHRAIN_MODE
    UAE_EAD_SDI  Abu Dhabi SDI hourly table (step 06c), whole-number ug/m3, Nov 2025 -> .  Same monitors
                 as UAE_EAD: a feed station joins the OpenAQ series of a monitor only if the agency names
                 correspond AND the published positions are < 3 km apart (dune_ground.py).  Time basis
                 (local vs UTC) is decided from the diurnal NO2 minimum; units are checked against the
                 OpenAQ values of the same monitors.  Stuck rule 8 h (whole numbers repeat by chance).

Rules (all values from dune_config, all counts written to the report)
  hourly   missing / negative / > GROUND_HOURLY_MAX / stuck (>= GROUND_FLATLINE_H identical consecutive
           hours) / spike (> GROUND_SPIKE_MIN and > 5x the centred 25 h median)  -> removed
  daily    mean of >= GROUND_MIN_HOURS_DAY valid hours (local calendar day); overpass mean of
           OVERPASS_HOURS_LOCAL kept for a sensitivity test (Oman only, needs >= 2 of 3 hours)
  weekly   ISO week mean of >= GROUND_MIN_DAYS_WEEK valid days
  monthly  from daily values if >= GROUND_MIN_FRAC_MONTH of the days are valid  (source 'daily');
           otherwise the agency's reported monthly mean (source 'reported')
  sites    Site_ID = one physical place, used to GROUP cross-validation folds.  Step 01 merged monitors
           < 1 km apart; here UAE monthly series (rounded coordinates) join an EAD monitor < 5 km away
           if their data agree (r >= 0.8 and median relative difference <= 15 %) or they carry the
           same station name.  Grouping errs on the side of merging: it can only make validation stricter.
  targets  Target_ID = one internally consistent series.  Inside a site, values are spliced only
           between series whose overlapping data agree; non-agreeing series (e.g. a street and a
           school monitor in one 1 km cell, or two agencies reporting different levels) become
           S019a, S019b ...  In a target-period the best source wins (daily network > reported monthly).
           Predictors are taken at each target's anchor monitor.

EAD date correction (found while writing this step)
    The OpenAQ harvest stored date = period.datetimeFrom.utc.  OpenAQ aggregates days in LOCAL time,
    and a UAE local day starts at 20:00 UTC of the previous day, so every stored date is one day
    early.  Proof in the files: Al Ain Street's first measurement is 2022-10-28T16:00Z (= 20:00 local
    28 Oct), yet its first daily row is dated 2022-10-27 with 4 hours.  Fixed by +1 day.

Forward test (FORWARD_START -> FORWARD_END, Jan-Sep 2026)
    Rows of the forward period are written ONLY to tables/forward/ground_*_forward_2026.csv, never to the
    main tables, and their SHA-256 hashes are appended to logs/forward_test_hashes.txt each run.  The
    modelling code reads the main tables for training and testing; the forward files are opened once,
    after the models are frozen.  Identical hashes across reruns show the set was not altered.

Outputs
    tables/ground_sites.csv            one row per Target_ID: Site_ID, anchor monitor, coordinates, cell, district, use
    tables/ground_daily.csv            Target_ID, Site_ID, date, NO2, NO2_ovp, n_hours, n_series, network, pollutant
    tables/ground_weekly.csv           Target_ID, Site_ID, iso_year, iso_week, week_start, NO2, NO2_ovp, n_days, ...
    tables/ground_monthly.csv          Target_ID, Site_ID, month, NO2, n_days, source, network, pollutant
    tables/ground_series_monthly.csv   every series before site combination (audit trail)
    logs/ground_qc_hourly.csv          hourly QC counts per Oman station
    logs/ground_uae_dedup.csv          every UAE-monthly vs EAD comparison and the grouping decision
    logs/ground_site_agreement.csv     every same-site pair: splice or keep apart
    logs/ground_truth_report.txt       the report
"""
import os, sys, glob, hashlib
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
dune_env.ensure("ground_truth")
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np, pandas as pd, geopandas as gpd
import dune_config as C
import dune_ground

OUT = []
# lower number = preferred source for the same site and period
PRIORITY = {"UAE_EAD": 1, "UAE_EAD_SDI": 1, "Oman": 1, "KSA_Riyadh": 2, "UAE_monthly": 3, "Bahrain_SCE": 4}
UAE_BOX = (51.0, 22.0, 57.0, 26.5)      # lon/lat box used to drop the stray non-GCC OpenAQ location
# UAE monthly code -> EAD monitor it PROBABLY is.  Only a cross-check printed beside the data-based
# decision; used to decide on its own only if there is no overlapping data and the two are < 5 km apart.
NAME_HINTS = {"AIS": "Al Ain Islamic Institute", "ALAINST": "Al Ain Street", "MAFRAQ": "Al Mafraq",
              "ALMAQTA": "Al Maqtaa", "QUAA": "Al Qua'a", "TAWIA": "Al Tawia", "BANSCHOOL": "Baniyas School",
              "BZYD": "Bida Zayed", "WST": "Bida Zayed", "GTSCHOOL": "Gayathi", "HABSHAN": "Habshan South",
              "HAMD": "Hamdan Street", "HMSTREET": "Hamdan Street", "KJASCHOOL": "Khadeeja School",
              "KCA": "Khalifa City A", "LWO": "Liwa", "MUSSAFAH": "Mussafah", "RUWAIS": "Ruwais",
              "SWEIHAN": "Sweihan", "ZAKHER": "Zakher", "KHLSCHOOL": "Khalifa School"}
# KHLSCHOOL is Khalifa School, an EAD monitor that OpenAQ never carried (added from the SDI feed)


def say(s=""):
    print(s)
    OUT.append(str(s))


def table(df, indent="   "):
    for line in df.to_string().splitlines():
        say(indent + line)


def find(name):
    for root in (C.BASE_DIR, C.OLD_DIR):
        p = os.path.join(root, name)
        if os.path.exists(p):
            return p
    for root in (C.BASE_DIR, C.OLD_DIR):
        hits = glob.glob(os.path.join(root, "**", name), recursive=True)
        if hits:
            return hits[0]
    return None


def read(key):
    p = find(C.GROUND_FILES[key])
    if p is None:
        say(f"   [missing] {C.GROUND_FILES[key]} - {key} skipped")
        return None
    for enc in ("utf-8-sig", "utf-8", "cp1256", "latin-1"):
        try:
            df = pd.read_csv(p, encoding=enc, low_memory=False)
            say(f"   {key:<14} <- {p}  ({len(df):,} rows)")
            return df
        except UnicodeDecodeError:
            continue
    raise SystemExit(f"cannot decode {p}")


def km(lat1, lon1, lat2, lon2):
    r = np.pi / 180
    a = np.sin((lat2 - lat1) * r / 2) ** 2 + np.cos(lat1 * r) * np.cos(lat2 * r) * np.sin((lon2 - lon1) * r / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(a))


def month_start(s):
    return pd.to_datetime(s).dt.to_period("M").dt.to_timestamp()


# =============================================================================== readers
def hourly_qc(h, flat_h, label, quiet=False):
    """h: Station (stations.csv key), t (hour-beginning, local), NO2.  Returns cleaned hourly rows and a
    QC table.  Flags: missing / negative / above_max / stuck (>= flat_h identical consecutive hours) /
    spike (> GROUND_SPIKE_MIN and > 5x the centred 25 h median of the remaining values)."""
    qc_rows, keep = [], []
    for st, g in h.groupby("Station"):
        g = g.sort_values("t")
        ndup = int(g.duplicated("t").sum())
        g = g.drop_duplicates("t", keep="first")
        s = pd.Series(pd.to_numeric(g.NO2, errors="coerce").values, index=pd.DatetimeIndex(g.t.values))
        flag = pd.Series("ok", index=s.index)
        flag[s.isna()] = "missing"
        flag[(flag == "ok") & (s < 0)] = "negative"
        flag[(flag == "ok") & (s > C.GROUND_HOURLY_MAX)] = "above_max"
        gap_h = s.index.to_series().diff().dt.total_seconds().div(3600).values
        same = (np.r_[np.nan, np.diff(s.values)] == 0) & (gap_h == 1)
        grp = np.cumsum(~same)
        run = pd.Series(grp).map(pd.Series(grp).value_counts()).values
        flag[(flag == "ok") & (run >= flat_h)] = "stuck"
        clean = s.where(flag == "ok")
        med = clean.rolling("25h", center=True, min_periods=6).median()
        flag[(flag == "ok") & (s > C.GROUND_SPIKE_MIN) & (s > 5 * med)] = "spike"
        c = flag.value_counts()
        bad = (flag != "ok") & s.notna()
        qc_rows.append(dict(Station=st, hours=len(s), duplicate_stamps=ndup, zeros=int((s == 0).sum()),
                            **{k: int(c.get(k, 0)) for k in ["ok", "missing", "negative", "above_max", "stuck", "spike"]},
                            max_removed=float(s[bad].max()) if bad.any() else np.nan,
                            stuck_median_value=float(s[flag == "stuck"].median()) if (flag == "stuck").any() else np.nan,
                            stuck_share_of_zero=float((s[flag == "stuck"] == 0).mean()) if (flag == "stuck").any() else np.nan))
        keep.append(pd.DataFrame({"Station": st, "t": s.index, "NO2": s.where(flag == "ok").values}))
    out = pd.concat(keep, ignore_index=True)
    qc = pd.DataFrame(qc_rows).set_index("Station")
    if not quiet:
        say(f"\n   {label} hourly QC (hours per flag; stuck = >= {flat_h} identical consecutive hours):")
        table(qc)
    return out, qc


STUCK_LEVELS = [-np.inf, 5, 15, 40, np.inf]   # ug/m3 classes: equal neighbours are likelier at low levels


def constant_runs(h):
    """every run of identical consecutive hourly values: Station, length (h), value"""
    out = []
    for st, g in h.groupby("Station"):
        g = g.sort_values("t").drop_duplicates("t")
        v = pd.to_numeric(g.NO2, errors="coerce").values
        consec = np.r_[False, np.diff(g.t.values) == np.timedelta64(1, "h")]
        same = np.r_[False, np.diff(v) == 0] & consec
        r = pd.DataFrame({"g": np.cumsum(~same), "v": v}).groupby("g").agg(length=("v", "size"), value=("v", "first"))
        r = r[r.value.notna()]
        r["Station"] = st
        out.append(r)
    return pd.concat(out, ignore_index=True)


def stuck_threshold(h, label, lengths=range(3, 25)):
    """Observed constant runs vs the number expected from rounding alone.
    Model: within a station and concentration class, the chance that the next hour repeats the value is
    p = N(runs >= 2 h) / N(runs >= 1 h), so runs of >= L hours are expected N(>=1) * p^(L-1) times
    (geometric run lengths).  Real persistence (calm nights) makes long runs somewhat likelier than this,
    so the chosen L is checked with the daily-mean sensitivity run."""
    R = constant_runs(h)
    R["level"] = pd.cut(R.value, STUCK_LEVELS)
    g = R.groupby(["Station", "level"], observed=True).length
    stats = pd.DataFrame({"N1": g.size(), "N2": g.apply(lambda x: int((x >= 2).sum()))})
    stats["p"] = stats.N2 / stats.N1
    n_valid = int(h.NO2.notna().sum())
    rows = []
    for L in lengths:
        o = R.length >= L
        rows.append(dict(L=L, observed_runs=int(o.sum()), expected_by_rounding=float((stats.N1 * stats.p ** (L - 1)).sum()),
                         hours_in_runs=int(R.length[o].sum()), share_of_hours=R.length[o].sum() / max(n_valid, 1)))
    T = pd.DataFrame(rows)
    T["false_share"] = (T.expected_by_rounding / T.observed_runs.replace(0, np.nan)).clip(upper=9.99)
    ok = (T.false_share <= C.STUCK_MAX_FALSE_SHARE) | ((T.observed_runs == 0) & (T.expected_by_rounding < 0.5))
    good = [L for L in T.L if ok[T.L >= L].all()]
    chosen = min(good) if good else C.STUCK_FALLBACK_H
    say(f"\n   {label}: constant runs observed vs expected from rounding (p by station x level "
        f"{[str(x) for x in stats.reset_index().level.cat.categories]}):")
    table(T.set_index("L").round({"expected_by_rounding": 1, "share_of_hours": 4, "false_share": 3}).head(14))
    say(f"   -> data-supported stuck threshold {chosen} h: shortest run length at which rounding explains <= "
        f"{C.STUCK_MAX_FALSE_SHARE:.0%} of the observed runs at that and every longer length"
        + ("" if good else f"  [no length qualified - fallback {C.STUCK_FALLBACK_H} h]"))
    return chosen, T


def diurnal_min_hours(h):
    """hour (0-23 of the stamps) of the lowest mean NO2, per station"""
    x = h.dropna(subset=["NO2"])
    x = x[x.NO2 <= C.GROUND_HOURLY_MAX]
    prof = x.groupby([x.Station, x.t.dt.hour]).NO2.mean().unstack()
    return prof.idxmin(axis=1).astype(int), prof.idxmax(axis=1).astype(int)


def hourly_to_daily(h, network, label, quiet=False):
    h = h.copy()
    h["hour"], h["date"] = h.t.dt.hour, h.t.dt.floor("D")
    valid = h.dropna(subset=["NO2"])
    d = valid.groupby(["Station", "date"]).NO2.agg(["mean", "size"]).rename(columns={"mean": "NO2", "size": "n_hours"})
    ov = valid[valid.hour.isin(C.OVERPASS_HOURS_LOCAL)].groupby(["Station", "date"]).NO2.agg(["mean", "size"])
    d["NO2_ovp"] = ov["mean"].where(ov["size"] >= 2)
    d = d.reset_index()
    n0 = len(d)
    d = d[d.n_hours >= C.GROUND_MIN_HOURS_DAY].copy()
    if not quiet:
        say(f"   {label} days with any valid hour {n0:,} -> {len(d):,} with >= {C.GROUND_MIN_HOURS_DAY} h")
    d["network"] = network
    return d


def read_oman():
    df = read("Oman")
    if df is None:
        return None, None
    # hour-ENDING local stamps (01:00 ... 24:00 written as next day 00:00) -> hour-beginning
    h = pd.DataFrame({"Station": "Oman|" + df.Station.astype(str),
                      "t": pd.to_datetime(df.Timestamp) - pd.Timedelta(hours=1), "NO2": df.NO2})
    stuck_threshold(h, "Oman (decimals, reference only - fixed rule GROUND_FLATLINE_H)")
    h, qc = hourly_qc(h, C.GROUND_FLATLINE_H, "Oman")
    qc.to_csv(os.path.join(C.DIRS["logs"], "ground_qc_hourly.csv"))
    # timestamp check: surface NO2 is LOWEST around 13-16 local (deep mixed layer).  If the minimum
    # sat ~4 h earlier, the stamps would be UTC, not local.
    lo, hi = diurnal_min_hours(h)
    say("   diurnal check (local hour-beginning): hour of minimum / maximum mean NO2")
    for st in lo.index:
        say(f"      {st:<16} min {lo[st]:02d}h  max {hi[st]:02d}h  -> "
            f"{'consistent with LOCAL time' if 11 <= lo[st] <= 17 else 'CHECK: looks shifted'}")
    return hourly_to_daily(h, "Oman", "Oman"), qc


def parse_hour(x):
    """'11 AM' / '12 AM' / '3 PM' / '14' -> 0..23"""
    m = pd.Series(x, dtype="object").astype(str).str.extract(r"(\d{1,2})\s*([AaPp][Mm])?")
    h = pd.to_numeric(m[0], errors="coerce")
    ap = m[1].str.upper()
    h = np.where(ap == "AM", h % 12, np.where(ap == "PM", h % 12 + 12, h))
    return pd.Series(h, index=getattr(x, "index", None))


def read_sdi(st):
    """Abu Dhabi SDI hourly table (step 06c) -> daily values under the SAME stations.csv keys as the
    OpenAQ series of the same monitors (one physical monitor = one series, whatever the feed)."""
    p = os.path.join(C.SDI_DIR, "EAD_SDI_hourly_raw.csv")
    if not os.path.exists(p):
        say("   UAE_EAD_SDI: no download (run 06c) - skipped")
        return None
    raw = pd.read_csv(p, low_memory=False, encoding="utf-8-sig")
    n0 = len(raw)
    raw = raw.drop_duplicates("OBJECTID")
    say(f"   UAE_EAD_SDI    <- {p}  ({n0:,} rows, {n0 - len(raw)} repeated OBJECTIDs dropped)")
    t = pd.to_datetime(raw.ObservedAt_iso)
    if "Hour" in raw:
        hr = parse_hour(raw.Hour)
        say(f"   'Hour' text equals the hour of ObservedAt in {(hr.values == t.dt.hour.values).mean():.1%} of rows "
            f"(examples {raw.Hour.dropna().unique()[:4].tolist()})")
    h = pd.DataFrame({"name": raw.StationName.astype(str), "t": t, "NO2": pd.to_numeric(raw.NO2, errors="coerce")})

    # ---- time basis: the SDI serves the time as a date field; decide whether it is local or UTC
    lo, _ = diurnal_min_hours(h.rename(columns={"name": "Station"}))
    med = float(np.median(lo.values))
    basis = C.SDI_TIME_BASIS
    if basis == "auto":
        if med >= 13:
            basis = "local"
        elif med <= 11:
            basis = "utc"
        else:
            raise SystemExit(f"SDI time basis undecidable (median NO2 minimum at {med:.1f} h) - set SDI_TIME_BASIS "
                             f"in dune_config.py and check the report")
    say(f"   SDI time basis: NO2 minimum at {med:.1f} h of the served stamps (per station {sorted(lo.values.tolist())}) "
        f"-> stamps are {basis.upper()} time{' (+4 h applied)' if basis == 'utc' else ''}; taken as hour-beginning")
    if basis == "utc":
        h["t"] = h.t + pd.Timedelta(hours=4)

    # ---- station identity
    sdi = dune_ground.sdi_stations(C.SDI_DIR)
    mp = dune_ground.map_sdi(sdi, st[st.Network == "UAE_EAD"])
    say("   SDI stations -> stations.csv:")
    table(mp[["sdi_key", "Station", "decision", "km"]].set_index("sdi_key"), "      ")
    key = dict(zip(mp.sdi_key, mp.Station))
    h["Station"] = h.name.map(dune_ground.norm).map(key)
    unknown = sorted(h[h.Station.isna()].name.unique())
    missing = sorted(set(h.Station.dropna()) - set(st.Station))
    if unknown:
        say(f"   [warning] SDI names without a station-layer entry (dropped): {unknown}")
    if missing:
        say(f"   [ACTION] SDI monitors not yet in stations.csv: {missing} -> run 01_build_inputs.py, 02 and 03 first")
    h = h.dropna(subset=["Station"])
    h = h[h.Station.isin(st.Station)]
    v = h.NO2.dropna()
    integer = bool(np.allclose(v, np.round(v)))
    h0 = h[["Station", "t", "NO2"]].copy()
    if integer and C.GROUND_FLATLINE_H_INTEGER == "auto":
        flat, T = stuck_threshold(h0, "UAE_EAD_SDI (whole numbers)")
        T.to_csv(os.path.join(C.DIRS["logs"], "ground_stuck_threshold.csv"), index=False)
    else:
        flat = C.GROUND_FLATLINE_H_INTEGER if integer else C.GROUND_FLATLINE_H
    say(f"   SDI NO2 reported as {'whole numbers' if integer else 'decimals'} -> stuck rule {flat} h")
    h, qc = hourly_qc(h0, flat, "UAE_EAD_SDI")
    qc.to_csv(os.path.join(C.DIRS["logs"], "ground_qc_hourly_sdi.csv"))
    d = hourly_to_daily(h, "UAE_EAD_SDI", "UAE_EAD_SDI")

    # sensitivity: the same daily means with other stuck thresholds
    rows = []
    for L in sorted(set(C.STUCK_SENS_H) | {flat}):
        hs, qs = hourly_qc(h0, L, "", quiet=True)
        ds = hourly_to_daily(hs, "UAE_EAD_SDI", "", quiet=True)
        j = d.merge(ds, on=["Station", "date"], suffixes=("", "_s"))
        rows.append(dict(threshold_h=L, hours_flagged_stuck=int(qs.stuck.sum()),
                         valid_days=len(ds), mean_daily_NO2=round(ds.NO2.mean(), 2),
                         median_abs_diff_vs_chosen=round(float(np.median(np.abs(j.NO2 - j.NO2_s))) if len(j) else np.nan, 3),
                         p95_abs_diff_vs_chosen=round(float(np.percentile(np.abs(j.NO2 - j.NO2_s), 95)) if len(j) else np.nan, 3)))
    say("   daily-mean sensitivity to the stuck threshold (all other QC identical):")
    table(pd.DataFrame(rows).set_index("threshold_h"))
    return d


def read_ead():
    df = read("UAE_EAD")
    if df is None:
        return None
    lo0, la0, lo1, la1 = UAE_BOX
    bad = df[~(df.lon.between(lo0, lo1) & df.lat.between(la0, la1))]
    if len(bad):
        say(f"   dropped {bad.location.nunique()} location(s) outside the UAE: {sorted(bad.location.unique())}")
    df = df.drop(bad.index)
    units = set(df.unit.dropna())
    if not units <= {"µg/m³", "ug/m3", "microg/m3"}:
        raise SystemExit(f"EAD units {units} - conversion not defined")
    df["date"] = pd.to_datetime(df.date) + pd.Timedelta(days=1)      # UTC-start label -> local day (see top)
    df["Station"] = "UAE_EAD|EAD_" + df.location
    ndup = int(df.duplicated(["Station", "date"]).sum())
    d = df.groupby(["Station", "date"]).agg(NO2=("value", "mean"), n_hours=("n_hours", "max")).reset_index()
    say(f"   EAD duplicate station-days (two sensors) averaged: {ndup}")
    for thr in (12, 18, 20):
        say(f"      days kept with >= {thr:2d} observed hours: {(d.n_hours >= thr).mean():.1%}")
    d = d[d.n_hours >= C.GROUND_MIN_HOURS_DAY].copy()
    d["NO2_ovp"] = np.nan
    d["network"] = "UAE_EAD"
    # weekday profile (UAE weekend = Sat-Sun since 2022): a weekend minimum supports the date fix
    wd = d.groupby(d.date.dt.day_name()).NO2.mean()
    wd = wd.reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    say("   EAD weekday means after the +1 day fix: " + " | ".join(f"{k[:3]} {v:.1f}" for k, v in wd.items()))
    return d


def read_uae_monthly():
    df = read("UAE_monthly")
    if df is None:
        return None
    say(f"   UAE monthly: Freq_Type {df.Freq_Type.value_counts().to_dict()} - annual rows ignored")
    df = df[df.Freq_Type.str.lower() == "monthly"].copy()
    if not set(df.Unit.dropna()) <= {"microg/m3", "ug/m3", "µg/m³"}:
        raise SystemExit(f"UAE monthly units {set(df.Unit)}")
    df["month"] = month_start(df.Timestamp)
    df["Station"] = "UAE_monthly|" + df.Station.astype(str)
    df["NO2"] = pd.to_numeric(df.NO2, errors="coerce")
    df = df.dropna(subset=["NO2"])
    hi = df.NO2 > C.GROUND_MONTHLY_MAX
    say(f"   UAE monthly values > {C.GROUND_MONTHLY_MAX:.0f}: {int(hi.sum())} removed | "
        f"duplicate station-months averaged: {int(df.duplicated(['Station', 'month']).sum())}")
    m = df[~hi].groupby(["Station", "month"]).NO2.mean().reset_index()
    m["n_days"], m["network"], m["source"] = np.nan, "UAE_monthly", "reported"
    return m


def read_riyadh():
    df = read("KSA_Riyadh")
    if df is None:
        return None, None
    n = df[df.Component == "NO2"].copy()
    if set(n.Unit) != {"ppb"}:
        raise SystemExit(f"Riyadh units {set(n.Unit)} - expected ppb only")
    n["Value"] = n.Value * C.NO2_PPB_TO_UGM3
    n["Station"] = "KSA_Riyadh|" + n.Station
    n["Date"] = pd.to_datetime(n.Date)
    daily = n[n.Periodicity == "Daily"].pivot_table(index=["Station", "Date"], columns="Indicator",
                                                    values="Value", aggfunc="mean")
    avg, mn, mx = "Avg / Hourly", "Min / Hourly", "Max / Hourly"
    incons = (daily[avg] < daily[mn] - 1e-6) | (daily[avg] > daily[mx] + 1e-6)
    say(f"   Riyadh daily rows {len(daily):,} | avg outside its own min-max (removed): {int(incons.sum())}")
    d = daily.loc[~incons, [avg]].rename(columns={avg: "NO2"}).dropna().reset_index().rename(columns={"Date": "date"})
    d["n_hours"], d["NO2_ovp"], d["network"] = np.nan, np.nan, "KSA_Riyadh"
    rep = n[(n.Periodicity == "Monthly") & (n.Indicator == avg)].copy()
    rep["month"] = month_start(rep.Date)
    rep = rep.groupby(["Station", "month"]).Value.mean().rename("NO2").reset_index()
    rep["n_days"], rep["network"], rep["source"] = np.nan, "KSA_Riyadh", "reported"
    say(f"   Riyadh: {d.Station.nunique()} stations, {len(d):,} valid days "
        f"({d.date.min():%Y-%m-%d} -> {d.date.max():%Y-%m-%d}); {len(rep)} reported monthly means")
    return d, rep


def read_bahrain():
    if C.BAHRAIN_MODE == "exclude":
        say("   Bahrain: BAHRAIN_MODE='exclude' - not used")
        return None
    raw = read("Bahrain_SCE")
    if raw is None:
        return None
    n = raw[raw.Indicator.str.contains("Nitrogen", case=False, na=False)].copy()
    say(f"   Bahrain indicator used: {n.Indicator.unique().tolist()} | unit {n.Unit.unique().tolist()}")
    n["month"] = pd.to_datetime(n.Year.astype(str) + "-" + n.Month.str.strip(), format="%Y-%B")
    n["Station"] = "Bahrain_SCE|" + n["Station Name"].str.strip()
    co = read("Bahrain_coords")
    if co is not None:      # proves the old 'Bahrain_..._NO2' file is this NOx series relabelled
        co["month"] = month_start(co.Date)
        co["Station"] = "Bahrain_SCE|" + co.Station.str.strip()
        chk = n.merge(co[["Station", "month", "NO2"]], on=["Station", "month"])
        same = np.isclose(chk.Value, chk.NO2).mean() if len(chk) else np.nan
        say(f"   old 'Bahrain_Ground_NO2' file equals the SCE 'Nitrogen Oxides' values in {same:.0%} of "
            f"{len(chk)} matched rows (i.e. it is the same series, relabelled)")
    m = n.groupby(["Station", "month"]).Value.mean().rename("NO2").reset_index()
    m["n_days"], m["network"], m["source"] = np.nan, "Bahrain_SCE", "reported"
    return m


# =============================================================================== aggregation
def daily_to_monthly(d):
    d = d.copy()
    d["month"] = d.date.dt.to_period("M").dt.to_timestamp()
    m = d.groupby(["Station", "month"]).agg(NO2=("NO2", "mean"), n_days=("NO2", "size"),
                                            network=("network", "first")).reset_index()
    m = m[m.n_days >= np.ceil(C.GROUND_MIN_FRAC_MONTH * m.month.dt.days_in_month)]
    m["source"] = "daily"
    return m


def overlap_stats(a, b, key="month"):
    """overlapping periods, Pearson r, median ratio a/b, median absolute relative difference |a-b|/b"""
    j = a.merge(b, on=key, suffixes=("_a", "_b"))
    j = j[j.NO2_b > 0]
    if len(j) < 3:
        return len(j), np.nan, np.nan, np.nan
    r = np.corrcoef(j.NO2_a, j.NO2_b)[0, 1] if j.NO2_a.std() > 0 and j.NO2_b.std() > 0 else np.nan
    return (len(j), r, float(np.median(j.NO2_a / j.NO2_b)),
            float(np.median(np.abs(j.NO2_a - j.NO2_b) / j.NO2_b)))


def dedup_uae(st, uae_m, ead_m):
    """Data-based matching of UAE monthly series to EAD monitors; returns {uae_station: ead_station}."""
    mon = st[st.Network == "UAE_monthly"]
    ead = st[st.Network == "UAE_EAD"]
    rows, match = [], {}
    for _, u in mon.iterrows():
        code = u.Station.split("|", 1)[1]
        hint = NAME_HINTS.get(code)
        best = None
        dist = km(u.Lat, u.Lon, ead.Lat.values, ead.Lon.values)
        for (_, e), dk in zip(ead.iterrows(), dist):
            if dk > C.DEDUP_RADIUS_KM:
                continue
            n, r, ratio, mard = overlap_stats(uae_m[uae_m.Station == u.Station][["month", "NO2"]],
                                              ead_m[ead_m.Station == e.Station][["month", "NO2"]])
            ename = e.Station.split("|EAD_", 1)[1]
            by_data = n >= C.DEDUP_MIN_MONTHS and r >= C.DEDUP_MIN_R and mard <= C.DEDUP_MAX_MARD
            by_name = hint == ename and not by_data      # same name, < 5 km: same place even if values differ
            rows.append(dict(uae_station=code, ead_station=ename, km=round(dk, 2), overlap_months=n,
                             r=round(r, 3) if r == r else np.nan, median_ratio=round(ratio, 3) if ratio == ratio else np.nan,
                             mard=round(mard, 3) if mard == mard else np.nan,
                             name_hint=hint == ename, same_site=by_data or by_name,
                             decided_by="data" if by_data else (
                                 ("name+distance, data DISAGREE" if n >= C.DEDUP_MIN_MONTHS else "name+distance, no overlap")
                                 if by_name else "")))
            if by_data or by_name:
                score = -mard if by_data else -9       # closest agreement wins
                if best is None or score > best[0]:
                    best = (score, e.Station)
        if best:
            match[u.Station] = best[1]
    log = pd.DataFrame(rows)
    log.to_csv(os.path.join(C.DIRS["logs"], "ground_uae_dedup.csv"), index=False)
    say(f"\n   UAE monthly series grouped with an EAD monitor (same Site_ID for cross-validation): {len(match)}")
    say("   decision counts (pair rows): " + str(log.decided_by.replace("", "not the same site").value_counts().to_dict()))
    hits = log[log.same_site].sort_values("uae_station")
    if len(hits):
        table(hits.set_index("uae_station"))
    miss = [c for c, e in NAME_HINTS.items() if f"UAE_monthly|{c}" in set(mon.Station)
            and f"UAE_monthly|{c}" not in match]
    if miss:
        say(f"   name hints without an EAD monitor within {C.DEDUP_RADIUS_KM:.0f} km (separate sites): {miss}")
    return match


def relabel_sites(st, match):
    """Union the Site_IDs of matched series; the smallest id of a group survives."""
    parent = {s: s for s in st.Site_ID.unique()}

    def f(x):
        while parent[x] != x:
            x = parent[x]
        return x
    sid = st.set_index("Station").Site_ID
    for u, e in match.items():
        a, b = f(sid[u]), f(sid[e])
        if a != b:
            parent[max(a, b)] = min(a, b)
    st = st.copy()
    st["Site_ID"] = st.Site_ID.map(f)
    return st


def combine(series, key, value_cols, st):
    """One row per (Target_ID, key): keep the best-priority level, average agreeing series within it."""
    s = series.merge(st[["Station", "Site_ID", "Target_ID"]], on="Station", how="left")
    lost = s[s.Target_ID.isna()].Station.unique()
    if len(lost):
        say(f"   [warning] series without a stations.csv entry (dropped): {list(lost)}")
    s = s.dropna(subset=["Target_ID"])
    s["prio"] = s.network.map(PRIORITY)
    if "source" in s:
        s["prio"] = s.prio + np.where(s.source == "daily", 0, 10)   # any daily-based value beats 'reported'
    s = s[s.prio == s.groupby(["Target_ID", key]).prio.transform("min")]
    agg = {c: (c, "mean") for c in value_cols}
    agg.update(n_series=("Station", "nunique"), network=("network", lambda x: "+".join(sorted(set(x)))))
    if "source" in s:
        agg["source"] = ("source", "first")
    return s.groupby(["Target_ID", "Site_ID", key]).agg(**agg).reset_index()


DAILY_NETS = {"Oman", "UAE_EAD", "UAE_EAD_SDI", "KSA_Riyadh"}


def assign_targets(st, daily, series_m):
    """Inside each Site_ID, link series whose overlapping data agree (r >= DEDUP_MIN_R and median relative
    difference <= DEDUP_MAX_MARD; >= GROUND_AGREE_MIN_DAYS days if both are daily networks, else
    >= DEDUP_MIN_MONTHS months).  Each linked group is one target series; a site with non-agreeing
    series gets Target_IDs S019a, S019b, ...  Unverifiable pairs (too little overlap) are NOT spliced."""
    rows, out = [], {}
    for sid, g in st.sort_values(["prio", "n_records"], ascending=[True, False]).groupby("Site_ID", sort=True):
        mem = g.Station.tolist()
        net = dict(zip(g.Station, g.Network))
        dec = {}
        for i, a in enumerate(mem):
            for b in mem[i + 1:]:
                if net[a] in DAILY_NETS and net[b] in DAILY_NETS:
                    key, nmin, src = "date", C.GROUND_AGREE_MIN_DAYS, daily
                else:
                    key, nmin, src = "month", C.DEDUP_MIN_MONTHS, series_m
                n, r, ratio, mard = overlap_stats(src[src.Station == a][[key, "NO2"]], src[src.Station == b][[key, "NO2"]], key)
                d = ("too little overlap" if n < nmin else
                     "splice" if (r >= C.DEDUP_MIN_R and mard <= C.DEDUP_MAX_MARD) else "DISAGREE - kept apart")
                dec[(a, b)] = dec[(b, a)] = d
                rows.append(dict(Site_ID=sid, a=a, b=b, compared_on=key, n=n, r=round(r, 2) if r == r else np.nan,
                                 ratio=round(ratio, 2) if ratio == ratio else np.nan,
                                 mard=round(mard, 2) if mard == mard else np.nan, decision=d))
        # greedy, no chaining: a series joins a group only if it agrees with at least one member and
        # disagrees with none (A~B and B~C must not splice A with C when A and C disagree)
        groups = []
        for m in mem:
            for grp in groups:
                if any(dec[(m, x)] == "splice" for x in grp) and not any(dec[(m, x)].startswith("DISAGREE") for x in grp):
                    grp.append(m)
                    break
            else:
                groups.append([m])
        for k, grp in enumerate(groups):
            for m in grp:
                out[m] = sid if len(groups) == 1 else sid + "abcdefghijklmnopqrstuvwxyz"[k]
    if rows:
        ag = pd.DataFrame(rows)
        ag.to_csv(os.path.join(C.DIRS["logs"], "ground_site_agreement.csv"), index=False)
        table(ag.set_index("Site_ID"))
    return st.Station.map(out)


# =============================================================================== main
def main():
    C.make_dirs()
    say("=== Ground truth assembly (step 06b)")
    say(f"   BAHRAIN_MODE = {C.BAHRAIN_MODE} | ppb->ug/m3 factor {C.NO2_PPB_TO_UGM3} | daily >= {C.GROUND_MIN_HOURS_DAY} h | "
        f"weekly >= {C.GROUND_MIN_DAYS_WEEK} d | monthly >= {C.GROUND_MIN_FRAC_MONTH:.0%} of days")
    st = pd.read_csv(C.STATIONS_CSV)
    st = st[st.Kind == "monitor"].copy()

    say("\n--- reading")
    oman_d, _ = read_oman()
    ead_d = read_ead()
    uae_m = read_uae_monthly()
    riy_d, riy_m = read_riyadh()
    bah_m = read_bahrain()
    sdi_d = read_sdi(st)

    if sdi_d is not None:
        late = sdi_d.date >= pd.Timestamp(C.FORWARD_END)
        if late.any():
            say(f"   SDI days on/after FORWARD_END {C.FORWARD_END} left out (kept in the raw file): {int(late.sum()):,}")
        sdi_d = sdi_d[~late]
        # unit check: same monitors, same calendar months (Jan-May exist in both feeds), different years.
        # ug/m3 in both -> ratio near 1 (minus any real trend); ppb in the SDI feed -> ~0.53
        if ead_d is not None:
            a = ead_d[ead_d.date.dt.month <= 5].groupby("Station").NO2.median()
            b = sdi_d[sdi_d.date.dt.month <= 5].groupby("Station").NO2.median()
            ratio = (b / a).dropna()
            say(f"   unit check, SDI 2026 / OpenAQ 2023-24 median NO2 (Jan-May, {len(ratio)} monitors): median ratio "
                f"{ratio.median():.2f} (range {ratio.min():.2f}-{ratio.max():.2f})")
            if len(ratio) and ratio.median() < C.SDI_MIN_UNIT_RATIO:
                raise SystemExit("SDI NO2 looks like ppb, not ug/m3 - stop and check the report")

    daily = pd.concat([x for x in (oman_d, ead_d, riy_d, sdi_d) if x is not None], ignore_index=True)
    daily["pollutant"] = "NO2"

    # ----- reported monthly vs monthly built from daily values (Riyadh): consistency of the agency data
    m_from_daily = daily_to_monthly(daily)
    if riy_m is not None and len(riy_m):
        dd = riy_d.assign(month=riy_d.date.dt.to_period("M").dt.to_timestamp())
        loose = dd.groupby(["Station", "month"]).NO2.agg(["mean", "size"]).reset_index()
        j = riy_m.merge(loose[loose["size"] >= 10], on=["Station", "month"])
        if len(j) >= 3:
            say(f"   Riyadh reported monthly vs mean of its daily rows (months with >= 10 days, n={len(j)}): "
                f"r = {np.corrcoef(j.NO2, j['mean'])[0, 1]:.2f}, median ratio {np.median(j.NO2 / j['mean']):.2f}")

    # ----- duplicates between UAE monthly and EAD
    say("\n--- duplicate sites")
    ead_m = m_from_daily[m_from_daily.network == "UAE_EAD"]
    match = dedup_uae(st, uae_m if uae_m is not None else pd.DataFrame(columns=["Station", "month", "NO2"]), ead_m)
    n_before = st.Site_ID.nunique()
    st = relabel_sites(st, match)
    say(f"   distinct sites: {n_before} (1 km rule, step 01) -> {st.Site_ID.nunique()} after data matching")

    # ----- Bahrain: stations without coordinates cannot be placed on the grid
    if bah_m is not None:
        no_xy = sorted(set(bah_m.Station) - set(st.Station))
        if no_xy:
            say(f"   Bahrain stations without coordinates (not usable until SCE sends them): {no_xy}")
        bah_m = bah_m[bah_m.Station.isin(st.Station)]

    # ----- monthly series table (audit trail) and site-level tables
    reported = [x for x in (uae_m, riy_m, bah_m) if x is not None]
    series_m = pd.concat([m_from_daily] + reported, ignore_index=True)
    series_m["pollutant"] = np.where((series_m.network == "Bahrain_SCE") & (C.BAHRAIN_MODE != "as_no2"),
                                     "NOx_reported", "NO2")
    out_s = series_m.assign(month=series_m.month.dt.strftime("%Y-%m-%d"))
    out_s.to_csv(os.path.join(C.DIRS["tables"], "ground_series_monthly.csv"), index=False)

    # ----- target series: inside a site, values are spliced ONLY between series whose data agree
    cnt = pd.concat([daily.groupby("Station").size(), series_m.groupby("Station").size()]).groupby(level=0).sum()
    st["n_records"] = st.Station.map(cnt).fillna(0)
    st["prio"] = st.Network.map(PRIORITY)
    st = st[st.n_records > 0].copy()
    say("\n--- agreement between series of the same site (daily overlap if both are daily networks, else monthly)")
    st["Target_ID"] = assign_targets(st, daily, series_m)
    n_split = st.groupby("Site_ID").Target_ID.nunique()
    say(f"   sites: {st.Site_ID.nunique()} | target series: {st.Target_ID.nunique()} "
        f"({int((n_split > 1).sum())} sites hold more than one non-agreeing series; they share the Site_ID, so "
        f"cross-validation still keeps them together)")

    tgt_daily = combine(daily, "date", ["NO2", "NO2_ovp", "n_hours"], st)
    tgt_daily["pollutant"] = "NO2"
    tgt_month = combine(series_m.drop(columns="pollutant"), "month", ["NO2", "n_days"], st)
    tgt_month["pollutant"] = np.where(tgt_month.network.str.contains("Bahrain") & (C.BAHRAIN_MODE != "as_no2"),
                                      "NOx_reported", "NO2")

    # weekly (ISO, Monday start) from target daily values
    w = tgt_daily.copy()
    iso = w.date.dt.isocalendar()
    w["iso_year"], w["iso_week"] = iso.year.astype(int), iso.week.astype(int)
    w["week_start"] = w.date - pd.to_timedelta(w.date.dt.weekday, unit="D")
    tgt_week = w.groupby(["Target_ID", "Site_ID", "iso_year", "iso_week", "week_start"]).agg(
        NO2=("NO2", "mean"), NO2_ovp=("NO2_ovp", "mean"), n_days=("NO2", "size"),
        network=("network", "first")).reset_index()
    tgt_week = tgt_week[tgt_week.n_days >= C.GROUND_MIN_DAYS_WEEK]
    tgt_week["pollutant"] = "NO2"

    # ----- target table: anchor monitor = best-priority network, most data (predictors are taken there)
    anchor = st.sort_values(["Target_ID", "prio", "n_records"], ascending=[True, True, False]).groupby("Target_ID").head(1)
    sites = anchor.rename(columns={"Station": "Anchor_Station"})[
        ["Target_ID", "Site_ID", "Anchor_Station", "Network", "Country", "Lat", "Lon", "Col", "Row"]].copy()
    sites["Members"] = sites.Target_ID.map(st.groupby("Target_ID").Station.apply(lambda x: ";".join(sorted(x))))
    g = gpd.read_file(C.GEOJSON).to_crs(4326)[[C.DISTRICT_ID, "geometry"]]
    pts = gpd.GeoDataFrame(sites, geometry=gpd.points_from_xy(sites.Lon, sites.Lat), crs=4326)
    j = gpd.sjoin(pts, g, how="left", predicate="within")
    j = j[~j.index.duplicated()]
    sites[C.DISTRICT_ID] = j[C.DISTRICT_ID].reindex(sites.index).values
    outside = sites[C.DISTRICT_ID].isna()
    if outside.any():   # coastal monitors just outside a polygon -> nearest district, distance reported
        near = gpd.sjoin_nearest(pts[outside.values].to_crs(6933), g.to_crs(6933), how="left", distance_col="d_m")
        near = near[~near.index.duplicated()]
        near = near.reindex(sites.index[outside.values])
        sites.loc[outside, C.DISTRICT_ID] = near[C.DISTRICT_ID].values
        say(f"\n   targets outside every district polygon -> nearest district: "
            + ", ".join(f"{s} ({d / 1000:.1f} km)" for s, d in zip(sites.loc[outside, 'Target_ID'], near.d_m)))
    sites[C.DISTRICT_ID] = sites[C.DISTRICT_ID].astype("Int64")
    sites["Use"] = np.where(sites.Network.eq("Bahrain_SCE") & (C.BAHRAIN_MODE != "as_no2"), "NOx_independent", "NO2")

    # ----- period labels (daily / monthly by calendar year, weekly by ISO year, as the rasters)
    def period(y):
        return np.select([np.isin(y, C.TRAIN_YEARS), np.isin(y, C.TEST_YEARS), y >= int(C.FORWARD_START[:4])],
                         ["train", "test", "forward"], "other")
    tgt_daily["period"] = period(tgt_daily.date.dt.year)
    tgt_week["period"] = period(tgt_week.iso_year)
    tgt_month["period"] = period(tgt_month.month.dt.year)

    # ----- write.  Forward-test rows go ONLY to tables/forward/ (sealed: hashed, never in the main tables)
    T = C.DIRS["tables"]
    F = os.path.join(T, "forward")
    os.makedirs(F, exist_ok=True)
    sites.to_csv(os.path.join(T, "ground_sites.csv"), index=False)
    outs = {"daily": (tgt_daily, "date"), "weekly": (tgt_week, "week_start"), "monthly": (tgt_month, "month")}
    hashes = []
    for name, (df, tcol) in outs.items():
        df = df.assign(**{tcol: df[tcol].dt.strftime("%Y-%m-%d")})
        fwd = df.period == "forward"
        df[~fwd].to_csv(os.path.join(T, f"ground_{name}.csv"), index=False)
        fp = os.path.join(F, f"ground_{name}_forward_{C.FORWARD_START[:4]}.csv")
        df[fwd].to_csv(fp, index=False)
        with open(fp, "rb") as fh:
            hashes.append(f"{hashlib.sha256(fh.read()).hexdigest()}  {os.path.basename(fp)}  ({int(fwd.sum())} rows)")
    with open(os.path.join(C.DIRS["logs"], "forward_test_hashes.txt"), "a", encoding="utf-8") as fh:
        fh.write(f"# {pd.Timestamp.now():%Y-%m-%d %H:%M:%S} local | forward test {C.FORWARD_START} -> {C.FORWARD_END}\n"
                 + "\n".join(hashes) + "\n")
    say("\n=== Forward test set (sealed: written only to tables/forward/, SHA-256 appended to logs/forward_test_hashes.txt)")
    for line in hashes:
        say("   " + line)

    # ----- coverage: what the train / test design can actually use
    say("\n=== Coverage (independent sites | target-periods) by country and year")
    cty = sites.set_index("Target_ID").Country
    site_week, site_month = tgt_week, tgt_month
    for name, df, tcol in (("WEEKLY (NO2, from daily networks)", site_week, "week_start"),
                           ("MONTHLY (all sources)", site_month, "month")):
        x = df.assign(Country=df.Target_ID.map(cty), year=pd.to_datetime(df[tcol]).dt.year)
        x["Country"] = np.where(x.pollutant == "NOx_reported", x.Country + " (NOx)", x.Country)
        a = x.groupby(["Country", "year"]).Site_ID.nunique().unstack(fill_value=0)
        b = x.groupby(["Country", "year"]).size().unstack(fill_value=0)
        say(f"\n   {name}: sites")
        table(a)
        say(f"   {name}: site-periods")
        table(b)
    if len(site_month):
        sm2 = site_month.assign(year=site_month.month.dt.year)
        say("\n   monthly values by source and year:")
        table(sm2.groupby(["source", "year"]).size().unstack(fill_value=0))

    say("\n=== Train / test / forward windows")
    for lab, yrs in (("train", C.TRAIN_YEARS), ("test", C.TEST_YEARS),
                     ("forward", list(range(int(C.FORWARD_START[:4]), int(C.FORWARD_END[:4]) + (C.FORWARD_END[5:] != "01-01"))))):
        for y in yrs:
            wk = site_week[site_week.iso_year == y]
            mo = site_month[(site_month.month.dt.year == y) & (site_month.pollutant == "NO2")]
            nx = site_month[(site_month.month.dt.year == y) & (site_month.pollutant != "NO2")]
            line = (f"   {lab} {y}: weekly NO2 {wk.Site_ID.nunique():3d} sites / {len(wk):5d} site-weeks | "
                    f"monthly NO2 {mo.Site_ID.nunique():3d} sites / {len(mo):5d} site-months | "
                    f"Bahrain NOx {nx.Site_ID.nunique()} sites / {len(nx)} months")
            if len(wk) == 0:
                line += "   <-- NO weekly NO2 ground truth"
            say(line)
    ksa = site_month[site_month.Target_ID.map(cty).eq("Saudi Arabia")]
    if len(ksa):
        say(f"   Saudi Arabia ground data period: {ksa.month.min():%Y-%m} -> {ksa.month.max():%Y-%m}")

    say(f"\n   targets written: {len(sites)} in {sites.Site_ID.nunique()} sites ({(sites.Use == 'NO2').sum()} NO2, "
        f"{(sites.Use != 'NO2').sum()} NOx-independent) | daily rows {len(tgt_daily):,} | weekly rows {len(tgt_week):,} | "
        f"monthly rows {len(tgt_month):,}")
    p = os.path.join(C.DIRS["logs"], "ground_truth_report.txt")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("\n".join(OUT))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
