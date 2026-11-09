"""
STEP 7b2 - Stage 1 diagnostics.  Local only, ~2-4 min (one optional refit of X2 / H, nothing is overwritten).

Run AFTER 07b_stage1.py.  Reads its outputs (model/v2/*) and those of 07b_xgboost_v1.py (model/*);
the forward set (tables/forward/) is NOT read.

Questions answered
    1. Why do the INDEPENDENT UAE agency stations fail?  Is it the LEVEL of each station (between-site
       offsets) or the month-to-month PATTERN (within-site anomalies)?
    2. Does the error depend on the area (emirate, by coordinate boxes), on the distance to the nearest
       training station, on how precise the published coordinates are, or on whether the station's
       satellite column lies inside the range seen in training?
    3. Agreement ceiling: where an agency monitor and an EAD monitor stand within 5 km, how well does
       one network's monthly value predict the other's?  A model trained on one network cannot be
       expected to beat that against the other.
    4. v1 (07b_xgboost_v1.py) vs Stage 1 v2 (07b_stage1.py): side by side from the two metric files, then
       on IDENTICAL rows (same target, same month / week) so the comparison is fair.
    5. X1 vs X2 vs E1/E2/H on identical out-of-fold rows, and - refitted here on 2022-2023 exactly as in
       07b_stage1 - X2 and H at the independent stations (07b_stage1 froze only the winner X1 and E2).

Outputs  logs/stage1_diagnostics.txt 
         model/v2/diag_independent_sites.csv   one row per independent site and period
         model/v2/diag_v1_v2_same_rows.csv
"""
import os, sys, importlib.util, warnings
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
dune_env.ensure("xgboost")
import numpy as np, pandas as pd
import dune_config as C

for w in (FutureWarning, RuntimeWarning):
    warnings.filterwarnings("ignore", category=w)

# the Stage 1 code itself (metrics, model classes, weekly->monthly) - imported, not copied, so the
# diagnostics use exactly the same definitions; importing does not run its main()
_spec = importlib.util.spec_from_file_location("stage1", os.path.join(CODE_DIR, "07b_stage1.py"))
S1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S1)

REFIT = ["X2", "H"]          # set to [] to skip the refit (section 5b)
MODEL_DIR = os.path.join(C.BASE_DIR, "model")
V2 = os.path.join(MODEL_DIR, "v2")
OUT = []


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


def table(df):
    for ln in df.to_string().splitlines():
        say("   " + ln)


def m_row(df, pcol="pred"):
    m = S1.metrics(df.y, df[pcol])
    m["sites"] = df.Site_ID.nunique()
    m["anomaly_r"] = S1.anomaly_r(df, pcol)
    return m


def short(m):
    return (f"n {m['n']:>5}  sites {m.get('sites', 0):>3}  R2 {m.get('R2', np.nan):5.2f}  r {m.get('r', np.nan):5.2f}  "
            f"RMSE {m.get('RMSE', np.nan):5.1f}  bias {m.get('bias', np.nan):+5.1f}  anomaly r {m.get('anomaly_r', np.nan):5.2f}")


def km(lat1, lon1, lat2, lon2):
    r = np.pi / 180
    a = np.sin((lat2 - lat1) * r / 2) ** 2 + np.cos(lat1 * r) * np.cos(lat2 * r) * np.sin((lon2 - lon1) * r / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(a))


def uae_area(lat, lon):
    """approximate area from coordinate boxes (not official boundaries; station codes are listed to check)"""
    if lon >= 56.0:
        return "East coast (Fujairah / Khor Fakkan)"
    if lat >= 25.62:
        return "Ras Al Khaimah"
    if lat >= 25.25 and lon >= 55.36:
        return "Sharjah / Ajman / UAQ"
    if 24.75 <= lat < 25.40 and 54.85 <= lon < 55.70:
        return "Dubai"
    if 24.20 <= lat < 24.70 and 54.20 <= lon < 54.80:
        return "Abu Dhabi city"
    if lat < 24.45 and lon >= 55.40:
        return "Al Ain"
    if lon < 54.20:
        return "Al Dhafra (west)"
    return "other inland"


def decomposition(d, pcol="pred"):
    """MSE = (mean bias)^2 + between-site offset spread + within-site (temporal) error"""
    d = d.dropna(subset=["y", pcol]).copy()
    e = d[pcol] - d.y
    site_bias = e.groupby(d.Site_ID).transform("mean")
    mse = float(np.mean(e ** 2))
    mb2 = float(e.mean() ** 2)
    between = float(np.mean((site_bias - e.mean()) ** 2))
    within = float(np.mean((e - site_bias) ** 2))
    sm = d.groupby("Site_ID")[["y", pcol]].mean()
    r_lvl = np.corrcoef(sm.y, sm[pcol])[0, 1] if len(sm) >= 4 else np.nan
    return dict(MSE=mse, mean_bias_sq=mb2, between_site=between, within_site=within,
                share_level=(mb2 + between) / mse if mse > 0 else np.nan, r_site_means=r_lvl, n_sites=len(sm))


def main():
    for p in [os.path.join(V2, f) for f in ("metrics_stage1.csv", "oof_predictions.csv", "station_weekly_predictions.csv",
                                             "stage1_meta.json")]:
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} - run 07b_stage1.py first")
    TW = pd.read_csv(os.path.join(MODEL_DIR, "weekly_table.csv"), parse_dates=["week_start"])
    TM = pd.read_csv(os.path.join(MODEL_DIR, "monthly_table.csv"), parse_dates=["month"])
    for T in (TW, TM):
        if (T.period == "forward").any():
            raise SystemExit("forward rows found in a model table - the forward set must stay sealed")
    sites = pd.read_csv(os.path.join(C.DIRS["tables"], "ground_sites.csv"))
    st_raw = pd.read_csv(C.STATIONS_CSV, dtype=str)
    SW = pd.read_csv(os.path.join(V2, "station_weekly_predictions.csv"), parse_dates=["week_start"])
    say(f"=== Stage 1 diagnostics (step 07b2) | pipeline {getattr(C, 'PIPELINE_VERSION', '')}")

    # training set exactly as 07b_stage1
    TWn = TW[np.isfinite(TW.y) & (TW.pollutant == "NO2") & np.isfinite(TW[S1.VCD]) & np.isfinite(TW.PBLH)]
    train = TWn[TWn.period == "train"].reset_index(drop=True)
    train_sites = set(train.Site_ID)
    tr_xy = sites[sites.Site_ID.isin(train_sites)].drop_duplicates("Site_ID")[["Site_ID", "Lat", "Lon"]]
    say(f"   training: {len(train)} station-weeks, {len(train_sites)} sites")

    # monthly predictions from the frozen weekly models (same aggregation as 07b_stage1)
    pcols = [c for c in SW.columns if c.startswith("pred_")]
    mon = None
    for c in pcols:
        m = S1.weekly_to_monthly(SW[["Station", "week_start", c]].rename(columns={c: "pred"})).rename(columns={"pred": c})
        mon = m if mon is None else mon.merge(m, on=["Station", "month"], how="outer")
    mon = mon.rename(columns={"Station": "Anchor_Station"})
    TMn = TM[np.isfinite(TM.y)].copy()
    TMn["network"] = TMn.network.astype(str)
    indep = (TMn.pollutant == "NO2") & ~TMn.Site_ID.isin(train_sites) & TMn.network.str.contains("UAE_monthly", na=False)
    I = TMn[indep].merge(mon, on=["Anchor_Station", "month"], how="inner")
    I["pset"] = np.select([I.raster_set.eq("T3"), I.period.eq("train"), I.period.eq("test")],
                          ["2019-2021", "2022-2023", "2024"], "other")
    I = I[I.pset != "other"]
    I = I.merge(sites[["Target_ID", "Lat", "Lon"]], on="Target_ID", how="left")
    winner = "pred_X1" if "pred_X1" in I else pcols[0]

    # ------------------------------------------------------------------ 1. level or pattern?
    say("\n=== 1. INDEPENDENT stations: is the error in each station's LEVEL or in its month-to-month PATTERN?")
    say("   MSE = mean bias^2 + between-site offset spread + within-site error.  share_level = part of the MSE")
    say("   that a per-station constant would remove.  r_site_means = do high stations get high predictions?")
    rows = []
    for ps, d in I.groupby("pset"):
        for c in pcols:
            m = m_row(d, c)
            m.update(decomposition(d, c))
            rows.append(dict(period=ps, model=c[5:], **{k: m.get(k) for k in
                        ["n", "sites", "R2", "r", "RMSE", "bias", "anomaly_r", "r_site_means", "share_level",
                         "mean_bias_sq", "between_site", "within_site"]}))
    table(pd.DataFrame(rows).set_index(["period", "model"]).round(2))

    # ------------------------------------------------------------------ 2. where does it fail?
    say("\n=== 2. Per independent site (frozen winner, all periods pooled)")
    # coordinate precision: decimals published in stations.csv; shared coordinates flag placeholders
    def decs(s):
        s = str(s)
        return len(s.split(".")[1].rstrip("0")) if "." in s else 0
    st_raw["dec"] = [min(decs(a), decs(b)) for a, b in zip(st_raw.Lat, st_raw.Lon)]
    st_raw["shared_xy"] = st_raw.groupby(["Lat", "Lon"]).Station.transform("size") > 1
    vcd_tr = train.groupby("Site_ID")[S1.VCD].mean()
    lo, hi = vcd_tr.min(), vcd_tr.max()
    S = []
    for sid, d in I.groupby("Site_ID"):
        lat, lon = d.Lat.iloc[0], d.Lon.iloc[0]
        dist = km(lat, lon, tr_xy.Lat.values, tr_xy.Lon.values)
        a = d.Anchor_Station.iloc[0]
        rr = st_raw[st_raw.Station == a]
        v = d[S1.VCD].mean() if S1.VCD in d else np.nan
        S.append(dict(Site_ID=sid, station=a.split("|", 1)[-1], area=uae_area(lat, lon),
                      km_nearest_train=float(dist.min()), nearest_train=tr_xy.Site_ID.values[dist.argmin()],
                      coord_decimals=int(rr.dec.iloc[0]) if len(rr) else np.nan,
                      shared_coords=bool(rr.shared_xy.iloc[0]) if len(rr) else False,
                      months=len(d), obs=d.y.mean(), pred=d[winner].mean(), bias=(d[winner] - d.y).mean(),
                      ratio=d[winner].mean() / d.y.mean() if d.y.mean() > 0 else np.nan,
                      r=np.corrcoef(d.y, d[winner])[0, 1] if len(d) >= 4 and d.y.std() > 0 else np.nan,
                      vcd_mean=v, vcd_inside_train=bool(lo <= v <= hi) if v == v else False))
    S = pd.DataFrame(S)
    S["dist_band"] = pd.cut(S.km_nearest_train, [0, 5, 20, 50, 1e4], labels=["<5 km", "5-20 km", "20-50 km", ">50 km"])
    S["precision"] = np.where(S.coord_decimals >= 3, ">=3 decimals (~100 m)", "<=2 decimals (~1 km or worse)")
    S["obs_level"] = pd.qcut(S.obs, 3, labels=["low", "mid", "high"]) if len(S) >= 6 else "all"
    table(S.sort_values("bias").set_index("station")[["area", "km_nearest_train", "coord_decimals", "shared_coords",
                                                      "months", "obs", "pred", "bias", "ratio", "r", "vcd_inside_train"]].round(2))
    I = I.merge(S[["Site_ID", "area", "dist_band", "precision", "obs_level", "vcd_inside_train"]], on="Site_ID", how="left")

    for key, title in (("area", "approximate area"), ("dist_band", "distance to the nearest training site"),
                       ("precision", "published coordinate precision"), ("obs_level", "observed level (site mean tercile)"),
                       ("vcd_inside_train", "site-mean column inside the training sites' range")):
        say(f"\n   by {title}:")
        g = []
        for k, d in I.groupby(key, observed=True):
            m = m_row(d, winner)
            g.append(dict(group=k, n=m["n"], sites=m["sites"], R2=m.get("R2"), r=m.get("r"), RMSE=m.get("RMSE"),
                          bias=m.get("bias"), obs=d.y.mean(), pred=d[winner].mean(), anomaly_r=m.get("anomaly_r")))
        table(pd.DataFrame(g).set_index("group").round(2))

    # the same distance question for the training sites themselves (scheme A out-of-fold, weekly)
    oof = pd.read_csv(os.path.join(V2, "oof_predictions.csv"), parse_dates=["week_start"])
    a = oof[oof.key.str.startswith("A new station") & oof.key.str.endswith("|" + winner[5:])]
    if len(a):
        xy = tr_xy.set_index("Site_ID")
        nn = {s: np.sort(km(xy.Lat[s], xy.Lon[s], xy.Lat.values, xy.Lon.values))[1] for s in xy.index}
        a = a.assign(dist_band=pd.cut(a.Site_ID.map(nn), [0, 5, 20, 50, 1e4], labels=["<5 km", "5-20 km", "20-50 km", ">50 km"]))
        say(f"\n   training sites, scheme A out-of-fold ({winner[5:]}, weekly), by distance to the nearest OTHER training site:")
        table(pd.DataFrame([dict(group=k, **{x: m_row(d)[x] for x in ("n", "sites")}, R2=m_row(d).get("R2"),
                                 RMSE=m_row(d).get("RMSE"), bias=m_row(d).get("bias"))
                            for k, d in a.groupby("dist_band", observed=True)]).set_index("group").round(2))

    # ------------------------------------------------------------------ 3. agreement ceiling between networks
    say("\n=== 3. Agreement ceiling: agency monthly monitor vs EAD monitor within "
        f"{C.DEDUP_RADIUS_KM} km (EAD value used as the 'prediction' of the agency value)")
    dd = os.path.join(C.DIRS["logs"], "ground_uae_dedup.csv")
    sm = os.path.join(C.DIRS["tables"], "ground_series_monthly.csv")
    if os.path.exists(dd) and os.path.exists(sm):
        D = pd.read_csv(dd)
        say("   pair decisions: " + str(D.decided_by.fillna("").replace("", "not the same site").value_counts().to_dict()))
        G = pd.read_csv(sm, parse_dates=["month"])
        G = G[G.pollutant == "NO2"]
        pairs = D[D.overlap_months >= C.DEDUP_MIN_MONTHS]
        J = []
        for p in pairs.itertuples():
            u = G[G.Station == f"UAE_monthly|{p.uae_station}"][["month", "NO2"]].rename(columns={"NO2": "y"})
            e = G[G.Station == f"UAE_EAD|EAD_{p.ead_station}"][["month", "NO2"]].rename(columns={"NO2": "ead"})
            j = u.merge(e, on="month")
            J.append(j.assign(Site_ID=p.uae_station, ead=j.ead, km=p.km, same_site=p.same_site))
        if J:
            J = pd.concat(J, ignore_index=True)
            for lab, d in (("all pairs within the radius", J), ("pairs judged the same site", J[J.same_site]),
                           ("pairs judged DIFFERENT sites", J[~J.same_site])):
                if len(d) >= 3:
                    say(f"   {lab:<34} {short(m_row(d, 'ead'))}  pairs {d.Site_ID.nunique()}")
            say("   -> a model that reproduced the EAD monitors perfectly would score about this against the agency")
            say("      monitors; the independent-station R2 should be read against this ceiling, not against 1.")
        else:
            say("   no pair with enough overlapping months")
        sa = os.path.join(C.DIRS["logs"], "ground_site_agreement.csv")
        if os.path.exists(sa):
            A = pd.read_csv(sa)
            say("   same-site series (06b): " + str(A.decision.value_counts().to_dict())
                + f" | median r {A.r.median():.2f}, median MARD {A.mard.median():.2f}, median ratio {A.ratio.median():.2f}")
    else:
        say("   ground_uae_dedup.csv or ground_series_monthly.csv not found - skipped")

    # ------------------------------------------------------------------ 4. v1 vs v2
    say("\n=== 4. v1 (07b_xgboost_v1) vs Stage 1 v2 (07b_stage1) - from the two metric files")
    say("   v1: weekly AND monthly models (monthly trained on all networks incl. agency stations);")
    say("   v2: weekly models on the daily networks only; monthly = day-weighted average of weekly predictions")
    m1p, m2p = os.path.join(MODEL_DIR, "metrics_xgb.csv"), os.path.join(V2, "metrics_stage1.csv")
    if os.path.exists(m1p):
        M1 = pd.read_csv(m1p)
        M2 = pd.read_csv(m2p)
        v = winner[5:]
        pairs = [("weekly A new station", ("weekly", "A new"), "A new station"),
                 ("weekly B random", ("weekly", "B random"), "B random"),
                 ("weekly C new country", ("weekly", "C leave"), "C new country (leave one out) | " + v),
                 ("weekly D gap filling", ("weekly", "D gap"), "D gap"),
                 ("weekly E new station + year", None, "E new station"),
                 ("weekly TEST 2024", ("weekly", "TEST"), "TEST 2024 weekly | " + v),
                 ("weekly T2 Riyadh", ("weekly", "T2"), "T2 weekly"),
                 ("monthly A new station", ("monthly", "A new"), None),
                 ("monthly TEST 2024", ("monthly", "TEST"), "OWN STATIONS monthly 2024"),
                 ("monthly INDEPENDENT 2022-23", None, "INDEPENDENT 2022-2023"),
                 ("monthly INDEPENDENT 2024", None, "INDEPENDENT 2024"),
                 ("monthly T3 UAE 2019-21", ("monthly", "T3"), "T3 all")]
        R = []
        for lab, k1, k2 in pairs:
            a = M1[(M1.resolution == k1[0]) & M1.scheme.astype(str).str.strip().str.startswith(k1[1])].head(1) if k1 else M1.iloc[0:0]
            b = M2[M2.label.astype(str).str.startswith(k2) & M2.label.astype(str).str.contains(f"| {v}", regex=False)].head(1) if k2 else M2.iloc[0:0]
            R.append(dict(test=lab, v1_n=a.n.values[0] if len(a) else np.nan, v1_R2=a.R2.values[0] if len(a) else np.nan,
                          v1_r=a.r.values[0] if len(a) else np.nan, v1_RMSE=a.RMSE.values[0] if len(a) else np.nan,
                          v2_n=b.n.values[0] if len(b) else np.nan, v2_R2=b.R2.values[0] if len(b) else np.nan,
                          v2_r=b.r.values[0] if len(b) else np.nan, v2_RMSE=b.RMSE.values[0] if len(b) else np.nan))
        table(pd.DataFrame(R).set_index("test").round(2))
        say("   NOTE rows are NOT the same where n differs (e.g. v1 monthly TEST includes agency stations that v1")
        say("   trained on; v2 'OWN STATIONS' is the daily networks only).  Section 4b compares identical rows.")
    else:
        say("   model/metrics_xgb.csv not found - v1 not run; skipped")

    # ------------------------------------------------------------------ 4b. identical rows
    say("\n=== 4b. v1 vs v2 on IDENTICAL rows")
    same = []
    p1m = os.path.join(MODEL_DIR, "predictions_monthly.csv")
    p1w = os.path.join(MODEL_DIR, "predictions_weekly.csv")
    anchor = sites.set_index("Target_ID").Anchor_Station
    if os.path.exists(p1m):
        P1 = pd.read_csv(p1m, parse_dates=["date"])
        P1["Anchor_Station"] = P1.Target_ID.map(anchor)
        P1 = P1.rename(columns={"date": "month", "pred": "v1"}).merge(mon[["Anchor_Station", "month", winner]],
                                                                      on=["Anchor_Station", "month"], how="inner")
        P1 = P1.rename(columns={winner: "v2"})
        net = TMn.drop_duplicates(["Target_ID", "month"]).set_index(["Target_ID", "month"]).network
        P1["network"] = [net.get((t, m_), "") for t, m_ in zip(P1.Target_ID, P1.month)]
        P1["indep"] = ~P1.Site_ID.isin(train_sites) & P1.network.astype(str).str.contains("UAE_monthly")
        groups = [("monthly A out-of-fold, INDEPENDENT sites, 2022-23 (neither saw the site; v1 trained on OTHER agency stations, v2 on daily networks only)",
                   P1.scheme.astype(str).str.startswith("A new") & P1.indep),
                  ("monthly A out-of-fold, daily-network sites, 2022-23 (v2 value is IN-SAMPLE - reference only)",
                   P1.scheme.astype(str).str.startswith("A new") & P1.Site_ID.isin(train_sites)),
                  ("monthly TEST 2024, daily-network sites (both: known site, new year)",
                   P1.scheme.eq("TEST") & P1.Site_ID.isin(train_sites)),
                  ("monthly TEST 2024, INDEPENDENT sites (v1 TRAINED on them in 2022-23, v2 never saw them)",
                   P1.scheme.eq("TEST") & P1.indep),
                  ("monthly T3 2019-21, all UAE sites", P1.scheme.eq("T3")),
                  ("monthly T3 2019-21, INDEPENDENT sites", P1.scheme.eq("T3") & P1.indep)]
        for lab, sel in groups:
            d = P1[sel].dropna(subset=["y", "v1", "v2"])
            if len(d) >= 3:
                a, b = m_row(d, "v1"), m_row(d, "v2")
                say(f"   {lab}\n      v1  {short(a)}\n      v2  {short(b)}")
                same.append(dict(test=lab, **{f"v1_{k}": a.get(k) for k in ("n", "R2", "r", "RMSE", "bias", "anomaly_r")},
                                 **{f"v2_{k}": b.get(k) for k in ("R2", "r", "RMSE", "bias", "anomaly_r")}))
    if os.path.exists(p1w):
        P1w = pd.read_csv(p1w, parse_dates=["date"])
        A2 = oof[oof.key.str.startswith("A new station") & oof.key.str.endswith("|" + winner[5:])]
        d = P1w[P1w.scheme.astype(str).str.startswith("A new")].rename(columns={"date": "week_start", "pred": "v1"}).merge(
            A2[["Target_ID", "week_start", "pred"]].rename(columns={"pred": "v2"}), on=["Target_ID", "week_start"], how="inner")
        if len(d) >= 3:
            a, b = m_row(d, "v1"), m_row(d, "v2")
            say(f"   weekly A out-of-fold, same station-weeks\n      v1  {short(a)}\n      v2  {short(b)}")
            same.append(dict(test="weekly A same rows", **{f"v1_{k}": a.get(k) for k in ("n", "R2", "r", "RMSE", "bias", "anomaly_r")},
                             **{f"v2_{k}": b.get(k) for k in ("R2", "r", "RMSE", "bias", "anomaly_r")}))
        t1 = P1w[P1w.scheme.eq("TEST")].rename(columns={"date": "week_start", "pred": "v1"})
        sw = SW.rename(columns={"Station": "Anchor_Station"})
        t1["Anchor_Station"] = t1.Target_ID.map(anchor)
        d = t1.merge(sw[["Anchor_Station", "week_start", winner]], on=["Anchor_Station", "week_start"], how="inner").rename(columns={winner: "v2"})
        if len(d) >= 3:
            a, b = m_row(d, "v1"), m_row(d, "v2")
            say(f"   weekly TEST 2024, same station-weeks\n      v1  {short(a)}\n      v2  {short(b)}")
            same.append(dict(test="weekly TEST 2024 same rows", **{f"v1_{k}": a.get(k) for k in ("n", "R2", "r", "RMSE", "bias", "anomaly_r")},
                             **{f"v2_{k}": b.get(k) for k in ("R2", "r", "RMSE", "bias", "anomaly_r")}))
    if same:
        pd.DataFrame(same).to_csv(os.path.join(V2, "diag_v1_v2_same_rows.csv"), index=False)
    else:
        say("   v1 prediction files not found - skipped")

    # ------------------------------------------------------------------ 5. variants on identical rows
    say("\n=== 5. Variants on identical out-of-fold rows (07b_stage1 section 2/3)")
    sch = oof.key.str.split("|").str[0]
    var = oof.key.str.split("|").str[1]
    R = []
    for s in sch.unique():
        for v_ in var.unique():
            d = oof[(sch == s) & (var == v_)]
            if len(d) >= 3:
                m = m_row(d)
                R.append(dict(scheme=s[:28], variant=v_, n=m["n"], R2=m.get("R2"), r=m.get("r"), RMSE=m.get("RMSE"),
                              bias=m.get("bias"), slope=m.get("slope"), anomaly_r=m.get("anomaly_r")))
    table(pd.DataFrame(R).set_index(["scheme", "variant"]).round(2))
    say("   slope = regression of prediction on observation: < 1 means high values are under-predicted")

    # ------------------------------------------------------------------ 5b. X2 / H at the independent stations
    if REFIT:
        say(f"\n=== 5b. {', '.join(REFIT)} refitted on 2022-2023 exactly as in 07b_stage1 (diagnostic only, not frozen)")
        FW = pd.read_csv(os.path.join(C.BASE_DIR, "features", "station_weekly_features.csv"), parse_dates=["week_start"])
        for v_ in REFIT:
            feats = S1.REDUCED
            mdl = S1.fit_tuned(v_, feats, train, train.y.values, k_inner=min(10, len(train_sites)))
            pw = FW[["Station", "week_start"]].assign(pred=mdl.predict(FW))
            mm = S1.weekly_to_monthly(pw).rename(columns={"Station": "Anchor_Station", "pred": f"pred_{v_}"})
            d0 = I.drop(columns=[c for c in I if c == f"pred_{v_}"]).merge(mm, on=["Anchor_Station", "month"], how="inner")
            for ps, d in d0.groupby("pset"):
                m = m_row(d, f"pred_{v_}")
                m.update(decomposition(d, f"pred_{v_}"))
                say(f"   INDEPENDENT {ps:<9} {v_:<3} {short(m)}  r_site_means {m['r_site_means']:.2f}  share_level {m['share_level']:.2f}")
                d_ = d.dropna(subset=[winner])
                mw = m_row(d_, winner)
                say(f"   INDEPENDENT {ps:<9} {winner[5:]:<3} {short(mw)}  (same rows)")
            I = I.merge(mm, on=["Anchor_Station", "month"], how="left") if f"pred_{v_}" not in I else I

    S.to_csv(os.path.join(V2, "diag_independent_sites.csv"), index=False)
    p = os.path.join(C.DIRS["logs"], "stage1_diagnostics.txt")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("\n".join(OUT))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
