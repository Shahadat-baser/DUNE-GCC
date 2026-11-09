"""
STEP 7b (v2) - Stage 1: satellite column -> surface NO2.

Why v2 (v1 = 07b_xgboost_v1.py, kept and reported in the supplement)
    v1 generalised well to a new YEAR at known stations (2024 R2 0.73) but poorly to NEW STATIONS
    (weekly R2 0.40, monthly ~0).  SHAP and the Riyadh curfew showed why: with 23-61 stations, static
    layers (settlement class, elevation, surface pressure ...) act as station fingerprints, so the model
    learned each station's typical level instead of following the satellite column.  v2 makes the
    column carry the signal and tests generalisation to new places directly.

Model variants - fixed BEFORE this run; the winner is the lowest scheme-A RMSE on 2022-2023 only
    E1  equation   ln C = a + b ln(VCD) - c ln(PBLH)                (mixed-layer physics: b = c = 1)
    E2  equation   E1 + wind speed + temperature + season (sin/cos day of year)
    X1  XGBoost    all v1 features (reference)
    X2  XGBoost    reduced features (no elevation / pressure / land / coast fingerprints; emission
                   proxies as 3x3 means only), monotone non-decreasing in every column feature
    H   hybrid     E2 (as offset, log scale) + XGBoost correction on the reduced features, monotone in
                   the column; retransformed with Duan's smearing factor
    VCD = NO2_trop_3x3 (~3 km, close to the TROPOMI footprint), floored at VCD_FLOOR; PBLH at 10 UTC.

Training data: WEEKLY ground values of the daily networks (Oman, Abu Dhabi), 2022-2023.  Monthly values
are day-weighted averages of the weekly predictions (the committee's "monthly from weekly").

Validation
    A   new station          leave-one-site-out on 2022-2023                       <- selection metric
    B   random 10-fold       reference only (optimistic)
    C   new country          leave-one-country-out (UAE <-> Oman)
    D   gap filling          12 calendar-month folds
    E   new station AND new year: train 2022-2023 without a site, predict that site in 2024  <- strictest
    TEST 2024                frozen model, known stations, against site-month climatology
    INDEPENDENT              UAE agency monthly stations that share no Site_ID with any training
                             monitor (never used): monthly averages of the frozen weekly predictions,
                             2022-2023, 2024 and 2019-2021 (T3 rasters)
    T3 / T2 / Bahrain        supplementary (processor test, Riyadh transfer, labelled NOx comparison)
    Equation coefficients with 95 % site-bootstrap intervals; the same fit on Oman's overpass-hour
    ground means (NO2_ovp) shows how much of the departure from b = c = 1 comes from 24 h vs 13:30.

Outputs  model/v2/*  and  logs/stage1_report.txt
"""
import os, sys, json, hashlib, time, itertools, warnings
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
from scipy import stats
import xgboost as xgb
import dune_config as C

for w in (FutureWarning, RuntimeWarning):
    warnings.filterwarnings("ignore", category=w)
warnings.filterwarnings("ignore", message=".*constant.*")
warnings.filterwarnings("ignore", message=".*poorly conditioned.*")

MODEL_DIR = os.path.join(C.BASE_DIR, "model")
V2 = os.path.join(MODEL_DIR, "v2")
SEED = 20261009
OUT = []
M_NO2 = 46.0055e6                        # ug per mol
VCD_FLOOR = 1e-6                         # mol m-2 (~6e14 molec cm-2), far below GCC background columns
VCD = "NO2_trop_3x3"
LOG_LO, LOG_HI = np.log(0.1), np.log(1000.0)    # predictions kept within 0.1-1000 ug/m3 before smearing
GRID = [dict(max_depth=d, min_child_weight=m, n_estimators=n)
        for d, m, n in itertools.product([2, 3, 5], [5, 10], [200, 500])]
FIXED = dict(eta=0.03, subsample=0.8, colsample_bytree=0.7, reg_lambda=10, tree_method="hist",
             objective="reg:squarederror", nthread=4, seed=SEED, verbosity=0)
COLUMN_FEATS = ["NO2_trop", "NO2_trop_3x3", "NO2_trop_5x5", "NO2_PBLH", "NO2_PBLH_3x3"]
REDUCED = COLUMN_FEATS + ["NO2_valid_frac", "S5P_cloud_3x3", "S5P_AAI_3x3", "PBLH", "WS", "U10", "V10", "T2m", "RH",
                          "PRECIP", "doy_sin", "doy_cos", "BUILT_frac_3x3", "POP_3x3", "SMOD_3x3", "VIIRS_3x3", "NDVI_3x3"]
NOT_FEATURES = {"y", "Target_ID", "Site_ID", "iso_year", "iso_week", "week_start", "month", "n_days", "n_series",
                "network", "source", "pollutant", "period", "Anchor_Station", "Country", "Network", "Use",
                "NO2_ndays", "raster_set", "days_covered", "days_in_month", "Year", "NO2_ovp", "n_hours",
                "S5P_proc", "S5P_proc_ge24", "Lat", "Lon", "Col", "Row", "Station"}
FAMILY = {"satellite": COLUMN_FEATS + ["NO2_valid_frac", "S5P_cloud", "S5P_AAI", "S5P_cloud_3x3", "S5P_AAI_3x3",
                                       "S5P_cloud_5x5", "S5P_AAI_5x5"],
          "weather": ["PBLH", "WS", "U10", "V10", "T2m", "Td2m", "RH", "PRECIP", "SP", "MET_fill"],
          "season": ["doy_sin", "doy_cos"]}


def family(f):
    for k, v in FAMILY.items():
        if f in v:
            return k
    return "emission proxies / surface" if f.startswith(("BUILT", "POP", "SMOD", "VIIRS", "NDVI")) else "fixed geography"


def say(s=""):
    print(s)
    OUT.append(str(s))


def table(df):
    for line in df.to_string().splitlines():
        say("   " + line)


# ================================================================================ metrics
def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) < 3:
        return dict(n=len(y))
    sse, sst = np.sum((y - p) ** 2), np.sum((y - y.mean()) ** 2)
    return dict(n=len(y), R2=1 - sse / sst if sst > 0 else np.nan, r=np.corrcoef(y, p)[0, 1],
                rho=stats.spearmanr(y, p)[0], RMSE=np.sqrt(sse / len(y)), MAE=np.mean(np.abs(y - p)),
                bias=np.mean(p - y), NMB=np.sum(p - y) / np.sum(y), slope=np.polyfit(y, p, 1)[0])


def anomaly_r(df, pcol="pred"):
    d = df.dropna(subset=["y", pcol])
    d = d[d.groupby("Site_ID").y.transform("size") >= 4]
    if len(d) < 10:
        return np.nan
    return np.corrcoef(d.y - d.groupby("Site_ID").y.transform("mean"),
                       d[pcol] - d.groupby("Site_ID")[pcol].transform("mean"))[0, 1]


def boot_ci(df, pcol="pred", n=500):
    rng = np.random.default_rng(SEED)
    sites = df.Site_ID.unique()
    if len(sites) < 4 or len(df) < 20:
        return {}
    g = {s: x for s, x in df.groupby("Site_ID")}
    out = {"R2": [], "r": [], "RMSE": []}
    for _ in range(n):
        b = pd.concat([g[s] for s in rng.choice(sites, len(sites))])
        m = metrics(b.y, b[pcol])
        for k in out:
            out[k].append(m.get(k, np.nan))
    return {f"{k}_ci": f"[{np.nanpercentile(v, 2.5):.2f}, {np.nanpercentile(v, 97.5):.2f}]" for k, v in out.items()}


ROWS = []


def record(label, df, **extra):
    m = metrics(df.y, df.pred)
    m.update(label=label, sites=df.Site_ID.nunique(), anomaly_r=anomaly_r(df), **boot_ci(df), **extra)
    ROWS.append(m)
    return m


def line(label, m):
    say(f"   {label:<44} n {m['n']:>5}  sites {m.get('sites', 0):>3}  R2 {m.get('R2', np.nan):5.2f}  r {m.get('r', np.nan):.2f}  "
        f"RMSE {m.get('RMSE', np.nan):5.1f}  bias {m.get('bias', np.nan):+5.1f}  anomaly r {m.get('anomaly_r', np.nan):.2f}  "
        f"R2 CI {m.get('R2_ci', '')}")


# ================================================================================ model variants
def design_eq(D, kind):
    v = np.log(np.maximum(D[VCD].values.astype(float), VCD_FLOOR))
    h = np.log(np.maximum(D.PBLH.values.astype(float), 10.0))
    cols = [np.ones(len(D)), v, h]
    if kind == "E2":
        cols += [D.WS.values, D.T2m.values - 300.0, D.doy_sin.values, D.doy_cos.values]
    return np.column_stack(cols)


class Equation:
    def __init__(self, kind):
        self.kind = kind

    def fit(self, D, y):
        X, ly = design_eq(D, self.kind), np.log(np.maximum(y, 0.1))
        ok = np.isfinite(X).all(axis=1) & np.isfinite(ly)
        self.coef, *_ = np.linalg.lstsq(X[ok], ly[ok], rcond=None)
        self.smear = np.mean(np.exp(ly[ok] - X[ok] @ self.coef))         # Duan's smearing estimator
        return self

    def log_pred(self, D):
        return design_eq(D, self.kind) @ self.coef

    def predict(self, D):
        return np.exp(np.clip(self.log_pred(D), LOG_LO, LOG_HI)) * self.smear


def booster(X, y, prm, base_margin=None, monotone=None):
    p = {**FIXED, "max_depth": prm["max_depth"], "min_child_weight": prm["min_child_weight"]}
    if monotone is not None:
        p["monotone_constraints"] = "(" + ",".join(str(int(m)) for m in monotone) + ")"
    d = xgb.DMatrix(np.asarray(X, "float32"), label=np.asarray(y, "float32"))
    if base_margin is not None:
        d.set_base_margin(np.asarray(base_margin, "float32"))
    return xgb.train(p, d, num_boost_round=prm["n_estimators"])


class Tree:
    """X1 / X2 (raw target) and H (log target with the E2 equation as offset)"""
    def __init__(self, kind, feats):
        self.kind, self.feats = kind, feats
        self.mono = [1 if f in COLUMN_FEATS else 0 for f in feats] if kind in ("X2", "H") else None

    def _xy(self, D):
        return D[self.feats].values.astype("float32")

    def fit(self, D, y, prm):
        self.prm = prm
        if self.kind == "H":
            self.eq = Equation("E2").fit(D, y)
            base = self.eq.log_pred(D)
            ly = np.log(np.maximum(y, 0.1))
            self.b = booster(self._xy(D), ly, prm, base_margin=base, monotone=self.mono)
            dm = xgb.DMatrix(self._xy(D))
            dm.set_base_margin(base.astype("float32"))
            self.smear = np.mean(np.exp(ly - self.b.predict(dm)))
        else:
            self.b = booster(self._xy(D), y, prm, monotone=self.mono)
        return self

    def predict(self, D):
        dm = xgb.DMatrix(self._xy(D))
        if self.kind == "H":
            dm.set_base_margin(self.eq.log_pred(D).astype("float32"))
            return np.exp(np.clip(self.b.predict(dm), LOG_LO, LOG_HI)) * self.smear
        return self.b.predict(dm)


def group_folds(groups, k, seed=SEED):
    u = np.array(sorted(pd.unique(groups)))
    np.random.default_rng(seed).shuffle(u)
    return [np.where(np.isin(groups, p))[0] for p in np.array_split(u, min(k, len(u)))]


def make(kind, feats):
    return Equation(kind) if kind in ("E1", "E2") else Tree(kind, feats)


def fit_tuned(kind, feats, D, y, k_inner=5):
    """equations: plain fit; trees: grid search with grouped inner folds (never sees the outer fold)"""
    if kind in ("E1", "E2"):
        return Equation(kind).fit(D, y)
    folds = group_folds(D.Site_ID.values, k_inner, SEED + 1)
    best, best_rmse = None, np.inf
    for prm in GRID:
        sse = 0.0
        for te in folds:
            tr = np.setdiff1d(np.arange(len(D)), te)
            p = Tree(kind, feats).fit(D.iloc[tr], y[tr], prm).predict(D.iloc[te])
            sse += np.sum((y[te] - p) ** 2)
        rmse = np.sqrt(sse / len(D))
        if np.isfinite(rmse) and rmse < best_rmse:
            best, best_rmse = prm, rmse
    if best is None:
        raise SystemExit(f"{kind}: no grid setting gave a finite inner-fold error - check the training table "
                         f"(rows {len(D)}, missing VCD {int(D[VCD].isna().sum())}, missing PBLH {int(D.PBLH.isna().sum())})")
    return Tree(kind, feats).fit(D, y, best)


def cv(kind, feats, D, folds):
    y = D.y.values
    pred = np.full(len(D), np.nan)
    for te in folds:
        tr = np.setdiff1d(np.arange(len(D)), te)
        pred[te] = fit_tuned(kind, feats, D.iloc[tr].reset_index(drop=True), y[tr]).predict(D.iloc[te])
    return pred


# ================================================================================ weekly -> monthly
def weekly_to_monthly(P):
    """P: Station, week_start, pred -> Station, month, pred (day-weighted, >= 90 % of the month covered)"""
    rows = []
    for wsd in P.week_start.unique():
        for mth, n in pd.Series(pd.date_range(wsd, periods=7, freq="D").to_period("M")).value_counts().items():
            rows.append((wsd, mth.to_timestamp(), n))
    E = pd.DataFrame(rows, columns=["week_start", "month", "days"])
    M = P.dropna(subset=["pred"]).merge(E, on="week_start")      # weeks without a prediction do not count as covered
    M["wp"] = M.pred * M.days
    g = M.groupby(["Station", "month"]).agg(wp=("wp", "sum"), days=("days", "sum")).reset_index()
    g = g[g.days >= 0.9 * g.month.dt.days_in_month]
    g["pred"] = g.wp / g.days
    return g[["Station", "month", "pred"]]


# ================================================================================ main
def main():
    os.makedirs(V2, exist_ok=True)
    TW = pd.read_csv(os.path.join(MODEL_DIR, "weekly_table.csv"), parse_dates=["week_start"])
    TM = pd.read_csv(os.path.join(MODEL_DIR, "monthly_table.csv"), parse_dates=["month"])
    FW = pd.read_csv(os.path.join(C.BASE_DIR, "features", "station_weekly_features.csv"), parse_dates=["week_start"])
    for T in (TW, TM):
        if (T.period == "forward").any():
            raise SystemExit("forward rows found in a model table - the forward set must stay sealed")
    TW = TW[np.isfinite(TW.y) & (TW.pollutant == "NO2")]
    # every variant is scored on the SAME rows: station-weeks without a satellite column (no valid
    # retrieval in the 3x3 cells all week, e.g. persistent cloud / dust) or without PBLH are left out
    has = np.isfinite(TW[VCD]) & np.isfinite(TW.PBLH)
    dropped = TW[~has].groupby("period").size().to_dict()
    TW = TW[has]
    train = TW[TW.period == "train"].reset_index(drop=True)
    test = TW[TW.period == "test"].reset_index(drop=True)
    allf = [c for c in TW.columns if c not in NOT_FEATURES and pd.api.types.is_numeric_dtype(TW[c])]
    variants = {"E1": [], "E2": [], "X1": allf, "X2": REDUCED, "H": REDUCED}
    say(f"=== Stage 1 v2 (step 07b_stage1) | xgboost {xgb.__version__} | VCD = {VCD} | grid {len(GRID)} tuned inside folds")
    say(f"   training (weekly, daily networks, 2022-2023): {len(train)} rows, {train.Site_ID.nunique()} sites, "
        f"{sorted(train.Country.unique())} | VCD below the floor: {(train[VCD] < VCD_FLOOR).sum()}")
    say(f"   station-weeks left out for having no satellite column / PBLH (same for every variant): {dropped}")
    say(f"   reduced feature set ({len(REDUCED)}): {', '.join(REDUCED)}")

    # ---------------- 1. the physical equation on all training rows: coefficients and site-bootstrap CI
    say("\n=== 1. Column-to-surface equation  ln C = a + b ln VCD - c ln PBLH (...)")
    rng = np.random.default_rng(SEED)
    sites = train.Site_ID.unique()
    for kind in ("E1", "E2"):
        eq = Equation(kind).fit(train, train.y.values)
        boots = []
        g = {s: x for s, x in train.groupby("Site_ID")}
        for _ in range(500):
            b = pd.concat([g[s] for s in rng.choice(sites, len(sites))])
            boots.append(Equation(kind).fit(b, b.y.values).coef)
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
        names = ["a", "b (ln VCD)", "-c (ln PBLH)"] + (["d (WS)", "e (T2m-300)", "f (doy sin)", "g (doy cos)"] if kind == "E2" else [])
        say(f"   {kind}: " + " | ".join(f"{n} {c:+.3f} [{l:+.3f}, {h:+.3f}]" for n, c, l, h in zip(names, eq.coef, lo, hi))
            + f" | smearing {eq.smear:.3f}")
        if kind == "E1":
            k = np.exp(eq.coef[0]) * eq.smear / M_NO2
            say(f"      mixed-layer physics expects b = +1 and -c = -1; with b = c = 1, exp(a)/M = {k:.3g} would be the "
                f"fraction of the column in the mixed layer")
    ovp = train.dropna(subset=["NO2_ovp"])
    if len(ovp) > 30:
        e24 = Equation("E1").fit(ovp, ovp.y.values).coef
        eov = Equation("E1").fit(ovp, ovp.NO2_ovp.values).coef
        say(f"   Oman rows with overpass-hour ground means (n {len(ovp)}): E1 on 24 h means  b {e24[1]:+.3f}  -c {e24[2]:+.3f} | "
            f"on 12-15 h means  b {eov[1]:+.3f}  -c {eov[2]:+.3f}")

    # ---------------- 2. validation of every variant on 2022-2023
    say("\n=== 2. Validation on 2022-2023 (every variant refitted - and trees re-tuned - inside every fold)")
    schemes = {
        "A new station (leave one site out)": group_folds(train.Site_ID.values, train.Site_ID.nunique()),
        "B random 10-fold (optimistic)": np.array_split(np.random.default_rng(SEED).permutation(len(train)), 10),
        "C new country (leave one out)": [np.where(train.Country.values == c)[0] for c in sorted(train.Country.unique())],
        "D gap filling (calendar months)": [np.where(train.week_start.dt.month.values == m)[0] for m in range(1, 13)
                                            if (train.week_start.dt.month.values == m).any()],
    }
    oof = {}
    t0 = time.time()
    for sname, folds in schemes.items():
        if sname.startswith("C") and len(folds) < 2:
            continue
        say(f"\n   --- {sname}")
        for v, feats in variants.items():
            p = cv(v, feats, train, folds)
            d = train[["Target_ID", "Site_ID", "Country", "week_start", "y"]].assign(pred=p)
            oof[(sname, v)] = d
            line(f"{v}", record(f"{sname} | {v}", d, scheme=sname, variant=v))
            if sname.startswith("C"):
                for c_, dd in d.groupby("Country"):
                    line(f"   {v} held out: {c_}", record(f"{sname} | {v} | held out {c_}", dd, scheme=sname, variant=v))
        say(f"   ({time.time() - t0:.0f} s)")

    # ---------------- 3. scheme E: new station AND new year (train 2022-23 without the site, predict it in 2024)
    say("\n=== 3. Scheme E: new station AND new year (2024) - the strictest test")
    test_sites = sorted(set(test.Site_ID) & set(train.Site_ID))
    for v, feats in variants.items():
        parts = []
        for s in test_sites:
            tr = train[train.Site_ID != s].reset_index(drop=True)
            te = test[test.Site_ID == s]
            parts.append(te[["Target_ID", "Site_ID", "Country", "week_start", "y"]]
                         .assign(pred=fit_tuned(v, feats, tr, tr.y.values).predict(te)))
        d = pd.concat(parts, ignore_index=True)
        oof[("E", v)] = d
        line(v, record(f"E new station + new year | {v}", d, scheme="E", variant=v))

    # ---------------- 4. selection (pre-declared): lowest scheme-A RMSE
    A = {v: metrics(oof[("A new station (leave one site out)", v)].y, oof[("A new station (leave one site out)", v)].pred)["RMSE"]
         for v in variants}
    winner = min(A, key=A.get)
    say(f"\n=== 4. Selection by scheme-A RMSE (2022-2023 only): " + " | ".join(f"{k} {v:.2f}" for k, v in A.items())
        + f"  ->  WINNER {winner}")

    # ---------------- 5. frozen winner (and E2 as the interpretable companion)
    frozen = {}
    for v in dict.fromkeys([winner, "E2"]):
        frozen[v] = fit_tuned(v, variants[v], train, train.y.values, k_inner=min(10, train.Site_ID.nunique()))
    fz = frozen[winner]
    meta = dict(winner=winner, variants=list(variants), selection_rmse_A=A, VCD=VCD, VCD_floor=VCD_FLOOR,
                features=variants[winner], train_rows=len(train), train_sites=int(train.Site_ID.nunique()), seed=SEED,
                xgboost=xgb.__version__)
    if isinstance(fz, Equation):
        meta.update(coef=fz.coef.tolist(), smearing=fz.smear)
    else:
        meta.update(params=fz.prm, smearing=getattr(fz, "smear", None),
                    offset_coef=fz.eq.coef.tolist() if fz.kind == "H" else None)
        fz.b.save_model(os.path.join(V2, f"stage1_{winner}.json"))
    blob = json.dumps(meta, indent=1, default=float)
    meta_hash = hashlib.sha256(blob.encode()).hexdigest()
    with open(os.path.join(V2, "stage1_meta.json"), "w") as fh:
        fh.write(blob)
    say(f"   frozen {winner}: {meta.get('params', meta.get('coef'))} | meta sha256 {meta_hash[:16]}...")

    def pred_rows(model, D):
        return D.assign(pred=model.predict(D))

    # ---------------- 6. TEST 2024 (known stations, new year) and climatology
    say("\n=== 5. TEST 2024 (frozen; known stations, new year)")
    clim = train.assign(m=train.week_start.dt.month).groupby(["Target_ID", "m"]).y.mean()
    tt = pred_rows(fz, test)
    tt["clim"] = [clim.get((t, d.month), np.nan) for t, d in zip(tt.Target_ID, tt.week_start)]
    line(f"{winner} weekly", record(f"TEST 2024 weekly | {winner}", tt, scheme="TEST", variant=winner))
    if "E2" != winner:
        line("E2 weekly", record("TEST 2024 weekly | E2", pred_rows(frozen["E2"], test), scheme="TEST", variant="E2"))
    c = tt.dropna(subset=["clim"])
    line("climatology (2022-23 site-week-of-month means)", record("TEST 2024 weekly | climatology", c.assign(pred=c.clim),
                                                                  scheme="TEST", variant="climatology"))
    line(f"{winner}, same rows", record(f"TEST 2024 weekly | {winner} same rows", c, scheme="TEST", variant=winner))
    mb = []
    for mo, d in tt.groupby(tt.week_start.dt.month):
        m = metrics(d.y, d.pred)
        mb.append(dict(month=mo, n=m["n"], sites=d.Site_ID.nunique(), R2=m.get("R2"), r=m.get("r"), RMSE=m.get("RMSE"),
                       bias=m.get("bias"), obs=d.y.mean(), pred=d.pred.mean()))
    say("   month by month (weekly values, 2024):")
    table(pd.DataFrame(mb).set_index("month").round(2))

    # ---------------- 7. monthly products from weekly predictions at every point
    FW = FW.copy()
    for v, mdl in frozen.items():
        FW[f"pred_{v}"] = mdl.predict(FW)
    mon = {v: weekly_to_monthly(FW[["Station", "week_start", f"pred_{v}"]].rename(columns={f"pred_{v}": "pred"}))
           for v in frozen}
    TMn = TM[np.isfinite(TM.y)].copy()

    def monthly_eval(sel, label, v=winner):
        d = TMn[sel].merge(mon[v].rename(columns={"Station": "Anchor_Station"}), on=["Anchor_Station", "month"], how="inner")
        return d

    train_sites = set(train.Site_ID)
    indep = (TMn.pollutant == "NO2") & ~TMn.Site_ID.isin(train_sites) & TMn.network.str.contains("UAE_monthly")
    say(f"\n=== 6. INDEPENDENT new stations: UAE agency monthly stations sharing no Site_ID with any training monitor "
        f"({TMn[indep].Site_ID.nunique()} sites; monthly averages of the frozen weekly model)")
    for lab, sel in (("2022-2023 (training years, new stations)", indep & (TMn.period == "train")),
                     ("2024 (new stations, new year)", indep & (TMn.period == "test")),
                     ("2019-2021 (T3 years, new stations)", indep & (TMn.raster_set == "T3"))):
        for v in frozen:
            d = monthly_eval(sel, lab, v)
            if len(d):
                line(f"{lab} | {v}", record(f"INDEPENDENT {lab} | {v}", d, scheme="INDEPENDENT", variant=v))

    say("\n=== 7. Monthly accuracy at the training networks' own stations (monthly from daily ground data)")
    own = (TMn.pollutant == "NO2") & TMn.Site_ID.isin(train_sites) & (TMn.source == "daily")
    for lab, sel in (("2022-2023", own & (TMn.period == "train")), ("2024", own & (TMn.period == "test"))):
        d = monthly_eval(sel, lab)
        if len(d):
            line(f"{lab} | {winner}", record(f"OWN STATIONS monthly {lab} | {winner}", d, scheme="OWN", variant=winner))

    # ---------------- 8. supplementary: T3, T2, Bahrain
    say("\n=== 8. Supplementary")
    d = monthly_eval((TMn.pollutant == "NO2") & (TMn.raster_set == "T3"), "T3")
    if len(d):
        line(f"T3 UAE 2019-21 monthly | {winner}", record(f"T3 all | {winner}", d, scheme="T3", variant=winner))
        d["proc"] = np.where(d.S5P_proc < 2.0, "S5P v1.x", "S5P v2.x")
        for k, dd in d.groupby("proc"):
            line(f"   T3 {k}", record(f"T3 {k} | {winner}", dd, scheme="T3", variant=winner))
    t2w = TW[TW.raster_set == "T2"].reset_index(drop=True) if "raster_set" in TW else TW.iloc[0:0]
    if len(t2w):
        t2w = pred_rows(fz, t2w)
        line(f"T2 Riyadh 2019-20 weekly | {winner}", record(f"T2 weekly | {winner}", t2w, scheme="T2", variant=winner))
        lo, hi = (pd.Timestamp(x) for x in C.TRANSFER_TESTS["T2"]["lockdown"])
        mid = t2w.week_start + pd.Timedelta(days=3)
        cur = t2w[(mid >= lo) & (mid < hi)]
        pre = t2w[mid < lo]
        if len(cur) and len(pre):
            say(f"   curfew check (does the prediction follow the observed drop?): obs {pre.y.mean():.1f} -> {cur.y.mean():.1f} | "
                f"pred {pre.pred.mean():.1f} -> {cur.pred.mean():.1f} | column {pre[VCD].mean():.2e} -> {cur[VCD].mean():.2e}")
    bah = monthly_eval((TMn.pollutant != "NO2"), "Bahrain")
    if len(bah):
        say(f"   Bahrain (reported as NOx, {len(bah)} site-months): rank correlation {stats.spearmanr(bah.y, bah.pred)[0]:.2f}")

    # ---------------- 9. explanation (trees: SHAP summed by feature family; equations: coefficients above)
    if not isinstance(fz, Equation):
        dm = xgb.DMatrix(train[fz.feats].values.astype("float32"))
        if fz.kind == "H":
            dm.set_base_margin(fz.eq.log_pred(train).astype("float32"))
        contrib = fz.b.predict(dm, pred_contribs=True)[:, :-1]
        s = pd.Series(np.abs(contrib).mean(axis=0), index=fz.feats)
        fam = s.groupby([family(f) for f in s.index]).sum().sort_values(ascending=False)
        say(f"\n=== 9. {winner}: mean |SHAP| summed by feature family ({'log units, the correction on top of E2' if fz.kind == 'H' else 'ug/m3'})")
        table(fam.round(3).to_frame("mean_abs_SHAP"))
        say("   top single features:")
        table(s.sort_values(ascending=False).head(10).round(3).to_frame("mean_abs_SHAP"))

    M = pd.DataFrame(ROWS)
    cols = ["label", "n", "sites", "R2", "R2_ci", "r", "r_ci", "rho", "RMSE", "RMSE_ci", "MAE", "bias", "NMB", "slope", "anomaly_r"]
    M[[c for c in cols if c in M]].to_csv(os.path.join(V2, "metrics_stage1.csv"), index=False)
    pd.concat([d.assign(key=f"{k[0]}|{k[1]}") for k, d in oof.items()]).to_csv(os.path.join(V2, "oof_predictions.csv"), index=False)
    FW[["Station", "week_start"] + [f"pred_{v}" for v in frozen]].to_csv(os.path.join(V2, "station_weekly_predictions.csv"), index=False)
    p = os.path.join(C.DIRS["logs"], "stage1_report.txt")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("\n".join(OUT))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
