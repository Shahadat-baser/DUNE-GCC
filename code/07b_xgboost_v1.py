"""
(v1 - FIRST model, kept unchanged for transparency; superseded by 07b_stage1.py, see README)

STEP 7b - Stage 1, XGBoost satellite-to-surface NO2 (local only, ~30-60 min on a laptop CPU).

Two models, the same recipe: WEEKLY (daily networks, 23 sites) and MONTHLY (all networks, ~64 sites).
Training data: 2022-2023 only (period == "train", pollutant NO2).  Nothing from 2024-2026 or Riyadh is
seen during training or tuning.

Validation on the training years (out-of-fold predictions, hyper-parameters tuned INSIDE every fold):
    A  new station      grouped K-fold by Site_ID (weekly: leave-one-site-out)   <- headline number
    B  random           10-fold by row (comparability with literature; known to be optimistic)
    C  new country      leave-one-country-out (UAE <-> Oman)
    D  gap filling      12 folds by calendar month (same sites, unseen months)
    anomaly skill       within-site correlation of A's predictions (each site's own mean removed)
    95 % intervals      cluster bootstrap by Site_ID
Frozen model (tuned by grouped CV on all 2022-2023 rows, then fitted once, saved with a hash):
    TEST 2024-2025      overall, month by month, known vs new sites, against site-month climatology
    Bahrain (NOx)       labelled comparison only: rank / anomaly correlation, 2024 -> 2025 direction
    T2 Riyadh 2019-20   new country, inland, S5P v1.x: overall, before / during (strict, easing) / after curfew
    T3 UAE 2019-21      same stations as training, S5P v1.x until mid-2021 then v2.2-2.3 (monthly only:
                        daily UAE data start Oct 2022): overall, by processor, by year, by lockdown phase,
                        against the 2022-23 site-month climatology.  T3 isolates the processor + period
                        effect; comparing T2 with T3 shows what is specific to Saudi Arabia.
    high-value check    a linear recalibration fitted on A's out-of-fold predictions, scored on TEST
Explanation: SHAP values (XGBoost's exact TreeSHAP), gain importance; residuals by S5P processor.
Sensitivity: monthly model trained only on S5P processor >= 2.4.

Features: no latitude / longitude / country / processor version (the model must not identify places or
years); the processor version is used only to inspect residuals.

Outputs  model/xgb_{weekly,monthly}.json (+ _meta.json), model/predictions_*.csv,
         model/metrics_xgb.csv, model/shap_*.csv, logs/xgb_report.txt
"""
import os, sys, json, hashlib, time
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
import itertools, warnings
import numpy as np, pandas as pd
from scipy import stats
import xgboost as xgb
import dune_config as C

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*constant.*")
warnings.filterwarnings("ignore", message=".*poorly conditioned.*")
MODEL_DIR = os.path.join(C.BASE_DIR, "model")
SEED = 20261008
OUT = []
GRID = [dict(max_depth=d, min_child_weight=m, reg_lambda=l, n_estimators=n)
        for d, m, l, n in itertools.product([3, 5, 7], [1, 5], [1, 10], [300, 700])]
FIXED = dict(learning_rate=0.03, subsample=0.8, colsample_bytree=0.7, tree_method="hist",
             objective="reg:squarederror", n_jobs=4, random_state=SEED, verbosity=0)
NOT_FEATURES = {"y", "Target_ID", "Site_ID", "iso_year", "iso_week", "week_start", "month", "n_days", "n_series",
                "network", "source", "pollutant", "period", "Anchor_Station", "Country", "Network", "Use",
                "NO2_ndays", "raster_set", "days_covered", "days_in_month", "Year", "NO2_ovp", "n_hours",
                "S5P_proc", "S5P_proc_ge24", "Lat", "Lon", "Col", "Row"}


def say(s=""):
    print(s)
    OUT.append(str(s))


def table(df):
    for line in df.to_string().splitlines():
        say("   " + line)


# =============================================================================== metrics
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
    """within-site temporal skill: Pearson r after removing each site's own mean (obs and pred separately)"""
    d = df.dropna(subset=["y", pcol])
    d = d[d.groupby("Site_ID").y.transform("size") >= 4]
    if len(d) < 10:
        return np.nan
    ya = d.y - d.groupby("Site_ID").y.transform("mean")
    pa = d[pcol] - d.groupby("Site_ID")[pcol].transform("mean")
    return np.corrcoef(ya, pa)[0, 1]


def boot_ci(df, pcol="pred", n=500, seed=SEED):
    """95 % cluster-bootstrap interval (resampling sites) for R2, r and RMSE"""
    rng = np.random.default_rng(seed)
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


# =============================================================================== folds and tuning
def group_folds(groups, k, seed):
    u = np.array(sorted(pd.unique(groups)))
    rng = np.random.default_rng(seed)
    rng.shuffle(u)
    k = min(k, len(u))
    parts = np.array_split(u, k)
    return [np.where(np.isin(groups, p))[0] for p in parts]


class Model:
    """XGBoost through its native API (no scikit-learn needed).  Same algorithm and settings as
    XGBRegressor: n_estimators -> boosting rounds, learning_rate -> eta, reg_lambda -> lambda."""
    def __init__(self, booster):
        self.booster = booster

    def predict(self, X):
        return self.booster.predict(xgb.DMatrix(np.asarray(X, dtype="float32")))

    def save_model(self, path):
        self.booster.save_model(path)

    def get_booster(self):
        return self.booster


def fit(X, y, params):
    p = {**FIXED, **params}
    native = dict(eta=p["learning_rate"], max_depth=p["max_depth"], min_child_weight=p["min_child_weight"],
                  reg_lambda=p["reg_lambda"], subsample=p["subsample"], colsample_bytree=p["colsample_bytree"],
                  tree_method=p["tree_method"], objective=p["objective"], nthread=p["n_jobs"], seed=p["random_state"],
                  verbosity=p["verbosity"])
    dtrain = xgb.DMatrix(np.asarray(X, dtype="float32"), label=np.asarray(y, dtype="float32"))
    return Model(xgb.train(native, dtrain, num_boost_round=p["n_estimators"]))


def tune(X, y, groups, k_inner=5, seed=SEED):
    """grid search with grouped inner folds (never sees the outer test fold); returns best params"""
    folds = group_folds(groups, k_inner, seed + 1)
    best, best_rmse = None, np.inf
    for prm in GRID:
        sse = 0.0
        for te in folds:
            tr = np.setdiff1d(np.arange(len(y)), te)
            p = fit(X[tr], y[tr], prm).predict(X[te])
            sse += np.sum((y[te] - p) ** 2)
        rmse = np.sqrt(sse / len(y))
        if rmse < best_rmse:
            best, best_rmse = prm, rmse
    return best


def cross_val(D, feats, folds, label, nested=True):
    X, y, g = D[feats].values.astype("float32"), D.y.values, D.Site_ID.values
    pred = np.full(len(D), np.nan)
    chosen = []
    for te in folds:
        tr = np.setdiff1d(np.arange(len(D)), te)
        prm = tune(X[tr], y[tr], g[tr]) if nested else GRID[0]
        chosen.append(prm)
        pred[te] = fit(X[tr], y[tr], prm).predict(X[te])
    out = D[["Target_ID", "Site_ID", "Country", "y"]].copy()
    out["pred"], out["scheme"] = pred, label
    return out, chosen


# =============================================================================== one resolution
def run(res, T, tcol):
    say(f"\n{'=' * 100}\n=== {res.upper()} model")
    T = T[np.isfinite(T.y)].copy()
    T[tcol] = pd.to_datetime(T[tcol])
    feats = [c for c in T.columns if c not in NOT_FEATURES and pd.api.types.is_numeric_dtype(T[c])]
    no2 = T.pollutant == "NO2"
    train = T[no2 & (T.period == "train")].reset_index(drop=True)
    say(f"   features ({len(feats)}): {', '.join(feats)}")
    say(f"   training rows {len(train)} | sites {train.Site_ID.nunique()} | countries {sorted(train.Country.unique())} | "
        f"{train[tcol].min():%Y-%m-%d} -> {train[tcol].max():%Y-%m-%d}")
    rows, preds = [], []

    def record(name, df, extra=None):
        m = metrics(df.y, df.pred)
        m.update(scheme=name, resolution=res, sites=df.Site_ID.nunique(), anomaly_r=anomaly_r(df))
        m.update(boot_ci(df))
        if extra:
            m.update(extra)
        rows.append(m)
        return m

    # ---- validation on 2022-2023
    t0 = time.time()
    k_sites = train.Site_ID.nunique() if train.Site_ID.nunique() <= 25 else 10
    schemes = {
        "A new station (grouped by site)": group_folds(train.Site_ID.values, k_sites, SEED),
        "B random 10-fold (optimistic)": np.array_split(np.random.default_rng(SEED).permutation(len(train)), 10),
        "C leave one country out": [np.where(train.Country.values == c)[0] for c in sorted(train.Country.unique())],
        "D gap filling (by calendar month)": [np.where(train[tcol].dt.month.values == m)[0] for m in range(1, 13)
                                              if (train[tcol].dt.month.values == m).any()],
    }
    for name, folds in schemes.items():
        if name.startswith("C") and len(folds) < 2:
            say(f"   {name}: skipped (one country)")
            continue
        oof, chosen = cross_val(train, feats, folds, name)
        preds.append(oof.assign(resolution=res, date=train[tcol].values))
        m = record(name, oof)
        if name.startswith("C"):
            for c in sorted(train.Country.unique()):
                record(f"   C held out: {c}", oof[oof.Country == c])
        say(f"   {name:<38} R2 {m.get('R2', np.nan):.2f}  r {m.get('r', np.nan):.2f}  RMSE {m.get('RMSE', np.nan):.1f}  "
            f"anomaly r {m['anomaly_r']:.2f}  ({time.time() - t0:.0f} s)")
        if name.startswith("A"):
            oof_A = oof

    # ---- frozen model: tuned by grouped CV on all training rows, fitted once
    X = train[feats].values.astype("float32")
    best = tune(X, train.y.values, train.Site_ID.values, k_inner=min(10, train.Site_ID.nunique()))
    model = fit(X, train.y.values, best)
    path = os.path.join(MODEL_DIR, f"xgb_{res}.json")
    model.save_model(path)
    with open(path, "rb") as fh:
        mhash = hashlib.sha256(fh.read()).hexdigest()
    meta = dict(resolution=res, features=feats, params={**FIXED, **best}, train_rows=len(train),
                train_sites=int(train.Site_ID.nunique()), train_period=[str(train[tcol].min().date()), str(train[tcol].max().date())],
                xgboost=xgb.__version__, sha256=mhash, seed=SEED)
    with open(os.path.join(MODEL_DIR, f"xgb_{res}_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    say(f"   frozen model: {best} | sha256 {mhash[:16]}...")

    # ---- recalibration for high values, fitted on A's out-of-fold predictions only
    a, b = np.polyfit(oof_A.pred.dropna(), oof_A.y[oof_A.pred.notna()], 1)[::-1]
    say(f"   high-value recalibration (from A out-of-fold): obs = {a:.2f} + {b:.3f} x pred")

    def predict(df):
        p = model.predict(df[feats].values.astype("float32"))
        return p, a + b * p

    # ---- TEST 2024-2025 (NO2)
    test = T[no2 & (T.period == "test")].reset_index(drop=True)
    if len(test):
        test["pred"], test["pred_recal"] = predict(test)
        known = set(train.Site_ID)
        test["site_type"] = np.where(test.Site_ID.isin(known), "known site", "new site")
        clim = (train.assign(m=train[tcol].dt.month).groupby(["Target_ID", "m"]).y.mean())
        test["clim"] = [clim.get((t, d.month), np.nan) for t, d in zip(test.Target_ID, test[tcol])]
        preds.append(test[["Target_ID", "Site_ID", "Country", "y", "pred", "pred_recal", "clim", "site_type"]]
                     .assign(scheme="TEST", resolution=res, date=test[tcol].values))
        say(f"\n   TEST {', '.join(str(y) for y in sorted(test[tcol].dt.year.unique()))} (frozen model): "
            f"{len(test)} rows, {test.Site_ID.nunique()} sites")
        for nm, d, pc in (("TEST all", test, "pred"), ("TEST all, recalibrated", test, "pred_recal")):
            m = record(nm, d.assign(pred=d[pc]))
            say(f"   {nm:<38} R2 {m.get('R2', np.nan):.2f}  r {m.get('r', np.nan):.2f}  RMSE {m.get('RMSE', np.nan):.1f}  "
                f"bias {m.get('bias', np.nan):+.1f}  slope {m.get('slope', np.nan):.2f}  anomaly r {m['anomaly_r']:.2f}")
        for st_, d in test.groupby("site_type"):
            m = record(f"TEST {st_}", d)
            say(f"   TEST {st_:<33} R2 {m.get('R2', np.nan):.2f}  r {m.get('r', np.nan):.2f}  RMSE {m.get('RMSE', np.nan):.1f}  n {m['n']}")
        c = test.dropna(subset=["clim"])
        if len(c) > 10:
            mc, mm = record("TEST climatology baseline", c.assign(pred=c.clim)), record("TEST model, same rows", c)
            say(f"   climatology vs model on the same {len(c)} rows: RMSE {mc['RMSE']:.1f} vs {mm['RMSE']:.1f} | "
                f"anomaly r {mc['anomaly_r']:.2f} vs {mm['anomaly_r']:.2f}")
        say("   month by month (frozen model):")
        mb = []
        for (yy, mo), d in test.groupby([test[tcol].dt.year, test[tcol].dt.month]):
            m = metrics(d.y, d.pred)
            mb.append(dict(month=f"{yy}-{mo:02d}", n=m["n"], sites=d.Site_ID.nunique(), R2=m.get("R2"), r=m.get("r"),
                           RMSE=m.get("RMSE"), bias=m.get("bias"), obs_mean=d.y.mean(), pred_mean=d.pred.mean()))
        mbt = pd.DataFrame(mb).set_index("month").round(2)
        mbt.to_csv(os.path.join(MODEL_DIR, f"test_month_by_month_{res}.csv"))
        table(mbt)

    # ---- Bahrain NOx (labelled comparison, never a NO2 metric)
    bah = T[(T.pollutant != "NO2") & T.period.isin(["test", "train"])].reset_index(drop=True)
    if len(bah):
        bah["pred"], _ = predict(bah)
        rho = stats.spearmanr(bah.y, bah.pred)[0]
        say(f"\n   Bahrain (reported as NOx, {len(bah)} site-months): rank correlation {rho:.2f} | "
            f"within-site anomaly r {anomaly_r(bah):.2f}")
        yrs = bah.assign(yr=pd.to_datetime(bah[tcol]).dt.year).groupby(["Site_ID", "yr"])[["y", "pred"]].mean().unstack()
        if (2024 in yrs["y"].columns) and (2025 in yrs["y"].columns):
            dy, dp = yrs["y"][2025] - yrs["y"][2024], yrs["pred"][2025] - yrs["pred"][2024]
            ok = dy.notna() & dp.notna()
            say(f"   2024 -> 2025 direction of change agrees at {int((np.sign(dy[ok]) == np.sign(dp[ok])).sum())} of {int(ok.sum())} sites")
        preds.append(bah[["Target_ID", "Site_ID", "Country", "y", "pred"]].assign(scheme="BAHRAIN_NOx", resolution=res,
                                                                              date=bah[tcol].values))

    # ---- transfer tests (T2 Riyadh, T3 UAE backcast): frozen model, never trained or tuned on
    clim_tr = train.assign(m=train[tcol].dt.month).groupby(["Target_ID", "m"]).y.mean()
    for key, tdef in getattr(C, "TRANSFER_TESTS", {}).items():
        tt = T[no2 & (T.raster_set == key)].reset_index(drop=True) if "raster_set" in T else T.iloc[0:0]
        if not len(tt):
            say(f"\n   {key} {tdef['title']}: no {res} rows")
            continue
        tt["pred"], tt["pred_recal"] = predict(tt)
        mid = pd.to_datetime(tt[tcol]) + pd.Timedelta(days=3 if res == "weekly" else 14)
        lo, hi = pd.Timestamp(tdef["lockdown"][0]), pd.Timestamp(tdef["lockdown"][1])
        ea = pd.Timestamp(tdef["easing_from"]) if tdef.get("easing_from") else hi
        tt["phase"] = np.select([mid < lo, mid < ea, mid < hi],
                                ["before restrictions", "restrictions" + (" (strict)" if tdef.get("easing_from") else ""),
                                 "restrictions (easing)"], "after restrictions")
        tt["processor"] = np.where(tt.S5P_proc < 2.0, "S5P v1.x", "S5P v2.x")
        tt["year"] = mid.dt.year.values
        tt["site_type"] = np.where(tt.Site_ID.isin(set(train.Site_ID)), "known site", "new site")
        tt["clim"] = [clim_tr.get((t_, d.month), np.nan) for t_, d in zip(tt.Target_ID, mid)]
        preds.append(tt[["Target_ID", "Site_ID", "Country", "y", "pred", "pred_recal", "clim", "phase", "processor",
                         "site_type"]].assign(scheme=key, resolution=res, date=tt[tcol].values))
        say(f"\n   {key} {tdef['title']} (frozen model): {len(tt)} rows, {tt.Site_ID.nunique()} sites, "
            f"processor {sorted(tt.S5P_proc.dropna().unique().tolist())}")
        m = record(f"{key} all", tt)
        say(f"   {'all':<26} R2 {m.get('R2', np.nan):.2f}  r {m.get('r', np.nan):.2f}  rho {m.get('rho', np.nan):.2f}  "
            f"RMSE {m.get('RMSE', np.nan):.1f}  bias {m.get('bias', np.nan):+.1f}  anomaly r {m['anomaly_r']:.2f}  r CI {m.get('r_ci', '')}")
        for col in ("phase", "processor", "year", "site_type"):
            if tt[col].nunique() < 2 and col != "phase":
                continue
            for v, d in tt.groupby(col):
                mm = record(f"{key} {col}: {v}", d)
                say(f"   {str(v):<26} n {mm['n']:>4}  r {mm.get('r', np.nan):.2f}  RMSE {mm.get('RMSE', np.nan):.1f}  "
                    f"bias {mm.get('bias', np.nan):+.1f}  obs mean {d.y.mean():.1f}  pred mean {d.pred.mean():.1f}")
        c = tt.dropna(subset=["clim"])
        if len(c) > 10:
            mc, mm = record(f"{key} climatology baseline", c.assign(pred=c.clim)), record(f"{key} model, same rows", c)
            say(f"   climatology (2022-23 site-month means) vs model on {len(c)} rows: RMSE {mc['RMSE']:.1f} vs {mm['RMSE']:.1f} | "
                f"anomaly r {mc['anomaly_r']:.2f} vs {mm['anomaly_r']:.2f}")

    # ---- explanation: SHAP (exact TreeSHAP from XGBoost) and gain
    bst = model.get_booster()
    bst.feature_names = feats
    contrib = bst.predict(xgb.DMatrix(X, feature_names=feats), pred_contribs=True)[:, :-1]
    shap = pd.Series(np.abs(contrib).mean(axis=0), index=feats).sort_values(ascending=False)
    gain = pd.Series(bst.get_score(importance_type="gain")).reindex(feats).fillna(0)
    imp = pd.DataFrame({"mean_abs_SHAP": shap, "gain_share": (gain / gain.sum()).reindex(shap.index)})
    imp.to_csv(os.path.join(MODEL_DIR, f"shap_importance_{res}.csv"))
    say("\n   mean |SHAP| (ug/m3) on the training rows, top 12:")
    table(imp.head(12).round(3))

    # ---- residuals by S5P processor version (diagnostic only; the version is not a feature)
    allp = pd.concat([d for d in preds if "pred" in d], ignore_index=True)
    pv = T[["Target_ID", tcol, "S5P_proc"]].rename(columns={tcol: "date"})
    pv["date"] = pd.to_datetime(pv.date)
    r_ = allp[allp.scheme.isin(["A new station (grouped by site)", "TEST"] + list(getattr(C, "TRANSFER_TESTS", {})))].copy()
    r_["date"] = pd.to_datetime(r_.date)
    r_ = r_.merge(pv, on=["Target_ID", "date"], how="left")
    r_["resid"] = r_.pred - r_.y
    say("\n   residual (pred - obs) by S5P processor version (A out-of-fold, TEST, transfer tests):")
    table(r_.groupby(["scheme", "S5P_proc"]).resid.agg(["size", "mean", "median"]).round(2))

    pd.concat(preds, ignore_index=True).to_csv(os.path.join(MODEL_DIR, f"predictions_{res}.csv"), index=False)
    return rows, train, feats, best


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    say(f"=== Stage 1 XGBoost (step 07b) | xgboost {xgb.__version__} | grid {len(GRID)} settings, tuned inside every fold")
    TW = pd.read_csv(os.path.join(MODEL_DIR, "weekly_table.csv"))
    TM = pd.read_csv(os.path.join(MODEL_DIR, "monthly_table.csv"))
    for T in (TW, TM):
        if (T.period == "forward").any():
            raise SystemExit("forward rows found in a model table - the forward set must stay sealed")
    rows = []
    r, _, _, _ = run("weekly", TW, "week_start")
    rows += r
    r, trainM, featsM, bestM = run("monthly", TM, "month")
    rows += r

    # sensitivity: monthly model trained only on S5P processor >= 2.4 (scheme A)
    sub = trainM[trainM.S5P_proc >= 2.4].reset_index(drop=True)
    if 0 < len(sub) < len(trainM):
        oof, _ = cross_val(sub, featsM, group_folds(sub.Site_ID.values, 10, SEED), "A, processor >= 2.4 only")
        m = metrics(oof.y, oof.pred)
        m.update(scheme="sensitivity: monthly A, S5P processor >= 2.4 only", resolution="monthly",
                 sites=oof.Site_ID.nunique(), anomaly_r=anomaly_r(oof))
        rows.append(m)
        say(f"\n   sensitivity, monthly scheme A on processor >= 2.4 only ({len(sub)} of {len(trainM)} rows): "
            f"R2 {m.get('R2', np.nan):.2f}  RMSE {m.get('RMSE', np.nan):.1f}")

    M = pd.DataFrame(rows)
    cols = ["resolution", "scheme", "n", "sites", "R2", "R2_ci", "r", "r_ci", "rho", "RMSE", "RMSE_ci", "MAE", "bias", "NMB", "slope", "anomaly_r"]
    M = M[[c for c in cols if c in M.columns]]
    M.to_csv(os.path.join(MODEL_DIR, "metrics_xgb.csv"), index=False)
    say(f"\n{'=' * 100}\n=== Summary (all metrics in ug/m3 where applicable; CI = 95 % cluster bootstrap by site)")
    table(M.set_index(["resolution", "scheme"]).round(2))
    p = os.path.join(C.DIRS["logs"], "xgb_report.txt")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("\n".join(OUT))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
