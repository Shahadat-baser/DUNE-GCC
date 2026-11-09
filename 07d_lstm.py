"""
STEP 7d - Stage 2: LSTM models (PyTorch, CPU); 2024 validated against ground stations month by month.
Local only, ~0.5-2 h on a laptop CPU (installs PyTorch CPU the first time).  Run AFTER 07b_stage1, 07b2
and 07c.  The sealed 2026 ground data (tables/forward/) are NOT read here - that is 07e, run once after.

Two LSTM models (every candidate and setting is fixed below BEFORE the run; nothing is tuned on 2024)
  LSTM-E  estimation with the satellite.  Target: GROUND NO2 at the 23 training sites in 2023 (the same
          sites and year as Stage 1).  Two input designs compete:
            weekly     sequences of L = 4, 8 or 13 weekly means (the 07a features, reduced set)
            daily      TEMPORAL DENSIFICATION: sequences of L = 28 or 56 DAILY values (S5P column with a
                       'no retrieval today' flag, ERA5 at the overpass hour, emission proxies), trained on
                       the daily ground values (~7x more training samples than weekly); its daily
                       predictions are averaged to weeks and months.  Days without a satellite retrieval
                       are bridged by the sequence, so weeks with no valid column still get an estimate.
          The candidate is chosen ONLY by its out-of-site RMSE on the 2023 weekly rows (5 grouped site
          folds, 3 seeds); the winner is then scored leave-one-site-out and on every test.  The 1 km
          product stays X1 (declared before this run); LSTM-E is the station-level temporal alternative.
  LSTM-P  projection WITHOUT the satellite.  Input: L weeks of ERA5 weather + season + the location's
          Stage 1 seasonal level for that calendar month.  Target: the Stage 1 weekly estimate of 2022-2023
          (744 districts + every station cell), so the model learns how weather moves NO2 around each
          place's satellite-based level.  L = 4, 8, 13 or 26 weeks, chosen by the loss on held-out SERIES
          in 2022-2023.  Two baselines for the projected years:
            P (fixed)  seasonal level from 2022-2023 for every projected year (pure projection from the
                       training period)
            P (roll)   seasonal level from the two PREVIOUS years: 2024 <- 2022-23, 2025 <- 2023-24,
                       2026 <- 2024-25 (Stage 1 estimates of those years; no ground data); the network
                       itself is the same, trained on 2022-2023 only
Baselines (no learning): Stage 1 climatology 2022-23, ground climatology 2023, seasonal-naive /
persistence (same week / month of 2023 observed); at KNOWN stations also 'site-calibrated' models (one
2023 ratio per series from leave-one-site-out predictions), for monitored stations only, never the maps.

Reported in logs/stage2_report.txt
  0. candidate selection tables (E: weekly 4/8/13 vs daily 28/56; P: 4/8/13/26 weeks)
  1. LSTM-E vs X1: leave one site out 2023, 2024 weekly, Riyadh 2019-20, independent stations
  2-3. 2024 at the stations, weekly and monthly, every model and baseline on IDENTICAL rows
  4. month by month 2024 and the month-wise CHANGE 2024 vs 2023
  5. 2025 (no NO2 ground data exist): P fixed / P roll vs Stage 1, 744 districts, month by month; Bahrain NOx
  6. applicability (07c dissimilarity index) at Riyadh, independent stations, T3
Outputs
  model/v2/lstm_E.pt, model/v2/lstm_P.pt, model/v2/stage2_meta.json (SHA-256, chosen settings, selection)
  model/v2/stage2_point_weekly.csv     Station x week: X1, LSTM-E, P fixed, P roll, Stage 1 climatology
  model/v2/site_calibration_2023.csv   one 2023 ratio per known series (out-of-site predictions)
  tables/District_NO2_weekly_LSTMP.csv, tables/District_NO2_monthly_LSTMP.csv   (projection database)
  logs/stage2_report.txt 
"""
import os, sys, json, time, hashlib, importlib.util, warnings
CODE_DIR = os.getcwd()                      # fallback when __file__ is undefined (notebook)
try:
    CODE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    if os.path.exists(os.path.join(os.getcwd(), "dune_config.py")):
        CODE_DIR = os.getcwd()
sys.path.insert(0, CODE_DIR)
for _m in ("dune_config", "dune_env", "dune_gee", "dune_ground", "dune_lstm"):
    sys.modules.pop(_m, None)
import dune_env
dune_env.ensure("lstm")
import numpy as np, pandas as pd
import torch
import dune_config as C
import dune_lstm as DL

for w in (FutureWarning, RuntimeWarning, UserWarning):
    warnings.filterwarnings("ignore", category=w)


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(CODE_DIR, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S1 = _load("stage1", "07b_stage1.py")              # metrics, bootstrap, weekly->monthly, REDUCED, VCD
A7 = _load("features07a", "07a_build_features.py") # humidity formula

MODEL_DIR = os.path.join(C.BASE_DIR, "model")
V2 = os.path.join(MODEL_DIR, "v2")
SEED = 20261013
# ---------------------------------------------------------------- fixed before the run
E_CAND = [("weekly", 4), ("weekly", 8), ("weekly", 13), ("daily", 28), ("daily", 56)]
E_SET = {"weekly": dict(H=16, dropout=0.2, wd=1e-3, lr=3e-3, epochs=200, batch=64, patience=20),
         "daily": dict(H=16, dropout=0.2, wd=1e-3, lr=3e-3, epochs=80, batch=256, patience=10)}
P_CAND = [4, 8, 13, 26]
P_SET = dict(H=32, dropout=0.1, wd=1e-5, lr=2e-3, epochs=40, batch=512, patience=5)
SEL_FOLDS, SEL_SEEDS, LOSO_SEEDS, FINAL_SEEDS = 5, 3, 3, 5
E_WFEATS = S1.REDUCED + ["col_missing"]
STATIC_D = ["BUILT_frac_3x3", "POP_3x3", "SMOD_3x3", "VIIRS_3x3", "NDVI_3x3"]
E_DFEATS = ["NO2_trop_3x3", "NO2_PBLH_3x3", "NO2_valid", "col_missing", "S5P_cloud_3x3", "S5P_AAI_3x3", "T2m", "RH",
            "WS", "U10", "V10", "PBLH", "SP", "PRECIP", "doy_sin", "doy_cos"] + STATIC_D
P_MET = ["T2m", "RH", "WS", "U10", "V10", "PBLH", "SP", "PRECIP", "doy_sin", "doy_cos"]
P_FEATS = P_MET + ["log_clim"]
P_SEL_FROM = pd.Timestamp("2022-07-04")       # all P candidates are compared on the same target weeks
OUT = []
NAMES = {"pred_X1": "X1", "pred_E": "LSTM-E", "pred_P": "LSTM-P", "pred_Proll": "LSTM-P roll", "pred_mean": "X1+E mean",
         "pred_S1clim": "S1 clim", "pred_groundclim": "ground clim", "pred_naive": "naive", "pred_persist": "persist",
         "pred_X1cal": "X1 site-cal", "pred_Ecal": "LSTM-E site-cal"}


def say(s=""):
    print(s, flush=True)
    OUT.append(str(s))


def table(df):
    for ln in df.to_string().splitlines():
        say("   " + ln)


def mrow(d, pcol):
    m = S1.metrics(d.y, d[pcol])
    m.update(sites=d.Site_ID.nunique(), anomaly_r=S1.anomaly_r(d, pcol), **S1.boot_ci(d, pcol))
    return m


def line(label, m):
    say(f"   {label:<15} n {m['n']:>5}  sites {m.get('sites', 0):>3}  R2 {m.get('R2', np.nan):5.2f} {m.get('R2_ci', ''):<15}"
        f"r {m.get('r', np.nan):5.2f}  RMSE {m.get('RMSE', np.nan):5.1f}  bias {m.get('bias', np.nan):+5.1f}  "
        f"anomaly r {m.get('anomaly_r', np.nan):5.2f}")


def compare(d, models, title):
    """every model on the SAME rows (rows where all models have a value)"""
    d = d.dropna(subset=["y"] + models)
    say(f"\n   {title}  (identical rows: n {len(d)}, sites {d.Site_ID.nunique()})")
    if len(d) >= 3:
        for mname in models:
            line(NAMES.get(mname, mname), mrow(d, mname))
    return d


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def week_of(d):
    d = pd.to_datetime(d)
    return d - pd.to_timedelta(d.dt.weekday, unit="D")


def seq_index(keys, times):
    return pd.MultiIndex.from_arrays([np.asarray(keys), pd.to_datetime(np.asarray(times))])


def sequences(F, index, keys, times, L, step_days):
    """X[n, L, D]: rows of F for times t-(L-1)*step ... t of the same key; missing steps stay NaN"""
    times = pd.to_datetime(np.asarray(times)).values
    X = np.full((len(keys), L, F.shape[1]), np.nan, dtype="float32")
    for k in range(L):
        ii = index.get_indexer(seq_index(keys, times - np.timedelta64(step_days * (L - 1 - k), "D")))
        ok = ii >= 0
        X[ok, k, :] = F[ii[ok]]
    return X


def site_folds(sites, k, seed):
    u = np.array(sorted(set(sites)))
    np.random.default_rng(seed).shuffle(u)
    return [set(p) for p in np.array_split(u, min(k, len(u)))]


def inner_split(sites, frac, seed):
    u = np.array(sorted(set(sites)))
    val = set(np.random.default_rng(seed).choice(u, max(2, int(round(frac * len(u)))), replace=False))
    return np.array([s not in val for s in sites])


def to_ug(e, p):
    return np.exp(np.clip(p, np.log(0.1), np.log(1000))) * e.smear


# ===================================================================== LSTM-E: data for both designs
class EData:
    """weekly and daily inputs at every station point; ground targets for training"""

    def __init__(self, FW, train, sites_tab):
        self.FW = FW
        FW["col_missing"] = (~np.isfinite(FW[S1.VCD])).astype(float)
        self.FWm = FW[E_WFEATS].values.astype("float32")
        self.idxW = seq_index(FW.Station.values, FW.week_start.values)
        # daily inputs (station daily table: cell value and 3x3 mean of every band, 2019 -> 2026-09)
        cols = ["Station", "date", "NO2_trop_3x3", "NO2_valid", "S5P_cloud_3x3", "S5P_AAI_3x3", "T2m", "Td2m", "U10",
                "V10", "SP", "PBLH", "PRECIP"]
        D = pd.read_csv(os.path.join(C.DIRS["tables"], C.STATION_DAILY_CSV), usecols=cols, parse_dates=["date"])
        D["PRECIP"] = D.PRECIP.clip(lower=0)
        D["RH"] = np.clip(A7.rh_from_t_td(D.T2m, D.Td2m), 0, 100)
        D["WS"] = np.hypot(D.U10, D.V10)
        D["NO2_PBLH_3x3"] = D.NO2_trop_3x3 / D.PBLH.where(D.PBLH > 10)
        D["col_missing"] = (~np.isfinite(D.NO2_trop_3x3)).astype(float)
        doy = D.date.dt.dayofyear
        D["doy_sin"], D["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
        S = pd.read_csv(os.path.join(C.DIRS["tables"], "Station_Static_by_Year.csv"))[["Station", "Year"] + STATIC_D]
        D["Year"] = D.date.dt.year
        D = D.merge(S, on=["Station", "Year"], how="left").sort_values(["Station", "date"]).reset_index(drop=True)
        self.D = D
        self.FD = D[E_DFEATS].values.astype("float32")
        self.idxD = seq_index(D.Station.values, D.date.values)
        # daily ground targets of the training series (the same series and year as Stage 1)
        gd = pd.read_csv(os.path.join(C.DIRS["tables"], "ground_daily.csv"), parse_dates=["date"])
        gd = gd[(gd.period == "train") & (gd.pollutant == "NO2") & np.isfinite(gd.NO2) & gd.Target_ID.isin(set(train.Target_ID))]
        gd["Anchor_Station"] = gd.Target_ID.map(sites_tab.set_index("Target_ID").Anchor_Station)
        self.gd = gd.reset_index(drop=True)

    # sequences for training rows
    def X_train(self, kind, L, rows):
        if kind == "weekly":
            return sequences(self.FWm, self.idxW, rows.Anchor_Station.values, rows.week_start.values, L, 7)
        return sequences(self.FD, self.idxD, rows.Anchor_Station.values, rows.date.values, L, 1)

    def fit(self, kind, L, rows, y_log, seeds, seed):
        tr = inner_split(rows.Site_ID.values, 0.2, seed)          # early stopping on held-out SITES
        X = self.X_train(kind, L, rows)
        e = DL.Ensemble(E_WFEATS if kind == "weekly" else E_DFEATS, L, E_SET[kind]["H"], seeds, E_SET[kind]["wd"],
                        E_SET[kind]["dropout"], seed, "log", 7 if kind == "weekly" else 1)
        return e.fit(X[tr], y_log[tr], X[~tr], y_log[~tr], epochs=E_SET[kind]["epochs"], batch=E_SET[kind]["batch"],
                     lr=E_SET[kind]["lr"], patience=E_SET[kind]["patience"])

    def weekly_pred(self, e, kind, stations, weeks):
        """weekly predictions for (station, week) pairs; daily models: mean of the 7 daily predictions"""
        if kind == "weekly":
            return to_ug(e, e.predict_raw(sequences(self.FWm, self.idxW, stations, weeks, e.L, 7))[0])
        st = np.repeat(np.asarray(stations), 7)
        dd = np.repeat(pd.to_datetime(np.asarray(weeks)).values, 7) + np.tile(np.arange(7) * np.timedelta64(1, "D"), len(stations))
        p = self.daily_pred(e, st, dd)
        return np.nanmean(p.reshape(-1, 7), axis=1)

    def daily_pred(self, e, stations, dates, chunk=30000):
        out = np.empty(len(stations))
        for s in range(0, len(stations), chunk):
            X = sequences(self.FD, self.idxD, stations[s:s + chunk], dates[s:s + chunk], e.L, 1)
            out[s:s + chunk] = to_ug(e, e.predict_raw(X)[0])
        return out


def e_rows(ed, kind, train, sites_subset):
    """training rows (weekly ground rows or daily ground rows) of the given sites"""
    if kind == "weekly":
        r = train[train.Site_ID.isin(sites_subset)].reset_index(drop=True)
        return r, np.log(np.maximum(r.y.values, 0.1))
    r = ed.gd[ed.gd.Site_ID.isin(sites_subset)].reset_index(drop=True)
    return r, np.log(np.maximum(r.NO2.values, 0.1))


def e_cv(ed, kind, L, train, folds, seeds):
    """out-of-site weekly predictions for every training row"""
    pred = np.full(len(train), np.nan)
    all_sites = set(train.Site_ID)
    for k, f in enumerate(folds):
        rows, ylog = e_rows(ed, kind, train, all_sites - f)
        e = ed.fit(kind, L, rows, ylog, seeds, SEED + 17 * k)
        te = train.Site_ID.isin(f).values
        pred[te] = ed.weekly_pred(e, kind, train.Anchor_Station.values[te], train.week_start.values[te])
    return pred


def main():
    t0 = time.time()
    meta1 = json.load(open(os.path.join(V2, "stage1_meta.json")))
    winner = meta1["winner"]
    say(f"=== Stage 2 LSTM (step 07d) | Stage 1 frozen {winner} | PyTorch {torch.__version__} CPU, {DL.THREADS} threads, "
        f"deterministic | seed {SEED}")
    say(f"   LSTM-E candidates {E_CAND} {E_SET}\n   LSTM-P candidates L = {P_CAND} weeks {P_SET}")
    say(f"   seeds: selection {SEL_SEEDS} ({SEL_FOLDS} site folds), leave-one-site-out {LOSO_SEEDS}, frozen models {FINAL_SEEDS}")

    TW = pd.read_csv(os.path.join(MODEL_DIR, "weekly_table.csv"), parse_dates=["week_start"])
    TM = pd.read_csv(os.path.join(MODEL_DIR, "monthly_table.csv"), parse_dates=["month"])
    for T in (TW, TM):
        if (T.period == "forward").any():
            raise SystemExit("forward rows found in a model table - the forward set must stay sealed")
    FW = pd.read_csv(os.path.join(C.BASE_DIR, "features", "station_weekly_features.csv"), parse_dates=["week_start"])
    SW = pd.read_csv(os.path.join(V2, "station_weekly_predictions.csv"), parse_dates=["week_start"])
    sites_tab = pd.read_csv(os.path.join(C.DIRS["tables"], "ground_sites.csv"))
    TWn = TW[np.isfinite(TW.y) & (TW.pollutant == "NO2") & np.isfinite(TW[S1.VCD]) & np.isfinite(TW.PBLH)]
    train = TWn[TWn.period == "train"].reset_index(drop=True)
    test = TWn[TWn.period == "test"].reset_index(drop=True)
    train_sites = set(train.Site_ID)
    FW = FW.merge(SW[["Station", "week_start", f"pred_{winner}"]].rename(columns={f"pred_{winner}": "pred_X1"}),
                  on=["Station", "week_start"], how="left")
    valid = (np.isfinite(FW[S1.VCD]) & np.isfinite(FW.PBLH)).values
    FW.loc[~valid, "pred_X1"] = np.nan                  # as the maps: no column / PBLH -> no estimate
    key = ["Anchor_Station", "week_start"]

    # ================================================================= 0-1. LSTM-E
    ed = EData(FW, train, sites_tab)
    say(f"\n=== 0. LSTM-E candidate selection (2023 only; out-of-site RMSE on the {len(train)} weekly ground rows, "
        f"{len(train_sites)} sites)")
    say(f"   training samples: weekly {len(train)} station-weeks | daily {len(ed.gd)} station-days "
        f"(temporal densification x{len(ed.gd) / max(len(train), 1):.1f})")
    folds = site_folds(train.Site_ID.values, SEL_FOLDS, SEED)
    cand = pd.DataFrame(index=train.index)
    for kind, L in E_CAND:
        cand[f"{kind}_{L}"] = e_cv(ed, kind, L, train, folds, SEL_SEEDS)
        print(f"   candidate {kind} {L}: done ({time.time() - t0:.0f} s)", flush=True)
    ok = cand.notna().all(axis=1) & np.isfinite(train.y)
    sel = pd.DataFrame([dict(candidate=c, RMSE=np.sqrt(np.mean((cand.loc[ok, c] - train.y[ok]) ** 2)),
                             R2=S1.metrics(train.y[ok], cand.loc[ok, c]).get("R2")) for c in cand.columns]).set_index("candidate")
    table(sel.round(3))
    best = sel.RMSE.idxmin()
    e_kind, e_L = best.split("_")[0], int(best.split("_")[1])
    say(f"   -> LSTM-E = {e_kind} sequences of {e_L} {'weeks' if e_kind == 'weekly' else 'days'} (lowest out-of-site RMSE; "
        "chosen before any test data is used)")

    say("\n=== 1. LSTM-E vs X1")
    oofE = e_cv(ed, e_kind, e_L, train, [{s} for s in sorted(train_sites)], LOSO_SEEDS)
    say(f"   leave-one-site-out finished ({time.time() - t0:.0f} s)")
    rows, ylog = e_rows(ed, e_kind, train, train_sites)
    eE = ed.fit(e_kind, e_L, rows, ylog, FINAL_SEEDS, SEED)
    eE.save(os.path.join(V2, "lstm_E.pt"), extra=dict(kind=e_kind, L=e_L, train_rows=len(rows), train_sites=len(train_sites)))
    FW["pred_E"] = ed.weekly_pred(eE, e_kind, FW.Station.values, FW.week_start.values)
    say(f"   frozen LSTM-E: members' best epochs {eE.best_epochs()}, smearing {eE.smear:.3f}")
    if e_kind == "daily":
        gain = (~valid) & np.isfinite(FW.pred_E.values) & FW.raster_set.eq("main").values
        say(f"   temporal densification: station-weeks with NO valid weekly column (X1 empty) that LSTM-E still estimates: "
            f"{int(gain.sum()):,} of {int((~valid & FW.raster_set.eq('main').values).sum()):,}")

    oof = pd.read_csv(os.path.join(V2, "oof_predictions.csv"), parse_dates=["week_start"])
    oX = oof[oof.key == f"A new station (leave one site out)|{winner}"][["Target_ID", "week_start", "pred"]]
    A = train[["Target_ID", "Site_ID", "week_start", "y"]].assign(pred_E=oofE).merge(
        oX.rename(columns={"pred": "pred_X1"}), on=["Target_ID", "week_start"], how="left")
    A["pred_mean"] = (A.pred_X1 + A.pred_E) / 2
    compare(A, ["pred_X1", "pred_E", "pred_mean"], "Scheme A: leave one site out, weekly, 2023")
    say("   (the LSTM-E design was selected on these 2023 rows, so its scheme-A score is slightly optimistic; 2024,\n"
        "    the independent stations and the 2026 forward test never influenced any choice)")
    g_ = A.dropna(subset=["pred_X1", "pred_E"]).groupby("Target_ID")
    cal = pd.DataFrame({"ratio_X1": g_.y.mean() / g_.pred_X1.mean(), "ratio_E": g_.y.mean() / g_.pred_E.mean(),
                        "weeks_2023": g_.size()}).clip(lower=1 / 3, upper=3).reset_index()
    cal.to_csv(os.path.join(V2, "site_calibration_2023.csv"), index=False)
    pts = FW.rename(columns={"Station": "Anchor_Station"})
    Tt = test[["Target_ID", "Site_ID", "y"] + key].merge(pts[key + ["pred_X1", "pred_E"]], on=key, how="left")
    Tt["pred_mean"] = (Tt.pred_X1 + Tt.pred_E) / 2
    compare(Tt, ["pred_X1", "pred_E", "pred_mean"], "TEST 2024 weekly, known stations (new year)")
    t2 = TW[(TW.raster_set == "T2") & np.isfinite(TW.y) & (TW.pollutant == "NO2")][["Target_ID", "Site_ID", "y"] + key]
    if len(t2):
        compare(t2.merge(pts[key + ["pred_X1", "pred_E"]], on=key), ["pred_X1", "pred_E"], "Riyadh 2019-2020 weekly (T2)")

    # ================================================================= 2. LSTM-P
    say("\n=== 2. LSTM-P (projection without satellite: weather + Stage 1 seasonal level)")
    P1 = FW[FW.raster_set == "main"][["Station", "week_start"] + P_MET + ["pred_X1"]].rename(
        columns={"Station": "key", "pred_X1": "target"})
    P1["key"] = "P|" + P1.key
    dd = pd.read_csv(os.path.join(C.DIRS["tables"], C.DISTRICT_DAILY_CSV),
                     usecols=[C.DISTRICT_ID, "date", "T2m", "Td2m", "U10", "V10", "SP", "PBLH", "PRECIP"], parse_dates=["date"])
    dd = dd[dd.date >= pd.Timestamp("2021-06-28")]
    dd["week_start"] = week_of(dd.date)
    g = dd.groupby([C.DISTRICT_ID, "week_start"])
    D = g[["T2m", "Td2m", "U10", "V10", "SP", "PBLH"]].mean()
    D["PRECIP"] = g.PRECIP.mean().clip(lower=0) * 7.0           # weekly total, as PRECIP_wk of the rasters
    D = D.reset_index()
    del dd
    D["RH"] = np.clip(A7.rh_from_t_td(D.T2m, D.Td2m), 0, 100)
    D["WS"] = np.hypot(D.U10, D.V10)
    mid = D.week_start + pd.Timedelta(days=3)
    D["doy_sin"], D["doy_cos"] = np.sin(2 * np.pi * mid.dt.dayofyear / 365.25), np.cos(2 * np.pi * mid.dt.dayofyear / 365.25)
    DW = pd.read_csv(os.path.join(C.DIRS["tables"], "District_NO2_weekly.csv"), parse_dates=["week_start"])
    D = D.merge(DW[[C.DISTRICT_ID, "week_start", "NO2_mean"]], on=[C.DISTRICT_ID, "week_start"], how="left")
    D = D.rename(columns={"NO2_mean": "target"})
    D["key"] = "D|" + D[C.DISTRICT_ID].astype(str)
    PP = pd.concat([P1, D[["key", "week_start"] + P_MET + ["target"]]], ignore_index=True)
    PP = PP.sort_values(["key", "week_start"]).reset_index(drop=True)
    mid = PP.week_start + pd.Timedelta(days=3)
    PP["cmonth"], PP["iso_year"] = mid.dt.month.values, PP.week_start.dt.isocalendar().year.astype(int).values
    okt = np.isfinite(PP.target) & (PP.target > 0)

    def seasonal(years):
        s = PP[okt & PP.iso_year.isin(years)]
        c = s.groupby(["key", "cmonth"]).target.mean()
        v = c.reindex(pd.MultiIndex.from_arrays([PP.key, PP.cmonth])).values
        return np.where(np.isfinite(v), v, PP.key.map(s.groupby("key").target.mean()).values)

    PP["clim"] = seasonal(C.TRAIN_YEARS)                         # fixed: 2022-2023
    PP["clim_roll"] = PP.clim
    for y in sorted(set(PP.iso_year)):
        if y > max(C.TRAIN_YEARS) + 1:                           # 2025 <- 2023-24, 2026 <- 2024-25
            rc = seasonal([y - 2, y - 1])
            PP.loc[PP.iso_year == y, "clim_roll"] = rc[(PP.iso_year == y).values]
    PP["log_clim"] = np.log(np.maximum(PP.clim, 0.1))
    FP = PP[P_FEATS].values.astype("float32")
    FPr = PP[P_MET].assign(log_clim=np.log(np.maximum(PP.clim_roll, 0.1))).values.astype("float32")
    idxP = seq_index(PP.key.values, PP.week_start.values)
    trw = (PP.iso_year.isin(C.TRAIN_YEARS) & okt & np.isfinite(PP.clim)).values
    uk = np.array(sorted(set(PP.key[trw])))
    vkeys = set(np.random.default_rng(SEED).choice(uk, max(2, int(0.15 * len(uk))), replace=False))
    isv_all = PP.key.isin(vkeys).values
    ylog_all = np.log(np.maximum(PP.target.values, 0.1)) - PP.log_clim.values

    def p_fit(L, rows, seeds):
        X = sequences(FP, idxP, PP.key.values[rows], PP.week_start.values[rows], L, 7)
        v = isv_all[rows]
        e = DL.Ensemble(P_FEATS, L, P_SET["H"], seeds, P_SET["wd"], P_SET["dropout"], SEED, "log", 7)
        e.fit(X[~v], ylog_all[rows][~v], X[v], ylog_all[rows][v], epochs=P_SET["epochs"], batch=P_SET["batch"],
              lr=P_SET["lr"], patience=P_SET["patience"])
        return e, X, v

    sel_rows = np.where(trw & (PP.week_start.values >= np.datetime64(P_SEL_FROM)))[0]
    say(f"   series: {sum(k.startswith('P|') for k in uk)} station cells + {sum(k.startswith('D|') for k in uk)} districts; "
        f"{len(vkeys)} series (15 %) held out for early stopping and selection")
    rows_sel = []
    for L in P_CAND:
        e, X, v = p_fit(L, sel_rows, SEL_SEEDS)
        pv = e.predict_raw(X[v])[0]
        rows_sel.append(dict(L_weeks=L, val_MSE_log=float(np.mean((pv - ylog_all[sel_rows][v]) ** 2)),
                             best_epochs=str(e.best_epochs())))
        print(f"   P candidate {L}: done ({time.time() - t0:.0f} s)", flush=True)
    selP = pd.DataFrame(rows_sel).set_index("L_weeks")
    table(selP.round(5))
    p_L = int(selP.val_MSE_log.idxmin())
    say(f"   -> LSTM-P = {p_L} weeks (lowest held-out-series loss on 2022-2023 target weeks from {P_SEL_FROM:%Y-%m-%d})")
    fin_rows = np.where(trw & (PP.week_start.values >= np.datetime64(pd.Timestamp("2022-01-03") + pd.Timedelta(weeks=p_L - 1))))[0]
    eP, _, _ = p_fit(p_L, fin_rows, FINAL_SEEDS)
    eP.save(os.path.join(V2, "lstm_P.pt"), extra=dict(L=p_L, train_sequences=int((~isv_all[fin_rows]).sum())))
    say(f"   frozen LSTM-P: {len(fin_rows):,} sequences, members' best epochs {eP.best_epochs()}, smearing {eP.smear:.3f} "
        f"({time.time() - t0:.0f} s)")

    def p_all(F, clim_col):
        out = np.empty(len(PP))
        for s in range(0, len(PP), 50000):
            X = sequences(F, idxP, PP.key.values[s:s + 50000], PP.week_start.values[s:s + 50000], p_L, 7)
            out[s:s + 50000] = eP.predict_raw(X)[0]
        return PP[clim_col].values * np.exp(np.clip(out, -5, 5)) * eP.smear

    PP["LSTM_P"] = p_all(FP, "clim")
    PP["LSTM_P_roll"] = p_all(FPr, "clim_roll")
    early = PP.week_start < pd.Timestamp("2022-01-03") + pd.Timedelta(weeks=p_L - 1)
    PP.loc[early, ["LSTM_P", "LSTM_P_roll"]] = np.nan
    pm = PP[PP.key.str.startswith("P|")].assign(Station=lambda x: x.key.str[2:])
    FW = FW.merge(pm[["Station", "week_start", "LSTM_P", "LSTM_P_roll", "clim"]].rename(
        columns={"LSTM_P": "pred_P", "LSTM_P_roll": "pred_Proll", "clim": "pred_S1clim"}), on=["Station", "week_start"], how="left")
    pts = FW.rename(columns={"Station": "Anchor_Station"})

    # ================================================================= 3. 2024 at the stations
    say("\n=== 3. 2024 at the ground stations (LSTM-P and the climatologies use no 2024 data at all)")
    gclim = train.assign(m=train.week_start.dt.month).groupby(["Target_ID", "m"]).y.mean()
    naive = train.assign(w=train.week_start.dt.isocalendar().week.astype(int)).groupby(["Target_ID", "w"]).y.mean()
    Tt = test[["Target_ID", "Site_ID", "y"] + key].merge(pts[key + ["pred_X1", "pred_E", "pred_P", "pred_S1clim"]], on=key, how="left")
    Tt["pred_groundclim"] = gclim.reindex(pd.MultiIndex.from_arrays([Tt.Target_ID, Tt.week_start.dt.month])).values
    Tt = Tt.merge(cal, on="Target_ID", how="left")
    Tt["pred_X1cal"], Tt["pred_Ecal"] = Tt.pred_X1 * Tt.ratio_X1, Tt.pred_E * Tt.ratio_E
    Tt["pred_naive"] = naive.reindex(pd.MultiIndex.from_arrays(
        [Tt.Target_ID, Tt.week_start.dt.isocalendar().week.astype(int)])).values
    compare(Tt, ["pred_X1", "pred_E", "pred_P", "pred_S1clim", "pred_groundclim"], "WEEKLY 2024, known stations")
    compare(Tt, ["pred_X1", "pred_E", "pred_P", "pred_naive"], "WEEKLY 2024 vs seasonal-naive (same ISO week of 2023)")
    compare(Tt, ["pred_X1cal", "pred_Ecal", "pred_groundclim", "pred_naive"],
            "WEEKLY 2024, models calibrated with the site's 2023 data vs baselines that use the same data")

    mon = None
    for c in ("pred_X1", "pred_E", "pred_P", "pred_Proll", "pred_S1clim"):
        m = S1.weekly_to_monthly(FW[["Station", "week_start", c]].rename(columns={c: "pred"})).rename(columns={"pred": c})
        mon = m if mon is None else mon.merge(m, on=["Station", "month"], how="outer")
    mon = mon.rename(columns={"Station": "Anchor_Station"})
    TMn = TM[np.isfinite(TM.y)].copy()
    TMn["network"] = TMn.network.astype(str)
    TMn["indep"] = (TMn.pollutant == "NO2") & ~TMn.Site_ID.isin(train_sites) & TMn.network.str.contains("UAE_monthly", na=False)
    TMn["own"] = (TMn.pollutant == "NO2") & TMn.Site_ID.isin(train_sites)
    prev = TMn[TMn.month.dt.year == 2023].assign(m=lambda x: x.month.dt.month).groupby(["Target_ID", "m"]).y.mean()
    M = TMn.merge(mon, on=["Anchor_Station", "month"], how="left")
    M["pred_persist"] = prev.reindex(pd.MultiIndex.from_arrays([M.Target_ID, M.month.dt.month])).values
    M = M.merge(cal, on="Target_ID", how="left")
    M["pred_X1cal"], M["pred_Ecal"] = M.pred_X1 * M.ratio_X1, M.pred_E * M.ratio_E
    M24 = M[M.month.dt.year == 2024]
    mm = ["pred_X1", "pred_E", "pred_P", "pred_S1clim", "pred_persist"]
    d_own = compare(M24[M24.own], mm, "MONTHLY 2024, daily-network sites (all their monthly series)")
    compare(M24[M24.own], ["pred_X1cal", "pred_Ecal", "pred_persist"], "MONTHLY 2024, daily-network sites, site-calibrated")
    d_ind = compare(M24[M24.indep], mm, "MONTHLY 2024, INDEPENDENT agency stations (never used)")
    compare(M24[M24.indep], ["pred_X1", "pred_E", "pred_P", "pred_S1clim"], "MONTHLY 2024, INDEPENDENT, without persistence")
    for lab, sel_ in (("2022-2023", M.month.dt.year.isin([2022, 2023])), ("2019-2021", M.raster_set.eq("T3"))):
        if (sel_ & M.indep).any():
            compare(M[sel_ & M.indep], ["pred_X1", "pred_E"], f"MONTHLY {lab}, INDEPENDENT agency stations")

    # ================================================================= 4. month by month 2024, month-wise change
    say("\n=== 4. Month by month 2024 (monthly values from weekly predictions; same rows for every model; ug/m3)")
    for lab, d in (("daily-network sites", d_own), ("independent agency stations", d_ind)):
        if not len(d):
            continue
        rows_ = []
        for mo, x in d.groupby(d.month.dt.month):
            r = dict(month=mo, n=len(x), obs=x.y.mean())
            for c in mm:
                r[f"RMSE_{NAMES[c]}"] = np.sqrt(np.mean((x[c] - x.y) ** 2))
            for c in ("pred_X1", "pred_E", "pred_P"):
                r[f"bias_{NAMES[c]}"] = (x[c] - x.y).mean()
            rows_.append(r)
        say(f"\n   {lab}:")
        table(pd.DataFrame(rows_).set_index("month").round(1))
    say("\n   Month-wise CHANGE 2024 minus 2023 (same station, same calendar month):")
    both = M[M.month.dt.year.isin([2023, 2024]) & (M.pollutant == "NO2")].assign(yr=lambda x: x.month.dt.year,
                                                                                  m=lambda x: x.month.dt.month)
    rows_ = []
    for c in ("pred_X1", "pred_E", "pred_P"):
        w = both.pivot_table(index=["Target_ID", "Site_ID", "m"], columns="yr", values=["y", c]).dropna()
        if len(w) < 5 or ("y", 2023) not in w or ("y", 2024) not in w:
            continue
        do, dp = (w[("y", 2024)] - w[("y", 2023)]).values, (w[(c, 2024)] - w[(c, 2023)]).values
        grp = np.where(w.index.get_level_values("Site_ID").isin(train_sites), "daily-network", "independent")
        for gname in ("daily-network", "independent"):
            k = grp == gname
            if k.sum() >= 5:
                rows_.append(dict(sites=gname, model=NAMES[c], n=int(k.sum()),
                                  r_change=np.corrcoef(do[k], dp[k])[0, 1] if np.std(dp[k]) > 0 else np.nan,
                                  same_sign=float(np.mean(np.sign(do[k]) == np.sign(dp[k]))),
                                  obs_mean_change=float(do[k].mean()), pred_mean_change=float(dp[k].mean())))
    if rows_:
        table(pd.DataFrame(rows_).set_index(["sites", "model"]).round(2))
    say("   (a 2023 climatology predicts no change: the no-skill reference; for X1 and LSTM-E at daily-network sites\n"
        "    2023 is the training year, so their 2023 values fit closely)")

    # ================================================================= 5. 2025 and the district projection database
    say("\n=== 5. 2025 - no NO2 ground data exist: LSTM-P (no satellite) vs Stage 1 (satellite), 744 districts")
    Dp = PP[PP.key.str.startswith("D|")].copy()
    Dp[C.DISTRICT_ID] = Dp.key.str[2:].astype(int)
    Dp = Dp[[C.DISTRICT_ID, "week_start", "target", "LSTM_P", "LSTM_P_roll", "clim", "clim_roll"]].rename(
        columns={"target": "NO2_stage1", "LSTM_P": "NO2_LSTM_P", "LSTM_P_roll": "NO2_LSTM_P_roll",
                 "clim": "NO2_stage1_clim_2022_23", "clim_roll": "NO2_stage1_clim_prev2yr"})
    Dp.to_csv(os.path.join(C.DIRS["tables"], "District_NO2_weekly_LSTMP.csv"), index=False)
    dm = None
    for c in ("NO2_stage1", "NO2_LSTM_P", "NO2_LSTM_P_roll"):
        m = S1.weekly_to_monthly(Dp[[C.DISTRICT_ID, "week_start", c]].rename(columns={C.DISTRICT_ID: "Station", c: "pred"}))
        m = m.rename(columns={"Station": C.DISTRICT_ID, "pred": c})
        dm = m if dm is None else dm.merge(m, on=[C.DISTRICT_ID, "month"], how="outer")
    dm.to_csv(os.path.join(C.DIRS["tables"], "District_NO2_monthly_LSTMP.csv"), index=False)
    dm["year"], dm["m"] = dm.month.dt.year, dm.month.dt.month
    rows_ = []
    for (yr, mo), x in dm[dm.year >= 2024].groupby(["year", "m"]):
        x = x.dropna()
        if len(x) > 10:
            rows_.append(dict(month=f"{yr}-{mo:02d}", districts=len(x), stage1=x.NO2_stage1.mean(), P=x.NO2_LSTM_P.mean(),
                              P_roll=x.NO2_LSTM_P_roll.mean(), r_P=np.corrcoef(x.NO2_stage1, x.NO2_LSTM_P)[0, 1],
                              r_Proll=np.corrcoef(x.NO2_stage1, x.NO2_LSTM_P_roll)[0, 1],
                              MAD_P=np.mean(np.abs(x.NO2_LSTM_P - x.NO2_stage1)),
                              MAD_Proll=np.mean(np.abs(x.NO2_LSTM_P_roll - x.NO2_stage1))))
    if rows_:
        say("   district means (ug/m3), correlation across districts and mean absolute difference to Stage 1,\n"
            "   month by month (2024: P roll = P fixed; 2026 = Jan-Sep):")
        table(pd.DataFrame(rows_).set_index("month").round(2))
    w = dm.pivot_table(index=[C.DISTRICT_ID, "m"], columns="year", values=["NO2_stage1", "NO2_LSTM_P", "NO2_LSTM_P_roll"]).dropna()
    if ("NO2_stage1", 2025) in w and ("NO2_stage1", 2024) in w and len(w) > 10:
        ds = w[("NO2_stage1", 2025)] - w[("NO2_stage1", 2024)]
        for c in ("NO2_LSTM_P", "NO2_LSTM_P_roll"):
            dp = w[(c, 2025)] - w[(c, 2024)]
            say(f"   month-wise change 2025 minus 2024 ({len(w)} district-months): Stage 1 {ds.mean():+.2f}, {c[4:]} "
                f"{dp.mean():+.2f} ug/m3; r {np.corrcoef(ds, dp)[0, 1]:.2f}; same sign {np.mean(np.sign(ds) == np.sign(dp)):.0%}")
    bah = M[(M.pollutant != "NO2") & M.month.dt.year.isin([2024, 2025])].assign(yr=lambda x: x.month.dt.year,
                                                                                 m=lambda x: x.month.dt.month)
    if len(bah):
        wb = bah.pivot_table(index=["Target_ID", "m"], columns="yr", values=["y", "pred_X1", "pred_P", "pred_Proll"]).dropna()
        if len(wb) >= 5 and ("y", 2025) in wb:
            do = wb[("y", 2025)] - wb[("y", 2024)]
            say(f"   Bahrain (reported as NOx; labelled comparison) 2025 vs 2024, {len(wb)} station-months: observed change "
                f"{do.mean():+.1f}; same sign as " + ", ".join(
                    f"{NAMES[c]} {np.mean(np.sign(do) == np.sign(wb[(c, 2025)] - wb[(c, 2024)])):.0%}"
                    for c in ("pred_X1", "pred_P", "pred_Proll")))

    # ================================================================= 6. applicability (07c dissimilarity index)
    say("\n=== 6. Applicability of the frozen Stage 1 model (07c dissimilarity index) at the test stations")
    try:
        import xgboost as xgb
        M7 = _load("maps07c", "07c_maps.py")
        bst = xgb.Booster()
        bst.load_model(os.path.join(V2, f"stage1_{winner}.json"))
        feats = meta1["features"]
        Xt = train[feats].values.astype("float32")
        wts = np.abs(bst.predict(xgb.DMatrix(Xt), pred_contribs=True)[:, :-1]).mean(axis=0)
        aoa = M7.AOA(Xt.astype("float64"), train.Site_ID.values, wts)
        dem = "DEM_3x3" if "DEM_3x3" in train else None
        rws = [dict(group="training sites (leave one site out)", n=len(train), DI_median=np.median(aoa.train_di),
                    inside_AOA=float(np.mean(aoa.train_di <= aoa.thr)), DEM_median=train[dem].median() if dem else np.nan)]
        indep_st = TMn[TMn.indep].Anchor_Station.unique()
        for lab, sel_ in (("Riyadh 2019-2020 (T2)", FW.raster_set.eq("T2")),
                          ("independent agency stations 2022-24", FW.Station.isin(indep_st) & FW.raster_set.eq("main")
                           & (FW.week_start.dt.year <= 2024)),
                          ("UAE 2019-2021 (T3)", FW.raster_set.eq("T3"))):
            d = FW[sel_.values & valid]
            if len(d):
                di = aoa.di(d[feats].values.astype("float64"))
                rws.append(dict(group=lab, n=len(d), DI_median=np.median(di), inside_AOA=float(np.mean(di <= aoa.thr)),
                                DEM_median=d[dem].median() if dem else np.nan))
        say(f"   AOA threshold {aoa.thr:.3f}" + (f"; training-site elevation {train[dem].min():.0f}-{train[dem].max():.0f} m" if dem else ""))
        table(pd.DataFrame(rws).set_index("group").round(3))
    except Exception as ex:                       # a diagnostic: never stops the step
        say(f"   skipped: {type(ex).__name__}: {ex}")

    # ================================================================= outputs
    FW[["Station", "week_start", "raster_set", "pred_X1", "pred_E", "pred_P", "pred_Proll", "pred_S1clim"]].to_csv(
        os.path.join(V2, "stage2_point_weekly.csv"), index=False)
    meta = dict(stage1_winner=winner, E_candidates=E_CAND, E_settings=E_SET, E_selected=dict(kind=e_kind, L=e_L),
                E_selection=sel.round(4).reset_index().to_dict("records"), P_candidates=P_CAND, P_settings=P_SET,
                P_selected_L=p_L, P_selection=selP.round(6).reset_index().to_dict("records"),
                seeds=dict(base=SEED, selection=SEL_SEEDS, loso=LOSO_SEEDS, final=FINAL_SEEDS), torch=torch.__version__,
                lstm_E_sha256=sha(os.path.join(V2, "lstm_E.pt")), lstm_P_sha256=sha(os.path.join(V2, "lstm_P.pt")),
                stage1_meta_sha256=sha(os.path.join(V2, "stage1_meta.json")),
                stage1_model_sha256=sha(os.path.join(V2, f"stage1_{winner}.json")),
                point_predictions_sha256=sha(os.path.join(V2, "stage2_point_weekly.csv")),
                site_calibration_sha256=sha(os.path.join(V2, "site_calibration_2023.csv")),
                frozen_at=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
    with open(os.path.join(V2, "stage2_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1, default=str)
    say(f"\n   frozen: lstm_E {meta['lstm_E_sha256'][:16]}... | lstm_P {meta['lstm_P_sha256'][:16]}... | "
        f"finished in {(time.time() - t0) / 60:.0f} min")
    say("   NEXT: 07e_forward_check.py - the sealed 2026 forward test, run once")
    p = os.path.join(C.DIRS["logs"], "stage2_report.txt")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("\n".join(OUT))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
