"""
STEP 7g - Sensitivity runs.  Local only, ~20-60 min.  Run AFTER 07e (and 07f).

Rules (so that no result can leak into the frozen product)
  - the frozen models, maps and tables are NEVER changed; every variant writes only under
    sensitivity/<variant>/ and is reported side by side with a CONTROL rebuilt the same way
  - models are refitted with the FROZEN hyperparameters of X1 (no re-tuning), except the pre-declared
    alternative model X2, which is tuned exactly as in 07b_stage1 (inner grouped folds, 2023 only)
  - the 2026 forward set has been scored (07e); any 2026 number here is post-hoc and is labelled as such

A. Ground-data processing (06b re-run into sensitivity/<variant>/ with one rule changed; features are the
   same station rasters as always)
     control         the rules of the main run (checks that the re-run reproduces the main tables)
     daily_12h       a daily mean needs >= 12 valid hours instead of 18 (Oman, OpenAQ EAD, SDI)
     sdi_stuck_4h    SDI whole-number feed: stuck instrument at >= 4 identical hours (main: data-driven)
     sdi_stuck_12h   ... at >= 12 identical hours
   For each: X1 refitted with the frozen settings -> leave-one-site-out 2023, 2024 test; and the FROZEN
   X1 scored against that variant's 2024 and 2026 ground values (only the truth changes)
B. Model and input choices (no 06b re-run)
     X2              the pre-declared reduced, monotone XGBoost (07b_stage1) - 2024, independent, 2026
     static_2026     2026 annual layers from Jan-Sep 2026 instead of carried forward from 2025 (frozen X1)
     processor       X1 refitted on station-weeks with TROPOMI processor >= 2.4 only (if 2023 has any < 2.4)
     bahrain_no2     Bahrain's 'Nitrogen Oxides' series scored as if it were NO2 (frozen X1, monthly)
Output  logs/sensitivity_report.txt, logs/sensitivity_*.csv (for 08_tables);  sensitivity/<variant>/ (re-run tables and logs)
"""
import os, sys, json, time, hashlib, importlib.util, warnings
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
dune_env.ensure("maps")
import numpy as np, pandas as pd
import xgboost as xgb
import dune_config as C

for w in (FutureWarning, RuntimeWarning, UserWarning):
    warnings.filterwarnings("ignore", category=w)


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(CODE_DIR, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S1 = _load("stage1", "07b_stage1.py")
V2 = os.path.join(C.BASE_DIR, "model", "v2")
SENS = os.path.join(C.BASE_DIR, "sensitivity")
VARIANTS = {"control": {}, "daily_12h": {"GROUND_MIN_HOURS_DAY": 12},
            "sdi_stuck_4h": {"GROUND_FLATLINE_H_INTEGER": 4}, "sdi_stuck_12h": {"GROUND_FLATLINE_H_INTEGER": 12}}
OUT = []


def say(s=""):
    print(s, flush=True)
    OUT.append(str(s))


def table(df):
    for ln in df.to_string().splitlines():
        say("   " + ln)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def met(y, p):
    m = S1.metrics(y, p)
    return {k: m.get(k) for k in ("n", "R2", "r", "RMSE", "bias")}


def weekly_table(gw, sites, FW):
    """ground weekly targets + station features, exactly as 07a joins them"""
    anchor = sites.set_index("Target_ID")[["Anchor_Station", "Country", "Network", "Use"]]
    T = gw.join(anchor, on="Target_ID", rsuffix="_site").rename(columns={"NO2": "y"})
    T = T.merge(FW.drop(columns=["iso_year", "iso_week"], errors="ignore").rename(columns={"Station": "Anchor_Station"}),
                on=["Anchor_Station", "week_start"], how="inner")
    return T[np.isfinite(T.y) & (T.pollutant == "NO2") & np.isfinite(T[S1.VCD]) & np.isfinite(T.PBLH)].reset_index(drop=True)


def loso(feats, prm, D):
    """leave-one-site-out with FIXED settings (no tuning)"""
    p = np.full(len(D), np.nan)
    for s in sorted(set(D.Site_ID)):
        te = (D.Site_ID == s).values
        p[te] = S1.Tree("X1", feats).fit(D[~te].reset_index(drop=True), D.y.values[~te], prm).predict(D[te])
    return p


def run_06b(variant, changes):
    """06b re-run with one rule changed; outputs only under sensitivity/<variant>/"""
    base = os.path.join(SENS, variant)
    dirs = {"tables": os.path.join(base, "tables"), "logs": os.path.join(base, "logs")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    done = os.path.join(dirs["tables"], "ground_weekly.csv")
    if os.path.exists(done) and os.path.exists(os.path.join(base, "variant.json")) and \
            json.load(open(os.path.join(base, "variant.json"))) == changes:
        say(f"   {variant}: re-using sensitivity/{variant}/ (same rule change)")
        return dirs
    G6 = _load(f"gt06b_{variant}", "06b_ground_truth.py")      # fresh module = fresh state for every variant
    G6.C.DIRS = {**G6.C.DIRS, **dirs}                          # writes go to the variant folder ONLY
    for k, v in changes.items():
        if not hasattr(G6.C, k):
            raise SystemExit(f"dune_config has no setting {k}")
        setattr(G6.C, k, v)
    t0 = time.time()
    print(f"\n----- 06b re-run for variant '{variant}' {changes} -----", flush=True)
    G6.main()
    json.dump(changes, open(os.path.join(base, "variant.json"), "w"))
    say(f"   {variant}: 06b re-run in {time.time() - t0:.0f} s -> sensitivity/{variant}/")
    return dirs


def main():
    t0 = time.time()
    m1 = json.load(open(os.path.join(V2, "stage1_meta.json")))
    winner, feats, prm = m1["winner"], m1["features"], m1["params"]
    hashes_before = {f: sha(os.path.join(V2, f)) for f in (f"stage1_{winner}.json", "stage1_meta.json")}
    say(f"=== Sensitivity runs (step 07g) | frozen {winner} settings {prm} (refits use these; no re-tuning)")
    say("   2026 numbers are POST-HOC: the forward set was scored once by 07e before this step")
    FW = pd.read_csv(os.path.join(C.BASE_DIR, "features", "station_weekly_features.csv"), parse_dates=["week_start"])
    FWm = FW[FW.raster_set == "main"]
    SW = pd.read_csv(os.path.join(V2, "station_weekly_predictions.csv"), parse_dates=["week_start"])
    frozen = SW[["Station", "week_start", f"pred_{winner}"]].rename(columns={"Station": "Anchor_Station", f"pred_{winner}": "frozen"})
    bst = xgb.Booster()
    bst.load_model(os.path.join(V2, f"stage1_{winner}.json"))

    # ================================================================= A. ground-data processing
    say("\n=== A. Ground-data processing (06b re-run per variant; only the named rule differs)")
    rows = []
    for v, ch in VARIANTS.items():
        d = run_06b(v, ch)
        gw = pd.read_csv(os.path.join(d["tables"], "ground_weekly.csv"), parse_dates=["week_start"])
        if v == "control":                         # the re-run must reproduce the main tables
            main_w = pd.read_csv(os.path.join(C.DIRS["tables"], "ground_weekly.csv"), parse_dates=["week_start"])
            k = ["Target_ID", "week_start"]
            j = main_w.merge(gw, on=k, how="outer", suffixes=("_main", "_rerun"), indicator=True)
            both = j[j["_merge"] == "both"]
            dmax = float(np.nanmax(np.abs(both.NO2_main - both.NO2_rerun))) if len(both) else np.nan
            say(f"   control vs main tables: weekly rows main {len(main_w)}, re-run {len(gw)}, only in one of them "
                f"{int((j['_merge'] != 'both').sum())}, max |difference| {dmax:.2e} ug/m3 "
                + ("-> REPRODUCED" if (j['_merge'] != 'both').sum() == 0 and dmax < 1e-6 else "-> NOT identical: read with care"))
        sites = pd.read_csv(os.path.join(d["tables"], "ground_sites.csv"))
        T = weekly_table(gw, sites, FWm)
        tr, te = T[T.period == "train"].reset_index(drop=True), T[T.period == "test"].reset_index(drop=True)
        r = dict(variant=v, train_rows=len(tr), train_sites=tr.Site_ID.nunique(), train_mean=tr.y.mean())
        pA = loso(feats, prm, tr)
        r.update({f"A_{k}": x for k, x in met(tr.y, pA).items() if k in ("R2", "RMSE")})
        mdl = S1.Tree("X1", feats).fit(tr, tr.y.values, prm)
        r.update({f"test_refit_{k}": x for k, x in met(te.y, mdl.predict(te)).items() if k in ("R2", "RMSE")})
        tf = te.merge(frozen, on=["Anchor_Station", "week_start"], how="left")
        r.update({f"test_frozen_{k}": x for k, x in met(tf.y, tf.frozen).items() if k in ("n", "R2", "RMSE", "bias")})
        fp = os.path.join(d["tables"], "forward", f"ground_weekly_forward_{C.FORWARD_START[:4]}.csv")
        if os.path.exists(fp):
            fw = pd.read_csv(fp, parse_dates=["week_start"])
            fw = fw[fw.pollutant == "NO2"].rename(columns={"NO2": "y"})
            fw["Anchor_Station"] = fw.Target_ID.map(sites.set_index("Target_ID").Anchor_Station)
            fw = fw.merge(frozen, on=["Anchor_Station", "week_start"], how="left").dropna(subset=["frozen"])
            r.update(fwd_rows=len(fw), fwd_obs_mean=fw.y.mean(),
                     **{f"fwd_frozen_{k}": x for k, x in met(fw.y, fw.frozen).items() if k in ("R2", "r", "RMSE", "bias")})
        rows.append(r)
        print(f"   variant {v}: done ({time.time() - t0:.0f} s)", flush=True)
    A = pd.DataFrame(rows).set_index("variant")
    A.to_csv(os.path.join(C.DIRS["logs"], "sensitivity_processing.csv"))
    say("\n   training and 2024 (A_ = leave one site out with frozen settings; test_refit = refitted X1; test_frozen = the\n"
        "   frozen X1 against this variant's 2024 values):")
    table(A[[c for c in A.columns if not c.startswith("fwd")]].round(3))
    if any(c.startswith("fwd") for c in A.columns):
        say("\n   2026 forward (POST-HOC; frozen X1 against each variant's 2026 values - only the ground processing changes):")
        table(A[[c for c in A.columns if c.startswith("fwd")]].round(3))
    say("   reading: 'control' must reproduce the main run; a variant matters if its numbers move by more than the\n"
        "   bootstrap intervals of the main report (e.g. +-0.1 in R2)")

    # ================================================================= B. model and input choices
    say("\n=== B. Model and input choices")
    sites = pd.read_csv(os.path.join(C.DIRS["tables"], "ground_sites.csv"))
    TW = pd.read_csv(os.path.join(C.BASE_DIR, "model", "weekly_table.csv"), parse_dates=["week_start"])
    TM = pd.read_csv(os.path.join(C.BASE_DIR, "model", "monthly_table.csv"), parse_dates=["month"])
    TWn = TW[np.isfinite(TW.y) & (TW.pollutant == "NO2") & np.isfinite(TW[S1.VCD]) & np.isfinite(TW.PBLH)]
    train, test = TWn[TWn.period == "train"].reset_index(drop=True), TWn[TWn.period == "test"].reset_index(drop=True)
    fwp = os.path.join(C.DIRS["tables"], "forward", f"ground_weekly_forward_{C.FORWARD_START[:4]}.csv")
    fwd = pd.read_csv(fwp, parse_dates=["week_start"])
    fwd = fwd[fwd.pollutant == "NO2"].rename(columns={"NO2": "y"})
    fwd["Anchor_Station"] = fwd.Target_ID.map(sites.set_index("Target_ID").Anchor_Station)
    valid = (np.isfinite(FW[S1.VCD]) & np.isfinite(FW.PBLH)).values

    # ---- X2
    say("\n   X2 (pre-declared alternative: reduced features, monotone in the column; tuned as in 07b_stage1):")
    x2 = S1.fit_tuned("X2", S1.REDUCED, train, train.y.values, k_inner=min(10, train.Site_ID.nunique()))
    FWx = FW.assign(pred_X2=np.where(valid, x2.predict(FW), np.nan)).rename(columns={"Station": "Anchor_Station"})
    FWx = FWx.merge(frozen, on=["Anchor_Station", "week_start"], how="left")
    FWx.loc[~valid, "frozen"] = np.nan
    rows = []
    for lab, d in (("2024 weekly, known stations", test), ("2026 weekly, forward (post-hoc)", fwd)):
        x = d[["Target_ID", "Site_ID", "y", "Anchor_Station", "week_start"]].merge(
            FWx[["Anchor_Station", "week_start", "frozen", "pred_X2"]], on=["Anchor_Station", "week_start"]).dropna()
        for c, nm in (("frozen", winner), ("pred_X2", "X2")):
            rows.append(dict(test=lab, model=nm, **met(x.y, x[c])))
    mon = {}
    for c in ("frozen", "pred_X2"):
        mon[c] = S1.weekly_to_monthly(FWx[["Anchor_Station", "week_start", c]].rename(columns={"Anchor_Station": "Station", c: "pred"}))
    TMn = TM[np.isfinite(TM.y) & (TM.pollutant == "NO2")].copy()
    indep = ~TMn.Site_ID.isin(set(train.Site_ID)) & TMn.network.astype(str).str.contains("UAE_monthly", na=False)
    for lab, sel in (("2024 monthly, independent stations", indep & (TMn.period == "test")),
                     ("2022-23 monthly, independent stations", indep & (TMn.period == "train"))):
        x = TMn[sel]
        for c, nm in (("frozen", winner), ("pred_X2", "X2")):
            xx = x.merge(mon[c].rename(columns={"Station": "Anchor_Station"}), on=["Anchor_Station", "month"]).dropna(subset=["pred"])
            rows.append(dict(test=lab, model=nm, **met(xx.y, xx.pred)))
    pd.DataFrame(rows).to_csv(os.path.join(C.DIRS["logs"], "sensitivity_models.csv"), index=False)
    table(pd.DataFrame(rows).set_index(["test", "model"]).round(3))

    # ---- 2026 static layers from Jan-Sep 2026 instead of carried forward from 2025
    say("\n   static_2026: annual layers (VIIRS, NDVI) from Jan-Sep 2026 instead of carried forward from 2025 (frozen X1):")
    pw = os.path.join(C.DIRS["tables"], "Station_Static_partial_windows.csv")
    if os.path.exists(pw):
        PWt = pd.read_csv(pw)
        tag = [t for t in PWt.window.unique() if str(t).startswith(C.FORWARD_START[:4])]
        if tag:
            pw26 = PWt[PWt.window == tag[0]].set_index("Station")
            F26 = FW[(FW.raster_set == "main") & (FW.iso_year == int(C.FORWARD_START[:4]))].copy()
            swap = [c for c in pw26.columns if c in F26.columns and c.startswith(("VIIRS", "NDVI"))]
            base = bst.predict(xgb.DMatrix(F26[feats].values.astype("float32")))
            for c in swap:
                F26[c] = F26.Station.map(pw26[c]).values
            alt = bst.predict(xgb.DMatrix(F26[feats].values.astype("float32")))
            F26 = F26.assign(carry=base, partial=alt).rename(columns={"Station": "Anchor_Station"})
            ok26 = (np.isfinite(F26[S1.VCD]) & np.isfinite(F26.PBLH)).values
            x = fwd.merge(F26[ok26][["Anchor_Station", "week_start", "carry", "partial"]], on=["Anchor_Station", "week_start"])
            say(f"   window {tag[0]}, layers swapped {swap}; mean |change| of the 2026 estimates "
                f"{np.mean(np.abs(alt - base)[ok26]):.2f} ug/m3")
            st_tab = pd.DataFrame([dict(static="carried forward from 2025 (main)", **met(x.y, x.carry)),
                                   dict(static="Jan-Sep 2026", **met(x.y, x.partial))])
            st_tab.to_csv(os.path.join(C.DIRS["logs"], "sensitivity_static2026.csv"), index=False)
            table(st_tab.set_index("static").round(3))
        else:
            say("   no partial window for the forward year in Station_Static_partial_windows.csv - skipped")
    else:
        say("   tables/Station_Static_partial_windows.csv not found (written by 02_static_layers) - skipped")

    # ---- processor
    say("\n   processor: TROPOMI processor versions of the 2023 training rows: "
        + str(train.S5P_proc.value_counts().sort_index().to_dict()))
    old = train.S5P_proc < 2.4
    if old.any():
        sub = train[~old].reset_index(drop=True)
        p_all, p_sub = loso(feats, prm, train), loso(feats, prm, sub)
        say(f"   leave-one-site-out, frozen settings: all rows R2 {met(train.y, p_all)['R2']:.3f} | processor >= 2.4 only "
            f"({len(sub)} rows) R2 {met(sub.y, p_sub)['R2']:.3f}")
    else:
        say("   every training row is processor >= 2.4 - this sensitivity does not apply (stated in the paper)")

    # ---- Bahrain as NO2
    say("\n   bahrain_no2: Bahrain's 'Nitrogen Oxides' series scored as if NO2 (frozen X1, monthly from weekly):")
    bah = TM[np.isfinite(TM.y) & (TM.pollutant != "NO2")]
    if len(bah):
        x = bah.merge(mon["frozen"].rename(columns={"Station": "Anchor_Station"}), on=["Anchor_Station", "month"]).dropna(subset=["pred"])
        rows = [dict(years="all", **met(x.y, x.pred), rank_r=x[["y", "pred"]].corr("spearman").iloc[0, 1])]
        for yr, xx in x.groupby(x.month.dt.year):
            if len(xx) >= 5:
                rows.append(dict(years=str(yr), **met(xx.y, xx.pred), rank_r=xx[["y", "pred"]].corr("spearman").iloc[0, 1]))
        table(pd.DataFrame(rows).set_index("years").round(3))
        say("   (NOx >= NO2 at the same place, so a model of NO2 should come out BELOW these values (negative bias); a\n"
            "    positive bias means the series is not NOx in ug/m3 as labelled - units or definition unverified)")
    else:
        say("   no Bahrain rows in the monthly table - skipped")

    if {f: sha(os.path.join(V2, f)) for f in hashes_before} != hashes_before:
        raise SystemExit("a frozen model file changed during this step - this must never happen; report this")
    say(f"\n   frozen model files unchanged (verified) | finished in {(time.time() - t0) / 60:.0f} min")
    p = os.path.join(C.DIRS["logs"], "sensitivity_report.txt")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("\n".join(OUT))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
