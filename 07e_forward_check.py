"""
STEP 7e - the sealed 2026 forward test (Abu Dhabi SDI hourly feed, Jan-Sep 2026).  Run ONCE, after 07d.
Local only, < 1 min.

What makes it a real forward test
  - the 2026 ground rows were written by 06b ONLY to tables/forward/ and their SHA-256 hashes were logged
    (logs/forward_test_hashes.txt) before any model existed; no earlier step reads them
  - this script first checks that the frozen models are unchanged (SHA-256 recorded by 07b_stage1 / 07d)
    and that the forward files still have the hashes logged by 06b; it stops otherwise
  - every scoring is appended to logs/forward_scored.txt with the model hashes.  If the forward set was
    already scored with DIFFERENT models, the report says so in capitals: that must be disclosed in the
    paper (the set is then no longer unseen for the new models)
  - 2026 is reported month by month (Jan-Sep) and compared with the same months of 2024; no annual 2026 value
Models scored (all made without any 2026 ground data): Stage 1 X1 (satellite, 2026 inputs, static layers
carried forward from 2025), LSTM-E (satellite), LSTM-P fixed (no satellite after 2023; seasonal level
2022-23), LSTM-P roll (same network; seasonal level from the Stage 1 estimates of 2024-2025), Stage 1
climatology 2022-23.  Ground baselines: climatology 2023-2024 and persistence from 2024 - the latest
year with NO2 ground data (none exist for 2025).
Output  logs/forward_report.txt
"""
import os, sys, json, hashlib, importlib.util, warnings
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
_spec = importlib.util.spec_from_file_location("stage1", os.path.join(CODE_DIR, "07b_stage1.py"))
S1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S1)

V2 = os.path.join(C.BASE_DIR, "model", "v2")
FWD = os.path.join(C.DIRS["tables"], "forward")
OUT = []
NAMES = {"pred_X1": "X1", "pred_E": "LSTM-E", "pred_P": "LSTM-P", "pred_Proll": "LSTM-P roll", "pred_S1clim": "S1 clim",
         "pred_groundclim": "ground clim", "pred_persist": "persist 2024",
         "pred_X1cal": "X1 site-cal", "pred_Ecal": "E site-cal"}


def say(s=""):
    print(s, flush=True)
    OUT.append(str(s))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def compare(d, models, title):
    d = d.dropna(subset=["y"] + models)
    say(f"\n   {title}  (identical rows: n {len(d)}, sites {d.Site_ID.nunique()})")
    if len(d) < 3:
        return d
    for c in models:
        m = S1.metrics(d.y, d[c])
        m.update(S1.boot_ci(d, c))
        say(f"   {NAMES[c]:<15} n {m['n']:>4}  R2 {m.get('R2', np.nan):5.2f} {m.get('R2_ci', ''):<15}r {m.get('r', np.nan):5.2f}  "
            f"RMSE {m.get('RMSE', np.nan):5.1f}  bias {m.get('bias', np.nan):+5.1f}  anomaly r {S1.anomaly_r(d, c):5.2f}")
    return d


def logged_hashes():
    """the LAST block written by 06b to logs/forward_test_hashes.txt -> {file name: sha256}"""
    p = os.path.join(C.DIRS["logs"], "forward_test_hashes.txt")
    if not os.path.exists(p):
        raise SystemExit("logs/forward_test_hashes.txt not found - the forward set has no recorded hashes")
    blocks, cur = [], None
    for ln in open(p, encoding="utf-8"):
        if ln.startswith("#"):
            cur = {}
            blocks.append((ln.strip(), cur))
        elif ln.strip() and cur is not None:
            parts = ln.split()
            cur[parts[1]] = parts[0]
    if not blocks:
        raise SystemExit("no hash block in logs/forward_test_hashes.txt")
    return blocks[-1]


def main():
    # ---------------- 1. the models must be the frozen ones
    m1 = json.load(open(os.path.join(V2, "stage1_meta.json")))
    m2p = os.path.join(V2, "stage2_meta.json")
    if not os.path.exists(m2p):
        raise SystemExit("model/v2/stage2_meta.json not found - run 07d_lstm.py first (models must be frozen)")
    m2 = json.load(open(m2p))
    checks = {"stage1_meta.json": m2["stage1_meta_sha256"], f"stage1_{m1['winner']}.json": m2["stage1_model_sha256"],
              "lstm_E.pt": m2["lstm_E_sha256"], "lstm_P.pt": m2["lstm_P_sha256"],
              "stage2_point_weekly.csv": m2["point_predictions_sha256"],
              "site_calibration_2023.csv": m2["site_calibration_sha256"]}
    say(f"=== Sealed 2026 forward test (step 07e) | {pd.Timestamp.now():%Y-%m-%d %H:%M}")
    for f, h in checks.items():
        now = sha(os.path.join(V2, f))
        if now != h:
            raise SystemExit(f"{f} changed after the models were frozen by 07d (sha {now[:12]} vs {h[:12]}) - re-run 07d first")
    say(f"   frozen models verified (Stage 1 {m1['winner']}, LSTM-E, LSTM-P; frozen {m2['frozen_at']})")

    # ---------------- 2. the forward files must be the ones hashed by 06b
    head, logged = logged_hashes()
    say(f"   forward hashes logged by 06b: {head}")
    for f in ("weekly", "monthly"):
        name = f"ground_{f}_forward_{C.FORWARD_START[:4]}.csv"
        now = sha(os.path.join(FWD, name))
        if logged.get(name) != now:
            raise SystemExit(f"{name}: sha256 {now[:16]} differs from the logged {str(logged.get(name))[:16]} - stopped")
        say(f"   {name}: sha256 {now[:16]}... matches")

    # ---------------- 3. scoring record (disclosure if the set was scored before with other models)
    rec = os.path.join(C.DIRS["logs"], "forward_scored.txt")
    model_key = f"{m2['stage1_model_sha256'][:16]} {m2['lstm_E_sha256'][:16]} {m2['lstm_P_sha256'][:16]}"
    if os.path.exists(rec):
        prev = [ln.strip() for ln in open(rec, encoding="utf-8") if ln.strip()]
        other = [ln for ln in prev if model_key not in ln]
        say(f"   earlier scorings: {len(prev)}" + (" (same frozen models: identical result)" if not other else ""))
        if other:
            say("   !!! THE FORWARD SET WAS ALREADY SCORED WITH DIFFERENT MODELS - it is no longer unseen for the current\n"
                "   !!! models; this must be disclosed in the paper:\n      " + "\n      ".join(other))
    with open(rec, "a", encoding="utf-8") as fh:
        fh.write(f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} | models {model_key} | forward {head}\n")

    # ---------------- 4. data
    sites = pd.read_csv(os.path.join(C.DIRS["tables"], "ground_sites.csv"))
    anchor = sites.set_index("Target_ID").Anchor_Station
    fw = pd.read_csv(os.path.join(FWD, f"ground_weekly_forward_{C.FORWARD_START[:4]}.csv"), parse_dates=["week_start"])
    fm = pd.read_csv(os.path.join(FWD, f"ground_monthly_forward_{C.FORWARD_START[:4]}.csv"), parse_dates=["month"])
    fw = fw[fw.pollutant == "NO2"].rename(columns={"NO2": "y"})
    fm = fm[fm.pollutant == "NO2"].rename(columns={"NO2": "y"})
    for d in (fw, fm):
        d["Anchor_Station"] = d.Target_ID.map(anchor)
    say(f"\n   forward rows: weekly {len(fw)} ({fw.Site_ID.nunique()} sites, {fw.week_start.min():%Y-%m-%d} -> "
        f"{fw.week_start.max():%Y-%m-%d}) | monthly {len(fm)} ({fm.month.min():%Y-%m} -> {fm.month.max():%Y-%m})")
    P = pd.read_csv(os.path.join(V2, "stage2_point_weekly.csv"), parse_dates=["week_start"])
    P = P[P.raster_set == "main"]
    models = ["pred_X1", "pred_E", "pred_P", "pred_Proll", "pred_S1clim"]
    TW = pd.read_csv(os.path.join(C.BASE_DIR, "model", "weekly_table.csv"), parse_dates=["week_start"])
    TM = pd.read_csv(os.path.join(C.BASE_DIR, "model", "monthly_table.csv"), parse_dates=["month"])
    hist_w = TW[(TW.pollutant == "NO2") & TW.week_start.dt.year.isin([2023, 2024]) & np.isfinite(TW.y)]
    hist_m = TM[(TM.pollutant == "NO2") & TM.month.dt.year.isin([2023, 2024]) & np.isfinite(TM.y)]
    gclim = hist_w.assign(m=hist_w.week_start.dt.month).groupby(["Target_ID", "m"]).y.mean()
    pers_w = hist_w[hist_w.week_start.dt.year == 2024].assign(w=lambda x: x.week_start.dt.isocalendar().week.astype(int)) \
        .groupby(["Target_ID", "w"]).y.mean()
    pers_m = hist_m[hist_m.month.dt.year == 2024].assign(m=lambda x: x.month.dt.month).groupby(["Target_ID", "m"]).y.mean()

    # ---------------- 5. weekly
    W = fw.merge(P.rename(columns={"Station": "Anchor_Station"})[["Anchor_Station", "week_start"] + models],
                 on=["Anchor_Station", "week_start"], how="left")
    W["pred_groundclim"] = gclim.reindex(pd.MultiIndex.from_arrays([W.Target_ID, W.week_start.dt.month])).values
    W["pred_persist"] = pers_w.reindex(pd.MultiIndex.from_arrays([W.Target_ID, W.week_start.dt.isocalendar().week.astype(int)])).values
    cal = pd.read_csv(os.path.join(V2, "site_calibration_2023.csv"))
    W = W.merge(cal, on="Target_ID", how="left")
    W["pred_X1cal"], W["pred_Ecal"] = W.pred_X1 * W.ratio_X1, W.pred_E * W.ratio_E
    say("\n=== Weekly, Jan-Sep 2026")
    compare(W, models, "all frozen models")
    compare(W, models + ["pred_groundclim", "pred_persist"], "with the ground baselines (sites with 2023-2024 data)")
    compare(W, ["pred_X1cal", "pred_Ecal", "pred_groundclim", "pred_persist"],
            "site-calibrated models (2023 ratio, out-of-site) vs baselines using the same site data")

    # ---------------- 6. monthly, month by month, change vs 2024
    mon = None
    for c in models:
        m = S1.weekly_to_monthly(P[["Station", "week_start", c]].rename(columns={c: "pred"})).rename(columns={"pred": c})
        mon = m if mon is None else mon.merge(m, on=["Station", "month"], how="outer")
    Mf = fm.merge(mon.rename(columns={"Station": "Anchor_Station"}), on=["Anchor_Station", "month"], how="left")
    Mf = Mf.merge(cal, on="Target_ID", how="left")
    Mf["pred_X1cal"], Mf["pred_Ecal"] = Mf.pred_X1 * Mf.ratio_X1, Mf.pred_E * Mf.ratio_E
    Mf["pred_persist"] = pers_m.reindex(pd.MultiIndex.from_arrays([Mf.Target_ID, Mf.month.dt.month])).values
    say("\n=== Monthly, Jan-Sep 2026 (monthly values from weekly predictions)")
    d = compare(Mf, models + ["pred_persist"], "all models and persistence (same month of 2024)")
    compare(Mf, ["pred_X1cal", "pred_Ecal", "pred_persist"], "site-calibrated models and persistence")
    if len(d):
        rows = []
        for mo, x in d.groupby(d.month.dt.month):
            r = dict(month=f"2026-{mo:02d}", n=len(x), obs=x.y.mean(), obs_2024=x.pred_persist.mean())
            for c in models:
                r[f"{NAMES[c]}"] = x[c].mean()
                r[f"RMSE_{NAMES[c]}"] = np.sqrt(np.mean((x[c] - x.y) ** 2))
            rows.append(r)
        say("\n   month by month (means over the same stations; ug/m3):")
        for ln in pd.DataFrame(rows).set_index("month").round(1).to_string().splitlines():
            say("   " + ln)
        mon24 = mon[mon.month.dt.year == 2024].assign(m=lambda x: x.month.dt.month).rename(columns={"Station": "Anchor_Station"})
        x = d.assign(m=d.month.dt.month).merge(mon24.drop(columns="month"), on=["Anchor_Station", "m"], suffixes=("", "_2024"))
        x = x.dropna(subset=["pred_persist"])
        if len(x) >= 5:
            do = x.y - x.pred_persist
            say(f"\n   change 2026 minus 2024, same station and month ({len(x)} station-months): observed mean {do.mean():+.1f} ug/m3")
            for c in models:
                dp = x[c] - x[f"{c}_2024"]
                ok = np.isfinite(dp)
                if ok.sum() >= 5:
                    say(f"      {NAMES[c]:<8} predicted mean change {dp[ok].mean():+.1f} | r {np.corrcoef(do[ok], dp[ok])[0, 1]:.2f} | "
                        f"same sign {np.mean(np.sign(do[ok]) == np.sign(dp[ok])):.0%}")
    say("\n   Note: 2026 static layers are carried forward from 2025 (STATIC_CARRY_FORWARD); report 2026 by month only.")
    p = os.path.join(C.DIRS["logs"], "forward_report.txt")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("\n".join(OUT))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
