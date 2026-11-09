"""
STEP 7h - GRU instead of LSTM: a post-hoc architecture sensitivity for Stage 2.  Local only, about as long as 07d
(about twice as long with --control).  Optional; it never changes the frozen products.

Run it AFTER 07d, 07e, 07f and 07g (do NOT re-run 07d):
    python 07h_gru_sensitivity.py              GRU run + comparison with the frozen LSTMs
    python 07h_gru_sensitivity.py --control    also re-runs the LSTM through the same harness first
                                               (shows that the harness reproduces the frozen 07d run)

What is identical to 07d (the fair comparison)
    07d_lstm.py itself is executed unchanged: the same data, features, candidates (LSTM-E: weekly 4/8/13 weeks,
    daily 28/56 days; LSTM-P: 4/8/13/26 weeks), hyperparameters, seeds, selection rule (2023 out-of-site RMSE
    for E, held-out series loss in 2022-2023 for P), early stopping, ensembles and outputs.  The ONLY change is
    the recurrent cell: torch.nn.GRU replaces torch.nn.LSTM (same hidden size, one layer, dropout + linear head).

What can NOT happen (protection of the frozen product)
    - every file 07d would write is redirected to sensitivity/gru/ (models, point predictions, district
      tables, report); the frozen model/v2/, tables/ and logs/ files are hashed before and after the run, and
      the script stops with an error if any of them changed
    - nothing is tuned on 2024 or 2026; the 2026 forward set was already scored once by 07e, so every 2026
      number below is POST-HOC and labelled so

Decision rule, fixed BEFORE the run (written into the report)
    The GRU is called "promising" only if, on identical rows, it improves on the frozen LSTM of the same role
    (E or P) by >= 0.05 R2 in BOTH 2024 tests at the training-network sites (weekly and monthly) with a paired
    site-bootstrap 95 % interval of the difference that excludes 0, and it is not worse by > 0.05 R2 at the
    independent agency stations.  2026 (post-hoc) can support but never establish a gain.

Outputs
    sensitivity/gru/model/        gru_E/gru_P ensembles (named lstm_*.pt by 07d), stage2_point_weekly.csv, stage2_meta.json
    sensitivity/gru/              District_NO2_*_LSTMP.csv (GRU versions), stage2_report.txt (the 07d report, GRU)
    sensitivity/gru/gru_vs_lstm.csv   every comparison row
    logs/gru_sensitivity_report.txt  
"""
import os, sys, json, time, shutil, hashlib, builtins, importlib.util, warnings
CODE_DIR = os.getcwd()                      # fallback when __file__ is undefined (notebook)
try:
    CODE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    if os.path.exists(os.path.join(os.getcwd(), "dune_config.py")):
        CODE_DIR = os.getcwd()
sys.path.insert(0, CODE_DIR)
if not os.path.exists(os.path.join(CODE_DIR, "07d_lstm.py")):
    raise SystemExit("07d_lstm.py not found in the code folder")
for _m in ("dune_config", "dune_env", "dune_gee", "dune_ground", "dune_lstm", "dune_results"):
    sys.modules.pop(_m, None)
import dune_env
dune_env.ensure("lstm")
import numpy as np, pandas as pd
import torch
from torch import nn
import dune_config as C

for w in (FutureWarning, RuntimeWarning, UserWarning):
    warnings.filterwarnings("ignore", category=w)

V2 = os.path.join(C.BASE_DIR, "model", "v2")
SENS = os.path.join(C.BASE_DIR, "sensitivity")
BOOT_N, BOOT_SEED = 2000, 20261014
GAIN, LOSS = 0.05, 0.05                       # decision rule thresholds (R2), fixed before the run
OUT = []


def say(s=""):
    print(s, flush=True)
    OUT.append(str(s))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(CODE_DIR, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------- the only model change
class GRUNet(nn.Module):
    """same interface and head as dune_lstm.Net; nn.GRU instead of nn.LSTM"""

    def __init__(self, D, H, dropout):
        super().__init__()
        self.gru = nn.GRU(D, H, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(H, 1)

    def forward(self, x):
        o, _ = self.gru(x)
        return self.head(self.drop(o[:, -1])).squeeze(1)


# ---------------------------------------------------------------------------- protection of the frozen files
def frozen_files():
    fs = [os.path.join(V2, f) for f in ("lstm_E.pt", "lstm_P.pt", "stage2_meta.json", "stage2_point_weekly.csv",
                                        "site_calibration_2023.csv", "stage1_meta.json", "station_weekly_predictions.csv",
                                        "oof_predictions.csv")]
    fs += [os.path.join(C.DIRS["tables"], f) for f in ("District_NO2_weekly_LSTMP.csv", "District_NO2_monthly_LSTMP.csv")]
    fs += [os.path.join(C.DIRS["logs"], "stage2_report.txt")]
    return [f for f in fs if os.path.exists(f)]


def run_07d(net_cls, out_dir, label):
    """execute 07d_lstm.main() unchanged except for the recurrent cell; every write goes to out_dir"""
    model_dir = os.path.join(out_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    meta1 = json.load(open(os.path.join(V2, "stage1_meta.json")))
    for f in ("stage1_meta.json", f"stage1_{meta1['winner']}.json", "station_weekly_predictions.csv", "oof_predictions.csv"):
        shutil.copy2(os.path.join(V2, f), os.path.join(model_dir, f))          # read-only inputs of 07d
    redirect = {"District_NO2_weekly_LSTMP.csv", "District_NO2_monthly_LSTMP.csv", "stage2_report.txt"}

    def _r(p):
        if isinstance(p, (str, os.PathLike)) and os.path.basename(str(p)) in redirect:
            return os.path.join(out_dir, os.path.basename(str(p)))
        return p
    to_csv0, open0 = pd.DataFrame.to_csv, builtins.open

    def to_csv(self, path_or_buf=None, *a, **k):
        return to_csv0(self, _r(path_or_buf), *a, **k)

    def open_(file, mode="r", *a, **k):
        return open0(_r(file) if any(c in mode for c in "wax+") else file, mode, *a, **k)
    D7 = _load(f"stage2_{label}", "07d_lstm.py")
    D7.V2 = model_dir                       # models, point predictions and meta of this run
    D7.DL.Net = net_cls                     # used by _fit_one and Ensemble.load inside dune_lstm
    pd.DataFrame.to_csv, builtins.open = to_csv, open_
    try:
        D7.main()
    finally:
        pd.DataFrame.to_csv, builtins.open = to_csv0, open0
    return os.path.join(model_dir, "stage2_point_weekly.csv")


# ---------------------------------------------------------------------------- comparison helpers
def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    sst = np.sum((y - y.mean()) ** 2)
    return 1 - np.sum((y - p) ** 2) / sst if sst > 0 else np.nan


def paired_delta(d, a, b, seed=BOOT_SEED):
    """R2(b) - R2(a) on identical rows, 95 % interval from a cluster bootstrap over sites (paired)"""
    rng = np.random.default_rng(seed)
    g = {s: x for s, x in d.groupby("Site_ID")}
    sites = np.array(list(g))
    dl = []
    if len(sites) >= 4:
        for _ in range(BOOT_N):
            x = pd.concat([g[s] for s in rng.choice(sites, len(sites))])
            dl.append(r2(x.y, x[b]) - r2(x.y, x[a]))
    lo, hi = (np.nanpercentile(dl, 2.5), np.nanpercentile(dl, 97.5)) if dl else (np.nan, np.nan)
    return r2(d.y, d[b]) - r2(d.y, d[a]), lo, hi


def main():
    t0 = time.time()
    control = "--control" in sys.argv
    if not os.path.exists(os.path.join(V2, "stage2_meta.json")):
        raise SystemExit("run 07d (and 07e) first: the frozen Stage 2 models are the reference")
    before = {f: sha(f) for f in frozen_files()}
    out = os.path.join(SENS, "gru")
    os.makedirs(out, exist_ok=True)
    say(f"=== 7h GRU sensitivity (post-hoc) | PyTorch {torch.__version__} CPU | frozen files hashed: {len(before)}")
    say("   07d executed unchanged except nn.GRU for nn.LSTM; outputs -> sensitivity/gru/")
    say(f"   decision rule (fixed before the run): 'promising' only if GRU - LSTM >= +{GAIN} R2 in BOTH 2024 tests at the\n"
        f"   training-network sites (weekly, monthly) with a paired site-bootstrap 95 % CI excluding 0, and not worse by\n"
        f"   more than {LOSS} R2 at the independent stations; 2026 is post-hoc (supporting only)")

    if control:
        say("\n=== Control: LSTM through the same harness (must reproduce the frozen 07d predictions)")
        import dune_lstm
        pc = run_07d(dune_lstm.Net, os.path.join(SENS, "gru_control_lstm"), "control")
        A = pd.read_csv(os.path.join(V2, "stage2_point_weekly.csv"))
        B = pd.read_csv(pc)
        m = A.merge(B, on=["Station", "week_start", "raster_set"], suffixes=("", "_c"))
        for c in ("pred_E", "pred_P", "pred_Proll"):
            dmax = np.nanmax(np.abs(m[c] - m[c + "_c"]))
            say(f"   {c}: max |frozen - control| = {dmax:.2e} ug/m3 -> {'REPRODUCED' if dmax < 1e-3 else 'NOT reproduced (report the difference)'}")

    say("\n=== GRU run (full 07d pipeline)")
    pg = run_07d(GRUNet, out, "gru")
    say(f"   GRU run finished ({(time.time() - t0) / 60:.0f} min); its full 07d-style report: sensitivity/gru/stage2_report.txt")
    gm = json.load(open(os.path.join(out, "model", "stage2_meta.json")))
    fm = json.load(open(os.path.join(V2, "stage2_meta.json")))
    say(f"   selected designs: LSTM-E {fm['E_selected']} vs GRU-E {gm['E_selected']}; "
        f"LSTM-P L={fm['P_selected_L']} vs GRU-P L={gm['P_selected_L']}")
    sel = pd.DataFrame(fm["E_selection"]).set_index("candidate")[["RMSE"]].rename(columns={"RMSE": "LSTM_RMSE"}).join(
        pd.DataFrame(gm["E_selection"]).set_index("candidate")[["RMSE"]].rename(columns={"RMSE": "GRU_RMSE"}))
    say("   E candidates, 2023 out-of-site RMSE (ug/m3):")
    for ln in sel.round(3).to_string().splitlines():
        say("      " + ln)

    after = {f: sha(f) for f in before}
    changed = [f for f in before if before[f] != after[f]]
    if changed:
        raise SystemExit(f"FROZEN FILES CHANGED - do not use these results: {changed}")
    say(f"   frozen files unchanged (verified, {len(before)} files)")

    # ---------------------------------------------------------------- identical-row comparison
    R = _load("dune_results", "dune_results.py")
    r = R.Results()
    G = pd.read_csv(pg, parse_dates=["week_start"]).rename(columns={"Station": "Anchor_Station", "pred_E": "pred_GE",
                                                                    "pred_P": "pred_GP", "pred_Proll": "pred_GProll"})
    P = r.points.merge(G[["Anchor_Station", "week_start", "raster_set", "pred_GE", "pred_GP", "pred_GProll"]],
                       on=["Anchor_Station", "week_start", "raster_set"], how="left")
    r._c["points"] = P                                     # monthly / forward frames now carry the GRU columns
    T = r.weekly_test()
    M = r.monthly_eval()
    sets = [("2024 weekly, training-network sites", T, "test_w"),
            ("2024 monthly, training-network sites", M[M.own & (M.month.dt.year == 2024)], "test_m"),
            ("2024 monthly, independent stations", M[M.indep & (M.month.dt.year == 2024)], "indep"),
            ("2026 weekly, forward (POST-HOC)", r.forward("weekly"), "fwd_w"),
            ("2026 monthly, forward (POST-HOC)", r.forward("monthly"), "fwd_m")]
    rows, verdict = [], {}
    for lab, d, key in sets:
        for role, a, b in (("E (estimation)", "pred_E", "pred_GE"), ("P (projection)", "pred_P", "pred_GP"),
                           ("P rolling", "pred_Proll", "pred_GProll")):
            if a not in d or b not in d:
                continue
            x = d.dropna(subset=["y", a, b, "pred_X1"])
            if len(x) < 20 or x.Site_ID.nunique() < 4:
                continue
            dlt, lo, hi = paired_delta(x, a, b)
            rows.append(dict(test=lab, role=role, n=len(x), sites=x.Site_ID.nunique(), R2_X1=r2(x.y, x.pred_X1),
                             R2_LSTM=r2(x.y, x[a]), R2_GRU=r2(x.y, x[b]), dR2_GRU_minus_LSTM=dlt, CI_lo=lo, CI_hi=hi,
                             RMSE_LSTM=np.sqrt(np.mean((x.y - x[a]) ** 2)), RMSE_GRU=np.sqrt(np.mean((x.y - x[b]) ** 2)),
                             bias_LSTM=np.mean(x[a] - x.y), bias_GRU=np.mean(x[b] - x.y)))
            verdict[(key, role)] = (dlt, lo, hi)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(out, "gru_vs_lstm.csv"), index=False)
    say("\n=== GRU vs frozen LSTM on identical rows (R2; dR2 = GRU - LSTM with paired site-bootstrap 95 % CI)")
    if len(D):
        for ln in D.drop(columns=["RMSE_LSTM", "RMSE_GRU", "bias_LSTM", "bias_GRU"]).round(3).to_string(index=False).splitlines():
            say("   " + ln)
        say("\n   RMSE and bias (ug/m3):")
        for ln in D[["test", "role", "RMSE_LSTM", "RMSE_GRU", "bias_LSTM", "bias_GRU"]].round(2).to_string(index=False).splitlines():
            say("   " + ln)

    say("\n=== Verdict (pre-declared rule)")
    for role in ("E (estimation)", "P (projection)"):
        need = [verdict.get(("test_w", role)), verdict.get(("test_m", role))]
        ind = verdict.get(("indep", role))
        if any(v is None for v in need):
            say(f"   {role}: not enough identical rows to apply the rule")
            continue
        gain = all(v[0] >= GAIN and v[1] > 0 for v in need)
        safe = ind is None or ind[0] >= -LOSS
        say(f"   {role}: 2024 weekly dR2 {need[0][0]:+.3f} [{need[0][1]:+.3f}, {need[0][2]:+.3f}], monthly {need[1][0]:+.3f} "
            f"[{need[1][1]:+.3f}, {need[1][2]:+.3f}]" + (f", independent {ind[0]:+.3f}" if ind else "")
            + f" -> {'PROMISING (consider for the main pipeline; disclose as post-hoc)' if gain and safe else 'NOT promising - keep the LSTM; report GRU as a sensitivity'}")
    say("   2026 rows above are post-hoc: they may support but never establish a gain (the forward test was scored once by 07e)")
    say(f"\n   finished in {(time.time() - t0) / 60:.0f} min")
    p = os.path.join(C.DIRS["logs"], "gru_sensitivity_report.txt")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("\n".join(OUT))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
