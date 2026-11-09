"""
STEP 7f - POST-HOC diagnostics of the 2026 forward test.  Local only, ~2-5 min.  Run AFTER 07e.

Nothing is fitted, tuned or changed here: the frozen models and their predictions are only read.  The
2026 forward set was already scored once by 07e (logs/forward_scored.txt); this step explains that
result and must be reported as post-hoc analysis.  It stops if any frozen model changed after 07e.

Question: why are observed 2026 values 7-11 ug/m3 above the same stations in 2023-2024, and why is X1
biased low by ~5 ug/m3 in 2026?  Candidate explanations, each tested with the data we have
  A  station composition - different stations report in different months (month-by-month tables mix
     station sets); repeated on a FIXED PANEL of stations present in every month
  B  a real increase - if NO2 rose, the satellite column at the same stations should rise too.  Per
     station: observed change, column change (NO2_trop_3x3) and X1 change, 2026 vs the same calendar
     months of 2024 (the latest year with ground NO2)
  C  the satellite processor - all 2026 data are TROPOMI 02.09.01 (switch in late 2025).  A step in the
     column at the switch is estimated GCC-wide from the district daily table: column on each day /
     column on the same days one year earlier, 60 days before vs 60 days after the switch
     (difference-in-differences), with a placebo at the same calendar date one year earlier and a
     week-block bootstrap interval
  D  the ground data source - 2023-2024 Abu Dhabi values come from OpenAQ (agency-validated daily
     means), 2026 from the SDI hourly feed with our QC.  The two feeds never overlap in time, so this
     cannot be separated from a real change with ground data alone; B and C are the evidence.  The
     QC-rule sensitivity of the 2026 values is in 07g.
  E  applicability - dissimilarity index (07c) of the 2026 station-weeks: errors inside vs outside the AOA
Output  logs/forward_diagnostics.txt  
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
dune_env.ensure("maps")
import numpy as np, pandas as pd
import dune_config as C

for w in (FutureWarning, RuntimeWarning, UserWarning):
    warnings.filterwarnings("ignore", category=w)


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(CODE_DIR, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S1 = _load("stage1", "07b_stage1.py")
A7 = _load("features07a", "07a_build_features.py")
V2 = os.path.join(C.BASE_DIR, "model", "v2")
FWD = os.path.join(C.DIRS["tables"], "forward")
SEED = 20261014
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


def met(d, c):
    m = S1.metrics(d.y, d[c])
    return dict(n=m["n"], sites=d.Site_ID.nunique(), R2=m.get("R2"), r=m.get("r"), RMSE=m.get("RMSE"), bias=m.get("bias"))


def main():
    m1 = json.load(open(os.path.join(V2, "stage1_meta.json")))
    m2 = json.load(open(os.path.join(V2, "stage2_meta.json")))
    for f, h in {f"stage1_{m1['winner']}.json": m2["stage1_model_sha256"], "stage2_point_weekly.csv": m2["point_predictions_sha256"],
                 "lstm_E.pt": m2["lstm_E_sha256"], "lstm_P.pt": m2["lstm_P_sha256"]}.items():
        if sha(os.path.join(V2, f)) != h:
            raise SystemExit(f"{f} changed after the models were frozen - the diagnostics must use the scored models")
    rec = os.path.join(C.DIRS["logs"], "forward_scored.txt")
    if not os.path.exists(rec):
        raise SystemExit("logs/forward_scored.txt not found - run 07e first (the diagnostics explain a scored result)")
    say("=== 2026 forward test - POST-HOC diagnostics (step 07f) | frozen models verified, nothing refitted")
    say("   scored by 07e: " + open(rec, encoding="utf-8").read().strip().splitlines()[-1])

    sites = pd.read_csv(os.path.join(C.DIRS["tables"], "ground_sites.csv"))
    anchor = sites.set_index("Target_ID").Anchor_Station
    fw = pd.read_csv(os.path.join(FWD, f"ground_weekly_forward_{C.FORWARD_START[:4]}.csv"), parse_dates=["week_start"])
    fw = fw[fw.pollutant == "NO2"].rename(columns={"NO2": "y"})
    fw["Anchor_Station"] = fw.Target_ID.map(anchor)
    TW = pd.read_csv(os.path.join(C.BASE_DIR, "model", "weekly_table.csv"), parse_dates=["week_start"])
    hist = TW[(TW.pollutant == "NO2") & np.isfinite(TW.y) & TW.week_start.dt.year.isin([2023, 2024])].copy()
    FW = pd.read_csv(os.path.join(C.BASE_DIR, "features", "station_weekly_features.csv"), parse_dates=["week_start"])
    FW = FW[FW.raster_set == "main"]
    P = pd.read_csv(os.path.join(V2, "stage2_point_weekly.csv"), parse_dates=["week_start"])
    P = P[P.raster_set == "main"].merge(FW[["Station", "week_start", S1.VCD, "PBLH", "S5P_proc"]], on=["Station", "week_start"], how="left")
    pts = P.rename(columns={"Station": "Anchor_Station"})
    W = fw.merge(pts, on=["Anchor_Station", "week_start"], how="left")
    W["month"] = W.week_start.dt.month
    W = W.dropna(subset=["pred_X1"])
    say(f"   2026 station-weeks with an X1 estimate: {len(W)} at {W.Site_ID.nunique()} sites")

    # ------------------------------------------------------------- A. composition / fixed panel
    say("\n=== A. Station composition by month (weekly rows) and a FIXED PANEL")
    comp = W.groupby("month").agg(stations=("Target_ID", "nunique"), weeks=("y", "size"), obs=("y", "mean"),
                                  X1=("pred_X1", "mean"))
    comp["bias_X1"] = comp.X1 - comp.obs
    table(comp.round(1))
    months = sorted(W.month.unique())
    present = W.groupby("Target_ID").month.nunique()
    panel = present[present >= max(1, len(months) - 1)].index        # in all months but at most one
    Wp = W[W.Target_ID.isin(panel)]
    say(f"   fixed panel: {len(panel)} stations reporting in >= {max(1, len(months) - 1)} of {len(months)} months")
    if len(Wp) >= 20:
        rows = [dict(model=c, **met(Wp, c)) for c in ("pred_X1", "pred_E", "pred_P", "pred_S1clim")]
        table(pd.DataFrame(rows).set_index("model").round(2))
        mp = Wp.groupby("month").agg(stations=("Target_ID", "nunique"), obs=("y", "mean"), X1=("pred_X1", "mean"))
        mp["bias_X1"] = mp.X1 - mp.obs
        table(mp.round(1))

    # ------------------------------------------------------------- B. real change? obs vs column vs X1
    say("\n=== B. Change 2026 vs the same calendar months of 2024, per station (weekly means; same months only)")
    h24 = hist[hist.week_start.dt.year == 2024].assign(month=lambda x: x.week_start.dt.month)
    h24 = h24.merge(pts[["Anchor_Station", "week_start", "pred_X1"]], on=["Anchor_Station", "week_start"], how="left")
    rows = []
    for t, d26 in W.groupby("Target_ID"):
        d24 = h24[h24.Target_ID == t]
        mo = sorted(set(d26.month) & set(d24.month))
        if not mo:
            continue
        a, b = d26[d26.month.isin(mo)], d24[d24.month.isin(mo)]
        rows.append(dict(Target_ID=t, station=str(anchor.get(t, "")).split("|")[-1][:22], months=len(mo),
                         obs_2024=b.y.mean(), obs_2026=a.y.mean(), obs_change=a.y.mean() - b.y.mean(),
                         column_change_pct=100 * (a[S1.VCD].mean() / b[S1.VCD].mean() - 1),
                         X1_change=a.pred_X1.mean() - b.pred_X1.mean()))
    B = pd.DataFrame(rows)
    if len(B):
        table(B.set_index("Target_ID").round(1))
        say(f"   stations {len(B)} | observed change mean {B.obs_change.mean():+.1f} ug/m3 ({100 * B.obs_change.mean() / B.obs_2024.mean():+.0f} %), "
            f"rose at {np.mean(B.obs_change > 0):.0%} of stations | column change mean {B.column_change_pct.mean():+.0f} % | "
            f"X1 change mean {B.X1_change.mean():+.1f} ug/m3")
        if len(B) >= 5:
            say(f"   across stations: r(observed change, column change) {np.corrcoef(B.obs_change, B.column_change_pct)[0, 1]:.2f} | "
                f"r(observed change, X1 change) {np.corrcoef(B.obs_change, B.X1_change)[0, 1]:.2f}")
        say("   reading: a real increase should appear in the column as well; a rise in the ground values with a flat\n"
            "   column, at nearly all stations at once, points to the ground data (source / QC) or to surface-only change")
    else:
        say("   no station has 2024 and 2026 data in the same calendar months")

    # ------------------------------------------------------------- C. processor step at the 02.09.01 switch
    say("\n=== C. TROPOMI processor switch to 02.09.01: GCC-wide step in the column (district daily table)")
    dd = pd.read_csv(os.path.join(C.DIRS["tables"], C.DISTRICT_DAILY_CSV), usecols=[C.DISTRICT_ID, "date", "NO2_trop", "S5P_proc"],
                     parse_dates=["date"])
    day = dd.groupby("date").agg(col=("NO2_trop", "median"), proc=("S5P_proc", "first")).reset_index()
    day["procn"] = day.proc.map(A7.proc_number)
    first = day[day.procn >= 2.9].date.min()
    if pd.isna(first):
        say("   no 02.09.x days found - skipped")
    else:
        say(f"   first day processed with >= 2.9: {first:%Y-%m-%d} | days per version 2025-2026: "
            + str(day[day.date.dt.year >= 2025].proc.value_counts().to_dict()))
        col = day.set_index("date").col

        def ratio_series(t0):
            idx = pd.date_range(t0 - pd.Timedelta(days=60), t0 + pd.Timedelta(days=59))
            cur = col.reindex(idx)
            prev = col.rolling(7, center=True, min_periods=3).mean().reindex(idx - pd.DateOffset(years=1)).values
            r = np.log(cur.values / prev)
            return pd.DataFrame({"date": idx, "after": idx >= t0, "lr": r}).dropna()

        def step(t0):
            d = ratio_series(t0)
            est = d[d.after].lr.mean() - d[~d.after].lr.mean()
            d["blk"] = ((d.date - d.date.min()).dt.days // 7)
            rng = np.random.default_rng(SEED)
            bs = []
            for _ in range(1000):
                b = []
                for side in (True, False):
                    blks = d[d.after == side].blk.unique()
                    pick = rng.choice(blks, len(blks))
                    b.append(pd.concat([d[(d.blk == k) & (d.after == side)] for k in pick]).lr.mean())
                bs.append(b[0] - b[1])
            return 100 * (np.exp(est) - 1), 100 * (np.exp(np.percentile(bs, [2.5, 97.5])) - 1), len(d)

        s, ci, n = step(first)
        sp, cip, npl = step(first - pd.DateOffset(years=1))
        say(f"   column step at the switch (this year / same days last year, 60 days after vs before): {s:+.1f} % "
            f"[{ci[0]:+.1f}, {ci[1]:+.1f}] ({n} days)")
        say(f"   placebo, same calendar date one year earlier (no switch):                          {sp:+.1f} % "
            f"[{cip[0]:+.1f}, {cip[1]:+.1f}] ({npl} days)")
        say("   reading: a step clearly outside the placebo range means the new processor changed the column level;\n"
            "   X1 (trained on 2.4-2.6) then sees a shifted input.  A step near zero rules the processor out.")

    # ------------------------------------------------------------- D. source
    say("\n=== D. Ground data source")
    say("   2023-2024 Abu Dhabi weekly values: OpenAQ daily means (agency data); 2026: SDI hourly feed, our QC (daily >= "
        f"{C.GROUND_MIN_HOURS_DAY} h, stuck rule chosen from the data).\n   The feeds do not overlap in time, so a source "
        "offset cannot be measured directly; see B (column) and C (processor), and 07g for the 2026 QC sensitivity.")
    if len(B) and B.obs_change.mean() > 2 and abs(B.column_change_pct.mean()) < 10:
        say(f"   the observations rose by {B.obs_change.mean():+.1f} ug/m3 while the column changed {B.column_change_pct.mean():+.0f} %:\n"
            "   a source / QC offset (or a surface-only change) is the most likely explanation; state it as unresolved.")

    # ------------------------------------------------------------- E. applicability
    say("\n=== E. 2026 errors inside vs outside the area of applicability (07c definition)")
    try:
        import xgboost as xgb
        M7 = _load("maps07c", "07c_maps.py")
        bst = xgb.Booster()
        bst.load_model(os.path.join(V2, f"stage1_{m1['winner']}.json"))
        feats = m1["features"]
        tr = TW[(TW.period == "train") & np.isfinite(TW.y) & (TW.pollutant == "NO2") & np.isfinite(TW[S1.VCD]) & np.isfinite(TW.PBLH)]
        Xt = tr[feats].values.astype("float32")
        aoa = M7.AOA(Xt.astype("float64"), tr.Site_ID.values,
                     np.abs(bst.predict(xgb.DMatrix(Xt), pred_contribs=True)[:, :-1]).mean(axis=0))
        Wf = W.merge(FW.rename(columns={"Station": "Anchor_Station"})[["Anchor_Station", "week_start"] + [f for f in feats if f not in W]],
                     on=["Anchor_Station", "week_start"], how="left")
        Wf["DI"] = aoa.di(Wf[feats].values.astype("float64"))
        Wf["inside"] = np.where(Wf.DI <= aoa.thr, "inside AOA", "outside AOA")
        table(pd.DataFrame([dict(group=k, **met(d, "pred_X1")) for k, d in Wf.groupby("inside")]).set_index("group").round(2))
        say(f"   share of 2026 station-weeks inside the AOA {np.mean(Wf.DI <= aoa.thr):.0%} (threshold {aoa.thr:.3f})")
    except Exception as ex:
        say(f"   skipped: {type(ex).__name__}: {ex}")

    # ------------------------------------------------------------- bias-free skill
    say("\n=== F. Skill with the mean bias removed (what the 2026 errors are made of)")
    rows = []
    for c in ("pred_X1", "pred_E", "pred_P", "pred_S1clim"):
        d = W.dropna(subset=[c])
        e = d[c] - d.y
        mse = float(np.mean(e ** 2))
        rows.append(dict(model=c[5:], n=len(d), RMSE=np.sqrt(mse), bias=e.mean(), share_of_MSE_from_bias=e.mean() ** 2 / mse,
                         RMSE_bias_removed=np.sqrt(np.mean((e - e.mean()) ** 2)), r=np.corrcoef(d.y, d[c])[0, 1]))
    table(pd.DataFrame(rows).set_index("model").round(2))
    p = os.path.join(C.DIRS["logs"], "forward_diagnostics.txt")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("\n".join(OUT))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
