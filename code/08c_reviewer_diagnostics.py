"""
STEP 8c - error diagnostics and uncertainty of the FROZEN Stage 1 model (X1), requested in review.
Local only, ~5-15 min.  Run after 07e (and 07c for part B).  Reads frozen outputs only; nothing is refitted except the
post-hoc random-forest baseline (part E), which never replaces X1.  Every 2026 number is POST-HOC.

A. Residuals stratified by season, boundary-layer regime (PBLH terciles of the training rows) and dust conditions
   (UV absorbing aerosol index terciles of the training rows, and AAI > 1), for scheme A (2023, out-of-site),
   TEST 2024 and the 2026 forward set; per network (UAE EAD, Oman); Spearman correlation of the residual with PBLH
   and AAI (95 % interval from a cluster bootstrap over sites).
B. Point-to-pixel representativeness from the 1 km weekly maps (07c): spread of the estimate within the 3 x 3 and
   5 x 5 cells around each station, and skill when the station is matched to the cell, 3 x 3 or 5 x 5 mean.
C. Accumulated local effects (ALE; Apley & Zhu 2020) of X1 for the column, PBLH, temperature, wind speed, surface
   pressure and aerosol index on the training rows -> figures/FigS2_ale.tif/.png (no titles; same style as 08a).
D. Uncertainty propagation through X1: (1) systematic column error, all column predictors scaled by 0.75 / 1.25 / 1.5
   (TROPOMI tropospheric columns are biased low over polluted sites); (2) random column error, multiplicative
   Gaussian noise (sd 25 %) per station-week, 200 Monte Carlo draws; (3) resampling/footprint choice, the 3 x 3 and
   5 x 5 column means replaced by the cell value.  The quality filter (qa_value > 0.75) is fixed in the Earth Engine
   L3 product and cannot be varied without re-processing L2 data (stated in the paper).
E. Simpler baseline: random-forest inversion with the same 40 predictors (leave-one-site-out 2023, then 2024 and
   2026), next to X1 and the log-linear PBLH scaling (E1, from 07b) on identical rows.

Output  logs/reviewer_diagnostics.txt (send this), logs/rd_*.csv, figures/FigS2_ale.tif/.png
"""
import os, sys, glob, warnings
CODE_DIR = r"D:\Cities\GCC_Cities2\code"                     # fallback when __file__ is undefined (notebook)
try:
    CODE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    if os.path.exists(os.path.join(os.getcwd(), "dune_config.py")):
        CODE_DIR = os.getcwd()
sys.path.insert(0, CODE_DIR)
for _m in ("dune_config", "dune_env", "dune_gee", "dune_ground", "dune_results"):
    sys.modules.pop(_m, None)
import dune_env
dune_env.ensure("diagnostics")
import numpy as np, pandas as pd
import xgboost as xgb
from scipy import stats
import dune_config as C
import dune_results as R

warnings.filterwarnings("ignore")
OUT = []
SEED = 20261111
NBOOT = 1000


def say(s=""):
    print(s, flush=True)
    OUT.append(str(s))


def table(df):
    for ln in df.to_string().splitlines():
        say("   " + ln)


def met(d, p="pred"):
    d = d[np.isfinite(d.y) & np.isfinite(d[p])]
    if len(d) < 5:
        return dict(n=len(d))
    e = d[p] - d.y
    sst = np.sum((d.y - d.y.mean()) ** 2)
    return dict(n=len(d), sites=d.Site_ID.nunique(), obs_mean=d.y.mean(), bias=e.mean(), RMSE=np.sqrt(np.mean(e ** 2)),
                R2=1 - np.sum(e ** 2) / sst if sst > 0 else np.nan, r=np.corrcoef(d.y, d[p])[0, 1])


def season(t):
    m = (pd.to_datetime(t) + pd.Timedelta(days=3)).dt.month
    return np.select([m.isin([12, 1, 2]), m.isin([3, 4, 5]), m.isin([6, 7, 8])], ["DJF", "MAM", "JJA"], "SON")


def rho_ci(d, col, p="pred"):
    d = d[np.isfinite(d[col]) & np.isfinite(d.y) & np.isfinite(d[p])]
    if len(d) < 20:
        return np.nan, np.nan, np.nan
    res = d[p] - d.y
    r0 = stats.spearmanr(d[col], res)[0]
    g = {s: x for s, x in d.groupby("Site_ID")}
    sites = np.array(list(g))
    rng = np.random.default_rng(SEED)
    bs = []
    for _ in range(NBOOT):
        x = pd.concat([g[s] for s in rng.choice(sites, len(sites))])
        bs.append(stats.spearmanr(x[col], x[p] - x.y)[0])
    return r0, np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5)


def main():
    r = R.Results()
    S1 = r.S1
    feats = r.meta1["features"]
    bst = xgb.Booster()
    bst.load_model(os.path.join(R.V2, f"stage1_{r.winner}.json"))
    if r.winner != "X1":
        say(f"   note: frozen Stage 1 model is {r.winner}")

    def predict(D):
        return bst.predict(xgb.DMatrix(D[feats].values.astype("float32")))

    say(f"=== 8c Error diagnostics and uncertainty of the frozen Stage 1 model ({r.winner}) | 2026 numbers are POST-HOC")
    tr, te = r.train.copy(), r.test.copy()
    tr["pred_fit"] = predict(tr)
    te["pred"] = predict(te)
    A = r.scheme_a()[["Target_ID", "week_start", "pred"]].merge(tr, on=["Target_ID", "week_start"], how="inner")
    FWf = pd.read_csv(os.path.join(C.BASE_DIR, "features", "station_weekly_features.csv"), parse_dates=["week_start"])
    FWf = FWf[FWf.raster_set == "main"].rename(columns={"Station": "Anchor_Station"})
    Fw = r.forward("weekly")[["Target_ID", "Site_ID", "Anchor_Station", "week_start", "y"]]
    Fw = Fw.merge(FWf[["Anchor_Station", "week_start"] + [f for f in feats if f in FWf.columns]], on=["Anchor_Station", "week_start"], how="left")
    Fw = Fw[Fw[feats].notna().any(axis=1)].copy()
    Fw["pred"] = predict(Fw)
    net = r.sites.set_index("Target_ID").Network
    for D in (A, te, Fw):
        D["season"] = season(D.week_start)
        D["network"] = D.Target_ID.map(net).astype(str)
    sets = [("2023 scheme A (out-of-site)", A), ("2024 TEST (frozen)", te), ("2026 forward (POST-HOC)", Fw)]

    # ---------------------------------------------------------------- A. stratified residuals
    say("\n=== A. Residuals by season, boundary-layer regime and dust conditions (bias = estimate - observation, ug/m3)")
    q_pblh = np.nanpercentile(tr.PBLH, [100 / 3, 200 / 3])
    aai = "S5P_AAI_3x3" if "S5P_AAI_3x3" in tr else "S5P_AAI"
    q_aai = np.nanpercentile(tr[aai], [100 / 3, 200 / 3])
    say(f"   PBLH terciles (training rows): {q_pblh[0]:.0f} / {q_pblh[1]:.0f} m | {aai} terciles: {q_aai[0]:.2f} / {q_aai[1]:.2f}")
    rows = []
    for lab, D in sets:
        D["PBLH_class"] = pd.cut(D.PBLH, [-np.inf, q_pblh[0], q_pblh[1], np.inf], labels=["low", "mid", "high"]).astype(str)
        D["AAI_class"] = pd.cut(D[aai], [-np.inf, q_aai[0], q_aai[1], np.inf], labels=["low", "mid", "high"]).astype(str)
        D["dust_AAI_gt1"] = np.where(D[aai] > 1, "AAI>1", "AAI<=1")
        rows.append(dict(set=lab, stratum="all", level="all", **met(D)))
        for col in ("season", "PBLH_class", "AAI_class", "dust_AAI_gt1", "network"):
            for lev, x in D.groupby(col):
                rows.append(dict(set=lab, stratum=col, level=lev, **met(x)))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(C.DIRS["logs"], "rd_stratified_residuals.csv"), index=False)
    table(T.set_index(["set", "stratum", "level"]).round(2))
    say("\n   Spearman correlation of the residual with PBLH and with the aerosol index (95 % CI, cluster bootstrap by site):")
    rr = []
    for lab, D in sets:
        for col in ("PBLH", aai, "T2m", "WS"):
            if col in D:
                v, lo, hi = rho_ci(D, col)
                rr.append(dict(set=lab, variable=col, rho=v, CI_lo=lo, CI_hi=hi))
    RR = pd.DataFrame(rr)
    RR.to_csv(os.path.join(C.DIRS["logs"], "rd_residual_correlation.csv"), index=False)
    table(RR.set_index(["set", "variable"]).round(3))
    for lab, D in sets:
        say(f"   {lab}: station-weeks with AAI > 1 = {np.mean(D[aai] > 1):.1%}; sites {D.Site_ID.nunique()}, "
            f"weeks per site median {D.groupby('Site_ID').size().median():.0f}")

    # ---------------------------------------------------------------- B. representativeness from the weekly maps
    say("\n=== B. Point-to-pixel representativeness (1 km weekly maps around each station cell)")
    st = r.stations.set_index("Station")
    files = {os.path.basename(f)[10:17]: f for f in glob.glob(os.path.join(C.BASE_DIR, "maps", "weekly", "*", "DUNE_NO2_W*.tif"))}
    if not files or "Row" not in st or "Col" not in st:
        say("   skipped: weekly maps (07c) or station Row/Col not found")
    else:
        import rasterio
        from rasterio.windows import Window
        brows = []
        for lab, D in sets[1:]:
            D = D.copy()
            iso = D.week_start.dt.isocalendar()
            D["wk"] = iso.year.astype(str) + "-" + iso.week.astype(int).map("{:02d}".format)
            for wk, x in D.groupby("wk"):
                if wk not in files:
                    continue
                with rasterio.open(files[wk]) as src:
                    for i, row in x.iterrows():
                        if row.Anchor_Station not in st.index:
                            continue
                        rr_, cc_ = int(st.loc[row.Anchor_Station, "Row"]), int(st.loc[row.Anchor_Station, "Col"])
                        w = src.read(1, window=Window(cc_ - 2, rr_ - 2, 5, 5), boundless=True, fill_value=np.nan).astype(float)
                        if w.shape != (5, 5) or not np.isfinite(w[2, 2]):
                            continue
                        brows.append(dict(set=lab, Site_ID=row.Site_ID, y=row.y, cell=w[2, 2], m3=np.nanmean(w[1:4, 1:4]),
                                          m5=np.nanmean(w), sd3=np.nanstd(w[1:4, 1:4]), sd5=np.nanstd(w)))
        B = pd.DataFrame(brows)
        if not len(B):
            say("   no station-week matched a weekly map (check maps/weekly/<year>/ and the station Row/Col)")
        else:
            B.to_csv(os.path.join(C.DIRS["logs"], "rd_representativeness.csv"), index=False)
            out = []
            for lab, x in B.groupby("set"):
                o = dict(set=lab, n=len(x), sd3_median=x.sd3.median(), sd5_median=x.sd5.median(),
                         abs_cell_minus_m5=np.mean(np.abs(x.cell - x.m5)))
                for c in ("cell", "m3", "m5"):
                    m = met(x.assign(pred=x[c]))
                    o[f"R2_{c}"], o[f"RMSE_{c}"] = m.get("R2"), m.get("RMSE")
                out.append(o)
            table(pd.DataFrame(out).set_index("set").round(3))
            say("   (sd3 / sd5 = spread of the 1 km estimate within 3 x 3 / 5 x 5 km around the station; R2_cell / R2_m3 /\n"
                "    R2_m5 = skill when the station is matched to the cell, the 3 x 3 mean or the 5 x 5 mean)")

    # ---------------------------------------------------------------- C. ALE
    say("\n=== C. Accumulated local effects (first order, 20 quantile bins, training rows; ug/m3)")
    ale_feats = [f for f in (S1.VCD, "PBLH", "T2m", "WS", "SP", aai) if f in feats]
    X0 = tr[feats].copy()
    ale = {}
    for f in ale_feats:
        x = X0[f].values
        ok = np.isfinite(x)
        edges = np.unique(np.nanpercentile(x[ok], np.linspace(0, 100, 21)))
        if len(edges) < 3:
            continue
        b = np.clip(np.searchsorted(edges, x[ok], side="right") - 1, 0, len(edges) - 2)
        Xk = X0[ok]
        eff = np.zeros(len(edges) - 1)
        cnt = np.zeros(len(edges) - 1)
        for k in range(len(edges) - 1):
            sel = b == k
            if not sel.any():
                continue
            lo_, hi_ = Xk[sel].copy(), Xk[sel].copy()
            lo_[f], hi_[f] = edges[k], edges[k + 1]
            eff[k] = np.mean(predict(hi_) - predict(lo_))
            cnt[k] = sel.sum()
        acc = np.concatenate([[0], np.cumsum(eff)])
        centre = np.sum(((acc[:-1] + acc[1:]) / 2) * cnt) / cnt.sum()
        ale[f] = (edges, acc - centre, x[ok])
        say(f"   {f:<15} ALE range {np.ptp(acc):6.2f} ug/m3 over the 5-95 % data range "
            f"[{np.nanpercentile(x[ok], 5):.3g}, {np.nanpercentile(x[ok], 95):.3g}]")
    pd.concat([pd.DataFrame(dict(feature=f, x=e, ale=a)) for f, (e, a, _) in ale.items()]).to_csv(
        os.path.join(C.DIRS["logs"], "rd_ale.csv"), index=False)
    try:
        F8 = R.load_module("figs08", "08_figures.py")
        F8.style()
        plt = F8.plt
        lab_ = {S1.VCD: "NO$_2$ column, 3 $\\times$ 3 ($\\mu$mol m$^{-2}$)", "PBLH": "Boundary-layer height (m)",
                "T2m": "2-m temperature (K)", "WS": "10-m wind speed (m s$^{-1}$)", "SP": "Surface pressure (hPa)",
                aai: "UV aerosol index, 3 $\\times$ 3"}
        n = len(ale)
        fig, axs = plt.subplots(2, 3, figsize=(F8.W2, 4.9), sharey=True, gridspec_kw=dict(wspace=0.12, hspace=0.5))
        unit = {S1.VCD: 1e6, "SP": 0.01}                          # mol m-2 -> umol m-2, Pa -> hPa (no offset labels)
        for ax, (f, (e, a, xr)), L in zip(axs.flat, ale.items(), "abcdef"):
            e, xr = e * unit.get(f, 1), xr * unit.get(f, 1)
            ax.axhline(0, color="#9D9D9D", lw=0.6)
            ax.plot(e, a, color=F8.MCOL["pred_X1"], lw=1.6, marker="o", ms=2.5)
            lo_, hi_ = np.nanpercentile(xr, [1, 99])
            ax.set_xlim(lo_, hi_)
            yl = ax.get_ylim()
            ax.plot(np.nanpercentile(xr, np.linspace(1, 99, 60)), np.full(60, yl[0] + 0.02 * (yl[1] - yl[0])), "|",
                    color="#555555", ms=4, alpha=0.6)
            ax.set_xlabel(lab_.get(f, f), fontsize=10)
            ax.spines[["top", "right"]].set_visible(False)
            F8.panel(ax, L)
        for ax in list(axs.flat)[n:]:
            ax.axis("off")
        for ax in axs[:, 0]:
            ax.set_ylabel(f"ALE ({F8.UG})")
        F8.save(fig, "FigS2_ale")
        say("   figure: figures/FigS2_ale.tif / .png")
    except Exception as ex:
        say(f"   ALE figure skipped: {type(ex).__name__}: {ex}")

    # ---------------------------------------------------------------- D. uncertainty propagation
    say("\n=== D. Propagation of column and resampling uncertainty through the frozen model (weekly)")
    colf = [f for f in feats if f.startswith("NO2_trop") or f.startswith("NO2_PBLH")]
    drows = []
    rng = np.random.default_rng(SEED)
    for lab, D in sets[1:]:
        D = D[np.isfinite(D.y)].copy()
        base = predict(D)
        m0 = met(D.assign(pred=base))
        for s in (0.75, 1.25, 1.5):
            Dp = D.copy()
            Dp[colf] = Dp[colf] * s
            p = predict(Dp)
            m = met(D.assign(pred=p))
            drows.append(dict(set=lab, case=f"column x {s}", mean_change=np.mean(p - base), mean_abs_change=np.mean(np.abs(p - base)),
                              R2=m.get("R2"), dR2=m.get("R2") - m0.get("R2"), bias=m.get("bias")))
        r2s, sds = [], []
        P = []
        for _ in range(200):
            Dp = D.copy()
            noise = rng.normal(1, 0.25, len(Dp))
            for f in colf:
                Dp[f] = Dp[f] * noise
            p = predict(Dp)
            P.append(p)
            r2s.append(met(D.assign(pred=p)).get("R2"))
        P = np.array(P)
        drows.append(dict(set=lab, case="random column error 25 % (MC 200)", mean_change=np.mean(P.mean(0) - base),
                          mean_abs_change=float(np.median(P.std(0))), R2=float(np.median(r2s)),
                          dR2=float(np.median(r2s)) - m0.get("R2"), bias=np.nan,
                          R2_lo=np.percentile(r2s, 2.5), R2_hi=np.percentile(r2s, 97.5)))
        for case, src in (("3x3 and 5x5 column := cell", "NO2_trop"), ("cell and 5x5 column := 3x3", "NO2_trop_3x3")):
            Dp = D.copy()
            for f in ("NO2_trop", "NO2_trop_3x3", "NO2_trop_5x5"):
                if f in Dp and f != src:
                    Dp[f] = Dp[src]
            if "NO2_PBLH" in Dp and "NO2_PBLH_3x3" in Dp:
                Dp["NO2_PBLH_3x3" if src == "NO2_trop" else "NO2_PBLH"] = Dp["NO2_PBLH" if src == "NO2_trop" else "NO2_PBLH_3x3"]
            p = predict(Dp)
            m = met(D.assign(pred=p))
            drows.append(dict(set=lab, case=case, mean_change=np.mean(p - base), mean_abs_change=np.mean(np.abs(p - base)),
                              R2=m.get("R2"), dR2=m.get("R2") - m0.get("R2"), bias=m.get("bias")))
    DD = pd.DataFrame(drows)
    DD.to_csv(os.path.join(C.DIRS["logs"], "rd_uncertainty.csv"), index=False)
    table(DD.set_index(["set", "case"]).round(3))
    say("   (mean_abs_change for the Monte Carlo row = median over station-weeks of the prediction SD across draws)")

    # ---------------------------------------------------------------- E. random-forest baseline
    say("\n=== E. Simpler baselines on identical rows (post hoc; X1 stays the frozen model)")
    try:
        from sklearn.ensemble import RandomForestRegressor
        med = tr[feats].median()

        def rf_fit(D):
            return RandomForestRegressor(n_estimators=500, min_samples_leaf=5, max_features=1 / 3, random_state=SEED,
                                         n_jobs=4).fit(D[feats].fillna(med).values, D.y.values)
        oof = np.full(len(tr), np.nan)
        for s_ in sorted(tr.Site_ID.unique()):
            k = (tr.Site_ID == s_).values
            oof[k] = rf_fit(tr[~k]).predict(tr.loc[k, feats].fillna(med).values)
        rf = rf_fit(tr)
        Ao = tr[["Target_ID", "Site_ID", "week_start", "y"]].assign(pred_RF=oof).merge(
            A[["Target_ID", "week_start", "pred"]].rename(columns={"pred": "pred_X1"}), on=["Target_ID", "week_start"])
        E = []
        oofs = pd.read_csv(os.path.join(R.V2, "oof_predictions.csv"), parse_dates=["week_start"])
        e1 = oofs[oofs.key == "A new station (leave one site out)|E1"][["Target_ID", "week_start", "pred"]].rename(columns={"pred": "pred_E1"})
        Ao = Ao.merge(e1, on=["Target_ID", "week_start"], how="left")
        for lab, D in (("2023 scheme A (out-of-site)", Ao),
                       ("2024 TEST (frozen)", te.assign(pred_X1=te.pred, pred_RF=rf.predict(te[feats].fillna(med).values))),
                       ("2026 forward (POST-HOC)", Fw.assign(pred_X1=Fw.pred, pred_RF=rf.predict(Fw[feats].fillna(med).values)))):
            models = [c for c in ("pred_X1", "pred_RF", "pred_E1") if c in D]
            x = D.dropna(subset=["y"] + models)
            for c in models:
                m = met(x.assign(pred=x[c]))
                E.append(dict(set=lab, model=c[5:], **m, **S1.boot_ci(x.assign(pred=x[c]), "pred")))
        EE = pd.DataFrame(E)
        EE.to_csv(os.path.join(C.DIRS["logs"], "rd_baselines.csv"), index=False)
        table(EE.set_index(["set", "model"]).round(3))
    except Exception as ex:
        say(f"   skipped: {type(ex).__name__}: {ex}")

    p = os.path.join(C.DIRS["logs"], "reviewer_diagnostics.txt")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("\n".join(OUT))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
