"""
STEP 7c - 1 km surface NO2 maps and the 744-district database from the FROZEN Stage 1 model.
Local only.  Roughly 1-3 h on a CPU for the 247 weeks; resumable (finished weeks are skipped).

Run AFTER 07b_stage1.py (frozen model in model/v2/) and 07b2_diagnostics.py.  Nothing is re-trained or
re-tuned here, and the sealed forward targets (tables/forward/) are NOT read.

What it does, week by week (2022-W01 -> 2026-W39, the weekly 1 km rasters of step 05)
    1. builds every model feature for every 0.01 deg cell of the 744 districts with the SAME definitions
       used for the stations (step 07a): cell value, 3x3 and 5x5 nan-means, weekly-mean derived variables,
       static layers of the same year (02_static_layers.layers_for_year, imported, not copied)
    2. predicts weekly surface NO2 (ug/m3) with the frozen booster (model/v2/stage1_<winner>.json)
       - a cell gets NO prediction where the 3x3 satellite column or PBLH is missing all week, exactly the
         rule that removed rows from training; such cells stay empty (no gap filling, no interpolation)
    3. area of applicability (Meyer & Pebesma 2021, Methods Ecol. Evol. 12:1620): dissimilarity index
       DI = distance, in the standardised feature space weighted by mean |SHAP|, to the nearest training
       station-week, divided by the mean distance between training station-weeks.  Threshold = upper
       whisker (Q3 + 1.5 IQR) of the training DI computed leave-one-SITE-out (same grouping as scheme A).
       AOA = 1 where DI <= threshold.  The script then TESTS the layer: are errors larger outside the AOA
       at the scheme-A out-of-fold rows and at the independent agency stations?
    4. consistency check: map values at the station cells must equal the station predictions written by
       07b_stage1 (same model, same features) - this proves the map is the validated model
Monthly maps = day-weighted means of the weekly maps (a week's 7 days are split between months), kept only
where >= 90 % of the month's days are covered - the same rule used to validate monthly values.

Outputs
    maps/weekly/<iso year>/DUNE_NO2_W<yyyy>-<ww>_<monday>.tif   bands NO2 (ug/m3), DI, AOA (1/0)
    maps/monthly/DUNE_NO2_M<yyyy>-<mm>.tif                       bands NO2, days_covered, DI_mean, AOA_share
    tables/District_NO2_weekly.csv, tables/District_NO2_monthly.csv
        per district: cells, cells predicted, coverage, mean NO2, population-weighted mean (GHSL),
        mean NO2 inside the AOA, AOA share; tiny districts (no cell centre inside) use their
        representative-point cell, flagged
    maps/README_maps.txt   grid, units, effective resolution, model hash, AOA threshold
    logs/maps_report.txt  
"""
import os, sys, json, hashlib, shutil, time, importlib.util, warnings
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
import numpy as np, pandas as pd, geopandas as gpd, rasterio, xgboost as xgb
from rasterio.transform import Affine
import dune_config as C

for w in (FutureWarning, RuntimeWarning):
    warnings.filterwarnings("ignore", category=w)


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(CODE_DIR, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S2 = _load("static02", "02_static_layers.py")      # static layer definitions (same as the station table)
A7 = _load("features07a", "07a_build_features.py") # weekly_files, humidity formula
S1 = _load("stage1", "07b_stage1.py")              # metrics, weekly->monthly, column / PBLH names

MODEL_DIR = os.path.join(C.BASE_DIR, "model")
V2 = os.path.join(MODEL_DIR, "v2")
MAPS = os.path.join(C.BASE_DIR, "maps")
TRANSFORM = Affine(C.RES, 0, C.X0, 0, -C.RES, C.Y1)
MONTH_MIN_COVER = 0.9            # same rule as the monthly validation (07b_stage1.weekly_to_monthly)
CHUNK = 400_000                  # cells per prediction / DI block (memory)
EFFECTIVE_RES = ("Satellite column: TROPOMI L2 footprint ~3.5 x 5.5 km (nearest-neighbour onto 0.01 deg) and 3x3 / 5x5 "
                 "means (~3 / ~5 km); meteorology: ERA5-Land 0.1 deg (~9-11 km, bilinear); variation below ~5 km "
                 "comes only from the 1 km proxies (GHSL, VIIRS, NDVI, DEM, land / coast).  The 0.01 deg grid is the "
                 "reporting grid, not the resolution of the satellite signal.")
OUT = []


def say(s=""):
    print(s, flush=True)
    OUT.append(str(s))


def window_means(a, rows, cols, sizes=(3, 5), chunk=400_000):
    """cell value and k x k nan-means at the selected cells, computed EXACTLY as the station features
    (07a.sample_week / 02.sample_points): the same k x k windows (NaN outside the grid), the same dtype and
    the same numpy call, so the grid and the stations get bit-identical inputs."""
    from numpy.lib.stride_tricks import sliding_window_view
    pad = np.pad(a, 2, mode="constant", constant_values=np.nan)
    view = sliding_window_view(pad, (5, 5))                   # view[r, c] = 5x5 window centred on (r, c)
    out = {k: np.empty(len(rows), dtype=a.dtype) for k in ("px",) + tuple(f"{s}x{s}" for s in sizes)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i in range(0, len(rows), chunk):
            w = view[rows[i:i + chunk], cols[i:i + chunk]]    # (n, 5, 5) copy
            n = len(w)
            out["px"][i:i + n] = w[:, 2, 2]
            if 3 in sizes:
                out["3x3"][i:i + n] = np.nanmean(w[:, 1:4, 1:4].reshape(n, -1), axis=1)
            if 5 in sizes:
                out["5x5"][i:i + n] = np.nanmean(w.reshape(n, -1), axis=1)
    return out


def file_hash(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ===================================================================== features on the grid
class Features:
    """resolves a model feature name to a vector over the selected cells (rows, cols)"""
    STATIC = ("DEM", "LandFrac", "CoastDist_km", "BUILT_frac", "POP", "SMOD", "VIIRS", "NDVI")

    def __init__(self, feats, rows, cols):
        self.feats, self.rows, self.cols = feats, rows, cols
        self.static_cache = {}

    def static_for_year(self, iso_year, fixed, coast, ghsl):
        carry = getattr(C, "STATIC_CARRY_FORWARD", {})
        src = carry.get(iso_year, iso_year)
        if src not in C.STATIC_YEARS:
            raise SystemExit(f"no static layers for year {iso_year} (STATIC_YEARS / STATIC_CARRY_FORWARD)")
        if src not in self.static_cache:
            L = S2.layers_for_year(src, fixed, coast, ghsl)
            v, wm = {}, {}
            for f in self.feats:
                base, suf = f.rsplit("_", 1) if "_" in f else (f, "")
                if base in self.STATIC and suf in ("px", "3x3"):
                    if base not in wm:                   # keep the layer's own dtype, as 02_static_layers
                        wm[base] = window_means(np.asarray(L[base]), self.rows, self.cols, sizes=(3,))
                    v[f] = wm[base][suf].astype("float32")
            v["__POP"] = np.asarray(L["POP"], dtype="float64")[self.rows, self.cols]
            self.static_cache = {src: v}            # keep one year in memory
        return self.static_cache[src], src

    def matrix(self, B, static, monday):
        mid = monday + pd.Timedelta(days=3)
        doy = mid.dayofyear
        cache, wins = {}, {}
        r, c = self.rows, self.cols

        def band(n):
            if n not in B:
                raise SystemExit(f"band {n} not in the weekly raster")
            return B[n]

        def get(n):
            if n in cache:
                return cache[n]
            if n in static:
                v = static[n]
            elif n == "NO2_valid_frac":
                v = band("NO2_ndays")[r, c] / np.float32(7.0)
            elif n == "PRECIP":
                v = np.clip(band("PRECIP_wk")[r, c], 0, None)
            elif n == "RH":
                v = np.clip(A7.rh_from_t_td(band("T2m")[r, c], band("Td2m")[r, c]), 0, 100)
            elif n == "WS":
                v = np.hypot(band("U10")[r, c], band("V10")[r, c])
            elif n in ("NO2_PBLH", "NO2_PBLH_3x3"):
                p = band("PBLH")[r, c]
                p = np.where(p > 10, p, np.nan)
                v = get("NO2_trop" if n == "NO2_PBLH" else "NO2_trop_3x3") / p
            elif n == "doy_sin":
                v = np.full(len(r), np.sin(2 * np.pi * doy / 365.25), dtype="float32")
            elif n == "doy_cos":
                v = np.full(len(r), np.cos(2 * np.pi * doy / 365.25), dtype="float32")
            elif n.endswith(("_3x3", "_5x5")) and n[:-4] in B:
                if n[:-4] not in wins:
                    wins[n[:-4]] = window_means(B[n[:-4]], r, c)
                v = wins[n[:-4]][n[-3:]]
            elif n in B:
                v = B[n][r, c]
            else:
                raise SystemExit(f"feature {n} cannot be built from the weekly raster / static layers - check the feature definitions")
            cache[n] = np.asarray(v, dtype="float32")
            return cache[n]

        X = np.column_stack([get(f) for f in self.feats]).astype("float32")
        valid = np.isfinite(get(S1.VCD)) & np.isfinite(np.where(band("PBLH")[r, c] > 0, band("PBLH")[r, c], np.nan))
        return X, valid


# ===================================================================== dissimilarity index
class AOA:
    def __init__(self, Xtr, sites, weights):
        self.mu = np.nanmean(Xtr, axis=0)
        sd = np.nanstd(Xtr, axis=0)
        self.w = np.where(sd > 0, weights / np.where(sd > 0, sd, 1), 0.0)
        T = self.scale(Xtr)
        d2 = self._d2(T, T)
        n = len(T)
        dist = np.sqrt(d2)
        self.dbar = dist[~np.eye(n, dtype=bool)].mean()
        same = sites[:, None] == sites[None, :]
        dist_cv = np.where(same, np.inf, dist)                 # leave-one-site-out, as scheme A
        self.train_di = dist_cv.min(axis=1) / self.dbar
        q1, q3 = np.percentile(self.train_di, [25, 75])
        lim = q3 + 1.5 * (q3 - q1)
        self.thr = float(self.train_di[self.train_di <= lim].max())   # boxplot upper whisker
        self.T, self.Tn = T, (T ** 2).sum(axis=1)

    def scale(self, X):
        Z = (X - self.mu) * self.w
        return np.where(np.isfinite(Z), Z, 0.0).astype("float32")   # missing feature -> training mean

    @staticmethod
    def _d2(A, B):
        return np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T, 0)

    def di(self, X):
        out = np.empty(len(X), dtype="float32")
        for i in range(0, len(X), 50_000):
            Z = self.scale(X[i:i + 50_000])
            d2 = np.maximum((Z ** 2).sum(1)[:, None] + self.Tn[None, :] - 2 * Z @ self.T.T, 0)
            out[i:i + 50_000] = np.sqrt(d2.min(axis=1)) / self.dbar
        return out


# ===================================================================== writing
def write_tif(path, layers, names, tags):
    prof = dict(driver="GTiff", dtype="float32", count=len(layers), width=C.WIDTH, height=C.HEIGHT, crs=C.CRS,
                transform=TRANSFORM, nodata=np.nan, compress="deflate", predictor=3, tiled=True,
                blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER")
    tmp = path + ".part"
    with rasterio.open(tmp, "w", **prof) as dst:
        for i, (a, n) in enumerate(zip(layers, names)):
            dst.write(a.astype("float32"), i + 1)
            dst.set_band_description(i + 1, n)
        dst.update_tags(**{k: str(v) for k, v in tags.items()})
    os.replace(tmp, path)                       # never leave a half-written file under the final name


def to_grid(vec, rows, cols):
    g = np.full((C.HEIGHT, C.WIDTH), np.nan, dtype="float32")
    g[rows, cols] = vec
    return g


def district_rows(key, lab, ncell, pred, di_ok, pop, tiny_idx, tiny_cell, ids, extra=None):
    """aggregate cell vectors to districts.  lab: district index (1..n) per cell, 0 = not a district cell"""
    n = len(ids)
    ok = np.isfinite(pred) & (lab > 0)
    L = lab[ok]
    cnt = np.bincount(L, minlength=n + 1)[1:]
    s = np.bincount(L, weights=pred[ok], minlength=n + 1)[1:]
    pw = np.bincount(L, weights=(pred * pop)[ok], minlength=n + 1)[1:]
    pt = np.bincount(L, weights=pop[ok], minlength=n + 1)[1:]
    ina = ok & (di_ok > 0.5)
    ca = np.bincount(lab[ina], minlength=n + 1)[1:]
    sa = np.bincount(lab[ina], weights=pred[ina], minlength=n + 1)[1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        d = pd.DataFrame({C.DISTRICT_ID: ids, "cells": ncell, "cells_pred": cnt, "coverage": cnt / np.maximum(ncell, 1),
                          "NO2_mean": s / cnt, "NO2_popw": np.where(pt > 0, pw / pt, np.nan),
                          "NO2_mean_AOA": sa / ca, "AOA_share": ca / cnt, "point_fallback": False})
    if len(tiny_idx):                            # districts without a cell centre: representative-point cell
        v = pred[tiny_cell]
        d.loc[tiny_idx, ["cells_pred", "coverage", "NO2_mean", "NO2_popw"]] = np.column_stack(
            [np.isfinite(v).astype(float), np.isfinite(v).astype(float), v, v])
        d.loc[tiny_idx, "NO2_mean_AOA"] = np.where(di_ok[tiny_cell] > 0.5, v, np.nan)
        d.loc[tiny_idx, "AOA_share"] = np.where(np.isfinite(v), (di_ok[tiny_cell] > 0.5).astype(float), np.nan)
        d.loc[tiny_idx, "point_fallback"] = True
    for k, v in (extra or {}).items():
        d.insert(1, k, v)
    return d


# ===================================================================== main
def main():
    t_start = time.time()
    meta_p = os.path.join(V2, "stage1_meta.json")
    if not os.path.exists(meta_p):
        raise SystemExit("model/v2/stage1_meta.json not found - run 07b_stage1.py first")
    meta = json.load(open(meta_p))
    winner, feats = meta["winner"], meta["features"]
    if winner not in ("X1", "X2"):
        raise SystemExit(f"frozen winner is {winner}; this step maps tree models without an offset (X1 / X2)")
    model_p = os.path.join(V2, f"stage1_{winner}.json")
    bst = xgb.Booster()
    bst.load_model(model_p)
    mhash, fhash = file_hash(meta_p)[:16], file_hash(model_p)[:16]
    say(f"=== 1 km maps (step 07c) | frozen {winner} | {len(feats)} features | meta sha256 {mhash}... | model sha256 {fhash}...")

    # ---------------- training rows exactly as 07b_stage1 (for the AOA)
    TW = pd.read_csv(os.path.join(MODEL_DIR, "weekly_table.csv"), parse_dates=["week_start"])
    if (TW.period == "forward").any():
        raise SystemExit("forward rows found in a model table - the forward set must stay sealed")
    TW = TW[np.isfinite(TW.y) & (TW.pollutant == "NO2") & np.isfinite(TW[S1.VCD]) & np.isfinite(TW.PBLH)]
    train = TW[TW.period == "train"].reset_index(drop=True)
    if len(train) != meta["train_rows"]:
        raise SystemExit(f"training rows {len(train)} differ from the frozen model's {meta['train_rows']} - "
                         "the tables changed after 07b_stage1; re-run 07b_stage1 first")
    Xtr = train[feats].values.astype("float32")
    contrib = bst.predict(xgb.DMatrix(Xtr), pred_contribs=True)[:, :-1]
    shap_w = np.abs(contrib).mean(axis=0)
    aoa = AOA(Xtr.astype("float64"), train.Site_ID.values, shap_w)
    say(f"   AOA: training DI (leave-one-site-out) median {np.median(aoa.train_di):.3f}, threshold {aoa.thr:.3f}, "
        f"{(aoa.train_di > aoa.thr).mean():.1%} of training rows above it")
    say(f"   feature weights (mean |SHAP|, top 8): " + ", ".join(
        f"{f} {w:.2f}" for f, w in sorted(zip(feats, shap_w), key=lambda x: -x[1])[:8]))

    # ---------------- cells: every district cell + representative cells of tiny districts + station cells
    g = gpd.read_file(C.GEOJSON).to_crs(4326).reset_index(drop=True)
    g["Area_km2"] = g.to_crs(6933).area / 1e6
    labg = S2.district_labels(g).reshape(C.HEIGHT, C.WIDTH)
    ids = g[C.DISTRICT_ID].values
    ncell = np.bincount(labg.ravel(), minlength=len(g) + 1)[1:]
    rp = g.geometry.representative_point()
    rp_r = np.clip(np.floor((C.Y1 - rp.y.values) / C.RES).astype(int), 0, C.HEIGHT - 1)
    rp_c = np.clip(np.floor((rp.x.values - C.X0) / C.RES).astype(int), 0, C.WIDTH - 1)
    tiny_idx = np.where(ncell == 0)[0]
    st = pd.read_csv(C.STATIONS_CSV)
    sel = np.zeros((C.HEIGHT, C.WIDTH), bool)
    sel[labg > 0] = True
    sel[rp_r[tiny_idx], rp_c[tiny_idx]] = True
    inside = st.Row.between(0, C.HEIGHT - 1) & st.Col.between(0, C.WIDTH - 1)
    sel[st.Row[inside].astype(int), st.Col[inside].astype(int)] = True
    rows, cols = np.nonzero(sel)
    lab = labg[rows, cols]
    cell_id = np.full((C.HEIGHT, C.WIDTH), -1, dtype=np.int64)
    cell_id[rows, cols] = np.arange(len(rows))
    tiny_cell = cell_id[rp_r[tiny_idx], rp_c[tiny_idx]]
    st_cell = np.where(inside, cell_id[st.Row.clip(0, C.HEIGHT - 1).astype(int), st.Col.clip(0, C.WIDTH - 1).astype(int)], -1)
    say(f"   cells: {int((labg > 0).sum()):,} district cells in {len(g)} districts ({len(tiny_idx)} tiny districts use "
        f"their representative-point cell) + station cells -> {len(rows):,} cells predicted per week")

    # ---------------- static inputs (read once)
    fixed = S2.read("Static_Fixed_1km.tif")
    coast = S2.coast_distance()
    area = S2.cell_area_m2()
    ghsl = {ep: S2.read_ghsl(ep, area) for ep in C.GHSL_EPOCHS}
    F = Features(feats, rows, cols)

    weeks = A7.weekly_files()
    if not weeks:
        raise SystemExit("no weekly rasters found - run 05_weekly_rasters.py")
    os.makedirs(os.path.join(MAPS, "monthly"), exist_ok=True)
    todo = [w for w in weeks if not os.path.exists(os.path.join(MAPS, "weekly", str(w[0]),
                                                                 f"DUNE_NO2_W{w[0]}-{w[1]:02d}_{w[2]:%Y-%m-%d}.tif"))]
    free = shutil.disk_usage(C.BASE_DIR).free / 1e9
    est = len(todo) * 0.03 + 60 * 0.03
    say(f"   weekly rasters {len(weeks)} ({weeks[0][0]}-W{weeks[0][1]:02d} -> {weeks[-1][0]}-W{weeks[-1][1]:02d}); "
        f"{len(weeks) - len(todo)} already mapped; estimate {est:.1f} GB (+-50 %) | free {free:.0f} GB")
    if free < est * 1.5:
        raise SystemExit("not enough free disk space for the safe estimate")
    tags = dict(product="DUNE surface NO2, 1 km (0.01 deg)", units_NO2="ug m-3 (weekly / monthly mean of 24 h)",
                model=f"Stage 1 {winner} (07b_stage1.py), frozen", model_sha256=fhash, meta_sha256=mhash,
                training=(f"{meta['train_rows']} station-weeks at {meta['train_sites']} sites "
                          f"({', '.join(f'{k} {v}' for k, v in train.groupby('Country').Site_ID.nunique().items())}; "
                          f"networks {', '.join(sorted(train.network.astype(str).unique()))}), weeks "
                          f"{train.week_start.min():%Y-%m-%d} to {train.week_start.max():%Y-%m-%d}"),
                AOA=f"DI <= {aoa.thr:.4f} (Meyer & Pebesma 2021; SHAP-weighted, leave-one-site-out)",
                no_prediction="no valid 3x3 satellite column or PBLH in the week (no gap filling)",
                effective_resolution=EFFECTIVE_RES, pipeline=getattr(S2.G, "PIPELINE_VERSION", ""))

    # ---------------- weeks
    month_acc, wk_rows, mo_rows, chk, cov = {}, [], [], [], []
    di_station = []
    pop_by_year = {}

    def finalize(mstart):
        acc = month_acc.pop(mstart)
        dim = mstart.days_in_month
        okm = acc["d"] >= MONTH_MIN_COVER * dim
        with np.errstate(invalid="ignore", divide="ignore"):
            no2 = np.where(okm, acc["s"] / acc["d"], np.nan)
            dim_ = np.where(acc["d"] > 0, acc["di"] / acc["d"], np.nan)
            aoas = np.where(acc["d"] > 0, acc["a"] / acc["d"], np.nan)
        write_tif(os.path.join(MAPS, "monthly", f"DUNE_NO2_M{mstart:%Y-%m}.tif"),
                  [to_grid(no2, rows, cols), to_grid(acc["d"].astype("float32"), rows, cols),
                   to_grid(dim_, rows, cols), to_grid(aoas, rows, cols)],
                  ["NO2", "days_covered", "DI_mean", "AOA_share"],
                  {**tags, "month": f"{mstart:%Y-%m}", "rule": f"day-weighted weekly means, >= {MONTH_MIN_COVER:.0%} of days"})
        mo_rows.append(district_rows("month", lab, ncell, no2.astype("float64"), np.where(okm, aoas, np.nan) >= 0.5,
                                     pop_by_year.get(mstart.year, pop_by_year[max(pop_by_year)]),
                                     tiny_idx, tiny_cell, ids, {"month": f"{mstart:%Y-%m}"}))
        cov.append(dict(kind="month", period=f"{mstart:%Y-%m}", predicted=float(np.isfinite(no2[lab > 0]).mean())))

    for k, (y, wk, monday, f) in enumerate(weeks):
        t0 = time.time()
        out = os.path.join(MAPS, "weekly", str(y), f"DUNE_NO2_W{y}-{wk:02d}_{monday:%Y-%m-%d}.tif")
        static, src_y = F.static_for_year(y, fixed, coast, ghsl)
        pop_by_year[y] = static["__POP"]
        if os.path.exists(out):                           # resume: reuse, but only if made by THIS model
            with rasterio.open(out) as s_:
                if s_.tags().get("model_sha256") != fhash:
                    raise SystemExit(f"{out} was made by another model (sha {s_.tags().get('model_sha256')}). Delete "
                                     "the maps folder (or move it away) and run this step again.")
                pred, di = s_.read(1)[rows, cols].astype("float64"), s_.read(2)[rows, cols]
            status = "reused"
        else:
            with rasterio.open(f) as src:
                B = {d: src.read(i + 1).astype("float32") for i, d in enumerate(src.descriptions)}
            X, valid = F.matrix(B, static, monday)
            pred = np.full(len(rows), np.nan)
            di = np.full(len(rows), np.nan, dtype="float32")
            idx = np.where(valid)[0]
            for i in range(0, len(idx), CHUNK):
                j = idx[i:i + CHUNK]
                pred[j] = bst.predict(xgb.DMatrix(X[j]))
                di[j] = aoa.di(X[j].astype("float64"))
            okc = np.isfinite(pred)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            write_tif(out,
                      [to_grid(pred, rows, cols), to_grid(di, rows, cols),
                       to_grid(np.where(okc, (di <= aoa.thr).astype("float32"), np.nan), rows, cols)],
                      ["NO2", "DI", "AOA"], {**tags, "iso_week": f"{y}-W{wk:02d}", "week_start": f"{monday:%Y-%m-%d}",
                                             "static_layers_year": src_y})
            status = "mapped"
            del B, X
        aoa_ok = np.where(np.isfinite(pred), (di <= aoa.thr).astype(float), np.nan)

        # consistency at the station cells + DI of every station-week (for the AOA test)
        sc = st_cell >= 0
        chk.append(pd.DataFrame({"Station": st.Station[sc].values, "week_start": monday, "map": pred[st_cell[sc]],
                                 "DI": di[st_cell[sc]]}))
        # district table
        wk_rows.append(district_rows("week", lab, ncell, pred, aoa_ok, pop_by_year[y], tiny_idx, tiny_cell, ids,
                                     {"week_start": f"{monday:%Y-%m-%d}", "iso_week": f"{y}-W{wk:02d}"}))
        dc = lab > 0
        cov.append(dict(kind="week", period=f"{y}-W{wk:02d}", predicted=float(np.isfinite(pred[dc]).mean()),
                        in_AOA=float(np.nanmean(aoa_ok[dc])) if np.isfinite(aoa_ok[dc]).any() else np.nan,
                        NO2_median=float(np.nanmedian(pred[dc])) if np.isfinite(pred[dc]).any() else np.nan))
        # monthly accumulation (each day of the week counts for its month)
        days = pd.Series(pd.date_range(monday, periods=7, freq="D").to_period("M")).value_counts()
        for per, nd in days.items():
            m0 = per.to_timestamp()
            a = month_acc.setdefault(m0, dict(s=np.zeros(len(rows)), d=np.zeros(len(rows), np.int16),
                                              di=np.zeros(len(rows)), a=np.zeros(len(rows))))
            okc = np.isfinite(pred)
            a["s"][okc] += pred[okc] * nd
            a["d"][okc] += nd
            a["di"][okc] += di[okc] * nd
            a["a"][okc] += aoa_ok[okc] * nd
        nxt = weeks[k + 1][2] if k + 1 < len(weeks) else None
        for m0 in sorted(month_acc):
            if nxt is None or (m0 + pd.offsets.MonthEnd(0)) < nxt:
                finalize(m0)
        print(f"   {y}-W{wk:02d} {status}: predicted {np.isfinite(pred[dc]).mean():.1%} of district cells "
              f"({time.time() - t0:.0f} s, total {(time.time() - t_start) / 60:.0f} min)", flush=True)

    # ---------------- tables
    T = C.DIRS["tables"]
    pd.concat(wk_rows, ignore_index=True).to_csv(os.path.join(T, "District_NO2_weekly.csv"), index=False)
    pd.concat(mo_rows, ignore_index=True).to_csv(os.path.join(T, "District_NO2_monthly.csv"), index=False)
    CV = pd.DataFrame(cov)
    say("\n=== Coverage of the district cells (share of cells with a prediction; share inside the AOA)")
    w_ = CV[CV.kind == "week"].copy()
    w_["year"] = w_.period.str[:4]
    table_ = w_.groupby("year").agg(weeks=("period", "size"), predicted=("predicted", "mean"),
                                    in_AOA=("in_AOA", "mean"), NO2_median=("NO2_median", "median"))
    for ln in table_.round(3).to_string().splitlines():
        say("   " + ln)
    m_ = CV[CV.kind == "month"]
    say(f"   monthly maps: {len(m_)} months; mean share of district cells with a monthly value (>= 90 % of days) "
        f"{m_.predicted.mean():.1%} (lowest {m_.predicted.min():.1%} in {m_.loc[m_.predicted.idxmin(), 'period'] if len(m_) else '-'})")

    # ---------------- consistency: map at station cells vs 07b_stage1 station predictions
    K = pd.concat(chk, ignore_index=True)
    SW = pd.read_csv(os.path.join(V2, "station_weekly_predictions.csv"), parse_dates=["week_start"])
    J = K.merge(SW[["Station", "week_start", f"pred_{winner}"]], on=["Station", "week_start"], how="inner")
    J = J.dropna(subset=["map", f"pred_{winner}"])
    dif = (J["map"] - J[f"pred_{winner}"]).abs()
    say(f"\n=== Consistency: map value at the station cell vs 07b_stage1 station prediction ({len(J):,} station-weeks)")
    if len(J):
        say(f"   max |difference| {dif.max():.2e} ug/m3 | share > 0.001 ug/m3 {(dif > 1e-3).mean():.2%}  (expected 0: "
            "identical inputs, identical model)")
        if (dif > 1e-3).any():
            say("   WARNING: the map and the validated model do not get identical inputs at some station cells - "
                "check this report before using the maps")
    nn = K.merge(SW[["Station", "week_start", f"pred_{winner}"]], on=["Station", "week_start"], how="inner")
    say(f"   station-weeks predicted by 07b_stage1 but empty on the map: {int((nn['map'].isna() & nn[f'pred_{winner}'].notna()).sum()):,}"
        f" of {len(nn):,} (weeks without a valid 3x3 column or PBLH: never in training, so not mapped)")

    # ---------------- does the AOA separate good from bad predictions?
    say("\n=== Test of the AOA layer")
    oof = pd.read_csv(os.path.join(V2, "oof_predictions.csv"), parse_dates=["week_start"])
    a = oof[oof.key.str.startswith("A new station") & oof.key.str.endswith("|" + winner)].copy()
    tr_di = pd.DataFrame({"Target_ID": train.Target_ID, "week_start": train.week_start, "DI": aoa.train_di})
    a = a.merge(tr_di, on=["Target_ID", "week_start"], how="left")

    def by(d, label):
        d = d.assign(inside=np.where(d.DI <= aoa.thr, "inside AOA", "outside AOA"))
        rows_ = []
        for kk, dd in d.groupby("inside"):
            m = S1.metrics(dd.y, dd.pred)
            rows_.append(dict(group=kk, n=m["n"], sites=dd.Site_ID.nunique(), R2=m.get("R2"), r=m.get("r"),
                              RMSE=m.get("RMSE"), MAE=m.get("MAE"), bias=m.get("bias")))
        if len(d) >= 30:
            d["DI_tercile"] = pd.qcut(d.DI.rank(method="first"), 3, labels=["low DI", "mid DI", "high DI"])
            for kk, dd in d.groupby("DI_tercile", observed=True):
                m = S1.metrics(dd.y, dd.pred)
                rows_.append(dict(group=kk, n=m["n"], sites=dd.Site_ID.nunique(), R2=m.get("R2"), r=m.get("r"),
                                  RMSE=m.get("RMSE"), MAE=m.get("MAE"), bias=m.get("bias")))
        say(f"   {label}:")
        for ln in pd.DataFrame(rows_).set_index("group").round(2).to_string().splitlines():
            say("      " + ln)

    if len(a):
        by(a, f"scheme A out-of-fold, weekly ({winner}; DI of each row against the OTHER sites)")
    # independent agency stations: monthly obs vs monthly mean of the weekly predictions and DI
    TM = pd.read_csv(os.path.join(MODEL_DIR, "monthly_table.csv"), parse_dates=["month"])
    TM = TM[np.isfinite(TM.y)]
    indep = (TM.pollutant == "NO2") & ~TM.Site_ID.isin(set(train.Site_ID)) & TM.network.astype(str).str.contains("UAE_monthly", na=False)
    mp = S1.weekly_to_monthly(K[["Station", "week_start", "map"]].rename(columns={"map": "pred"}))
    md = S1.weekly_to_monthly(K[["Station", "week_start", "DI"]].rename(columns={"DI": "pred"})).rename(columns={"pred": "DI"})
    I = TM[indep & TM.period.isin(["train", "test"])].merge(mp.rename(columns={"Station": "Anchor_Station"}),
                                                             on=["Anchor_Station", "month"]).merge(
        md.rename(columns={"Station": "Anchor_Station"}), on=["Anchor_Station", "month"])
    if len(I):
        by(I, "INDEPENDENT agency stations 2022-2024, monthly (from the maps)")

    # ---------------- metadata file + report
    with open(os.path.join(MAPS, "README_maps.txt"), "w", encoding="utf-8") as fh:
        fh.write("DUNE surface NO2 maps (step 07c)\n\n")
        for kk, v in tags.items():
            fh.write(f"{kk}: {v}\n")
        fh.write(f"grid: {C.CRS}, {C.RES} deg, upper-left ({C.X0}, {C.Y1}), {C.WIDTH} x {C.HEIGHT}\n"
                 f"weekly bands: NO2, DI, AOA | monthly bands: NO2 (>= {MONTH_MIN_COVER:.0%} of days), days_covered, DI_mean, AOA_share\n"
                 "2026 weeks use 2025 static layers (STATIC_CARRY_FORWARD); 2026 ground data are a sealed forward test.\n")
    p = os.path.join(C.DIRS["logs"], "maps_report.txt")
    say(f"\n   finished in {(time.time() - t_start) / 60:.0f} min")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("\n".join(OUT))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
