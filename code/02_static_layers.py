"""
STEP 2 - static and annual layers on the 0.01 deg grid, then sampled at stations and districts.

Downloads (tiled computePixels, straight to disk, no Google Drive):
    static/Static_Fixed_1km.tif        DEM (m), LandFrac (0-1, Hansen/UMD 30 m data mask, area mean)
    static/GHSL_<epoch>_1km.tif        BUILT_frac (0-1), POP (persons per cell), SMOD (GHSL class)
    static/Annual_<year>_1km.tif       VIIRS (nW cm-2 sr-1, annual median), NDVI (annual mean)
Computed locally:
    static/CoastDist_km_1km.tif        distance to the nearest predominantly-water cell (km)
Tables:
    tables/Station_Static_by_Year.csv  Station x Year, pixel value and 3x3 mean of every layer
    tables/District_Static_by_Year.csv OBJECTID x Year, area mean (POP: total and density)
GHSL is published for 2020 and 2025; each year 2019-2025 is linearly interpolated between the two
epochs (2019 is a one-year linear extrapolation), and that rule is written into the table.

Sanity check printed at the end: GHSL population summed over each country's districts.

Forward-test year 2026: its annual layers are carried forward from 2025 (STATIC_CARRY_FORWARD).  As a
sensitivity check, part-year layers for STATIC_SENS_WINDOWS (Jan-Sep 2025 and Jan-Sep 2026) are also
built; logs/static_carry_forward_check.txt reports how they compare, and the modelling step re-scores
the forward test with them.
"""
import os, sys
# --- locate the code folder: works with "python script.py", "%run script.py" AND a pasted notebook cell
CODE_DIR = os.getcwd()                      # fallback when __file__ is undefined (notebook)
try:
    CODE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:                       # pasted into a notebook cell: __file__ does not exist
    if os.path.exists(os.path.join(os.getcwd(), "dune_config.py")):
        CODE_DIR = os.getcwd()
if not os.path.exists(os.path.join(CODE_DIR, "dune_config.py")):
    raise SystemExit(f"dune_config.py not found in {CODE_DIR} - unzip DUNE_GEE_v3 into that folder")
sys.path.insert(0, CODE_DIR)
for _m in ("dune_config", "dune_env", "dune_gee"):   # always load the CURRENT files, never a copy
    sys.modules.pop(_m, None)                         # cached in the notebook kernel from an earlier run
import dune_env
dune_env.ensure("static")      # installs / checks every package this step needs
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine
import dune_config as C
import dune_gee as G
if getattr(G, "PIPELINE_VERSION", None) != "2026-10-08a":
    raise SystemExit(f"{G.__file__} is an older version than this script. Unzip the WHOLE DUNE_GEE_v3.zip "
                     "again into the code folder and choose 'Replace the files in the destination'.")

S = C.DIRS["static"]
TRANSFORM = Affine(C.RES, 0, C.X0, 0, -C.RES, C.Y1)


def download_all():
    jobs = [(G.static_fixed(), ["DEM", "LandFrac"], "Static_Fixed_1km.tif", 128)]   # 30 m sources: small tiles
    for ep in C.GHSL_EPOCHS:
        jobs.append((G.static_ghsl(ep), ["BUILT_frac", "POP_mean100", "SMOD"], f"GHSL_{ep}_1km.tif", 256))
    for y in C.STATIC_YEARS:
        jobs.append((G.static_annual(y), ["VIIRS", "NDVI"], f"Annual_{y}_1km.tif", None))
    for ep in C.GHSL_EPOCHS:                    # discard GHSL files from earlier versions (no POP_mean100 band)
        f = os.path.join(S, f"GHSL_{ep}_1km.tif")
        if os.path.exists(f):
            with rasterio.open(f) as src:
                old = "POP_mean100" not in list(src.descriptions)
            if old:
                G.log(f"GHSL_{ep}_1km.tif is from an earlier version - deleting and rebuilding", "static")
                os.remove(f)
    for img, bands, name, tile in jobs:
        G.log(f"{name}: downloading ...", "static")
        G.log(f"{name}: {G.download_raster(img, bands, os.path.join(S, name), tile=tile)}", "static")


def read(name):
    with rasterio.open(os.path.join(S, name)) as src:
        return {src.descriptions[i]: src.read(i + 1) for i in range(src.count)}


def cell_area_m2():
    """Exact area of every 0.01 deg cell (m2), per row, on the authalic sphere of WGS84 (error < 0.3 %)."""
    R = 6371007.181
    lat_top = C.Y1 - np.arange(C.HEIGHT) * C.RES
    a = R ** 2 * np.deg2rad(C.RES) * (np.sin(np.deg2rad(lat_top)) - np.sin(np.deg2rad(lat_top - C.RES)))
    return np.repeat(a[:, None], C.WIDTH, axis=1).astype("float64")


def read_ghsl(ep, area):
    d = read(f"GHSL_{ep}_1km.tif")
    d["POP"] = np.nan_to_num(d["POP_mean100"], nan=0.0) * area / 1e4      # persons per 0.01 deg cell
    return d


def coast_distance():
    out = os.path.join(S, "CoastDist_km_1km.tif")
    lf = read("Static_Fixed_1km.tif")["LandFrac"]
    if not os.path.exists(out):
        from scipy import ndimage
        water = ~(np.nan_to_num(lf, nan=0.0) >= 0.5)
        mid = np.deg2rad((C.Y0 + C.Y1) / 2)
        sampling = (C.RES * 111.32, C.RES * 111.32 * np.cos(mid))      # km per cell (row, col); +-6% across the domain
        dist = ndimage.distance_transform_edt(~water, sampling=sampling).astype("float32")
        prof = dict(driver="GTiff", dtype="float32", count=1, width=C.WIDTH, height=C.HEIGHT, crs=C.CRS,
                    transform=TRANSFORM, nodata=np.nan, compress="deflate", predictor=3)
        with rasterio.open(out, "w", **prof) as dst:
            dst.write(dist, 1)
            dst.set_band_description(1, "CoastDist_km")
    with rasterio.open(out) as src:
        return src.read(1)


def layers_for_year(y, fixed, coast, ghsl):
    w = (y - C.GHSL_EPOCHS[0]) / (C.GHSL_EPOCHS[1] - C.GHSL_EPOCHS[0])
    a, b = ghsl[C.GHSL_EPOCHS[0]], ghsl[C.GHSL_EPOCHS[1]]
    ann = read(f"Annual_{y}_1km.tif")
    return {
        "DEM": fixed["DEM"], "LandFrac": fixed["LandFrac"], "CoastDist_km": coast,
        "BUILT_frac": (1 - w) * a["BUILT_frac"] + w * b["BUILT_frac"],
        "POP": np.clip((1 - w) * a["POP"] + w * b["POP"], 0, None),
        "SMOD": a["SMOD"] if y < 2023 else b["SMOD"],                     # class: nearest epoch, never averaged
        "VIIRS": ann["VIIRS"], "NDVI": ann["NDVI"],
    }


def sample_points(arr, rows, cols):
    px, m3 = [], []
    for r, c in zip(rows, cols):
        inside = 0 <= r < C.HEIGHT and 0 <= c < C.WIDTH
        px.append(arr[r, c] if inside else np.nan)
        win = arr[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2] if inside else np.array([np.nan])
        m3.append(np.nanmean(win) if np.isfinite(win).any() else np.nan)
    return np.array(px), np.array(m3)


def district_labels(g):
    order = g.sort_values("Area_km2", ascending=False)                   # small polygons drawn last
    lab = rasterize([(geom, i + 1) for i, geom in zip(order.index, order.geometry)],
                    out_shape=(C.HEIGHT, C.WIDTH), transform=TRANSFORM, fill=0, dtype="int32", all_touched=False)
    return lab.ravel()


def zonal(arr, lab, n, how="mean"):
    v = arr.ravel()
    ok = np.isfinite(v) & (lab > 0)
    tot = np.bincount(lab[ok], weights=v[ok], minlength=n + 1)[1:]
    cnt = np.bincount(lab[ok], minlength=n + 1)[1:]
    if how == "sum":
        return tot, cnt
    with np.errstate(invalid="ignore", divide="ignore"):
        return tot / cnt, cnt


def same_as_before(new, path, keys):
    """Integrity: rows that already existed in `path` must be identical in `new` (NaN == NaN).
    `new` is first passed through the same CSV write/read as the file on disk, so float32 layers are
    compared exactly as they would be stored (comparing an in-memory float32 with its shortest decimal
    text gives spurious differences of half a float32 step, e.g. 3e-5 at a DEM of 300 m).
    The earlier file is kept as <name>_before_<time>.csv."""
    import io
    if not os.path.exists(path):
        return
    old = pd.read_csv(path)
    new = pd.read_csv(io.StringIO(new.to_csv(index=False)))
    m = old.merge(new, on=keys, how="inner", suffixes=("_o", "_n"))
    cols = [c for c in old.columns if c not in keys and c + "_n" in m and pd.api.types.is_numeric_dtype(old[c])]
    worst, worst_col = 0.0, ""
    for c in cols:
        a, b = m[c + "_o"].astype(float), m[c + "_n"].astype(float)
        nan_mismatch = (a.isna() != b.isna()).sum()
        d = float(np.nanmax(np.abs(a - b))) if (a.notna() & b.notna()).any() else 0.0
        d = np.inf if nan_mismatch else d
        if d > worst:
            worst, worst_col = d, c
    lost = len(old) - len(m)
    print(f"integrity {os.path.basename(path)}: {len(m):,} earlier rows re-written, max |difference| {worst:.3g}"
          f"{' in ' + worst_col if worst_col else ''}, earlier rows missing {lost}")
    if worst > 0 or lost:
        raise SystemExit(f"{os.path.basename(path)}: earlier rows would change - nothing overwritten. Check this output.")
    os.replace(path, path.replace(".csv", f"_before_{pd.Timestamp.now():%Y%m%d_%H%M%S}.csv"))


def carry_forward_check(st, g, lab, n):
    """Part-year VIIRS / NDVI layers for STATIC_SENS_WINDOWS: tables for the modelling sensitivity run and a
    report of how far the carried-forward 2025 layers are from what 2026 actually looked like."""
    import ee
    wins = getattr(C, "STATIC_SENS_WINDOWS", [])
    if not wins:
        return
    lines = ["=== Carry-forward check of the annual layers (VIIRS, NDVI)"]
    layers = {}
    for a, b in wins:
        nv = ee.ImageCollection(C.VIIRS_MON).filterDate(a, b).size().getInfo()
        nn = ee.ImageCollection(C.MODIS_NDVI).filterDate(a, b).size().getInfo()
        tag = f"{a[:7]}_{b[:7]}"
        lines.append(f"   window {a} -> {b}: VIIRS months {nv} | MODIS 16-day composites {nn}")
        if nv == 0 or nn == 0:
            lines.append("      skipped - a layer has no images in this window yet")
            continue
        f = os.path.join(S, f"Partial_{tag}_1km.tif")
        G.log(f"{os.path.basename(f)}: {G.download_raster(G.static_period(a, b), ['VIIRS', 'NDVI'], f)}", "static")
        d = read(os.path.basename(f))
        layers[tag] = (d, nv, nn)
    if not layers:
        print("\n".join(lines))
        return
    srows, drows = [], []
    for tag, (d, nv, nn) in layers.items():
        s = st[["Station", "Network", "Kind", "Site_ID"]].copy()
        s["window"], s["VIIRS_months"], s["NDVI_composites"] = tag, nv, nn
        dd = pd.DataFrame({C.DISTRICT_ID: g[C.DISTRICT_ID].values, "window": tag, "VIIRS_months": nv, "NDVI_composites": nn})
        for k in ("VIIRS", "NDVI"):
            s[f"{k}_px"], s[f"{k}_3x3"] = sample_points(d[k], st.Row.values, st.Col.values)
            dd[k], _ = zonal(d[k], lab, n)
        srows.append(s)
        drows.append(dd)
    pd.concat(srows).to_csv(os.path.join(C.DIRS["tables"], "Station_Static_partial_windows.csv"), index=False)
    pd.concat(drows).to_csv(os.path.join(C.DIRS["tables"], "District_Static_partial_windows.csv"), index=False)

    full = read(f"Annual_{max(C.STATIC_YEARS)}_1km.tif")
    inside = lab.reshape(C.HEIGHT, C.WIDTH) > 0

    def cmp(x, y, name):
        ok = inside & np.isfinite(x) & np.isfinite(y)
        a_, b_ = x[ok].astype("float64"), y[ok].astype("float64")
        r = np.corrcoef(a_, b_)[0, 1]
        big = np.abs(b_) > (0.5 if name == "VIIRS" else 0.05)     # relative change only where the layer is not ~0
        rel = np.median(np.abs(a_[big] - b_[big]) / np.abs(b_[big])) if big.any() else np.nan
        return f"r {r:.3f} | median |relative difference| {rel:.1%} | median difference {np.median(a_ - b_):+.3g}"

    tags = list(layers)
    new = tags[-1]
    for k in ("VIIRS", "NDVI"):
        lines.append(f"   {k}, cells inside the 744 districts:")
        lines.append(f"      {new} vs full-year {max(C.STATIC_YEARS)} (what carrying forward replaces): "
                     + cmp(layers[new][0][k], full[k], k))
        if len(tags) > 1:
            lines.append(f"      {new} vs {tags[0]} (same months, one year apart = real change): "
                         + cmp(layers[new][0][k], layers[tags[0]][0][k], k))
    lines.append("   tables: Station_Static_partial_windows.csv, District_Static_partial_windows.csv "
                 "(used by the modelling sensitivity run)")
    txt = "\n".join(lines)
    print("\n" + txt)
    with open(os.path.join(C.DIRS["logs"], "static_carry_forward_check.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")


def main():
    G.init()
    download_all()
    fixed = read("Static_Fixed_1km.tif")
    coast = coast_distance()
    area = cell_area_m2()
    ghsl = {ep: read_ghsl(ep, area) for ep in C.GHSL_EPOCHS}
    for ep in C.GHSL_EPOCHS:
        print(f"GHSL {ep}: {ghsl[ep]['POP'].sum() / 1e6:.1f} M people in the whole 0.01 deg grid "
              f"(includes neighbouring countries inside the box)")

    st = pd.read_csv(C.STATIONS_CSV)
    g = gpd.read_file(C.GEOJSON).to_crs(4326).reset_index(drop=True)
    g["Area_km2"] = g.to_crs(6933).area / 1e6
    lab = district_labels(g)
    n = len(g)
    rp = g.geometry.representative_point()
    rp_r = np.floor((C.Y1 - rp.y.values) / C.RES).astype(int)
    rp_c = np.floor((rp.x.values - C.X0) / C.RES).astype(int)

    st_rows, d_rows = [], []
    carry = getattr(C, "STATIC_CARRY_FORWARD", {})
    for y in list(C.STATIC_YEARS) + sorted(carry):
        src_y = carry.get(y, y)                     # forward-test years reuse the last complete year
        L = layers_for_year(src_y, fixed, coast, ghsl)
        s = st[["Station", "Network", "Kind", "Site_ID"]].copy()
        s["Year"], s["Static_source_year"] = y, src_y
        for k, arr in L.items():
            s[f"{k}_px"], s[f"{k}_3x3"] = sample_points(arr, st.Row.values, st.Col.values)
        st_rows.append(s)

        d = pd.DataFrame({C.DISTRICT_ID: g[C.DISTRICT_ID].values, "Year": y, "Static_source_year": src_y,
                          "Area_km2": g.Area_km2.values})
        for k, arr in L.items():
            if k == "POP":
                tot, cnt = zonal(arr, lab, n, "sum")
                d["POP_total"], d["POP_density_km2"] = tot, tot / np.maximum(g.Area_km2.values, 1e-9)
            elif k == "SMOD":
                d["SMOD_urban_frac"], cnt = zonal((arr >= 21).astype("float32"), lab, n)   # GHSL classes 21-30
            else:
                d[k], cnt = zonal(arr, lab, n)
        _, cnt = zonal(fixed["DEM"], lab, n)
        d["n_cells"] = cnt
        tiny = cnt == 0                                                   # no cell centre inside the polygon
        for k, arr in L.items():
            col = {"POP": "POP_total", "SMOD": "SMOD_urban_frac"}.get(k, k)
            vals = arr[rp_r[tiny], rp_c[tiny]]
            if k == "POP":                                                # share of the cell's people, by area
                vals = vals * (g.Area_km2.values[tiny] * 1e6) / area[rp_r[tiny], rp_c[tiny]]
            d.loc[tiny, col] = (vals >= 21).astype(float) if k == "SMOD" else vals
        d["point_fallback"] = tiny
        d_rows.append(d)

    S_out = pd.concat(st_rows, ignore_index=True)
    D_out = pd.concat(d_rows, ignore_index=True)
    S_out["GHSL_rule"] = D_out["GHSL_rule"] = ("linear between 2020 and 2025 epochs; SMOD nearest epoch; "
                                               "years in STATIC_CARRY_FORWARD reuse Static_source_year")
    same_as_before(S_out, os.path.join(C.DIRS["tables"], "Station_Static_by_Year.csv"), ["Station", "Year"])
    S_out.to_csv(os.path.join(C.DIRS["tables"], "Station_Static_by_Year.csv"), index=False)
    D_path = os.path.join(C.DIRS["tables"], "District_Static_by_Year.csv")
    print(f"wrote Station_Static_by_Year.csv ({len(S_out)}); District_Static_by_Year.csv follows the population check")
    print(f"districts with no cell centre inside (point fallback): {int(D_out[D_out.Year == 2020].point_fallback.sum())}")

    # ---- population check against the native-resolution district sums from 02b (independent method)
    ctry = next((c for c in ["COUNTRY", "NAME_0", "Country"] if c in g.columns), None)
    rows = []
    for ep in C.GHSL_EPOCHS:
        nat_f = os.path.join(C.DIRS["tables"], f"District_GHSL_POP_native_{ep}.csv")
        grid = D_out[D_out.Year == ep][[C.DISTRICT_ID, "POP_total"]]
        if os.path.exists(nat_f):
            nat = pd.read_csv(nat_f)[[C.DISTRICT_ID, f"POP_native_{ep}"]]
            m = grid.merge(nat, on=C.DISTRICT_ID).merge(g[[C.DISTRICT_ID, ctry]], on=C.DISTRICT_ID)
            t = m.groupby(ctry)[["POP_total", f"POP_native_{ep}"]].sum() / 1e6
            t.columns = [f"grid_{ep}", f"native_{ep}"]
            t[f"ratio_{ep}"] = t.iloc[:, 0] / t.iloc[:, 1]
            rows.append(t)
    if rows:
        tab = pd.concat(rows, axis=1)
        tab.loc["GCC"] = tab.sum()
        for ep in C.GHSL_EPOCHS:
            if f"ratio_{ep}" in tab:
                tab.loc["GCC", f"ratio_{ep}"] = tab.loc["GCC", f"grid_{ep}"] / tab.loc["GCC", f"native_{ep}"]
        print("\nGHSL population (millions): 1 km grid summed over districts vs native 100 m district sums")
        print(tab.round(3).to_string())
        worst = max(abs(tab[c] - 1).max() for c in tab.columns if c.startswith("ratio"))
        if worst > 0.05:
            raise SystemExit(f"grid and native population differ by up to {worst:.1%} - check this output")
        print(f"agreement within {worst:.1%} - population layer accepted")
        # districts: the native 100 m sum is exact, so it becomes POP_total; the grid sum is kept for QA
        nat = {ep: pd.read_csv(os.path.join(C.DIRS["tables"], f"District_GHSL_POP_native_{ep}.csv"))
                       .set_index(C.DISTRICT_ID)[f"POP_native_{ep}"] for ep in C.GHSL_EPOCHS}
        w = (D_out.Static_source_year - C.GHSL_EPOCHS[0]) / (C.GHSL_EPOCHS[1] - C.GHSL_EPOCHS[0])
        a0 = D_out[C.DISTRICT_ID].map(nat[C.GHSL_EPOCHS[0]]).values
        a1 = D_out[C.DISTRICT_ID].map(nat[C.GHSL_EPOCHS[1]]).values
        D_out["POP_total_grid"] = D_out.POP_total
        D_out["POP_total"] = np.clip((1 - w) * a0 + w * a1, 0, None)
        D_out["POP_density_km2"] = D_out.POP_total / np.maximum(D_out.Area_km2, 1e-9)
        same_as_before(D_out, D_path, [C.DISTRICT_ID, "Year"])
        D_out.to_csv(D_path, index=False)
        print("District_Static_by_Year.csv: POP_total = native 100 m district sum (interpolated by year); "
              "POP_total_grid kept for QA")
    else:
        print("[note] run 02b_check_population.py once to create the native district totals for this check")
        D_out.to_csv(D_path, index=False)
    carry_forward_check(st, g, lab, n)
    G.write_manifest("static", {"stations": len(st), "districts": n})


if __name__ == "__main__":
    main()
