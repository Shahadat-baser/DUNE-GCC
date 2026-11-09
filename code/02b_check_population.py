"""
STEP 2b - population diagnosis (2-5 minutes).  Run once and check the whole printout.

This script measures the population at every stage, so the faulty stage will be identified instead of guessed:

  A. the GHSL source itself: projection, pixel size, value range, and the people inside Bahrain and
     inside a 10 km box over central Riyadh, summed at the NATIVE 100 m resolution
  B. the same two areas summed from our 1 km POP image, computed on the server
  C. the same from the downloaded file static/GHSL_2020_1km.tif, on this PC
  D. the authoritative district totals: people per district summed at NATIVE 100 m resolution for all
     744 districts -> tables/District_GHSL_POP_native_<epoch>.csv, then the per-country totals

If A is right and B is wrong, the server-side aggregation is at fault; if B is right and C is wrong,
the download/reading is at fault.  D is the standard method in any case (zonal sum at native scale)
and will be the district population used in the paper.
"""
import os, sys
CODE_DIR = os.getcwd()                      # fallback when __file__ is undefined (notebook)
try:
    CODE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    if os.path.exists(os.path.join(os.getcwd(), "dune_config.py")):
        CODE_DIR = os.getcwd()
sys.path.insert(0, CODE_DIR)
for _m in ("dune_config", "dune_env", "dune_gee"):
    sys.modules.pop(_m, None)
import dune_env
dune_env.ensure("static")
import numpy as np, pandas as pd, rasterio
import ee
import dune_config as C
import dune_gee as G
if getattr(G, "PIPELINE_VERSION", None) != "2026-10-08a":
    raise SystemExit(f"{G.__file__} is an older version than this script. Unzip the WHOLE DUNE_GEE_v3.zip "
                     "again into the code folder and choose 'Replace the files in the destination'.")


def main():
    G.init()
    ep = C.GHSL_EPOCHS[0]
    src = ee.Image(f"{C.GHSL_POP}/{ep}").select("population_count")
    proj = src.projection()
    dist = ee.FeatureCollection(C.DISTRICT_ASSET)
    bahrain = dist.filter(ee.Filter.eq("COUNTRY", "Bahrain")).geometry()
    riyadh = ee.Geometry.Rectangle([46.62, 24.62, 46.72, 24.72], None, False)

    print(f"=== A. GHSL source {C.GHSL_POP}/{ep}")
    info = ee.Dictionary({
        "crs": proj.crs(), "scale_m": proj.nominalScale(),
        "bands": src.bandNames(), "type": src.bandTypes(),
    }).getInfo()
    print(f"    projection {info['crs']} | nominal scale {info['scale_m']:.1f} m | type {info['type']}")
    for name, geom in [("Bahrain (all districts)", bahrain), ("Riyadh 10 km box", riyadh)]:
        st = src.reduceRegion(ee.Reducer.sum().combine(ee.Reducer.minMax(), "", True)
                              .combine(ee.Reducer.count(), "", True),
                              geom, crs=proj, scale=info["scale_m"], maxPixels=1e10).getInfo()
        print(f"    {name:<24} native sum {st['population_count_sum']:>14,.0f} people | "
              f"min {st['population_count_min']:.2f} max {st['population_count_max']:.2f} | "
              f"{st['population_count_count']:,} cells")

    print("\n=== B. our 1 km POP image, summed on the server")
    pop1 = G.static_ghsl(ep).select("POP")
    for name, geom in [("Bahrain (all districts)", bahrain), ("Riyadh 10 km box", riyadh)]:
        st = pop1.reduceRegion(ee.Reducer.sum().combine(ee.Reducer.count(), "", True), geom,
                               crs=C.CRS, crsTransform=C.CRS_TRANSFORM, maxPixels=1e10).getInfo()
        print(f"    {name:<24} 1 km sum {st['POP_sum']:>14,.0f} people | {st['POP_count']:,} cells")
    one = pop1.reduceRegion(ee.Reducer.first(), ee.Geometry.Point([46.68, 24.68]),
                            crs=C.CRS, crsTransform=C.CRS_TRANSFORM).getInfo()
    nat = src.reduceRegion(ee.Reducer.sum(), ee.Geometry.Rectangle([46.68, 24.67, 46.69, 24.68], None, False),
                           crs=proj, scale=info["scale_m"]).getInfo()
    print(f"    one Riyadh cell (46.68E 24.68N): 1 km image {one.get('POP')} | native 100 m sum over "
          f"the same 0.01 deg cell {nat.get('population_count')}")

    print("\n=== C. downloaded file on this PC")
    f = os.path.join(C.DIRS["static"], f"GHSL_{ep}_1km.tif")
    with rasterio.open(f) as r:
        names = list(r.descriptions)
        a = r.read(names.index("POP") + 1).astype("float64")
        rr, cc = r.index(46.685, 24.675)
        print(f"    {os.path.basename(f)}: bands {names} | dtype {r.dtypes[0]} | nodata {r.nodata}")
        print(f"    POP: nansum {np.nansum(a):,.0f} | min {np.nanmin(a):.2f} | max {np.nanmax(a):,.1f} | "
              f"negative cells {(a < 0).sum():,} | -9999 cells {(a == -9999).sum():,} | NaN {np.isnan(a).sum():,}")
        print(f"    POP at the same Riyadh cell (row {rr}, col {cc}): {a[rr, cc]:,.1f}")

    print("\n=== D. authoritative district totals (sum at native 100 m)")
    for e in C.GHSL_EPOCHS:
        img = ee.Image(f"{C.GHSL_POP}/{e}").select("population_count").rename("POP_native")
        img = img.updateMask(img.gte(0))                       # drop any negative no-data code
        fc = img.reduceRegions(collection=dist.select([C.DISTRICT_ID, "COUNTRY"]), reducer=ee.Reducer.sum(),
                               crs=proj, scale=info["scale_m"], tileScale=4)
        df = ee.data.computeFeatures({"expression": fc, "fileFormat": "PANDAS_DATAFRAME"})
        df = df[[C.DISTRICT_ID, "COUNTRY", "sum"]].rename(columns={"sum": f"POP_native_{e}"})
        out = os.path.join(C.DIRS["tables"], f"District_GHSL_POP_native_{e}.csv")
        df.to_csv(out, index=False)
        tot = (df.groupby("COUNTRY")[f"POP_native_{e}"].sum() / 1e6).round(2)
        print(f"    {e}: GCC {tot.sum():.1f} M | " + " | ".join(f"{k} {v}" for k, v in tot.items()))
        print(f"       -> {out}")


if __name__ == "__main__":
    main()
