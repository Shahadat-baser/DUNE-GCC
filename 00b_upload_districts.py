"""
STEP 0b - copy the 744 districts from the local GCC_Cities.geojson into Earth Engine.

Result: the table  config.DISTRICT_ASSET  (projects/<project>/assets/DUNE_v3/GCC_Districts_744).
You never have to look up an asset id - every later script reads it from dune_config.py.

Why upload from the local file instead of reusing an old asset: the paper's district means are
computed on the server and its maps / static means are computed locally.  Both must use the SAME
polygons.  This script uploads exactly the local geometries and then checks the asset against the
file (feature count, ids, total area).

How (no Google Cloud Storage needed):
  1. read the geojson, keep OBJECTID + names, repair invalid polygons (make_valid, polygon parts only)
  2. send the polygons in parts small enough for one request (< ~4 MB each) to temporary assets
  3. merge the parts on the server into ONE table, delete the parts
Edges are planar in lon/lat (geodesic = False), the same convention as the GeoJSON standard and as
the local rasterisation in STEP 2.

Takes 5-20 minutes, mostly waiting for Earth Engine tasks.  Safe to rerun: if the final table
already exists and matches the file, it does nothing.
"""
import os, sys, json, time
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
dune_env.ensure("upload_districts")      # installs / checks every package this step needs
import geopandas as gpd
from shapely.geometry import mapping, MultiPolygon, Polygon, GeometryCollection
from shapely.validation import make_valid
import ee
import dune_config as C
import dune_gee as G
if getattr(G, "PIPELINE_VERSION", None) != "2026-10-08a":
    raise SystemExit(f"{G.__file__} is an older version than this script. Unzip the WHOLE DUNE_GEE_v3.zip "
                     "again into the code folder and choose 'Replace the files in the destination'.")

MAX_PART_BYTES = 4_000_000


def polygonal(geom):
    g = geom if geom.is_valid else make_valid(geom)
    if isinstance(g, (Polygon, MultiPolygon)):
        return g
    if isinstance(g, GeometryCollection):
        polys = []
        for p in g.geoms:
            if isinstance(p, Polygon):
                polys.append(p)
            elif isinstance(p, MultiPolygon):
                polys.extend(p.geoms)
        return MultiPolygon(polys) if polys else None
    return None


def asset_exists(aid):
    try:
        ee.data.getAsset(aid)
        return True
    except Exception:
        return False


def wait(tasks):
    while True:
        st = [t.status() for t in tasks]
        states = [s["state"] for s in st]
        print("   tasks: " + ", ".join(f"{s['description']}={s['state']}" for s in st), flush=True)
        if all(s in ("COMPLETED", "FAILED", "CANCELLED") for s in states):
            bad = [s for s in st if s["state"] != "COMPLETED"]
            if bad:
                raise SystemExit(f"export failed: {bad[0].get('error_message', bad[0])}")
            return
        time.sleep(20)


def check(local):
    fc = ee.FeatureCollection(C.DISTRICT_ASSET)
    info = ee.Dictionary({
        "n": fc.size(),
        "ids": fc.aggregate_count_distinct(C.DISTRICT_ID),
        "area_km2": fc.map(lambda f: f.set("a", f.geometry().area(100))).aggregate_sum("a").divide(1e6),
    }).getInfo()
    loc_area = local.to_crs(6933).area.sum() / 1e6
    diff = 100 * (info["area_km2"] - loc_area) / loc_area
    ok = info["n"] == len(local) and info["ids"] == local[C.DISTRICT_ID].nunique() and abs(diff) < 0.5
    print(f"asset {C.DISTRICT_ASSET}\n   features {info['n']} (file {len(local)}) | unique {C.DISTRICT_ID} "
          f"{info['ids']} | area {info['area_km2']:,.0f} km2 vs file {loc_area:,.0f} km2 ({diff:+.3f}%)"
          f"\n   {'MATCHES THE LOCAL FILE' if ok else 'DOES NOT MATCH - check the numbers above'}")
    return ok


def main():
    G.init()
    g = gpd.read_file(C.GEOJSON).to_crs(4326)
    assert len(g) == 744 and g[C.DISTRICT_ID].is_unique, f"expected 744 unique {C.DISTRICT_ID}, got {len(g)}"
    keep = [c for c in C.DISTRICT_KEEP if c in g.columns]
    g = g[keep + ["geometry"]].copy()
    n_bad = (~g.geometry.is_valid).sum()
    g["geometry"] = g.geometry.apply(polygonal)
    if g.geometry.isna().any():
        raise SystemExit(f"{g.geometry.isna().sum()} districts have no polygon after repair: "
                         f"{g.loc[g.geometry.isna(), C.DISTRICT_ID].tolist()}")
    print(f"{len(g)} districts read from {C.GEOJSON} | repaired invalid geometries: {n_bad} | attributes {keep}")

    if asset_exists(C.DISTRICT_ASSET):
        print("final table already exists - checking it against the file")
        if check(g):
            return
        raise SystemExit("The existing asset differs from the file. Delete it in the Code Editor "
                         "(Assets tab) and rerun this script.")

    if not asset_exists(C.ASSET_FOLDER):
        ee.data.createFolder(C.ASSET_FOLDER)
        print(f"created folder {C.ASSET_FOLDER}")

    feats, sizes = [], []
    for r in g.itertuples(index=False):
        gj = mapping(r.geometry)
        props = {k: (int(getattr(r, k)) if k == C.DISTRICT_ID else str(getattr(r, k))) for k in keep}
        feats.append((gj, props))
        sizes.append(len(json.dumps(gj)) + 200)
    parts, cur, cur_b = [], [], 0
    for f, b in zip(feats, sizes):
        if cur and cur_b + b > MAX_PART_BYTES:
            parts.append(cur); cur, cur_b = [], 0
        cur.append(f); cur_b += b
    parts.append(cur)
    print(f"uploading in {len(parts)} part(s), {sum(sizes) / 1e6:.1f} MB of geometry")

    part_ids, tasks = [], []
    for i, part in enumerate(parts, 1):
        aid = f"{C.DISTRICT_ASSET}_part{i:02d}"
        part_ids.append(aid)
        if asset_exists(aid):
            continue
        fc = ee.FeatureCollection([ee.Feature(ee.Geometry(gj, None, False), p) for gj, p in part])
        t = ee.batch.Export.table.toAsset(collection=fc, description=f"DUNE_districts_part{i:02d}", assetId=aid)
        t.start(); tasks.append(t)
    if tasks:
        wait(tasks)

    merged = ee.FeatureCollection([ee.FeatureCollection(a) for a in part_ids]).flatten()
    t = ee.batch.Export.table.toAsset(collection=merged, description="DUNE_districts_merge",
                                      assetId=C.DISTRICT_ASSET)
    t.start(); wait([t])
    for a in part_ids:
        ee.data.deleteAsset(a)
    if not check(g):
        raise SystemExit("uploaded table does not match the file")
    G.write_manifest("upload_districts", {"asset": C.DISTRICT_ASSET, "features": len(g), "repaired": int(n_bad)})


if __name__ == "__main__":
    main()
