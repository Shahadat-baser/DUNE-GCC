"""
STEP 0 - preflight.  Fast (a few minutes).  Run it first and check logs/preflight_report.txt.

Checks, before a single large request is made:
  1. library versions and the 0.01 deg grid
  2. every source collection: date range actually available, image count, required band names
  3. ERA5 / ERA5-Land: one image at the overpass hour for EVERY day of every year (no silent gaps)
  4. GHSL epochs exist
  5. the local district file, and whether the district table is already in Earth Engine
  6. smoke test: one day's predictors at Riyadh, Dubai, Manama and Muscat, with plausibility ranges
  7. Sentinel-5P processor metadata (whatever version property the images carry)

Also proves the network/SSL setting works (every check is a real request).

Notebook:  %run 00_preflight.py        Terminal:  python 00_preflight.py
"""
import os, sys, datetime as dt
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
dune_env.ensure("preflight")      # installs / checks every package this step needs
import ee
import dune_config as C
import dune_gee as G
if getattr(G, "PIPELINE_VERSION", None) != "2026-10-08a":
    raise SystemExit(f"{G.__file__} is an older version than this script. Unzip the WHOLE DUNE_GEE_v3.zip "
                     "again into the code folder and choose 'Replace the files in the destination'.")

REPORT = []


def say(s=""):
    print(s, flush=True)
    REPORT.append(str(s))


def collection_summary(cid, start, end, need_bands):
    col = ee.ImageCollection(cid).filterDate(start, end)
    info = ee.Dictionary({
        "n": col.size(),
        "first": ee.Date(col.aggregate_min("system:time_start")).format("yyyy-MM-dd HH:mm"),
        "last": ee.Date(col.aggregate_max("system:time_start")).format("yyyy-MM-dd HH:mm"),
        "bands": ee.Image(col.first()).bandNames(),
    }).getInfo()
    missing = [b for b in need_bands if b not in info["bands"]]
    say(f"  {cid}\n      images {info['n']:,} | {info['first']} -> {info['last']}"
        f" | required bands {'OK' if not missing else 'MISSING ' + str(missing)}")
    return not missing


def year_windows():
    """(year, start, end) for every calendar year in [DATE_START, DATE_END), last year possibly partial."""
    s0, e0 = dt.date.fromisoformat(C.DATE_START), dt.date.fromisoformat(C.DATE_END)
    return [(y, max(dt.date(y, 1, 1), s0), min(dt.date(y + 1, 1, 1), e0)) for y in range(s0.year, C.LAST_YEAR + 1)]


def hourly_daily_coverage(cid, label):
    bad = False
    for y, a, b in year_windows():
        n = (ee.ImageCollection(cid).filterDate(a.isoformat(), b.isoformat())
             .filter(ee.Filter.calendarRange(C.MET_HOUR_UTC, C.MET_HOUR_UTC, "hour")).size().getInfo())
        ndays = (b - a).days
        flag = "OK" if n == ndays else "GAP"
        bad |= n != ndays
        say(f"      {label} {y}: {n}/{ndays} days at {C.MET_HOUR_UTC:02d} UTC  {flag}")
    return not bad


def list_tables(parent, depth=0, out=None):
    out = [] if out is None else out
    try:
        assets = ee.data.listAssets({"parent": parent}).get("assets", [])
    except Exception as exc:
        say(f"    cannot list {parent}: {exc}")
        return out
    for a in assets:
        say(f"    {'  ' * depth}{a['type']:<16} {a['name']}")
        if a["type"] == "TABLE":
            out.append(a["name"])
        elif a["type"] == "FOLDER" and depth < 2:
            list_tables(a["name"], depth + 1, out)
    return out


def main():
    G.init()
    ok = True
    say(f"DUNE v3 preflight  {dt.datetime.utcnow():%Y-%m-%d %H:%M} UTC")
    say(f"earthengine-api {ee.__version__} | python {sys.version.split()[0]}")
    say(f"grid {C.CRS} {C.RES} deg | {C.WIDTH} x {C.HEIGHT} cells | transform {C.CRS_TRANSFORM}")

    say("\n[1] source collections")
    s, e = C.DATE_START, C.DATE_END
    ok &= collection_summary(C.S5P_NO2, s, e, list(C.S5P_BANDS))
    ok &= collection_summary(C.ERA5L_HR, s, e, list(C.MET_BANDS))
    ok &= collection_summary(C.ERA5_HR, s, e, list(C.MET_BANDS) + ["boundary_layer_height"])
    ok &= collection_summary(C.ERA5L_DAY, s, e, ["total_precipitation_sum"])
    ok &= collection_summary(C.VIIRS_MON, s, e, ["avg_rad", "cf_cvg"])
    ok &= collection_summary(C.MODIS_NDVI, s, e, ["NDVI"])

    say("\n[2] one image per day at the overpass hour")
    ok &= hourly_daily_coverage(C.ERA5L_HR, "ERA5-Land")
    ok &= hourly_daily_coverage(C.ERA5_HR, "ERA5     ")
    for y, a, b in year_windows():
        a_, b_ = a.isoformat(), b.isoformat()
        n = ee.ImageCollection(C.ERA5L_DAY).filterDate(a_, b_).size().getInfo()
        nd = (b - a).days
        ok &= n == nd
        say(f"      ERA5-Land daily precip {y}: {n}/{nd} days  {'OK' if n == nd else 'GAP'}")
        so = (ee.ImageCollection(C.S5P_NO2).filterDate(a_, b_)
              .map(lambda i: i.set("d", i.date().format("YYYY-MM-dd"))).aggregate_count_distinct("d").getInfo())
        say(f"      S5P days with at least one orbit {y}: {so}/{nd}")
        v = ee.ImageCollection(C.VIIRS_MON).filterDate(a_, b_).size().getInfo()
        m = ee.ImageCollection(C.MODIS_NDVI).filterDate(a_, b_).size().getInfo()
        say(f"      VIIRS months {y}: {v} | MODIS 16-day composites {y}: {m}"
            + (f"   (annual layers for {y} are carried forward from {C.STATIC_CARRY_FORWARD[y]})"
               if y in getattr(C, "STATIC_CARRY_FORWARD", {}) else ""))

    say("\n[3] GHSL epochs")
    for ep in C.GHSL_EPOCHS:
        for cid, band in [(C.GHSL_BUILT, "built_surface"), (C.GHSL_POP, "population_count"), (C.GHSL_SMOD, "smod_code")]:
            try:
                bands = ee.Image(f"{cid}/{ep}").bandNames().getInfo()
                say(f"      {cid}/{ep}: {'OK' if band in bands else 'band missing ' + str(bands)}")
            except Exception as exc:
                ok = False
                say(f"      {cid}/{ep}: NOT FOUND ({str(exc)[:100]})")

    say("\n[4] district table in Earth Engine")
    say(f"    local file : {C.GEOJSON} ({'found' if os.path.exists(C.GEOJSON) else 'NOT FOUND'})")
    ok &= os.path.exists(C.GEOJSON)
    try:
        n = ee.FeatureCollection(C.DISTRICT_ASSET).size().getInfo()
        say(f"    asset      : {C.DISTRICT_ASSET} -> {n} features {'OK' if n == 744 else 'EXPECTED 744'}")
        ok &= n == 744
    except Exception:
        say(f"    asset      : {C.DISTRICT_ASSET} not uploaded yet -> run 00b_upload_districts.py next")
    say("    existing tables in your project (for information only):")
    list_tables(f"projects/{C.EE_PROJECT}/assets")

    say("\n[5] smoke test - 2023-01-15 predictors at four cities (values at the 0.01 deg cell)")
    pts = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([46.70, 24.70]), {"name": "Riyadh"}),
        ee.Feature(ee.Geometry.Point([55.30, 25.25]), {"name": "Dubai"}),
        ee.Feature(ee.Geometry.Point([50.58, 26.22]), {"name": "Manama"}),
        ee.Feature(ee.Geometry.Point([58.40, 23.59]), {"name": "Muscat"}),
    ])
    img = G.daily_image(ee.Date("2023-01-15"))
    props = img.toDictionary(G.DAY_PROPS).getInfo()
    say(f"    day properties: {props}")
    rows = img.reduceRegions(pts, ee.Reducer.first(), crs=C.CRS, crsTransform=C.CRS_TRANSFORM).getInfo()["features"]
    rng = {"NO2_trop": (-5e-5, 2e-3), "T2m": (270, 320), "Td2m": (230, 305), "U10": (-30, 30), "V10": (-30, 30),
           "SP": (85000, 104000), "PBLH": (10, 5000), "PRECIP": (0, 200), "S5P_cloud": (0, 1), "S5P_AAI": (-5, 10)}
    for f in rows:
        p = f["properties"]
        cells = []
        for b, (lo, hi) in rng.items():
            v = p.get(b)
            flag = "" if v is None or lo <= v <= hi else " <-- OUT OF RANGE"
            ok &= flag == ""
            cells.append(f"{b}={v:.4g}{flag}" if isinstance(v, (int, float)) else f"{b}=NA")
        say(f"    {p['name']:<7} " + ", ".join(cells) + f", MET_fill={p.get('MET_fill')}")

    say("\n[6] Sentinel-5P metadata (look for a processor / version property)")
    for day in ["2019-06-01", "2022-06-01", "2025-06-01"]:
        im = ee.Image(ee.ImageCollection(C.S5P_NO2).filterDate(day, (dt.date.fromisoformat(day) + dt.timedelta(1)).isoformat()).first())
        names = im.propertyNames().getInfo()
        keep = [n for n in names if any(k in n.upper() for k in ["VERSION", "PROCESSOR", "ALGORITHM", "COLLECTION"])]
        say(f"    {day}: {len(names)} properties; version-like: " +
            (str(im.toDictionary(keep).getInfo()) if keep else "none - record versions from the S5P ATBD instead"))

    say("\nRESULT: " + ("ALL CHECKS PASSED - next: 00b_upload_districts.py (if not done), then 01_build_inputs.py" if ok else
                        "SOMETHING NEEDS ATTENTION - read the lines marked GAP / MISSING / OUT OF RANGE / EXPECTED"))
    with open(os.path.join(C.DIRS["logs"], "preflight_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(REPORT))
    G.write_manifest("preflight", {"passed": bool(ok)})
    print(f"\nreport -> {os.path.join(C.DIRS['logs'], 'preflight_report.txt')}")


if __name__ == "__main__":
    main()
