"""
DUNE - shared Earth Engine logic.  Every image definition used by the paper lives in this file,
so a reviewer reads ONE place to see exactly how each predictor is built.

Conventions
    * Day = UTC calendar day [00:00, 24:00).  Dates are written by Python as YYYY-MM-DD.
      (Never use Earth Engine 'YYYYMMDD' for file names: in Joda-Time 'DD' is DAY-OF-YEAR, which is
      the bug that broke the original MET_1km_* file names.)
    * Grid = config.CRS + config.CRS_TRANSFORM (0.01 deg).  Every reduction passes both.
    * Resampling follows the resolution ratio:
        source coarser than 1 km (ERA5-Land 0.1 deg, ERA5 0.25 deg)  -> bilinear
        source at ~1 km (S5P L3 at a nominal 1113 m)                  -> nearest (no smoothing added)
        source finer than 1 km (VIIRS, MODIS, GHSL, SRTM, Hansen mask) -> reduceResolution (area-weighted)
"""
import os, io, json, time, random, datetime as dt, threading
import numpy as np
import pandas as pd
import ee
import dune_config as C

PIPELINE_VERSION = "2026-10-08a"   # every step script refuses to run against a different dune_gee.py

_LOCK = threading.Lock()


# =========================================================================== init / logging
_SSL_DONE = False


def configure_ssl():
    """Apply C.SSL_MODE once, BEFORE any network call.  Earth Engine, Google sign-in and our own
    downloads all go through the 'requests' library, so one switch covers every connection."""
    global _SSL_DONE
    if _SSL_DONE:
        return
    mode = getattr(C, "SSL_MODE", "default")
    if mode == "truststore":
        import truststore
        truststore.inject_into_ssl()
    elif mode == "off":
        import ssl, warnings, requests, urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        warnings.filterwarnings("ignore", message="Unverified HTTPS request")
        ssl._create_default_https_context = ssl._create_unverified_context      # urllib
        _orig = requests.Session.request

        def _no_verify(self, method, url, **kw):
            kw["verify"] = False
            return _orig(self, method, url, **kw)
        requests.Session.request = _no_verify                                   # EE, google-auth, downloads
    _SSL_DONE = True


def init():
    configure_ssl()
    try:
        ee.Initialize(project=C.EE_PROJECT)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=C.EE_PROJECT)
    # Without a deadline a request whose connection silently drops (network change, PC sleep, proxy
    # reset) waits FOREVER - that is what froze 05_weekly_rasters.py for days after week 2022-45.
    # With it, a stalled request fails after REQUEST_TIMEOUT_S and is retried like any other error.
    ee.data.setDeadline(int(C.REQUEST_TIMEOUT_S * 1000))
    C.make_dirs()


def log(msg, name="run"):
    line = f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with _LOCK, open(os.path.join(C.DIRS["logs"], f"{name}.log"), "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def write_manifest(name, extra=None):
    import sys, platform
    m = dict(script=name, utc=dt.datetime.utcnow().isoformat(timespec="seconds"),
             earthengine_api=ee.__version__, python=sys.version.split()[0], platform=platform.platform(),
             grid=dict(crs=C.CRS, crs_transform=C.CRS_TRANSFORM, width=C.WIDTH, height=C.HEIGHT),
             period=[C.DATE_START, C.DATE_END], collections=dict(
                 S5P=C.S5P_NO2, ERA5L_HR=C.ERA5L_HR, ERA5_HR=C.ERA5_HR, ERA5L_DAY=C.ERA5L_DAY,
                 VIIRS=C.VIIRS_MON, NDVI=C.MODIS_NDVI, GHSL=[C.GHSL_BUILT, C.GHSL_POP, C.GHSL_SMOD],
                 DEM=C.DEM, LANDMASK=C.LANDMASK),
             met_hour_utc=C.MET_HOUR_UTC)
    m.update(extra or {})
    p = os.path.join(C.DIRS["logs"], f"manifest_{name}_{dt.datetime.now():%Y%m%d_%H%M%S}.json")
    with open(p, "w") as fh:
        json.dump(m, fh, indent=2, default=str)
    return p


# =========================================================================== small helpers
def _empty(names):
    """A fully masked float image with the given band names (used when a source has no image)."""
    return ee.Image.constant([0] * len(names)).rename(names).toFloat().updateMask(ee.Image.constant(0))


def _first_or_empty(col, names):
    return ee.Image(ee.Algorithms.If(col.size().gt(0), ee.Image(col.first()).toFloat(), _empty(names)))


def to_grid(img):
    return img.reproject(crs=C.CRS, crsTransform=C.CRS_TRANSFORM)


def _fine_to_grid(img, reducer, max_pixels=4096):
    """Area-weighted aggregation of a FINER source onto the 0.01 deg grid (mean/sum/mode)."""
    return (img.reduceResolution(reducer=reducer, bestEffort=False, maxPixels=max_pixels)
               .reproject(crs=C.CRS, crsTransform=C.CRS_TRANSFORM))


# =========================================================================== DAILY predictor image
S5P_SRC, S5P_OUT = list(C.S5P_BANDS.keys()), list(C.S5P_BANDS.values())
MET_SRC, MET_OUT = list(C.MET_BANDS.keys()), list(C.MET_BANDS.values())


def daily_image(d):
    """All dynamic predictors for one UTC day d (ee.Date).  Bands = C.DYNAMIC_BANDS + MET_fill.

    NO2_trop   : mean of all OFFL L3 orbits that day (L3 already applies qa_value > 0.75); negatives kept
    NO2_valid  : 1 where NO2_trop is observed, 0 elsewhere (unmasked; its zonal mean = valid fraction)
    T2m..SP    : ERA5-Land at MET_HOUR_UTC, bilinear; masked sea/coast cells filled from ERA5
    MET_fill   : 1 where the ERA5 fill was used (coastal / island cells, e.g. Bahrain)
    PBLH       : ERA5 boundary_layer_height at MET_HOUR_UTC (ERA5-Land has no PBLH), bilinear
    PRECIP     : ERA5-Land daily total precipitation (mm), coast gaps filled from neighbouring land cells
    """
    d = ee.Date(d)
    nxt = d.advance(1, "day")

    s5p_col = ee.ImageCollection(C.S5P_NO2).filterDate(d, nxt).select(S5P_SRC, S5P_OUT)
    s5p = ee.Image(ee.Algorithms.If(s5p_col.size().gt(0), s5p_col.mean().toFloat(), _empty(S5P_OUT)))
    no2_valid = s5p.select("NO2_trop").mask().gt(0).rename("NO2_valid").toFloat()

    t0 = d.advance(C.MET_HOUR_UTC, "hour")
    t1 = t0.advance(1, "hour")
    land = _first_or_empty(ee.ImageCollection(C.ERA5L_HR).filterDate(t0, t1).select(MET_SRC, MET_OUT), MET_OUT)
    e5 = ee.ImageCollection(C.ERA5_HR).filterDate(t0, t1)
    sea = _first_or_empty(e5.select(MET_SRC, MET_OUT), MET_OUT)
    land = land.resample("bilinear")
    met = land.unmask(sea.resample("bilinear"))
    met_fill = land.select("T2m").mask().eq(0).rename("MET_fill").toFloat()
    pblh = _first_or_empty(e5.select(["boundary_layer_height"], ["PBLH"]), ["PBLH"]).resample("bilinear")

    pcol = ee.ImageCollection(C.ERA5L_DAY).filterDate(d, nxt).select(["total_precipitation_sum"], ["PRECIP"])
    p = _first_or_empty(pcol, ["PRECIP"]).multiply(1000)                      # m -> mm
    p_native = p.reproject(ee.ImageCollection(C.ERA5L_DAY).first().projection())
    precip = p_native.unmask(p_native.focalMean(2, "square", "pixels")).resample("bilinear").rename("PRECIP")

    img = ee.Image.cat([s5p, met, pblh, precip, no2_valid, met_fill]).toFloat()
    return img.set({
        "system:time_start": d.millis(),
        "date": d.format("yyyy-MM-dd"),
        "S5P_n_orbits": s5p_col.size(),
        "ERA5L_n": ee.ImageCollection(C.ERA5L_HR).filterDate(t0, t1).size(),
        "ERA5_n": e5.size(),
        "PRECIP_n": pcol.size(),
        # S5P processor version of the day's first orbit (v1.x before mid-2021, v2.x after) - documents
        # the retrieval version behind every row, so version changes can be tested as a covariate
        "S5P_proc": ee.Algorithms.If(s5p_col.size().gt(0),
                                     ee.Image(s5p_col.first()).get("PROCESSOR_VERSION"), "none"),
    })


ALL_DAILY_BANDS = C.DYNAMIC_BANDS + ["MET_fill"]
DAY_PROPS = ["date", "S5P_n_orbits", "S5P_proc", "ERA5L_n", "ERA5_n", "PRECIP_n"]


def daily_collection(start, end):
    """ImageCollection of daily_image for UTC days in [start, end)  (python 'YYYY-MM-DD' strings)."""
    s = dt.date.fromisoformat(start)
    n = (dt.date.fromisoformat(end) - s).days
    s_ee = ee.Date(start)
    return ee.ImageCollection(ee.List.sequence(0, n - 1).map(lambda i: daily_image(s_ee.advance(i, "day"))))


def weekly_image(week_start):
    """Mean of the 7 daily images starting at week_start (an ISO Monday), with counts and totals.
    NO2_trop and the MET bands: mean over days with data.  PRECIP_wk: 7-day total (mm).
    NO2_ndays: number of days with a valid NO2 retrieval in the cell (0-7)."""
    end = (dt.date.fromisoformat(week_start) + dt.timedelta(days=7)).isoformat()
    ic = daily_collection(week_start, end)
    mean_bands = S5P_OUT + MET_OUT + ["PBLH", "MET_fill"]
    img = ee.Image.cat([
        ic.select(mean_bands).mean(),
        ic.select("PRECIP").sum().rename("PRECIP_wk"),
        ic.select("NO2_valid").sum().rename("NO2_ndays"),
    ]).toFloat()
    return img.set({"week_start": week_start})


WEEKLY_BANDS = S5P_OUT + MET_OUT + ["PBLH", "MET_fill", "PRECIP_wk", "NO2_ndays"]


# =========================================================================== STATIC images
def static_fixed():
    """Elevation (m, area mean of SRTM 30 m) and land fraction (0-1, area mean of the Hansen/UMD 30 m
    data mask: 1 = mapped land, 2 = permanent water, 0 = no data / open sea).

    Why not ESA WorldCover 10 m: a 10 m source aggregated to 0.01 deg needs ~12,000 input pixels per
    output cell, so one download tile asks Earth Engine for more than 2^31 pixels and is refused
    ("Number of pixels requested ... exceeds the maximum allowed").  At 30 m the same tile needs
    ~9 x fewer pixels and the land/sea boundary is unchanged at 1 km resolution."""
    dem_src = ee.Image(C.DEM).select("elevation")
    dem = _fine_to_grid(dem_src.toFloat().setDefaultProjection(dem_src.projection()),
                        ee.Reducer.mean(), 4096).rename("DEM")
    dm = ee.Image(C.LANDMASK).select("datamask")
    land30 = dm.eq(1).unmask(0).toFloat().setDefaultProjection(dm.projection())
    land = _fine_to_grid(land30, ee.Reducer.mean(), 4096).rename("LandFrac")
    return ee.Image.cat([dem, land]).toFloat()


def static_ghsl(epoch):
    """GHSL P2023A at one epoch on the 0.01 deg grid.

    BUILT_frac  : built-up fraction (0-1), area mean of built_surface / 1e4 m2
    POP_mean100 : area-weighted MEAN number of persons per 100 m GHSL cell.  Persons per 0.01 deg cell
                  = POP_mean100 x (cell area / 1e4 m2); that multiplication is done in Python
                  (02_static_layers.py) with the exact cell area, because Earth Engine's reduceResolution
                  returned the cell mean even when asked for a sum (verified in 02b: one Riyadh cell 110.5
                  versus 12,511 people summed at native 100 m - a ratio equal to the ~113 GHSL cells per
                  0.01 deg cell).  District totals are additionally checked against native-resolution sums.
    SMOD        : settlement class (mode)
    GHSL marks the sea and no-data with -200; those cells are MASKED before any aggregation (left in,
    they produced 721,429 negative cells in the first download)."""
    b = ee.Image(f"{C.GHSL_BUILT}/{epoch}").select("built_surface")          # m2 of built surface per 100 m cell
    bm = b.updateMask(b.gte(0).And(b.lte(1e4)))
    built = _fine_to_grid(bm.toFloat().divide(1e4).setDefaultProjection(b.projection()),
                          ee.Reducer.mean(), 1024).rename("BUILT_frac")
    p = ee.Image(f"{C.GHSL_POP}/{epoch}").select("population_count")         # persons per 100 m cell
    pm = p.updateMask(p.gte(0))
    pop = _fine_to_grid(pm.toFloat().setDefaultProjection(p.projection()),
                        ee.Reducer.mean(), 1024).rename("POP_mean100")
    sm = ee.Image(f"{C.GHSL_SMOD}/{epoch}").select("smod_code")              # 1 km Mollweide, classes 10-30
    smm = sm.updateMask(sm.gte(10))
    smod = _fine_to_grid(smm.toFloat().setDefaultProjection(sm.projection()),
                         ee.Reducer.mode(), 64).rename("SMOD")
    return ee.Image.cat([built, pop, smod]).toFloat()


def static_annual(year):
    """VIIRS night lights (annual median of monthly avg_rad, cloud-free months only) and MODIS NDVI (annual mean)."""
    return static_period(f"{year}-01-01", f"{year + 1}-01-01")


def static_period(a, b):
    """Same layers as static_annual for any window [a, b) - used for the part-year 2026 sensitivity layers."""
    vcol = ee.ImageCollection(C.VIIRS_MON).filterDate(a, b)
    vproj = ee.Image(vcol.first()).projection()
    viirs = (vcol.map(lambda i: i.select("avg_rad").updateMask(i.select("cf_cvg").gt(0)))
                 .median().setDefaultProjection(vproj))
    viirs = _fine_to_grid(viirs.toFloat(), ee.Reducer.mean(), 256).rename("VIIRS")
    ncol = ee.ImageCollection(C.MODIS_NDVI).filterDate(a, b).select("NDVI")
    nproj = ee.Image(ncol.first()).projection()
    ndvi = ncol.mean().multiply(1e-4).setDefaultProjection(nproj)            # MODIS scale factor
    ndvi = _fine_to_grid(ndvi.toFloat(), ee.Reducer.mean(), 256).rename("NDVI")
    return ee.Image.cat([viirs, ndvi]).toFloat()


# =========================================================================== feature collections
def stations_fc(df):
    feats = [ee.Feature(ee.Geometry.Point([float(r.Lon), float(r.Lat)]),
                        {"Station": str(r.Station), "Network": str(r.Network)}) for r in df.itertuples()]
    return ee.FeatureCollection(feats)


def districts_fc():
    if not C.DISTRICT_ASSET:
        raise SystemExit("DISTRICT_ASSET is empty in dune_config.py - run 00_preflight.py and paste the table id.")
    return ee.FeatureCollection(C.DISTRICT_ASSET).select([C.DISTRICT_ID])


# =========================================================================== table download with retry + split
def _fetch_fc(fc, selectors):
    url = fc.getDownloadURL(filetype="CSV", selectors=selectors)
    import requests
    r = requests.Session().get(url, timeout=(60, C.REQUEST_TIMEOUT_S * 3))          # Session -> honours SSL_MODE
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    return pd.read_csv(io.StringIO(r.content.decode("utf-8")))


def fetch_range(build_fc, start, end, out_csv, selectors, logname):
    """Compute build_fc(start, end) and save it to out_csv.  Retries transient errors with backoff;
    if a range keeps failing (time-out / memory), it is split in two and each half fetched separately.
    A single day that still fails is written to logs/failed_<logname>.csv and skipped."""
    if os.path.exists(out_csv):
        return "skip"
    s, e = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    last = None
    for k in range(C.MAX_RETRIES):
        try:
            df = _fetch_fc(build_fc(start, end), selectors)
            tmp = out_csv + ".part"
            df.to_csv(tmp, index=False)
            os.replace(tmp, out_csv)
            return f"ok {len(df)} rows"
        except Exception as exc:
            last = str(exc)
            transient = any(t in last for t in ["429", "Too many", "503", "500", "Connection", "reset"])
            if not transient and (e - s).days > 1:
                break                                   # deterministic failure -> split
            time.sleep(min(60, 2 ** k + random.random() * 3))
    if (e - s).days > 1:
        mid = (s + (e - s) / 2).isoformat()
        log(f"split {start}..{end} -> halves ({last[:120] if last else ''})", logname)
        root, ext = os.path.splitext(out_csv)
        parts = [f"{root}__a{ext}", f"{root}__b{ext}"]
        r1 = fetch_range(build_fc, start, mid, parts[0], selectors, logname)
        r2 = fetch_range(build_fc, mid, end, parts[1], selectors, logname)
        frames = [pd.read_csv(p) for p in parts if os.path.exists(p)]
        if frames and len(frames) == 2:
            pd.concat(frames, ignore_index=True).to_csv(out_csv, index=False)
            for p in parts:
                os.remove(p)
        return f"split: {r1} | {r2}"
    with _LOCK, open(os.path.join(C.DIRS["logs"], f"failed_{logname}.csv"), "a") as fh:
        fh.write(f"{start},{end},\"{(last or '')[:300]}\"\n")
    return "FAILED"


def date_chunks(start, end, days):
    s, e = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    out = []
    while s < e:
        n = min(s + dt.timedelta(days=days), e)
        out.append((s.isoformat(), n.isoformat()))
        s = n
    return out


def month_chunks(start, end):
    s, e = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    out = []
    while s < e:
        n = dt.date(s.year + (s.month == 12), s.month % 12 + 1, 1)
        out.append((s.isoformat(), min(n, e).isoformat()))
        s = n
    return out


# =========================================================================== raster download (tiled computePixels)
def tile_grid(tile=None, box=None):
    """tiles (r0, c0, h, w) covering the grid, or only the cell box (row0, row1, col0, col1), end-exclusive"""
    t = tile or C.TILE
    R0, R1, C0, C1 = box if box else (0, C.HEIGHT, 0, C.WIDTH)
    tiles = []
    for r0 in range(R0, R1, t):
        for c0 in range(C0, C1, t):
            tiles.append((r0, c0, min(t, R1 - r0), min(t, C1 - c0)))
    return tiles


def lonlat_box_to_cells(lon0, lat0, lon1, lat1):
    """lon/lat box -> (row0, row1, col0, col1) on the analysis grid (end-exclusive, clipped)"""
    c0 = max(int(np.floor((lon0 - C.X0) / C.RES)), 0)
    c1 = min(int(np.ceil((lon1 - C.X0) / C.RES)), C.WIDTH)
    r0 = max(int(np.floor((C.Y1 - lat1) / C.RES)), 0)
    r1 = min(int(np.ceil((C.Y1 - lat0) / C.RES)), C.HEIGHT)
    return r0, r1, c0, c1


def _tile_request(img, bands, r0, c0, h, w):
    grid = {"dimensions": {"width": w, "height": h},
            "affineTransform": {"scaleX": C.RES, "shearX": 0, "translateX": C.X0 + c0 * C.RES,
                                "shearY": 0, "scaleY": -C.RES, "translateY": C.Y1 - r0 * C.RES},
            "crsCode": C.CRS}
    for k in range(C.MAX_RETRIES):
        try:
            arr = ee.data.computePixels({"expression": img, "fileFormat": "NUMPY_NDARRAY",
                                         "grid": grid, "bandIds": bands})
            return np.stack([arr[b].astype("float32") for b in bands])
        except Exception as exc:
            if k == C.MAX_RETRIES - 1:
                raise RuntimeError(f"tile r{r0} c{c0}: {exc}")
            time.sleep(min(60, 2 ** k + random.random() * 3))


def download_raster(img, bands, out_tif, descriptions=None, threads=None, tile=None, box=None):
    """Download img (any projection) on the exact 0.01 deg grid as a float32 GeoTIFF, NaN = no data.
    box = (row0, row1, col0, col1): only those cells are requested; the file still has the full grid
    size and transform (NaN elsewhere, which compresses to almost nothing), so every reader treats it
    exactly like a full-domain raster."""
    if os.path.exists(out_tif):
        return "skip"
    import rasterio
    from rasterio.transform import Affine
    from concurrent.futures import ThreadPoolExecutor
    expr = to_grid(img.select(bands).toFloat()).unmask(C.NODATA)
    data = np.full((len(bands), C.HEIGHT, C.WIDTH), np.nan, dtype="float32")

    def job(t):
        r0, c0, h, w = t
        block = _tile_request(expr, bands, r0, c0, h, w)
        block[block == C.NODATA] = np.nan
        data[:, r0:r0 + h, c0:c0 + w] = block

    with ThreadPoolExecutor(threads or C.N_THREADS) as ex:
        list(ex.map(job, tile_grid(tile, box)))
    prof = dict(driver="GTiff", dtype="float32", count=len(bands), width=C.WIDTH, height=C.HEIGHT,
                crs=C.CRS, transform=Affine(C.RES, 0, C.X0, 0, -C.RES, C.Y1), nodata=np.nan,
                compress="deflate", predictor=3, tiled=True, blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER")
    tmp = out_tif + ".part.tif"
    with rasterio.open(tmp, "w", **prof) as dst:
        dst.write(data)
        for i, b in enumerate(bands, 1):
            dst.set_band_description(i, (descriptions or {}).get(b, b))
    os.replace(tmp, out_tif)
    return "ok"
