"""
STEP 5 - weekly 1 km predictor rasters for the ISO years in RASTER_YEARS (default 2022-2025).

These feed the 1 km maps; training and validation use the tables from STEPS 3-4.
One GeoTIFF per ISO week:  rasters/weekly/<ISO year>/DUNE_W<YYYY>-<WW>_<Monday>.tif
Bands (described inside the file): NO2_trop, S5P_cloud, S5P_AAI, T2m, Td2m, U10, V10, SP, PBLH,
MET_fill, PRECIP_wk, NO2_ndays.  float32, NaN = no data, DEFLATE compressed.

Size: about 2534 x 1577 x 12 bands per week, roughly 60-120 MB compressed; 209 weeks for 2022-2025
is roughly 15-25 GB.  The script prints the estimate and the free disk space before it starts.
Resumable: finished weeks are skipped.
"""
import os, sys, shutil, datetime as dt
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
dune_env.ensure("weekly_rasters")      # installs / checks every package this step needs
import dune_config as C
import dune_gee as G
if getattr(G, "PIPELINE_VERSION", None) != "2026-10-08a":
    raise SystemExit(f"{G.__file__} is an older version than this script. Unzip the WHOLE DUNE_GEE_v3.zip "
                     "again into the code folder and choose 'Replace the files in the destination'.")

OUT = os.path.join(C.DIRS["rasters"], "weekly")


def iso_weeks(years):
    weeks = []
    for y in years:
        d = dt.date.fromisocalendar(y, 1, 1)
        while d.isocalendar()[0] == y:
            weeks.append((y, d.isocalendar()[1], d))
            d += dt.timedelta(days=7)
    return weeks


def main():
    G.init()
    end = dt.date.fromisoformat(C.DATE_END)
    weeks = [w for w in iso_weeks(C.RASTER_YEARS) if w[2] + dt.timedelta(days=7) <= end]
    todo = [w for w in weeks if not os.path.exists(os.path.join(OUT, str(w[0]), f"DUNE_W{w[0]}-{w[1]:02d}_{w[2].isoformat()}.tif"))]
    free = shutil.disk_usage(C.BASE_DIR).free / 1e9
    G.log(f"weekly rasters: {len(weeks)} complete ISO weeks ({C.RASTER_YEARS}, ending before {C.DATE_END}); "
          f"{len(todo)} still to download, estimate {len(todo) * 0.09:.1f} GB (+-50%) | free on disk {free:.0f} GB",
          "weekly_rasters")
    if free < len(todo) * 0.15:
        raise SystemExit("not enough free disk space for the safe estimate - free space or reduce RASTER_YEARS")
    failed = []
    for y, w, monday in weeks:
        if monday + dt.timedelta(days=7) > dt.date.fromisoformat(C.DATE_END):
            G.log(f"W{y}-{w:02d} runs past DATE_END, skipped", "weekly_rasters")
            continue
        folder = os.path.join(OUT, str(y))
        os.makedirs(folder, exist_ok=True)
        f = os.path.join(folder, f"DUNE_W{y}-{w:02d}_{monday.isoformat()}.tif")
        try:
            r = G.download_raster(G.weekly_image(monday.isoformat()), G.WEEKLY_BANDS, f)
        except Exception as exc:
            r = f"FAILED {str(exc)[:200]}"
            failed.append(f"W{y}-{w:02d}")
        G.log(f"W{y}-{w:02d} ({monday}): {r}", "weekly_rasters")
    done = sum(len(os.listdir(os.path.join(OUT, str(y)))) for y in C.RASTER_YEARS if os.path.isdir(os.path.join(OUT, str(y))))
    G.log(f"finished: {done} weekly files on disk | failed this run: {failed or 'none'}"
          + (" -> rerun this step; finished weeks are skipped" if failed else ""), "weekly_rasters")
    G.write_manifest("weekly_rasters", {"weeks": len(weeks), "failed": failed})


if __name__ == "__main__":
    main()
