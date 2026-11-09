"""
STEP 5b - weekly rasters for the transfer tests in dune_config.TRANSFER_TESTS, Earth Engine.
    T2  Riyadh 2019-2020       small box, ~15-30 min, ~25 MB  (already done if you ran the old 05b: skipped)
    T3  UAE 2019-2021 backcast UAE box, ~2-4 h, ~1-2 GB

Same weekly_image() and same 0.01 deg grid as step 05, but only the cells of each test's box are
requested.  Files keep the full grid size and transform (NaN outside the box, compressed to almost
nothing), so step 07a samples them with exactly the same code as the main rasters.  Each test has
its own folder, so they never mix with the main 2022-2026 rasters.  Resumable: finished weeks are
skipped.  Run a single test with:  %run 05b_weekly_rasters_tests.py T3
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
dune_env.ensure("weekly_rasters")
import datetime as dt
import pandas as pd
import dune_config as C
import dune_gee as G
if getattr(G, "PIPELINE_VERSION", None) != "2026-10-08a":
    raise SystemExit(f"{G.__file__} is an older version than this script. Unzip the WHOLE DUNE_GEE_v3.zip "
                     "again into the code folder and choose 'Replace the files in the destination'.")


def run_test(key, t):
    box = G.lonlat_box_to_cells(*t["box"])
    st = pd.read_csv(C.STATIONS_CSV)
    pts = st[st.Network.isin(t["networks"])]
    inside = pts.Row.between(box[0] + 2, box[1] - 3) & pts.Col.between(box[2] + 2, box[3] - 3)
    if not inside.all():
        raise SystemExit(f"{key}: monitors outside the box (with the 5x5 margin): {pts[~inside].Station.tolist()}")
    tag = f"weekly_rasters_{key.lower()}"
    G.log(f"{key} {t['title']}: box {t['box']} -> rows {box[0]}-{box[1]}, cols {box[2]}-{box[3]} "
          f"({(box[1] - box[0]) * (box[3] - box[2]):,} cells); all {len(pts)} monitors inside; ISO years {t['years']}", tag)
    failed, n = [], 0
    for y in t["years"]:
        d = dt.date.fromisocalendar(y, 1, 1)
        folder = os.path.join(t["raster_dir"], str(y))
        os.makedirs(folder, exist_ok=True)
        while d.isocalendar()[0] == y:
            w = d.isocalendar()[1]
            f = os.path.join(folder, f"DUNE_W{y}-{w:02d}_{d.isoformat()}.tif")
            try:
                r = G.download_raster(G.weekly_image(d.isoformat()), G.WEEKLY_BANDS, f, box=box)
                n += 1
            except Exception as exc:
                r = f"FAILED {str(exc)[:200]}"
                failed.append(f"W{y}-{w:02d}")
            G.log(f"{key} W{y}-{w:02d} ({d}): {r}", tag)
            d += dt.timedelta(days=7)
    G.log(f"{key} finished: {n} weekly files | failed: {failed or 'none'}"
          + (" -> rerun; finished weeks are skipped" if failed else ""), tag)


def main():
    G.init()
    keys = [a for a in sys.argv[1:] if a in C.TRANSFER_TESTS] or list(C.TRANSFER_TESTS)
    for k in keys:
        run_test(k, C.TRANSFER_TESTS[k])


if __name__ == "__main__":
    main()
