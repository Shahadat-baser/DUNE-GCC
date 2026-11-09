"""
DUNE - single configuration file.  Every script imports this; edit values HERE only.

Folder layout created under BASE_DIR (default D:\\, or the DUNE_BASE_DIR environment variable):
    code\\        these scripts (copy the whole DUNE_GEE_v3 folder here)
    GCC_Cities.geojson   the 744 districts (already there)
    inputs\\      stations.csv
    static\\      1 km static rasters (DEM, land fraction, GHSL, VIIRS, NDVI)
    tables\\      station / district tables (daily dynamic, static)
    rasters\\     weekly 1 km predictor rasters
    logs\\        preflight report, run manifests, failed-chunk lists
"""
import os

# ------------------------------------------------------------------ paths
BASE_DIR   = os.environ.get("DUNE_BASE_DIR", "D:\\")          # data root; set DUNE_BASE_DIR to use another folder
OLD_DIR    = os.environ.get("DUNE_OLD_DIR", BASE_DIR)        # folder holding the raw ground files (searched with sub-folders)
EE_PROJECT = os.environ.get("DUNE_EE_PROJECT", "your-earth-engine-project-id")   # your own Google Earth Engine cloud project

# ------------------------------------------------------------------ network / SSL
# University networks often inspect HTTPS with their own certificate, which breaks Python's
# certificate check ("CERTIFICATE_VERIFY_FAILED").  Options:
#   "off"        do not verify certificates (last resort on networks with HTTPS inspection)
#   "truststore" verify against the Windows certificate store, which on a university PC usually
#                already trusts the inspection certificate  (pip install truststore) - safer
#   "default"    normal verification
SSL_MODE = os.environ.get("DUNE_SSL_MODE", "default")

# The 744 districts.  The LOCAL file is the single source of truth; 00b_upload_districts.py copies
# it, unchanged, into the Earth Engine asset below, so the polygons used on the server and the
# polygons used locally are provably the same.  No id needs to be looked up by hand.
GEOJSON        = os.path.join(BASE_DIR, "GCC_Cities.geojson")
ASSET_FOLDER   = f"projects/{EE_PROJECT}/assets/DUNE_v3"
DISTRICT_ASSET = f"{ASSET_FOLDER}/GCC_Districts_744"
DISTRICT_ID    = "OBJECTID"                   # unique district key in the asset and the geojson
DISTRICT_KEEP  = ["OBJECTID", "COUNTRY", "NAME_1", "NAME_2"]   # attributes copied to the asset (if present)

DIRS = {k: os.path.join(BASE_DIR, k) for k in ["code", "inputs", "static", "tables", "rasters", "logs"]}
STATIONS_CSV = os.path.join(DIRS["inputs"], "stations.csv")

# ------------------------------------------------------------------ the 1 km analysis grid
# One fixed grid for every product: EPSG:4326, 0.01 deg cells (1.11 km N-S, ~1.0 km E-W at 25 N).
# Every reduction and every download passes CRS + CRS_TRANSFORM, never 'scale', so all layers are
# pixel-identical and a reviewer can rebuild the grid from these six numbers.
CRS = "EPSG:4326"
RES = 0.01
X0, Y1 = 34.50, 32.15                         # upper-left corner (lon, lat)
X1, Y0 = 59.84, 16.38                         # lower-right corner
WIDTH  = int(round((X1 - X0) / RES))          # 2534
HEIGHT = int(round((Y1 - Y0) / RES))          # 1577
CRS_TRANSFORM = [RES, 0, X0, 0, -RES, Y1]

# ------------------------------------------------------------------ periods
DATE_START   = "2019-01-01"                   # inclusive, UTC
DATE_END     = "2026-10-01"                   # exclusive, UTC (extended 7 Oct 2026 for the 2026 forward test)
LAST_YEAR    = int(DATE_END[:4]) if DATE_END[5:] != "01-01" else int(DATE_END[:4]) - 1   # last year with data
RASTER_YEARS = [2022, 2023, 2024, 2025, 2026] # ISO years written as weekly 1 km rasters (weeks past DATE_END skipped)
STATIC_YEARS = list(range(2019, 2026))        # annual VIIRS / NDVI layers DOWNLOADED (complete years only)
# Years whose annual layers are carried forward from an earlier complete year.  2026 is a forward-test
# year: only information available before it starts is used for slowly varying layers (VIIRS, NDVI,
# GHSL), and a part-year composite would not be comparable with the full-year ones.
STATIC_CARRY_FORWARD = {2026: 2025}
# Sensitivity check of the carry-forward rule (step 02): the same layers built for these part-year
# windows.  2026 Jan-Sep vs 2025 Jan-Sep = real year-to-year change (what carrying 2025 forward ignores);
# 2026 Jan-Sep vs full-year 2025 = what a part-year composite would add on top (seasonal artefact).
STATIC_SENS_WINDOWS = [("2025-01-01", "2025-10-01"), ("2026-01-01", "2026-10-01")]

# Daily tables are named after the period they hold.  The 2019-2025 files written before the extension
# are kept untouched and used by 04b to prove the extension did not change a single earlier value.
STATION_DAILY_CSV  = f"Station_Daily_{DATE_START[:4]}_{LAST_YEAR}.csv"
DISTRICT_DAILY_CSV = f"District_Daily_{DATE_START[:4]}_{LAST_YEAR}.csv"
FROZEN_DAILY = {"station": "Station_Daily_2019_2025.csv", "district": "District_Daily_2019_2025.csv"}
FROZEN_END   = "2026-01-01"   # the frozen tables hold dates before this

# ------------------------------------------------------------------ source collections
S5P_NO2    = "COPERNICUS/S5P/OFFL/L3_NO2"
ERA5L_HR   = "ECMWF/ERA5_LAND/HOURLY"
ERA5_HR    = "ECMWF/ERA5/HOURLY"              # PBLH, and sea/coast fill for ERA5-Land
ERA5L_DAY  = "ECMWF/ERA5_LAND/DAILY_AGGR"     # daily precipitation total
VIIRS_MON  = "NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG"
MODIS_NDVI = "MODIS/061/MOD13A1"
GHSL_BUILT = "JRC/GHSL/P2023A/GHS_BUILT_S"
GHSL_POP   = "JRC/GHSL/P2023A/GHS_POP"
GHSL_SMOD  = "JRC/GHSL/P2023A/GHS_SMOD_V2-0"
GHSL_EPOCHS = [2020, 2025]                    # interpolated to each year later, in Python
DEM        = "USGS/SRTMGL1_003"
LANDMASK   = "UMD/hansen/global_forest_change_2023_v1_11"   # band datamask: 1 land, 2 water, 0 sea/no data

S5P_BANDS = {  # source band -> output name
    "tropospheric_NO2_column_number_density": "NO2_trop",   # mol m-2, negatives KEPT (retrieval noise)
    "cloud_fraction": "S5P_cloud",
    "absorbing_aerosol_index": "S5P_AAI",                   # dust / smoke proxy
}
MET_HOUR_UTC = 10   # ERA5 hour matched to the S5P overpass (13:30 local solar ~ 10:10 UTC at 50 E)
MET_BANDS = {       # ERA5-Land hourly band -> output name (filled over sea/coast from ERA5)
    "temperature_2m": "T2m",
    "dewpoint_temperature_2m": "Td2m",
    "u_component_of_wind_10m": "U10",
    "v_component_of_wind_10m": "V10",
    "surface_pressure": "SP",
}
DYNAMIC_BANDS = list(S5P_BANDS.values()) + list(MET_BANDS.values()) + ["PBLH", "PRECIP", "NO2_valid"]

# ------------------------------------------------------------------ run control
N_THREADS      = 4       # parallel Earth Engine requests (stay well under the per-user limit)
MAX_RETRIES    = 5
REQUEST_TIMEOUT_S = 600  # any single Earth Engine request taking longer is abandoned and retried
STATION_CHUNK_DAYS  = 31
DISTRICT_CHUNK_DAYS = 7
TILE = 512               # computePixels tile edge (pixels); 512 x 512 x 12 bands x 4 B = 12.6 MB < 48 MB
NODATA = -9999.0
TINY_DISTRICT_KM2 = 4.0  # districts smaller than this get a point sample (no 0.01 deg cell centre inside)


# ------------------------------------------------------------------ ground truth (step 06b)
# Raw national files only.  Ground_Truth_Master.csv (old merged file) is NOT used: it mixes monthly
# values stored as daily rows and holds a 1394 ug/m3 value of unknown origin.
# File names are searched in BASE_DIR, OLD_DIR and their sub-folders.
GROUND_FILES = {
    "Oman":           "Oman_Ground_Observations_Master_2023_2024.csv",    # hourly, ug/m3
    "UAE_EAD":        "OpenAQ_v3_GCC_NO2_daily.csv",                      # daily (OpenAQ v3), ug/m3, n_hours
    "UAE_monthly":    "UAE_Ground_Observations_Final.csv",                # monthly + annual, ug/m3
    "KSA_Riyadh":     "Riyadh_Ground_Observations_NO2_SO2.csv",           # daily / monthly / quarterly, ppb
    "Bahrain_SCE":    "02-air-quality-by-month-station_Bahrain.csv",      # monthly, raw SCE download
    "Bahrain_coords": "Bahrain_Ground_NO2_Monthly_2021_2025.csv",         # only used for station coordinates
}
# Bahrain SCE labels its series "Nitrogen Oxides".  Until SCE confirms the values are NO2:
#   "independent"  kept OUT of training and of every NO2 metric; scored separately as a labelled
#                  NOx comparison (correlation / anomaly / trend direction)        <- default
#   "exclude"      dropped completely
#   "as_no2"       treated as NO2 (sensitivity run only; say so in the paper)
BAHRAIN_MODE = "independent"

NO2_PPB_TO_UGM3 = 1.8805      # 46.0055 g/mol * 101325 Pa / (8.314462 * 298.15 K) / 1000  (25 C, 1 atm)
GROUND_MIN_HOURS_DAY  = 18    # valid hours for a daily mean (75 %)
GROUND_MIN_DAYS_WEEK  = 4     # valid days for a weekly mean
GROUND_MIN_FRAC_MONTH = 0.75  # valid-day fraction for a monthly mean built from daily values
GROUND_HOURLY_MAX     = 1000.0  # ug/m3; hourly NO2 above this is an instrument fault
GROUND_SPIKE_MIN      = 300.0   # ug/m3; an hourly value above this AND > 5x its +-12 h median is a spike
GROUND_FLATLINE_H     = 4       # >= this many identical consecutive hourly values -> stuck instrument
GROUND_MONTHLY_MAX    = 300.0   # ug/m3; a monthly mean above this is rejected
OVERPASS_HOURS_LOCAL  = [12, 13, 14]   # hour-beginning local time around the 13:30 S5P overpass
DEDUP_RADIUS_KM  = 5.0        # UAE monthly vs EAD: only pairs closer than this can be the same site
DEDUP_MIN_MONTHS = 4          # overlapping months needed to decide from the data (EAD has only ~5)
GROUND_AGREE_MIN_DAYS = 30    # overlapping days needed to splice two daily series of one site
DEDUP_MIN_R      = 0.80       # same site only if r >= this AND the values are nearly identical:
DEDUP_MAX_MARD   = 0.15       # median |a-b| / b <= 15 %.  (r alone is not enough: every GCC series
                              # shares the winter-high seasonal cycle, so distinct sites also give r > 0.8)
TRAIN_YEARS = [2022, 2023]    # XGBoost training window
TEST_YEARS  = [2024, 2025]    # main hold-out window (LSTM, ground validation)
# Forward test: Abu Dhabi hourly data from the Abu Dhabi SDI, Jan-Sep 2026.  Its ground values are
# written to tables/forward/ ONLY (never into the training / test tables) and their SHA-256 hashes are
# logged, so it can be shown that they were not used for training or tuning and were not changed later.
FORWARD_START, FORWARD_END = "2026-01-01", "2026-10-01"

# Transfer tests - defined BEFORE any model is run.  The frozen 2022-2023 model is applied unchanged;
# these stations / years are never used for training or tuning.  Weekly rasters for each test are
# downloaded only for its box (05b), into their own folder, on the same grid and with the same weekly
# definition as the main rasters.
#   T2  Riyadh 2019-2020: NEW country, inland (~600 m), OLD S5P processor (v1.x)
#   T3  UAE 2019-2021:    SAME country / stations as training, OLD processor until mid-2021, then v2.2-2.3
#       -> T3 isolates the processor + period effect; T2 minus T3 is what is specific to Saudi Arabia.
#   lockdown dates: Saudi curfew 23 Mar - 21 Jun 2020 (first easing 28 May; SPA / MOI announcements);
#   UAE National Disinfection Programme / movement restrictions 26 Mar - 24 Jun 2020 (WAM / NCEMA)
TRANSFER_TESTS = {
    "T2": dict(title="Riyadh 2019-2020", box=(46.40, 24.40, 47.10, 25.00), years=[2019, 2020],
               raster_dir=os.path.join(DIRS["rasters"], "weekly_t2"), networks=["KSA_Riyadh"],
               lockdown=("2020-03-23", "2020-06-21"), easing_from="2020-05-28"),
    "T3": dict(title="UAE 2019-2021 backcast", box=(52.50, 22.90, 56.60, 26.20), years=[2019, 2020, 2021],
               raster_dir=os.path.join(DIRS["rasters"], "weekly_t3"), networks=["UAE_monthly", "UAE_EAD"],
               lockdown=("2020-03-26", "2020-06-24"), easing_from=None),
}
# kept for older scripts
T2_BOX, T2_YEARS = TRANSFER_TESTS["T2"]["box"], TRANSFER_TESTS["T2"]["years"]
T2_RASTER_DIR, T2_LOCKDOWN = TRANSFER_TESTS["T2"]["raster_dir"], TRANSFER_TESTS["T2"]["lockdown"]
T2_EASING_FROM = TRANSFER_TESTS["T2"]["easing_from"]

# Abu Dhabi SDI hourly table (downloaded by 06c)
SDI_DIR         = os.path.join(BASE_DIR, "ground_new")
SDI_TIME_BASIS  = "auto"      # "auto" = decided from the diurnal NO2 minimum; or "local" / "utc"
SDI_MIN_UNIT_RATIO = 0.65     # SDI / OpenAQ median NO2 at the same monitors; ppb instead of ug/m3 would give ~0.53
# Stuck-instrument rule for networks reporting WHOLE numbers (equal consecutive values also occur by
# rounding).  "auto" = the shortest run length L at which rounding explains at most STUCK_MAX_FALSE_SHARE
# of the observed constant runs of length >= L (and of every longer length), estimated per station and
# concentration level with a geometric run-length model (06b, logs/ground_stuck_threshold.csv).
# A number fixes L; STUCK_FALLBACK_H is used if "auto" finds no such L.
GROUND_FLATLINE_H_INTEGER = "auto"
STUCK_MAX_FALSE_SHARE = 0.05
STUCK_FALLBACK_H = 8
STUCK_SENS_H = [4, 12]        # daily means are also rebuilt with these thresholds (sensitivity, report only)


def make_dirs():
    for d in DIRS.values():
        os.makedirs(d, exist_ok=True)
