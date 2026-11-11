# DUNE — 1 km surface NO2 benchmark for the GCC (0.01° grid, 2019–2026)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24145692.svg)](https://doi.org/10.5281/zenodo.24145692)

Code for: S. Baser, B. S. Tawabini, M. Bilal and A. Koeshidayatullah, "DUNE: A Reproducible Two-Stage Benchmark for 1 km Surface NO2 Over the Gulf Cooperation Council," submitted to *IEEE Transactions on Geoscience and Remote Sensing*. Cite the archived release (see `CITATION.cff`).

This pipeline builds every predictor of the DUNE paper from Google Earth Engine and writes it straight to a local disk. It uses no Google Drive and no manual file handling. All raster, district and station products share one fixed grid and one set of image definitions (`dune_gee.py`). That way a reviewer can read the whole method in one file and regenerate every number.

## 1. Setup (once)

Use a Python 3.10+ environment (e.g. VS Code with a conda environment named `DUNE_GCC`). **No manual installation is needed.** Each step starts with `dune_env.ensure("<step>")`, which checks the packages that step needs and pip-installs any missing ones into the running Python, using pinned versions. Each package is test-imported in a separate process, so a library blocked by Windows Application Control is caught (for example, pandas 3.x DLLs blocked on a managed Windows PC) and an older pin is tried instead. Pins: earthengine-api 1.7.46, numpy 2.2.6, pandas 2.3.3, shapely 2.1.2, pyproj 3.7.2, pyogrio 0.11.1, geopandas 1.1.1, rasterio 1.4.4, scipy 1.15.3. A constraints file stops any dependency from pulling pandas 3 or numpy ≥ 2.3 back in.

(`environment.yml` builds the same environment from scratch: `conda env create -f environment.yml`.)

**Paths.** By default the data root is `D:\` (scripts in `D:\code\`, districts in `D:\GCC_Cities.geojson`). To use another folder, set the environment variable `DUNE_BASE_DIR` (and `DUNE_OLD_DIR` if the raw ground files are elsewhere). Set `DUNE_EE_PROJECT` to your own Google Earth Engine cloud project; authentication is per user and nothing is stored in the code.

**Networks with HTTPS inspection.** If downloads fail with `CERTIFICATE_VERIFY_FAILED`, set `DUNE_SSL_MODE=truststore` (uses the Windows certificate store) or, as a last resort, `DUNE_SSL_MODE=off`. The default is normal certificate verification. Credentials are never stored in the code.

Run each step from a terminal (`python 00_preflight.py`) or from a notebook opened in the `code` folder (`%run 00_preflight.py`). Every step can be resumed: if you rerun it, finished pieces are skipped.

## 2. Run order

| Step | Script | Runs on | Time (approx.) | Output |
|---|---|---|---|---|
| 0 | `00_preflight.py` | EE, metadata only | 2–5 min | `logs/preflight_report.txt` |
| 0b | `00b_upload_districts.py` | EE | 5–20 min | the table `DUNE_v3/GCC_Districts_744` in your EE assets, checked against the local geojson |
| 1 | `01_build_inputs.py` | local | seconds | `inputs/stations.csv` |
| 2 | `02_static_layers.py` | EE + local | 15–40 min | `static/*.tif`, `tables/*_Static_by_Year.csv` |
| 3 | `03_station_daily.py` | EE | 1–3 h | `tables/Station_Daily_2019_2025.csv` |
| 4 | `04_district_daily.py` | EE | 3–8 h | `tables/District_Daily_2019_2025.csv` |
| 5 | `05_weekly_rasters.py` | EE | 6–15 h, 15–25 GB | `rasters/weekly/<year>/DUNE_W<yyyy>-<ww>_<monday>.tif` |
| 4b | `04b_check_district_daily.py` | local | 1 min | `logs/daily_tables_report.txt` |
| 6a | `06a_profile_ground_files.py` | local | seconds | `logs/ground_profile.txt` (what each ground file contains) |
| 6c | `06c_download_ead_sdi.py` | internet | minutes | `ground_new/EAD_SDI_*.csv`, `logs/ead_sdi_report.txt` (raw EAD hourly table from the Abu Dhabi SDI, append-only archive) |
| 6b | `06b_ground_truth.py` | local | ~1 min | `tables/ground_sites.csv`, `ground_daily.csv`, `ground_weekly.csv`, `ground_monthly.csv`, `logs/ground_truth_report.txt` |
| 5b | `05b_weekly_rasters_tests.py` | EE | T2 15–30 min (~25 MB); T3 2–4 h (~1–2 GB) | `rasters/weekly_t2/`, `rasters/weekly_t3/` (transfer-test boxes only, same grid and weekly definition) |
| 7a | `07a_build_features.py` | local | 10–30 min | `features/station_*_features.csv`, `model/weekly_table.csv`, `model/monthly_table.csv`, `logs/features_report.txt` |
| 7b | `07b_stage1.py` (v2) | local | 45–90 min (CPU) | `model/v2/` (frozen model + meta with SHA-256, out-of-fold and station predictions, metrics), `logs/stage1_report.txt`. v1 (`07b_xgboost_v1.py`) is kept unchanged and reported in the supplement |
| 7b2 | `07b2_diagnostics.py` | local | 2–4 min | `logs/stage1_diagnostics.txt`, `model/v2/diag_*.csv`: independent-station error split into level vs pattern, by area / distance to training / coordinate precision; agency-vs-EAD agreement ceiling; v1 vs v2 on identical rows; X2 and H refitted (diagnostic only, nothing frozen is changed) |
| 7c | `07c_maps.py` | local | 1–3 h (resumable) | `maps/weekly/<year>/*.tif` (NO2, DI, AOA), `maps/monthly/*.tif` (NO2, days covered, DI mean, AOA share), `tables/District_NO2_weekly.csv`, `tables/District_NO2_monthly.csv`, `maps/README_maps.txt`, `logs/maps_report.txt`; about 4–6 GB |
| 7d | `07d_lstm.py` | local | 0.5–2 h (installs PyTorch CPU once) | LSTM-E (estimation with the satellite; weekly sequences of 4/8/13 weeks vs DAILY sequences of 28/56 days = temporal densification, chosen on 2023 out-of-site RMSE) and LSTM-P (projection without satellite data after 2023; 4/8/13/26 weeks of ERA5 weather + seasonal level, fixed 2022–23 or rolling previous two years); 2024 at the stations weekly and monthly, month by month, month-wise change vs 2023; 2025 district projection; Riyadh applicability. `model/v2/lstm_*.pt`, `stage2_meta.json`, `tables/District_NO2_*_LSTMP.csv`, `logs/stage2_report.txt` |
| 7e | `07e_forward_check.py` | local | < 1 min | the sealed 2026 forward test, once: verifies the frozen-model and forward-file hashes, scores Jan–Sep 2026 month by month, appends to `logs/forward_scored.txt`; `logs/forward_report.txt` |
| 7f | `07f_forward_diagnostics.py` | local | 2–5 min | POST-HOC explanation of the 2026 result, nothing refitted: station composition and a fixed panel; observed vs satellite-column vs X1 change 2026 vs 2024 per station; GCC-wide column step at the TROPOMI 02.09.01 switch (difference-in-differences with a placebo year); errors inside/outside the AOA; bias-free skill. `logs/forward_diagnostics.txt` |
| 7g | `07g_sensitivity.py` | local | 20–60 min | sensitivity runs; 06b re-run into `sensitivity/<variant>/` (control, daily ≥ 12 h, SDI stuck 4 h / 12 h) with X1 refitted at the frozen settings; X2; 2026 static layers from Jan–Sep 2026; processor ≥ 2.4; Bahrain as NO2. Frozen files are never touched. `logs/sensitivity_report.txt` |
| 7h | `07h_gru_sensitivity.py` | local | as 07d (×2 with `--control`) | optional, post hoc: 07d re-run with a GRU cell instead of the LSTM (all else identical), outputs only in `sensitivity/gru/`; frozen files hashed before/after; paired site-bootstrap GRU − LSTM differences and a pre-declared verdict. `logs/gru_sensitivity_report.txt`, `sensitivity/gru/gru_vs_lstm.csv` (Fig. S1, Table S10) |
| 8a | `08_figures.py` | local | 2–5 min | all manuscript figures at 1080 dpi (TIFF with LZW + PNG), Times New Roman, no titles inside figures; `figures/Fig01…Fig08`, `logs/figures_report.txt` |
| 8b | `08_tables.py` | local | 2–5 min | Tables I–VI as CSV (full precision) and `tables_paper/DUNE_tables.docx` (IEEE style); needs the CSVs written by 7g; `logs/tables_report.txt` |
| 8c | `08c_reviewer_diagnostics.py` | local | 5–15 min | post hoc error diagnostics of the frozen X1: residuals by season, PBLH regime, dust (aerosol index) and network; point-to-pixel representativeness from the weekly maps; accumulated local effects (`figures/FigS2_ale`); propagation of column and resampling uncertainty; random-forest baseline. `logs/reviewer_diagnostics.txt`, `logs/rd_*.csv` (Tables S11–S14 via 08b) |

Steps 3, 4 and 5 are independent once step 2 is done, so they can run on separate days.

### Extension to September 2026 (forward test) - run once, in this order

The period was extended from 2019–2025 to **2019-01-01 → 2026-09-30** so the models can be checked against Abu Dhabi data from 2026. The main hold-out period stays **2024–2025**. Nothing that was already downloaded is fetched or rewritten again:

1. `00_preflight.py`: confirms every source is complete up to 30 Sep 2026.
2. `06c_download_ead_sdi.py`: refreshes the archive of the Abu Dhabi feed.
3. `01_build_inputs.py`: adds the SDI monitors that are not yet known (Khalifa School). It **stops without writing** if any existing point would change position, cell or `Site_ID`.
4. `02_static_layers.py`: no new downloads. 2026 rows reuse the 2025 annual layers (`Static_source_year`). It stops if any earlier row would change.
5. `03_station_daily.py`: downloads the 2026 months, plus small add-on files that hold only the new point for the months already finished.
6. `04_district_daily.py`: downloads the 2026 weeks. It re-fetches the final 3 days of 2025 because the chunk boundary moved, and it reports any difference between the two fetches.
7. `05_weekly_rasters.py`: downloads ISO weeks 2026-01 to 2026-39.
8. `04b_check_district_daily.py`: proves that every row of the frozen `*_2019_2025.csv` tables is present and **identical** in the extended `*_2019_2026.csv` tables (row hashes), and counts the 2026 days, points and rasters.
9. `06b_ground_truth.py`: assembles the ground truth. Forward-test rows go only to `tables/forward/`, and their hashes go to `logs/forward_test_hashes.txt`.

Run `06c` once a month. The SDI service keeps only about 11 months of data, so the local archive (append-only) is the only lasting copy.

## 3. Variables

| Band | Source | Native grid | To 0.01° | Units / notes |
|---|---|---|---|---|
| NO2_trop | S5P OFFL L3 NO2 (`tropospheric_NO2_column_number_density`) | L3 at nominal 1113 m (L2 pixel 3.5 × 5.5 km) | nearest | mol m⁻²; daily mean of orbits; L3 already filters qa_value > 0.75; negatives kept |
| NO2_valid | derived | — | — | 1 if observed. Its district mean is the observed fraction. |
| S5P_cloud, S5P_AAI | S5P L3 | as above | nearest | cloud fraction; absorbing aerosol index (dust proxy) |
| T2m, Td2m, U10, V10, SP | ERA5-Land hourly at 10 UTC | 0.1° | bilinear | K, K, m s⁻¹, m s⁻¹, Pa. Sea/coast gaps filled from ERA5 (0.25°). |
| MET_fill | derived | — | — | 1 where the ERA5 fill was used (coastal and island cells) |
| PBLH | ERA5 hourly at 10 UTC | 0.25° | bilinear | m |
| PRECIP | ERA5-Land daily aggregate | 0.1° | bilinear | mm day⁻¹ (`total_precipitation_sum`, not the hourly accumulation) |
| DEM | SRTM 30 m | 30 m | area mean | m |
| LandFrac | Hansen/UMD Global Forest Change data mask (v1.11) | 30 m | area mean | 0–1 (share of the cell that is land; 10 m WorldCover exceeds Earth Engine's 2^31-pixel request limit) |
| CoastDist_km | derived locally from LandFrac | — | — | km to the nearest cell that is mostly water |
| BUILT_frac, POP, SMOD | GHSL P2023A, 2020 and 2025 epochs | 100 m / 1 km | area mean / area-weighted sum / mode | fraction, persons per cell, settlement class. Interpolated to each year. |
| VIIRS | VIIRS DNB monthly (VCMSLCFG) | 15″ | area mean of annual median | nW cm⁻² sr⁻¹; only months with cloud-free coverage |
| NDVI | MODIS MOD13A1 | 500 m | area mean of annual mean | scaled by 1e-4 |

## 4. Design decisions a reviewer will check

1. **One grid.** The grid is EPSG:4326, 0.01° cells, upper-left corner (34.50 E, 32.15 N), 2534 × 1577 cells. Every request passes `crs` and `crsTransform`; none uses `scale`. Cells are about 1.11 km north–south and about 1.0 km east–west at 25° N.
2. **The 1 km grid harmonises; it does not downscale.** S5P's L2 footprint is 3.5 × 5.5 km. Placing it on a 0.01° grid adds no information below that footprint. Sub-footprint structure comes only from the fine predictors (GHSL, VIIRS, NDVI).
3. **Dates are UTC days written by Python as `YYYY-MM-DD`.** The original MET file names used Earth Engine's `YYYYMMDD`, where `DD` means day of year. That paired only January days with the satellite data.
4. **Overpass-matched meteorology.** ERA5 and ERA5-Land are taken at 10 UTC. That is about 13:30 local solar time at 50° E, the S5P overpass.
5. **Coastal meteorology is not dropped.** ERA5-Land is land-only, so island and coastal cells (Bahrain, the UAE coast) are filled from ERA5 and flagged in `MET_fill`. The flag supports the coastal-bias analysis.
6. **Aggregation runs once, in Python.** Daily values come from Earth Engine. Weekly and monthly series are built locally with the same code for stations and districts. Rasters use the same daily definition (`weekly_image`).
7. **No leakage between networks.** Monitors closer than 1 km share a `Site_ID` in `stations.csv`, for example a UAE site reported both monthly and through OpenAQ. Cross-validation must group by `Site_ID`.
8. **One set of polygons.** `00b` uploads the local geojson to Earth Engine unchanged (planar edges) and verifies the feature count, ids and total area, so server-side and local district means use identical boundaries.
9. **Small districts.** Districts with no cell centre inside them get a point sample (`Kind = district_point`), and the tables flag this.
10. **Ground truth from raw national files only** (step 6b). Units are converted explicitly: Riyadh ppb × 1.8805 (25 °C, 1 atm). Hourly Oman data are quality-controlled (negative, > 1000 µg m⁻³, stuck ≥ 4 h, spikes). A daily mean needs ≥ 18 valid hours, a weekly mean ≥ 4 valid days, and a monthly mean built from daily values ≥ 75 % of days. Every rule is set in `dune_config.py`, and every removal is counted in the report.
11. **Duplicate sites are resolved by data, not names.** UAE monthly coordinates are rounded, so a UAE monthly series joins an EAD monitor only if the two are < 5 km apart **and** their overlapping months agree (r ≥ 0.8 and median relative difference ≤ 15 %). Correlation alone is not enough, because every GCC series shares the same seasonal cycle. Matching series share a `Site_ID`, which groups the cross-validation folds. Grouping leans towards merging, so it can only make validation stricter. Values are spliced into one target series (`Target_ID`) only when the series' overlapping data agree, and splicing never chains through a third series. Series that disagree (for example a street and a school monitor in the same cell, or two agencies reporting different levels for one site) stay as separate targets (`S019a`, `S019b`, ...) under the same `Site_ID`. Every decision is logged in `logs/ground_uae_dedup.csv` and `logs/ground_site_agreement.csv`.
12. **OpenAQ date fix.** The OpenAQ v3 harvest stored the UTC start of each local day, which is one day early for the UAE (UTC+4). Step 6b adds one day. Evidence: Al Ain Street's first hourly value is 2022-10-28T16:00Z, but its first daily row is dated 2022-10-27.
13. **Bahrain.** SCE reports the series as "Nitrogen Oxides". With `BAHRAIN_MODE = "independent"` (the default), it never enters training or the NO₂ metrics; it is scored separately as a labelled NOx comparison. Three of the six stations have no published coordinates and are not used.
14. **Forward test, sealed.** Abu Dhabi hourly data for January–September 2026 (SDI feed) are kept apart from the train and test tables (`tables/forward/`) and hashed (SHA-256) every time they are written. Slowly varying layers for 2026 are carried forward from 2025, so the forward year uses only information that existed before it started.
15. **Two choices checked against the data rather than asserted.**
    (a) *Carrying the 2025 annual layers forward:* step 2 also builds night lights and NDVI for January–September 2025 and January–September 2026. `logs/static_carry_forward_check.txt` compares (i) part-year 2026 with the carried-forward full-year 2025 layer, and (ii) part-year 2026 with part-year 2025 (same months, so this is the real year-to-year change). The modelling step re-scores the forward test with the part-year layers (`*_Static_partial_windows.csv`).
    (b) *Stuck-instrument threshold for whole-number data (Abu Dhabi feed):* each station's constant runs are compared with the number that rounding alone would produce (geometric run-length model, estimated per station and concentration class). The threshold is the shortest run length at which rounding explains at most 5 % of the observed runs, at that length and every longer one. The table is in `logs/ground_stuck_threshold.csv`, and the report shows daily means rebuilt with 4 h and 12 h thresholds as a sensitivity check. Oman (decimal data) gets the same analysis for reference and keeps the fixed 4 h rule.
16. **Transfer tests T2 and T3 (defined before modelling, `TRANSFER_TESTS` in `dune_config.py`).** The frozen 2022–2023 models are applied unchanged; neither test is used for training or tuning. Weekly rasters cover only each test's box (`05b`), on the same grid and with the same weekly definition.
    - **T2, Riyadh 2019–2020:** a new country, an inland environment at about 600 m elevation, and the old S5P processor (v1.x).
    - **T3, UAE 2019–2021 backcast:** the same stations as training, with processor v1.x until mid-2021 and v2.2–2.3 after. T3 is monthly only, because daily UAE data start in October 2022. Because the stations are the same, T3 isolates the effect of the processor and the period. Comparing T2 with T3 then shows what is specific to Saudi Arabia.

    Each test is scored overall, by processor, by year, before, during and after the 2020 COVID restrictions (Saudi curfew 23 Mar–21 Jun 2020, easing from 28 May; UAE movement restrictions 26 Mar–24 Jun 2020), and against the 2022–2023 site-month climatology. Each score has 95 % site-bootstrap intervals.
17. **Stage 1 v2 (column to surface).** v1 (XGBoost on all features) generalised to a new year at known stations but poorly to new stations, because static layers acted as station fingerprints. v2 compares five variants fixed before the run:
    - **E1, physical equation:** ln C = a + b ln VCD − c ln PBLH (mixed-layer physics: b = c = 1).
    - **E2:** E1 + wind speed, temperature and season.
    - **X1:** v1 XGBoost (reference).
    - **X2:** XGBoost on reduced, non-fingerprint features, monotone in the column.
    - **H:** E2 plus a monotone XGBoost correction.

    The winner is chosen only by scheme-A RMSE (leave one site out, 2022–2023). It is then tested on: new station and new year (scheme E); 2024; and 30+ independent UAE agency stations never used for training (monthly averages of the weekly predictions); with T3 and T2 as supplementary tests. Equation coefficients are reported with site-bootstrap intervals.
18. **Extension integrity.** Extending the period never rewrites earlier data. `01`, `02` and `04b` stop or report if any value from before the extension would change, and `03` and `04` check that rows fetched twice are identical.
19. **Audit trail.** Every run writes `logs/manifest_<step>_<time>.json` (collections, grid, library versions). Every failed chunk goes to `logs/failed_<step>.csv`.

20. **Training data, stated exactly.** The frozen Stage 1 model is trained on weekly means from the two networks that publish daily data: Abu Dhabi EAD (UAE, 18 sites) and Oman (5 sites), 549 station-weeks. The training window is 2022–2023, but the daily ground data start in 2023, so all training weeks are in 2023. No other GCC country contributes training data; Saudi Arabia (Riyadh) is a transfer test only, and Bahrain is a labelled NOx comparison.
21. **Maps are the validated model, cell for cell (`07c`).** Grid features are built with the same functions as the station features (same windows, same dtype, same numpy call; static layers from `02_static_layers.layers_for_year`). The report compares the map value at every station cell with the station prediction of `07b_stage1`; the expected difference is zero. A cell gets no value in a week without a valid 3×3 column or PBLH (the rule that removed rows from training); there is no gap filling. Monthly maps are day-weighted means of the weekly maps, kept where ≥ 90 % of days are covered, the same rule as the monthly validation.
22. **Area of applicability.** Every weekly map carries the dissimilarity index of Meyer & Pebesma (2021): SHAP-weighted, standardised feature distance to the nearest training station-week, divided by the mean training distance. The threshold is the upper whisker of the training DI computed leave-one-site-out, the same grouping as scheme A. `07c` tests whether errors are larger outside the AOA, both at the scheme-A rows and at the independent agency stations.
23. **Effective resolution.** The 0.01° grid is the reporting grid. The satellite signal resolves about 3.5 × 5.5 km (3×3 and 5×5 means about 3 and 5 km), meteorology about 9–11 km; variation below about 5 km comes only from the 1 km proxies. This is written into every map's metadata.

24. **LSTM (`07d`, PyTorch CPU, deterministic), every candidate fixed before the run.** LSTM-E asks whether the recent history of column and weather improves on X1's one-week estimate. Two input designs compete: weekly sequences (4, 8, 13 weeks) and daily sequences (28, 56 days) trained on the daily ground values, which multiplies the training samples about seven-fold and lets the sequence bridge days without a satellite retrieval (temporal densification). The design is chosen only by its out-of-site RMSE on the 2023 rows (5 grouped site folds); the winner is then scored leave one site out, in 2024, at the independent stations and in Riyadh, always on identical rows to X1. The 1 km product stays X1, as declared before the run. LSTM-P asks how well weekly NO2 can be projected with no satellite or ground data after 2023: it learns from the 2022–2023 Stage 1 estimates (744 districts and every station cell) how weather moves NO2 around each place's seasonal level; the sequence length (4, 8, 13, 26 weeks) is chosen on held-out series in 2022–2023. For the projected years the seasonal level is either fixed (2022–23) or rolling (the two previous years' Stage 1 estimates: 2025 from 2023–24, 2026 from 2024–25); the network is the same. Baselines: Stage 1 climatology, ground climatology, seasonal-naive / persistence; at monitored stations also site-calibrated models (one 2023 ratio from leave-one-site-out predictions), for monitored stations only. 2025 has no NO2 ground data, so 2025 is reported as LSTM-P against Stage 1.
25. **Forward test, opened once (`07e`).** The 2026 ground rows were hashed by `06b` before any model existed. `07e` refuses to run if a frozen model or a forward file changed, and logs every scoring with the model hashes; a second scoring with different models is flagged for disclosure.
26. **After the forward test (`07f`, `07g`).** Both steps only read the frozen models (they stop if a frozen file changed) and label every 2026 number as post-hoc. Sensitivity variants write only to `sensitivity/<variant>/`; a control variant re-runs 06b with the main rules and must reproduce the main tables; refits use the frozen X1 settings (no re-tuning), except the pre-declared alternative X2, tuned as in `07b_stage1`.
27. **Recurrent cell (`07h`).** A GRU replaces the LSTM cell with every other choice unchanged; `--control` re-runs the LSTM through the same harness and must reproduce the frozen predictions exactly. Pre-declared rule: the GRU counts as better only with >= +0.05 R2 in both 2024 tests at the training-network sites (paired site-bootstrap interval excluding 0). Result in the paper: not met.

## 5. Not in this release

- No API keys or credentials are stored in any script. Earth Engine authentication is per user.
- Ground observations are not downloaded here; steps 1 and 6b read the existing national files.

- Ground data from the agencies, OpenAQ and the Abu Dhabi SDI service are not redistributed; each user downloads them under the provider's terms. Station data collected through WAQI (aqicn.org) were used only for a private consistency check and are not part of this repository (WAQI terms forbid redistribution of cached data).

## 6. Release, licence and citation

- Code: MIT licence (`LICENSE`). Derived 1 km maps and district tables, when deposited, are released separately under CC BY 4.0 with their own DOI.
- `CITATION.cff` and `.zenodo.json` give the citation metadata that GitHub and Zenodo read automatically.
- `.gitignore` keeps data, rasters, model outputs, logs and credentials out of the repository.
- Reproduce the paper: steps 0 → 8b in the order of Section 2. Every frozen file carries a SHA-256 hash in `model/v2/*meta.json`; `07e` refuses to score if a hash changed.
