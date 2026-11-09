"""
STEP 6c - download the Abu Dhabi (EAD) hourly air-quality table from the Abu Dhabi Spatial Data
Infrastructure (public ArcGIS REST service), local + internet, minutes.

Why: EAD data reached us only through OpenAQ, which stops on 20 May 2024 and has many incomplete days.
The SDI service publishes the EAD stations and an hourly table (PM10, PM2.5, O3, SO2, NO2, CO).  Its
period is unknown until downloaded - a first look showed records from 2025, which is exactly the
year with no NO2 ground truth.  Nothing is merged here: this step only downloads, saves the raw
table unchanged and reports what it contains (period, stations, units, time stamps).  Merging into
06b follows once the content is known.

    service  https://arcgis.sdi.abudhabi.ae/agspublish/rest/services/ADPHC/AirQuality/MapServer
             layer 0 = AirQualityMonitoringStations (points), table 1 = AirQualityHourlyData

Outputs (BASE_DIR\\ground_new\\):
    EAD_SDI_stations.csv        station layer with lon/lat (EPSG:4326)
    EAD_SDI_hourly_raw.csv      every record, all fields, date fields also as ISO text (UTC as served)
    EAD_SDI_metadata.json       field list and service limits, as served
logs\\ead_sdi_report.txt

Resumable: if it stops, run it again - it continues after the last OBJECTID already saved.
"""
import os, sys, json, time, datetime as dt
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
dune_env.ensure("ground_download")
import numpy as np, pandas as pd, requests
import dune_config as C

SERVICE = "https://arcgis.sdi.abudhabi.ae/agspublish/rest/services/ADPHC/AirQuality/MapServer"
OUT_DIR = os.path.join(C.BASE_DIR, "ground_new")
RAW = os.path.join(OUT_DIR, "EAD_SDI_hourly_raw.csv")
PAGE = 1000          # lowered automatically if the server times out
TIMEOUT = 180
REPORT = []


def say(s=""):
    print(s)
    REPORT.append(str(s))


def session():
    s = requests.Session()
    if C.SSL_MODE == "off":
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        s.verify = False
    elif C.SSL_MODE == "truststore":
        import truststore
        truststore.inject_into_ssl()
    return s


def get(s, url, params, tries=C.MAX_RETRIES):
    for k in range(tries):
        try:
            r = s.get(url, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            js = r.json()
            if "error" in js:
                raise RuntimeError(js["error"])
            return js
        except Exception as exc:
            wait = 10 * (k + 1)
            print(f"   [retry {k + 1}/{tries}] {type(exc).__name__}: {str(exc)[:150]} - waiting {wait} s")
            time.sleep(wait)
    return None


def esri_dates(df, fields):
    for f in fields:
        if f in df:
            df[f + "_iso"] = pd.to_datetime(pd.to_numeric(df[f], errors="coerce"), unit="ms", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")
    return df


def main():
    global PAGE
    C.make_dirs()
    os.makedirs(OUT_DIR, exist_ok=True)
    s = session()
    say(f"=== EAD hourly table from the Abu Dhabi SDI  ({dt.datetime.utcnow():%Y-%m-%d %H:%M} UTC)")

    meta = {"service": SERVICE, "downloaded_utc": dt.datetime.utcnow().isoformat(timespec="seconds")}
    for lid in (0, 1):
        js = get(s, f"{SERVICE}/{lid}", {"f": "json"})
        if js is None:
            raise SystemExit(f"cannot reach {SERVICE}/{lid} - check the network / SSL_MODE")
        meta[f"layer_{lid}"] = {k: js.get(k) for k in ("name", "type", "description", "maxRecordCount", "fields",
                                                        "supportsPagination", "advancedQueryCapabilities")}
    with open(os.path.join(OUT_DIR, "EAD_SDI_metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1, default=str)
    flds = meta["layer_1"]["fields"] or []
    date_fields = [f["name"] for f in flds if f.get("type") == "esriFieldTypeDate"]
    say(f"   table fields: {', '.join(f['name'] + ':' + f.get('type', '').replace('esriFieldType', '') for f in flds)}")
    mrc = meta["layer_1"].get("maxRecordCount") or PAGE
    PAGE = min(PAGE, int(mrc))
    say(f"   server maxRecordCount {mrc} -> page size {PAGE}")

    # ---------------- stations
    js = get(s, f"{SERVICE}/0/query", {"where": "1=1", "outFields": "*", "returnGeometry": "true",
                                       "outSR": 4326, "f": "json"})
    if js:
        st = pd.DataFrame([dict(f["attributes"], lon=f.get("geometry", {}).get("x"), lat=f.get("geometry", {}).get("y"))
                           for f in js.get("features", [])])
        st.to_csv(os.path.join(OUT_DIR, "EAD_SDI_stations.csv"), index=False, encoding="utf-8-sig")
        say(f"   stations: {len(st)}  -> EAD_SDI_stations.csv")

    # ---------------- hourly table, paged by OBJECTID (resumable)
    oid = next((f["name"] for f in flds if f.get("type") == "esriFieldTypeOID"), "OBJECTID")
    last = -1
    if os.path.exists(RAW):
        last = int(pd.read_csv(RAW, usecols=[oid])[oid].max())
        say(f"   resuming after {oid} {last}")
    n_new, t0 = 0, time.time()
    while True:
        js = get(s, f"{SERVICE}/1/query", {"where": f"{oid} > {last}", "outFields": "*", "orderByFields": f"{oid} ASC",
                                           "resultRecordCount": PAGE, "returnGeometry": "false", "f": "json"})
        if js is None:
            if PAGE > 100:
                PAGE //= 2
                print(f"   lowering page size to {PAGE}")
                continue
            say(f"   [stopped] server not answering after {oid} {last} - run again later to resume")
            break
        feats = js.get("features", [])
        if not feats:
            break
        df = esri_dates(pd.DataFrame([f["attributes"] for f in feats]), date_fields)
        df.to_csv(RAW, mode="a", header=not os.path.exists(RAW), index=False, encoding="utf-8-sig")
        last = int(df[oid].max())
        n_new += len(df)
        if n_new % (PAGE * 20) < PAGE:
            print(f"   {n_new:,} records ({time.time() - t0:.0f} s), last {oid} {last}")
        if not js.get("exceededTransferLimit", len(feats) >= PAGE) and len(feats) < PAGE:
            break
    say(f"   downloaded this run: {n_new:,} records")

    # ---------------- report
    if not os.path.exists(RAW):
        say("   no records")
    else:
        d = pd.read_csv(RAW, low_memory=False)
        say(f"\n   total records {len(d):,} | columns {list(d.columns)}")
        tcol = next((c for c in (f + "_iso" for f in date_fields) if c in d), None)
        if tcol:
            t = pd.to_datetime(d[tcol])
            say(f"   {tcol}: {t.min()} -> {t.max()}")
            say("   records per year: " + str(t.dt.year.value_counts().sort_index().to_dict()))
            if "Hour" in d:      # does the served time equal local time stored as UTC?
                h = pd.to_numeric(d.Hour, errors="coerce")
                for off in (0, 4):
                    say(f"   share of rows with Hour == ({tcol} hour + {off}) mod 24: {((t.dt.hour + off) % 24 == h).mean():.1%}")
        scol = next((c for c in d.columns if "station" in c.lower() or "alias" in c.lower()), None)
        if scol:
            g = d.assign(_t=pd.to_datetime(d[tcol]) if tcol else pd.NaT).groupby(scol)
            say(f"\n   per station ({scol}):")
            for line in g.agg(rows=("_t", "size"), first=("_t", "min"), last=("_t", "max")).to_string().splitlines():
                say("      " + line)
        no2 = next((c for c in d.columns if c.upper() == "NO2"), None)
        if no2:
            v = pd.to_numeric(d[no2], errors="coerce")
            say(f"\n   NO2: n {v.notna().sum():,} | min {v.min():.2f} | p1 {v.quantile(.01):.2f} | median {v.median():.2f} | "
                f"p99 {v.quantile(.99):.2f} | max {v.max():.2f} | zero {(v == 0).sum():,} | negative {(v < 0).sum():,}")
            say("   (units are not stated by the service; EAD publishes NO2 in ug/m3 - to be confirmed against "
                "the OpenAQ overlap, if any)")
    p = os.path.join(C.DIRS["logs"], "ead_sdi_report.txt")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("\n".join(REPORT))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
