"""
STEP 1 - build the input point list (local only, seconds).

Writes inputs/stations.csv with one row per point that the station extraction will sample:
    Station, Network, Country, Lat, Lon, Kind, Site_ID, Col, Row
      Kind = 'monitor'         a ground station from the five ground files of the old project
             'district_point'  representative point of a district smaller than TINY_DISTRICT_KM2
                               (no 0.01 deg cell centre falls inside it, so a polygon mean is empty)
      Site_ID  monitors closer than 1 km to each other (e.g. the same Abu Dhabi site reported both by
               the UAE monthly file and by OpenAQ/EAD) share one Site_ID.  Cross-validation must group
               by Site_ID, otherwise the same physical site sits in train and test folds.
      Col, Row  the 0.01 deg grid cell of the point (0-based, from the upper-left corner)

Also checks that GCC_Cities.geojson (in BASE_DIR) holds 744 unique district ids.
"""
import os, sys, glob, shutil
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
for _m in ("dune_config", "dune_env", "dune_gee", "dune_ground"):   # always load the CURRENT files
    sys.modules.pop(_m, None)                         # cached in the notebook kernel from an earlier run
import dune_env
dune_env.ensure("build_inputs")      # installs / checks every package this step needs
import numpy as np, pandas as pd, geopandas as gpd
import dune_config as C
import dune_ground

GROUND = {  # network -> (file under OLD_DIR, country)
    "UAE_monthly":  ("UAE_Ground_Observations_Final.csv", "UAE"),
    "UAE_EAD":      (os.path.join("OpenAQ", "OpenAQ_NO2_Ground_Truth_Master.csv"), "UAE"),
    "KSA_Riyadh":   ("Riyadh_Ground_Observations_NO2_SO2.csv", "Saudi Arabia"),
    "Oman":         ("Oman_Ground_Observations_Master_2023_2024.csv", "Oman"),
    "Bahrain_SCE":  ("Bahrain_Ground_NO2_Monthly_2021_2025.csv", "Bahrain"),
}


def km(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(a))


def site_ids(df, radius_km=1.0):
    parent = list(range(len(df)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    lat, lon = df.Lat.values, df.Lon.values
    for i in range(len(df)):
        d = km(lat[i], lon[i], lat, lon)
        for j in np.where((d < radius_km) & (np.arange(len(df)) > i))[0]:
            parent[find(j)] = find(i)
    roots = [find(i) for i in range(len(df))]
    order = {r: k + 1 for k, r in enumerate(dict.fromkeys(roots))}
    return [f"S{order[r]:03d}" for r in roots]


def main():
    C.make_dirs()
    # ---------------- geojson
    if not os.path.exists(C.GEOJSON):
        raise SystemExit(f"district file not found: {C.GEOJSON}")
    g = gpd.read_file(C.GEOJSON).to_crs(4326)
    assert len(g) == 744 and g[C.DISTRICT_ID].is_unique, f"expected 744 unique {C.DISTRICT_ID}, got {len(g)}"
    g["Area_km2"] = g.to_crs(6933).area / 1e6
    print(f"districts: {len(g)} | area km2 min {g.Area_km2.min():.2f} median {g.Area_km2.median():.0f}")

    # ---------------- monitors
    rows = []
    for net, (f, country) in GROUND.items():
        p = os.path.join(C.OLD_DIR, f)
        if not os.path.exists(p):
            print(f"  [missing] {p}")
            continue
        s = pd.read_csv(p, usecols=["Station", "Lat", "Lon"]).dropna().drop_duplicates("Station")
        s["Network"], s["Country"], s["Kind"] = net, country, "monitor"
        rows.append(s)
        print(f"  {net:<12} {len(s):3d} stations  <- {f}")
    st = pd.concat(rows, ignore_index=True)
    st["Station"] = st.Network + "|" + st.Station.astype(str)

    # Abu Dhabi SDI feed (step 06c): monitors not already known are APPENDED, so every earlier
    # station keeps its row order and therefore its Site_ID.
    sdi = dune_ground.sdi_stations(C.SDI_DIR)
    if sdi is not None:
        mp = dune_ground.map_sdi(sdi, st[st.Network == "UAE_EAD"])
        print("\n  Abu Dhabi SDI stations vs known EAD monitors:")
        print("  " + mp[["sdi_key", "Station", "decision", "km"]].to_string(index=False).replace("\n", "\n  "))
        new = mp[mp.decision.str.startswith("NEW")]
        if len(new):
            st = pd.concat([st, pd.DataFrame({"Station": new.Station.values, "Lat": new.Lat.values, "Lon": new.Lon.values,
                                              "Network": "UAE_EAD", "Country": "UAE", "Kind": "monitor"})],
                           ignore_index=True)
            print(f"  added {len(new)} monitor(s) from the SDI feed: {new.Station.tolist()}")
    bad = st[(st.Lon < C.X0) | (st.Lon >= C.X1) | (st.Lat <= C.Y0) | (st.Lat > C.Y1)]
    if len(bad):
        print(f"  dropping {len(bad)} monitor(s) outside the grid: {bad.Station.tolist()}")
        st = st.drop(bad.index)
    st = st.reset_index(drop=True)
    st["Site_ID"] = site_ids(st)

    # ---------------- tiny districts
    tiny = g[g.Area_km2 < C.TINY_DISTRICT_KM2]
    rp = tiny.geometry.representative_point()
    dp = pd.DataFrame({"Station": "DIST|" + tiny[C.DISTRICT_ID].astype(str), "Lat": rp.y.values, "Lon": rp.x.values,
                       "Network": "district_point", "Country": tiny.get("COUNTRY", pd.Series("", index=tiny.index)).values,
                       "Kind": "district_point", "Site_ID": ""})
    dp["Country"] = dp.Country.replace({"United Arab Emirates": "UAE"})      # one spelling for every row
    out = pd.concat([st.reset_index(drop=True), dp], ignore_index=True)
    out["Col"] = np.floor((out.Lon - C.X0) / C.RES).astype(int)
    out["Row"] = np.floor((C.Y1 - out.Lat) / C.RES).astype(int)
    out = out[["Station", "Network", "Country", "Lat", "Lon", "Kind", "Site_ID", "Col", "Row"]]

    # integrity: an existing stations.csv may only GROW - every earlier point must be unchanged
    if os.path.exists(C.STATIONS_CSV):
        old = pd.read_csv(C.STATIONS_CSV, keep_default_na=False)
        cur = out.astype({"Site_ID": str}).fillna({"Site_ID": ""})
        m = old.merge(cur, on="Station", how="left", suffixes=("_old", ""), indicator=True)
        gone = m[m["_merge"] == "left_only"].Station.tolist()
        diff = []
        for c in ["Lat", "Lon", "Col", "Row", "Site_ID", "Network", "Kind"]:
            a, b = m[c + "_old"], m[c]
            if c in ("Lat", "Lon", "Col", "Row"):
                bad = ~np.isclose(pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce"), rtol=0, atol=1e-9)
            else:
                bad = a.astype(str) != b.astype(str)
            diff += [f"{st_} {c}: {x} -> {y}" for st_, x, y in zip(m.Station[bad & (m["_merge"] == "both")], a[bad], b[bad])]
        if gone or diff:
            raise SystemExit("stations.csv would change for EXISTING points - nothing written:\n  "
                             + "\n  ".join([f"removed {g}" for g in gone] + diff[:30]))
        added = sorted(set(out.Station) - set(old.Station))
        stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        if added:
            shutil.copy(C.STATIONS_CSV, C.STATIONS_CSV.replace(".csv", f"_before_{stamp}.csv"))
        print(f"\nintegrity: all {len(old)} existing points unchanged (position, cell, Site_ID); added {len(added)}: {added}")
    out.to_csv(C.STATIONS_CSV, index=False)

    m = out[out.Kind == "monitor"]
    print(f"\nmonitors: {len(m)} in {m.Site_ID.nunique()} distinct sites (1 km rule) | "
          f"{m.groupby(['Col', 'Row']).ngroups} distinct grid cells")
    print(m.groupby("Country").agg(stations=("Station", "size"), sites=("Site_ID", "nunique")).to_string())
    shared = m.groupby("Site_ID").filter(lambda x: x.Network.nunique() > 1)
    if len(shared):
        print(f"\nsites reported by more than one network ({shared.Site_ID.nunique()}):")
        print(shared.sort_values("Site_ID")[["Site_ID", "Station", "Lat", "Lon"]].to_string(index=False))
    print(f"\ntiny districts given a point sample: {len(dp)}")
    print(f"wrote {C.STATIONS_CSV}  ({len(out)} points)")


if __name__ == "__main__":
    main()
