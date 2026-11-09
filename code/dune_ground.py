"""
DUNE - helpers shared by 01_build_inputs.py and 06b_ground_truth.py for the Abu Dhabi SDI feed.
Only pandas / numpy.

The SDI feed (step 06c) names stations "EAD_AlMafraq", "EAD_KhadijaSchool" ...; OpenAQ (our earlier EAD
source) names the same monitors "Al Mafraq", "Khadeeja School" ...  A feed station is the SAME monitor as
an OpenAQ monitor only if (1) the agency names correspond (SDI_TO_OPENAQ) AND (2) the coordinates
published by the SDI station layer are within SDI_SAME_KM of the OpenAQ coordinates.  If the names
correspond but the positions do not, the station is treated as a NEW monitor (it may have moved), so
two different locations are never joined into one series.
"""
import os, re
import numpy as np, pandas as pd

SDI_SAME_KM = 3.0
SDI_TO_OPENAQ = {   # SDI name (without "EAD_") -> OpenAQ location name; None = not carried by OpenAQ
    "AlAinSchool": "Al Ain Islamic Institute", "AlAinStreet": "Al Ain Street", "AlMafraq": "Al Mafraq",
    "AlMaqta": "Al Maqtaa", "AlQuaa": "Al Qua'a", "AlTawia": "Al Tawia", "Baniyas": "Baniyas School",
    "BidaZayed": "Bida Zayed", "E11Road": "E11+Road", "Gayathi": "Gayathi", "Habshan": "Habshan South",
    "HamdanStreet": "Hamdan Street", "KhadijaSchool": "Khadeeja School", "KhalifaCity": "Khalifa City A",
    "Liwa": "Liwa", "Mussafah": "Mussafah", "RuwaisTransco": "Ruwais", "Sweihan": "Sweihan",
    "Zakher": "Zakher", "KhalifaSchool": None,
}


def norm(x):
    x = re.sub(r"[^a-z0-9]", "", str(x).lower())
    return x[3:] if x.startswith("ead") else x


def pretty(sdi_name):
    """'EAD_KhalifaSchool' -> 'Khalifa School'"""
    n = re.sub(r"^EAD_", "", str(sdi_name))
    return re.sub(r"(?<=[a-z])(?=[A-Z0-9])", " ", n)


def km(lat1, lon1, lat2, lon2):
    r = np.pi / 180
    a = np.sin((lat2 - lat1) * r / 2) ** 2 + np.cos(lat1 * r) * np.cos(lat2 * r) * np.sin((lon2 - lon1) * r / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(a))


# Other spellings of the same stations seen in the SDI station layer (full names) -> canonical SDI name.
# Every spelling is resolved explicitly; an unknown spelling stops the step instead of being guessed.
EXTRA_ALIASES = {
    "Al Ain Islamic Institute": "AlAinSchool", "Al Ain School": "AlAinSchool", "Baniyas School": "Baniyas",
    "Gayathi School": "Gayathi", "Habshan South": "Habshan", "Khadejah School": "KhadijaSchool",
    "Khadeeja School": "KhadijaSchool", "Khalifa City A": "KhalifaCity", "Ruwais": "RuwaisTransco",
    "Ruwais Transco": "RuwaisTransco", "Khalifa School": "KhalifaSchool", "E11 Road": "E11Road",
}


def canonical(label):
    """any known spelling (SDI short name, OpenAQ name, layer full name) -> canonical SDI name, else None"""
    table = {norm(k): k for k in SDI_TO_OPENAQ}
    table.update({norm(v): k for k, v in SDI_TO_OPENAQ.items() if v})
    table.update({norm(k): v for k, v in EXTRA_ALIASES.items()})
    return table.get(norm(label))


def sdi_stations(sdi_dir):
    """Station layer of the SDI feed -> DataFrame [sdi_key, sdi_label, Lat, Lon]; sdi_key = normalised
    canonical SDI name, i.e. the same key the hourly table's StationName gives."""
    p = os.path.join(sdi_dir, "EAD_SDI_stations.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, encoding="utf-8-sig")
    best, hits = None, -1
    for c in df.columns:
        if df[c].dtype == object:
            h = df[c].map(canonical).notna().sum()
            if h > hits:
                best, hits = c, h
    unresolved = df.loc[df[best].map(canonical).isna(), best].astype(str).tolist() if best else list(df.index)
    if best is None or unresolved:
        raise SystemExit(f"EAD_SDI_stations.csv: station names not recognised {unresolved} (column '{best}', "
                         f"columns {list(df.columns)}) - check the file; nothing was written")
    lon = next(c for c in df.columns if c.lower() in ("lon", "x", "longitude"))
    lat = next(c for c in df.columns if c.lower() in ("lat", "y", "latitude"))
    out = pd.DataFrame({"sdi_key": df[best].map(canonical).map(norm), "sdi_label": df[best].astype(str),
                        "canon": df[best].map(canonical),
                        "Lat": pd.to_numeric(df[lat], errors="coerce"), "Lon": pd.to_numeric(df[lon], errors="coerce")})
    if out.sdi_key.duplicated().any():
        raise SystemExit(f"EAD_SDI_stations.csv: two layer rows resolve to one station: "
                         f"{out[out.sdi_key.duplicated(keep=False)].sdi_label.tolist()}")
    return out.dropna(subset=["Lat", "Lon"])


def map_sdi(sdi, monitors):
    """sdi: from sdi_stations(); monitors: stations.csv rows of Network UAE_EAD.
    Returns one row per SDI station: sdi_key, Station (stations.csv key), decision, km, Lat, Lon."""
    ead = monitors.assign(oname=monitors.Station.str.split("|EAD_", n=1, regex=False).str[1])
    rows = []
    for r in sdi.itertuples():
        raw = r.canon
        oname = SDI_TO_OPENAQ.get(raw)
        hit = ead[ead.oname == oname] if oname else ead.iloc[0:0]
        if len(hit):
            d = float(km(r.Lat, r.Lon, hit.Lat.values[0], hit.Lon.values[0]))
            if d <= SDI_SAME_KM:
                rows.append(dict(sdi_key=r.sdi_key, Station=hit.Station.values[0], decision="same monitor (name + position)",
                                 km=round(d, 2), Lat=r.Lat, Lon=r.Lon))
                continue
            dec = f"NEW monitor: name matches '{oname}' but {d:.1f} km apart"
        else:
            d = np.nan
            dec = "NEW monitor (not carried by OpenAQ)"
        key = "UAE_EAD|EAD_" + pretty(raw)
        same = monitors[monitors.Station == key]
        if len(same):       # added from the SDI feed by an earlier run: must be the same place
            d2 = float(km(r.Lat, r.Lon, same.Lat.values[0], same.Lon.values[0]))
            if d2 > SDI_SAME_KM:
                raise SystemExit(f"SDI station {r.sdi_label} maps to existing {key} but lies {d2:.1f} km from it - stopped")
            rows.append(dict(sdi_key=r.sdi_key, Station=key, decision="same monitor (added from the SDI feed earlier)",
                             km=round(d2, 2), Lat=r.Lat, Lon=r.Lon))
            continue
        rows.append(dict(sdi_key=r.sdi_key, Station=key, decision=dec,
                         km=round(d, 2) if d == d else np.nan, Lat=r.Lat, Lon=r.Lon))
    return pd.DataFrame(rows)
