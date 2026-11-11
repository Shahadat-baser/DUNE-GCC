"""
STEP 8b - manuscript tables (local, ~2-5 min).  Run after 07g.  Reads frozen outputs only and rebuilds every
number with the same row definitions as the steps that reported them (identical rows per comparison,
95 % intervals from a cluster bootstrap by site).

Output  tables_paper/TableN_<name>.csv (full precision) and tables_paper/DUNE_tables.docx, logs/tables_report.txt
  Table I    input data sets, native resolution and resampling to the 0.01° grid
  Table II   ground observations by network and role (sites, station-weeks / site-months)
  Table III  Stage 1 variants on 2023 (schemes A-E): R2 [95 % CI], RMSE, bias, within-site r
  Table IV   frozen X1 on every external test (2024, independent network, transfer, 2026 forward)
  Table V    Stage 2: X1, LSTM-E, LSTM-P, climatology and persistence on identical rows
  Table VI   sensitivity runs (07g)
  Table S10  GRU instead of LSTM (07h; supplement), written only if 07h was run
"""
import os, sys, re, warnings
CODE_DIR = r"D:\Cities\GCC_Cities2\code"                      # fallback when __file__ is undefined (notebook)
try:
    CODE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    if os.path.exists(os.path.join(os.getcwd(), "dune_config.py")):
        CODE_DIR = os.getcwd()
sys.path.insert(0, CODE_DIR)
for _m in ("dune_config", "dune_env", "dune_gee", "dune_ground", "dune_results"):
    sys.modules.pop(_m, None)
import dune_env
dune_env.ensure("tables")
import numpy as np, pandas as pd
import dune_config as C
import dune_results as R

warnings.filterwarnings("ignore")
OUTD = os.path.join(C.BASE_DIR, "tables_paper")
OUT = []
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII"]
NM = {"pred_X1": "X1", "pred_E": "LSTM-E", "pred_P": "LSTM-P", "pred_Proll": "LSTM-P (rolling)", "pred_S1clim": "Stage 1 climatology",
      "pred_groundclim": "Ground climatology", "pred_persist": "Persistence", "pred_naive": "Seasonal naive"}


def say(s=""):
    print(s, flush=True)
    OUT.append(str(s))


def fmt(m):
    return dict(n=m.get("n"), sites=m.get("sites"), R2=m.get("R2"), R2_CI=m.get("R2_ci", ""), r=m.get("r"),
                RMSE=m.get("RMSE"), bias=m.get("bias"), anomaly_r=m.get("anomaly_r"))


def block(r, d, models, test):
    d = d.dropna(subset=["y"] + models)
    return [dict(test=test, model=NM.get(c, c), **fmt(r.metrics(d, c))) for c in models] if len(d) >= 3 else []


def table1():
    return pd.DataFrame([
        ("TROPOMI tropospheric NO2 column, cloud fraction, UV aerosol index", "COPERNICUS/S5P/OFFL/L3_NO2", "3.5 × 5.5 km (L2); L3 ~1.1 km", "Daily (13:30 LT)", "Nearest neighbour; daily mean of orbits; qa_value > 0.75 (L3)"),
        ("2 m temperature and dew point, 10 m wind, surface pressure", "ECMWF/ERA5_LAND/HOURLY (ERA5 over sea)", "0.1° (0.25°)", "10 UTC", "Bilinear"),
        ("Boundary-layer height", "ECMWF/ERA5/HOURLY", "0.25°", "10 UTC", "Bilinear"),
        ("Total precipitation", "ECMWF/ERA5_LAND/DAILY_AGGR", "0.1°", "Daily total", "Bilinear; weekly sum"),
        ("Night-time lights (VIIRS DNB)", "NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG", "~460 m", "Annual median of months", "Area-weighted mean"),
        ("NDVI (MOD13A1 v061)", "MODIS/061/MOD13A1", "500 m", "Annual mean", "Area-weighted mean"),
        ("Built-up surface, population, settlement class (GHSL R2023A)", "JRC/GHSL/P2023A", "100 m / 1 km", "2020, 2025 (linear in time)", "Area-weighted mean; class: nearest epoch"),
        ("Elevation (SRTM), land fraction (Hansen data mask)", "USGS/SRTMGL1_003; UMD/hansen", "30 m", "Static", "Area-weighted mean; coast distance from land fraction"),
    ], columns=["Variable", "Earth Engine collection", "Native resolution", "Time", "To the 0.01° grid"])


def table2(r):
    TW, TM = r.TW, r.TM
    rows = []
    tw = TW[(TW.pollutant == "NO2") & np.isfinite(TW.y)]
    for (net, per), d in tw.groupby(["Network", "period"]):
        rows.append(dict(network=net, resolution="weekly", period=per, sites=d.Site_ID.nunique(), rows=len(d)))
    tm = TM.copy()
    tm["role"] = np.where(r.indep_mask(tm), "independent", np.where(tm.pollutant != "NO2", "excluded (NOx)", tm.period))
    for (net, role), d in tm.groupby(["Network", "role"]):
        rows.append(dict(network=net, resolution="monthly", period=role, sites=d.Site_ID.nunique(), rows=len(d)))
    for res in ("weekly", "monthly"):
        f = r.forward(res)
        rows.append(dict(network="UAE_EAD (SDI feed)", resolution=res, period="forward 2026 (sealed)", sites=f.Site_ID.nunique(), rows=len(f)))
    return pd.DataFrame(rows)


def table3(r):
    M = pd.read_csv(os.path.join(R.V2, "metrics_stage1.csv"))
    rows = []
    for lab in M.label:
        parts = [p.strip() for p in str(lab).split("|")]
        if len(parts) == 2 and parts[0][:1] in "ABCDE" and parts[0][1:2] == " ":
            m = M[M.label == lab].iloc[0]
            rows.append(dict(scheme=parts[0], variant=parts[1], n=m.n, sites=m.sites, R2=m.R2, R2_CI=m.get("R2_ci", ""),
                             RMSE=m.RMSE, bias=m.bias, anomaly_r=m.anomaly_r))
    return pd.DataFrame(rows)


def table4(r):
    rows = []
    T = r.weekly_test()
    rows += block(r, T, ["pred_X1"], "2024 weekly, known stations")
    M = r.monthly_eval()
    rows += block(r, M[M.own & (M.month.dt.year == 2024)], ["pred_X1"], "2024 monthly, daily-network sites")
    for lab, sel in (("Independent 2022-2023 (monthly)", M.month.dt.year.isin([2022, 2023])),
                     ("Independent 2024 (monthly)", M.month.dt.year == 2024),
                     ("Independent 2019-2021 (monthly, T3 rasters)", M.raster_set.eq("T3"))):
        rows += block(r, M[M.indep & sel], ["pred_X1"], lab)
    rows += block(r, M[M.raster_set.eq("T3") & (M.pollutant == "NO2")], ["pred_X1"], "UAE 2019-2021 backcast (T3, monthly)")
    k = ["Anchor_Station", "week_start"]
    t2 = r.TW[(r.TW.raster_set == "T2") & np.isfinite(r.TW.y) & (r.TW.pollutant == "NO2")][["Target_ID", "Site_ID", "y"] + k]
    rows += block(r, t2.merge(r.points[k + ["pred_X1"]], on=k), ["pred_X1"], "Riyadh 2019-2020 (T2, weekly)")
    rows += block(r, r.forward("weekly"), ["pred_X1"], "Forward 2026 weekly (sealed)")
    rows += block(r, r.forward("monthly"), ["pred_X1"], "Forward 2026 monthly (sealed)")
    return pd.DataFrame(rows)


def table5(r):
    rows = []
    T = r.weekly_test()
    tr = r.train
    gclim = tr.assign(m=tr.week_start.dt.month).groupby(["Target_ID", "m"]).y.mean()
    T["pred_groundclim"] = gclim.reindex(pd.MultiIndex.from_arrays([T.Target_ID, T.week_start.dt.month])).values
    rows += block(r, T, ["pred_X1", "pred_E", "pred_P", "pred_S1clim", "pred_groundclim"], "2024 weekly (sites with 2023 baselines)")
    M = r.monthly_eval()
    mm = ["pred_X1", "pred_E", "pred_P", "pred_S1clim", "pred_persist"]
    rows += block(r, M[M.own & (M.month.dt.year == 2024)], mm, "2024 monthly, daily-network sites")
    rows += block(r, M[M.indep & (M.month.dt.year == 2024)], mm, "2024 monthly, independent stations")
    rows += block(r, r.forward("weekly"), ["pred_X1", "pred_E", "pred_P", "pred_Proll", "pred_S1clim"], "2026 weekly, forward")
    rows += block(r, r.forward("monthly"), ["pred_X1", "pred_E", "pred_P", "pred_Proll", "pred_S1clim", "pred_persist"],
                  "2026 monthly, forward")
    return pd.DataFrame(rows)


def table6():
    """one long table: block | variant | test set | n | R2 | RMSE | bias (frozen-model scoring unless stated)"""
    L = C.DIRS["logs"]
    rows = []
    p = os.path.join(L, "sensitivity_processing.csv")
    if os.path.exists(p):
        for _, a in pd.read_csv(p).iterrows():
            v = str(a.variant)
            rows.append(dict(block="Ground processing", variant=v, test="2023, scheme A (X1 refitted)", n=a.get("train_rows"),
                             R2=a.get("A_R2"), RMSE=np.nan, bias=np.nan))
            rows.append(dict(block="Ground processing", variant=v, test="2024 weekly (frozen X1)", n=np.nan,
                             R2=a.get("test_frozen_R2"), RMSE=np.nan, bias=np.nan))
            rows.append(dict(block="Ground processing", variant=v, test="2026 weekly (frozen X1)", n=np.nan,
                             R2=a.get("fwd_frozen_R2"), RMSE=np.nan, bias=a.get("fwd_frozen_bias")))
    p = os.path.join(L, "sensitivity_models.csv")
    if os.path.exists(p):
        for _, a in pd.read_csv(p).iterrows():
            rows.append(dict(block="Alternative model", variant=a.model, test=a.test, n=a.n, R2=a.R2, RMSE=a.RMSE, bias=a.bias))
    p = os.path.join(L, "sensitivity_static2026.csv")
    if os.path.exists(p):
        for _, a in pd.read_csv(p).iterrows():
            rows.append(dict(block="2026 annual layers", variant=a.static, test="2026 weekly (frozen X1)", n=a.n, R2=a.R2,
                             RMSE=a.RMSE, bias=a.bias))
    D = pd.DataFrame(rows)
    if len(D):
        D["n"] = D.n.map(lambda v: "" if pd.isna(v) else str(int(v)))
    return D


def table_s10():
    """supplementary Table S10: GRU vs LSTM from 07h (post hoc); empty if 07h was not run"""
    p = os.path.join(C.BASE_DIR, "sensitivity", "gru", "gru_vs_lstm.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    D = pd.read_csv(p)
    D = D[D.role.isin(["E (estimation)", "P (projection)"])]
    keep = ["test", "role", "n", "sites", "R2_X1", "R2_LSTM", "R2_GRU", "dR2_GRU_minus_LSTM", "CI_lo", "CI_hi", "bias_LSTM", "bias_GRU"]
    return D[[c for c in keep if c in D.columns]]


def _rd(name):
    p = os.path.join(C.DIRS["logs"], name)
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()


def table_s11():
    """supplementary Table S11: stratified residuals of X1 (08c)"""
    return _rd("rd_stratified_residuals.csv")


def table_s12():
    return _rd("rd_representativeness.csv").pipe(lambda D: D if D.empty else
        D.groupby("set").apply(lambda x: pd.Series(dict(n=len(x), sd3_median=x.sd3.median(), sd5_median=x.sd5.median(),
                                                          abs_cell_minus_m5=(x.cell - x.m5).abs().mean()))).reset_index())


def table_s13():
    return _rd("rd_uncertainty.csv")


def table_s14():
    return _rd("rd_baselines.csv")


HDR = {"network": "Network", "resolution": "Resolution", "period": "Role", "sites": "Sites", "rows": "Rows",
       "scheme": "Scheme", "variant": "Model", "n": "n", "R2": "R\u00b2", "R2_CI": "95 % CI (R\u00b2)", "RMSE": "RMSE",
       "bias": "Bias", "anomaly_r": "Within-site r", "test": "Test set", "model": "Model", "r": "r", "block": "Block"}


def _rules(table):
    """IEEE style: horizontal rules only (top, below the header, bottom)"""
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    tblPr = table._tbl.tblPr
    bd = OxmlElement("w:tblBorders")
    for edge, val in (("top", "single"), ("bottom", "single"), ("left", "nil"), ("right", "nil"),
                      ("insideH", "nil"), ("insideV", "nil")):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), val)
        e.set(qn("w:sz"), "6")
        e.set(qn("w:color"), "000000")
        bd.append(e)
    tblPr.append(bd)
    for cell in table.rows[0].cells:
        tcPr = cell._tc.get_or_add_tcPr()
        b = OxmlElement("w:tcBorders")
        e = OxmlElement("w:bottom")
        e.set(qn("w:val"), "single")
        e.set(qn("w:sz"), "4")
        e.set(qn("w:color"), "000000")
        b.append(e)
        tcPr.append(b)


def to_docx(tables, path):
    """one table per page block, full text width (7.16 in, IEEE double column), Times New Roman 8 pt, black"""
    from docx import Document
    from docx.shared import Pt, RGBColor, Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    sec.left_margin = sec.right_margin = Inches(0.67)
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    st.font.size = Pt(8)
    st.font.color.rgb = RGBColor(0, 0, 0)
    st.paragraph_format.space_after = Pt(0)
    for i, (title, df) in enumerate(tables):
        if df is None or df.empty:
            continue
        mS = re.search(r"Table (S\d+)", title)
        label = f"TABLE {mS.group(1)}" if mS else f"TABLE {ROMAN[i]}"
        for txt in (label, title.upper()):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.add_run(txt).font.size = Pt(8)
        df = df.copy()
        if "scheme" in df:
            df["scheme"] = df.scheme.astype(str).str[:1]
        for c in df.columns:
            if pd.api.types.is_float_dtype(df[c]):
                df[c] = df[c].map(lambda v: "" if pd.isna(v) else (f"{v:.2f}" if abs(v) < 100 else f"{v:.0f}"))
        cols = list(df.columns)
        t = doc.add_table(rows=1, cols=len(cols))
        t.autofit = False
        # width proportional to the longest entry (bounded), summing to the text width
        L = np.array([min(max(len(HDR.get(c, c)), df[c].astype(str).str.len().max() if len(df) else 0), 34) + 2 for c in cols], float)
        wid = L / L.sum() * 7.16
        for j, c in enumerate(cols):
            cell = t.rows[0].cells[j]
            cell.text = ""
            run = cell.paragraphs[0].add_run(HDR.get(c, c))
            run.bold = True
        for _, row in df.iterrows():
            cells = t.add_row().cells
            for j, v in enumerate(row):
                cells[j].text = "" if (isinstance(v, float) and np.isnan(v)) else str(v)
        for row in t.rows:
            for j, cell in enumerate(row.cells):
                cell.width = Inches(wid[j])
                for para in cell.paragraphs:
                    if j > 0:
                        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    for run in para.runs:
                        run.font.size = Pt(8)
                        run.font.name = "Times New Roman"
                        run.font.color.rgb = RGBColor(0, 0, 0)
        _rules(t)
        doc.add_paragraph()
    doc.save(path)


def main():
    os.makedirs(OUTD, exist_ok=True)
    say(f"=== Tables (step 08b) -> {OUTD}")
    r = R.Results()
    specs = [("Input data sets and their harmonisation to the 0.01° grid", "Table1_data", table1),
             ("Ground observations by network and role", "Table2_ground", lambda: table2(r)),
             ("Stage 1 variants on 2023 (validation schemes A–E)", "Table3_stage1", lambda: table3(r)),
             ("Frozen X1 on the external tests", "Table4_external", lambda: table4(r)),
             ("Stage 2 models and baselines on identical rows", "Table5_stage2", lambda: table5(r)),
             ("Sensitivity runs", "Table6_sensitivity", table6),
             ("GRU instead of LSTM in Stage 2 (post hoc, supplement Table S10)", "TableS10_gru", table_s10),
             ("Residuals of X1 by season, boundary-layer regime, dust and network (supplement Table S11)", "TableS11_residuals", table_s11),
             ("Point-to-pixel representativeness (supplement Table S12)", "TableS12_representativeness", table_s12),
             ("Propagation of column and resampling uncertainty (supplement Table S13)", "TableS13_uncertainty", table_s13),
             ("Random-forest baseline vs X1 and E1 (supplement Table S14)", "TableS14_baselines", table_s14)]
    done = []
    for title, name, f in specs:
        try:
            df = f()
            df.to_csv(os.path.join(OUTD, name + ".csv"), index=False)
            done.append((title, df))
            say(f"   {name}: {len(df)} rows")
        except Exception as ex:
            say(f"   {name} FAILED: {type(ex).__name__}: {ex}")
    to_docx(done, os.path.join(OUTD, "DUNE_tables.docx"))
    say("   DUNE_tables.docx written")
    p = os.path.join(C.DIRS["logs"], "tables_report.txt")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("\n".join(OUT))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
