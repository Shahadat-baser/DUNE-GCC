"""
STEP 8a - manuscript figures (local, ~5-15 min).  Run after 07g.  Reads frozen outputs only.

Figures (files figures/FigNN_<name>.tif/.png)
  Fig01  study area: districts, elevation, ground monitors by role
  Fig02  workflow of the framework (schematic)
  Fig03  observed vs estimated: (a) leave-one-site-out 2023, (b) 2024 known stations, (c) 2026 sealed forward,
         (d) independent agency stations 2024 (monthly)
  Fig04  maps 2022-2025: (a) winter DJF, (b) summer JJA, (c) all-month mean surface NO2, (d) share of months inside the AOA
  Fig05  mean |SHAP| of the frozen X1, top 15 features, coloured by feature family
  Fig06  month by month: (a) 2024 daily-network sites, (b) 2024 independent stations, (c) 2026 forward
  Fig07  2026 vs 2024 per station: observed change vs TROPOMI column change and X1 change (post-hoc)
  Fig08  GCC district-mean monthly NO2 2022-2026: Stage 1 vs LSTM-P (fixed and rolling baseline)
  FigS1  supplement: GRU minus LSTM (R2, paired site-bootstrap 95 % CI) from 07h (post hoc)
Output  figures/ and logs/figures_report.txt
"""
import os, sys, glob, warnings
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
dune_env.ensure("figures")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
import geopandas as gpd, rasterio
import dune_config as C
import dune_results as R

warnings.filterwarnings("ignore")
DPI = 1080
W1, W2 = 3.5, 7.16
FIG = os.path.join(C.BASE_DIR, "figures")
OUT = []
COL = {"obs": "#000000", "pred_X1": "#1F4E79", "pred_E": "#C55A11", "pred_P": "#548235", "pred_Proll": "#A9D18E",
       "pred_S1clim": "#7F7F7F", "pred_persist": "#BFBFBF"}
LAB = {"obs": "Observed", "pred_X1": "X1 (Stage 1)", "pred_E": "LSTM-E", "pred_P": "LSTM-P", "pred_Proll": "LSTM-P (rolling)",
       "pred_S1clim": "Stage 1 climatology", "pred_persist": "Persistence"}
UG = r"$\mu$g m$^{-3}$"
MCOL = {"obs": "#000000", "pred_X1": "#4E79A7", "pred_E": "#F28E2B", "pred_P": "#59A14F", "pred_Proll": "#B07AA1",
        "pred_S1clim": "#9D9D9D", "pred_persist": "#E15759"}


def say(s=""):
    print(s, flush=True)
    OUT.append(str(s))


def style():
    names = {f.name for f in font_manager.fontManager.ttflist}
    font = "Times New Roman" if "Times New Roman" in names else ("Liberation Serif" if "Liberation Serif" in names else "DejaVu Serif")
    if font != "Times New Roman":
        say(f"   [note] Times New Roman not installed - using {font}; run on Windows for the final figures")
    plt.rcParams.update({
        "font.family": font, "mathtext.fontset": "custom", "mathtext.rm": font, "mathtext.it": f"{font}:italic",
        "mathtext.bf": f"{font}:bold", "mathtext.sf": font, "mathtext.cal": font, "mathtext.tt": font,
        "font.size": 10, "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
        "text.color": "black", "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black",
        "axes.edgecolor": "black", "axes.linewidth": 0.8, "axes.titlesize": 11, "legend.frameon": False,
        "savefig.dpi": DPI, "figure.dpi": 150})


def panel(ax, letter):
    ax.text(-0.02, 1.02, f"({letter})", transform=ax.transAxes, fontsize=11, fontweight="bold", ha="right", va="bottom")


def save(fig, name):
    for ax in fig.axes:                              # map axes: never an axis title (degree labels only)
        if ax.get_xlabel().startswith("Geodetic") or ax.get_ylabel().startswith("Geodetic"):
            ax.set_xlabel("")
            ax.set_ylabel("")
    os.makedirs(FIG, exist_ok=True)
    base = os.path.join(FIG, name)
    fig.savefig(base + ".tif", dpi=DPI, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(base + ".png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    say(f"   {name}.tif / .png")


def scatter(ax, y, p, lim, color=None):
    """observed vs estimated with the validation scores in the upper-left corner (opaque box above the points)"""
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = np.asarray(y)[ok], np.asarray(p)[ok]
    ax.scatter(y, p, s=7, c=color or MCOL["pred_X1"], alpha=0.5, linewidths=0, rasterized=True, zorder=2)
    ax.plot([0, lim], [0, lim], color="black", lw=0.8, zorder=3)
    if len(y) > 2:
        b, a = np.polyfit(y, p, 1)
        ax.plot([0, lim], [a, a + b * lim], color=MCOL["pred_persist"], lw=1.0, ls="--", zorder=3)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    st = dict(n=int(len(y)), R2=float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)),
              r=float(np.corrcoef(y, p)[0, 1]), RMSE=float(np.sqrt(np.mean((y - p) ** 2))), bias=float(np.mean(p - y)))
    txt = (f"n = {st['n']}\n$R^2$ = {st['R2']:.2f}\nr = {st['r']:.2f}\nRMSE = {st['RMSE']:.1f}\nBias = {st['bias']:+.1f}").replace("-", "\u2212")
    ul = np.sum((y < 0.42 * lim) & (p > 0.58 * lim))           # points under an upper-left box
    lr = np.sum((y > 0.58 * lim) & (p < 0.42 * lim))           # points under a lower-right box
    x, yy, ha, va = (0.035, 0.965, "left", "top") if ul <= lr else (0.965, 0.035, "right", "bottom")
    ax.text(x, yy, txt, transform=ax.transAxes, va=va, ha=ha, fontsize=10, linespacing=1.15, zorder=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BFBFBF", lw=0.5, alpha=0.92))
    return st


def raster_extent():
    return [C.X0, C.X0 + C.WIDTH * C.RES, C.Y1 - C.HEIGHT * C.RES, C.Y1]


# ---------------------------------------------------------------------------- map helpers
SEA, LAND = "#D6E9F5", "#F4F1EA"


def countries(g):
    """country outlines from the district file (COUNTRY attribute if present, else the outer boundary)"""
    col = next((c for c in ("COUNTRY", "Country", "NAME_0", "ADM0_NAME") if c in g.columns), None)
    return g.dissolve(col) if col else g.dissolve()


def lonlat_ticks(ax, step=5):
    """degree labels; ticks closer than 1 deg to a map edge are dropped so labels of adjacent panels never touch"""
    from matplotlib.ticker import FuncFormatter, FixedLocator
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    ax.xaxis.set_major_locator(FixedLocator([v for v in np.arange(np.ceil(x0 / step) * step, x1, step) if x0 + 1 < v < x1 - 1]))
    ax.yaxis.set_major_locator(FixedLocator([v for v in np.arange(np.ceil(y0 / step) * step, y1, step) if y0 + 1 < v < y1 - 1]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}°E"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}°N"))
    ax.tick_params(length=2, pad=1.5)


def basemap(ax, g, cg, land=True):
    b = g.total_bounds
    ax.set_facecolor(SEA)
    if land:
        cg.plot(ax=ax, color=LAND, lw=0, zorder=0)
    ax.set_xlim(b[0] - 0.3, b[2] + 0.3)
    ax.set_ylim(b[1] - 0.3, b[3] + 0.3)
    ax.set_aspect(1 / np.cos(np.deg2rad((b[1] + b[3]) / 2)))
    lonlat_ticks(ax)                     # degree labels only (e.g. 45°E, 25°N); no axis titles


def no_axis_titles(ax):
    """GeoPandas writes 'Geodetic longitude/latitude [degree]' from the CRS; maps carry degree tick labels only"""
    ax.set_xlabel("")
    ax.set_ylabel("")


def borders(ax, g, cg):
    g.boundary.plot(ax=ax, color="#5A5A5A", lw=0.08, alpha=0.7, zorder=3)
    cg.boundary.plot(ax=ax, color="black", lw=0.6, zorder=4)
    no_axis_titles(ax)


# ============================================================================ figures
def fig01(r, g):
    from matplotlib.colors import LightSource, LinearSegmentedColormap, Normalize
    from matplotlib.cm import ScalarMappable
    cg = countries(g)
    fig, ax = plt.subplots(figsize=(W2, 5.6))
    basemap(ax, g, cg)
    terrain = LinearSegmentedColormap.from_list("dune_terrain", ["#E9E4C9", "#D9C48E", "#C29B5E", "#9C6B3E", "#6E4B2E", "#F2F2F2"])
    norm = Normalize(0, 2500)
    dem_p = os.path.join(C.DIRS["static"], "Static_Fixed_1km.tif")
    if os.path.exists(dem_p):
        with rasterio.open(dem_p) as s:
            dem = s.read(s.descriptions.index("DEM") + 1).astype(float)
        land = np.isfinite(dem) & (dem > 0)
        rgb = LightSource(azdeg=315, altdeg=40).shade(np.nan_to_num(dem), cmap=terrain, norm=norm, blend_mode="soft", vert_exag=0.02)
        rgba = np.dstack([rgb[..., :3], land.astype(float)])
        ax.imshow(rgba, extent=raster_extent(), interpolation="bilinear", zorder=1)
        cb = fig.colorbar(ScalarMappable(norm, terrain), ax=ax, fraction=0.025, pad=0.012, extend="max")
        cb.set_label("Elevation (m)")
    borders(ax, g, cg)
    st = r.sites.copy()
    tr, TM = r.train_sites, r.TM
    ind = set(TM[r.indep_mask(TM)].Site_ID)
    fwd = set(r.forward("weekly").Site_ID)
    role = np.select([st.Site_ID.isin(tr), st.Site_ID.isin(ind), st.Network.eq("KSA_Riyadh"), st.Site_ID.isin(fwd),
                      st.Network.eq("Bahrain_SCE")],
                     ["Training (EAD, Oman)", "Independent (UAE agencies)", "Transfer (Riyadh)",
                      "Forward test only (2026)", "Excluded (Bahrain)"], "Other")
    sty = {"Training (EAD, Oman)": ("o", "#E15759"), "Independent (UAE agencies)": ("^", "#4E79A7"),
           "Transfer (Riyadh)": ("s", "#59A14F"), "Forward test only (2026)": ("D", "#B07AA1"),
           "Excluded (Bahrain)": ("X", "#8C8C8C")}
    for k, (mk, c) in sty.items():
        d = st[role == k].drop_duplicates("Site_ID")
        if len(d):
            ax.scatter(d.Lon, d.Lat, marker=mk, s=22, c=c, edgecolors="black", linewidths=0.4,
                       label=f"{k} ({d.Site_ID.nunique()})", zorder=6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.075), ncol=3, fontsize=10, handletextpad=0.3, columnspacing=1.2,
              borderaxespad=0)
    save(fig, "Fig01_study_area")


# ---------------------------------------------------------------------------- workflow icons (patches only)
def _icons():
    from matplotlib.patches import Rectangle, Circle, Ellipse, Polygon, Wedge, FancyBboxPatch as FB, Arc

    def satellite(ax, x, y, s):
        ax.add_patch(Rectangle((x - .16 * s, y - .16 * s), .32 * s, .32 * s, fc="#B7950B", ec="black", lw=.6, zorder=5))
        for sx in (-1, 1):
            x0 = x + sx * .18 * s - (0 if sx > 0 else .42 * s)
            ax.add_patch(Rectangle((x0, y - .1 * s), .42 * s, .2 * s, fc="#2E86C1", ec="black", lw=.6, zorder=5))
            for k in (1, 2, 3):
                ax.plot([x0 + k * .105 * s] * 2, [y - .1 * s, y + .1 * s], color="white", lw=.5, zorder=6)
        ax.add_patch(Wedge((x, y - .2 * s), .12 * s, 200, 340, fc="#D5D8DC", ec="black", lw=.5, zorder=5))
        for rr in (.32, .44):
            ax.add_patch(Arc((x, y - .3 * s), rr * s, rr * s * .7, theta1=220, theta2=320, color="#C0392B", lw=.9, zorder=5))

    def weather(ax, x, y, s):
        ax.add_patch(Circle((x - .18 * s, y + .14 * s), .17 * s, fc="#F4D03F", ec="#B7950B", lw=.6, zorder=4))
        for dx, dy, rr in ((-.12, -.02, .17), (.06, .06, .21), (.24, -.03, .15)):
            ax.add_patch(Circle((x + dx * s, y + dy * s), rr * s, fc="#D6EAF8", ec="#2E86C1", lw=.6, zorder=5))
        ax.add_patch(Rectangle((x - .28 * s, y - .18 * s), .56 * s, .14 * s, fc="#D6EAF8", ec="none", zorder=6))
        for k, dx in enumerate((-.15, .0, .15)):
            ax.plot([x + dx * s, x + (dx - .05) * s], [y - .24 * s, y - .38 * s], color="#2E86C1", lw=1.0, zorder=5)

    def city(ax, x, y, s):
        for dx, h, w in ((-.3, .45, .16), (-.12, .7, .18), (.08, .55, .16), (.26, .38, .16)):
            ax.add_patch(Rectangle((x + dx * s, y - .35 * s), w * s, h * s, fc="#1B2631", ec="black", lw=.4, zorder=5))
            for wy in np.arange(-.28, -.35 + h - .05, .1):
                ax.add_patch(Rectangle((x + (dx + .04) * s, y + wy * s), .04 * s, .04 * s, fc="#F7DC6F", lw=0, zorder=6))
                ax.add_patch(Rectangle((x + (dx + w - .08) * s, y + wy * s), .04 * s, .04 * s, fc="#F7DC6F", lw=0, zorder=6))
        ax.add_patch(Circle((x + .3 * s, y + .3 * s), .08 * s, fc="#F9E79F", ec="none", zorder=5))

    def leaf(ax, x, y, s):
        ax.add_patch(Ellipse((x, y), .7 * s, .34 * s, angle=35, fc="#28B463", ec="#186A3B", lw=.7, zorder=5))
        ax.plot([x - .3 * s, x + .28 * s], [y - .21 * s, y + .2 * s], color="#186A3B", lw=.7, zorder=6)
        ax.plot([x - .3 * s, x - .4 * s], [y - .21 * s, y - .33 * s], color="#186A3B", lw=1.0, zorder=6)

    def mountain(ax, x, y, s):
        ax.add_patch(Polygon([(x - .45 * s, y - .3 * s), (x - .1 * s, y + .3 * s), (x + .25 * s, y - .3 * s)], fc="#A04000", ec="black", lw=.5, zorder=5))
        ax.add_patch(Polygon([(x - .05 * s, y - .3 * s), (x + .2 * s, y + .12 * s), (x + .45 * s, y - .3 * s)], fc="#CA6F1E", ec="black", lw=.5, zorder=5))
        ax.add_patch(Polygon([(x - .22 * s, y + .1 * s), (x - .1 * s, y + .3 * s), (x + .02 * s, y + .1 * s)], fc="white", ec="black", lw=.4, zorder=6))
        ax.plot([x - .5 * s, x + .5 * s], [y - .3 * s, y - .3 * s], color="#1F618D", lw=1.4, zorder=6)

    def pin(ax, x, y, s, c="#E74C3C"):
        ax.add_patch(Polygon([(x - .17 * s, y + .05 * s), (x + .17 * s, y + .05 * s), (x, y - .35 * s)], fc=c, ec="black", lw=.5, zorder=5))
        ax.add_patch(Circle((x, y + .12 * s), .2 * s, fc=c, ec="black", lw=.5, zorder=5))
        ax.add_patch(Circle((x, y + .12 * s), .08 * s, fc="white", ec="none", zorder=6))

    def stations(ax, x, y, s):
        for dx, dy, c in ((-.25, -.05, "#E74C3C"), (.05, .12, "#8E44AD"), (.3, -.1, "#2471A3")):
            pin(ax, x + dx * s, y + dy * s, .7 * s, c)

    def cube(ax, x, y, s):
        cols = ["#F39C12", "#27AE60", "#2E86C1", "#8E44AD"]
        for k, c in enumerate(cols):
            yy = y - .3 * s + k * .18 * s
            pts = [(x - .45 * s, yy), (x + .15 * s, yy), (x + .45 * s, yy + .22 * s), (x - .15 * s, yy + .22 * s)]
            ax.add_patch(Polygon(pts, fc=c, ec="black", lw=.5, alpha=.95, zorder=5 + k))
        yy = y - .3 * s + 3 * .18 * s
        for t in np.linspace(0, 1, 5)[1:-1]:
            ax.plot([x - .45 * s + t * .6 * s, x - .15 * s + t * .6 * s], [yy, yy + .22 * s], color="white", lw=.4, zorder=10)
            ax.plot([x - .45 * s + t * .3 * s, x + .15 * s + t * .3 * s], [yy + t * .22 * s] * 2, color="white", lw=.4, zorder=10)

    def funnel(ax, x, y, s):
        ax.add_patch(Polygon([(x - .4 * s, y + .3 * s), (x + .4 * s, y + .3 * s), (x + .08 * s, y - .05 * s), (x + .08 * s, y - .35 * s),
                              (x - .08 * s, y - .25 * s), (x - .08 * s, y - .05 * s)], fc="#E67E22", ec="black", lw=.6, zorder=5))
        for dx, dy in ((-.2, .42), (.05, .5), (.25, .4)):
            ax.add_patch(Circle((x + dx * s, y + dy * s), .06 * s, fc="#5DADE2", ec="black", lw=.4, zorder=5))
        ax.add_patch(Circle((x, y - .45 * s), .06 * s, fc="#5DADE2", ec="black", lw=.4, zorder=5))

    def tree(ax, x, y, s):
        nodes = {"r": (0, .3), "a": (-.25, 0), "b": (.25, 0), "c": (-.38, -.3), "d": (-.12, -.3), "e": (.12, -.3), "f": (.38, -.3)}
        for u, v in (("r", "a"), ("r", "b"), ("a", "c"), ("a", "d"), ("b", "e"), ("b", "f")):
            ax.plot([x + nodes[u][0] * s, x + nodes[v][0] * s], [y + nodes[u][1] * s, y + nodes[v][1] * s], color="black", lw=.7, zorder=4)
        for k, (u, (dx, dy)) in enumerate(nodes.items()):
            c = "#2E86C1" if u == "r" else ("#48C9B0" if u in "ab" else "#F5B041")
            ax.add_patch(Circle((x + dx * s, y + dy * s), .09 * s, fc=c, ec="black", lw=.5, zorder=5))

    def lstm(ax, x, y, s):
        for k, dx in enumerate((-.38, 0, .38)):
            ax.add_patch(FB((x + (dx - .12) * s, y - .12 * s), .24 * s, .24 * s, boxstyle="round,pad=0,rounding_size=%f" % (.05 * s),
                            fc=["#A569BD", "#8E44AD", "#6C3483"][k], ec="black", lw=.5, zorder=5))
            ax.add_patch(Arc((x + dx * s, y + .16 * s), .2 * s, .2 * s, theta1=0, theta2=180, color="#6C3483", lw=.9, zorder=5))
            ax.plot([x + dx * s, x + dx * s], [y - .3 * s, y - .12 * s], color="black", lw=.7, zorder=4)
            ax.plot([x + dx * s, x + dx * s], [y + .12 * s, y + .3 * s], color="black", lw=.7, zorder=4)
        for dx in (-.26, .12):
            ax.annotate("", xy=(x + (dx + .14) * s, y), xytext=(x + dx * s, y), arrowprops=dict(arrowstyle="-|>", lw=.7, color="black"), zorder=6)

    def mapicon(ax, x, y, s):
        from matplotlib import cm
        cols = cm.YlOrRd(np.linspace(.25, .85, 3))
        for k in range(3):
            x0 = x + (-.42 + k * .28) * s
            dy = .06 * s if k % 2 else 0
            ax.add_patch(Polygon([(x0, y - .28 * s + dy), (x0 + .28 * s, y - .28 * s + (.06 * s - dy)),
                                  (x0 + .28 * s, y + .28 * s + (.06 * s - dy)), (x0, y + .28 * s + dy)], fc=cols[k], ec="black", lw=.5, zorder=5))
        pin(ax, x + .1 * s, y + .05 * s, .55 * s, "#1F618D")

    def target(ax, x, y, s):
        for rr, c in ((.38, "#E74C3C"), (.27, "white"), (.17, "#E74C3C"), (.07, "white")):
            ax.add_patch(Circle((x, y), rr * s, fc=c, ec="black", lw=.4, zorder=5))
        ax.plot([x - .1 * s, x + .02 * s, x + .38 * s], [y + .02 * s, y - .12 * s, y + .3 * s], color="#1E8449", lw=2.2, zorder=7,
                solid_capstyle="round")

    def globe(ax, x, y, s):
        ax.add_patch(Circle((x, y), .36 * s, fc="#AED6F1", ec="black", lw=.6, zorder=5))
        for w in (.36, .2):
            ax.add_patch(Ellipse((x, y), w * s, .72 * s, fc="none", ec="#1F618D", lw=.6, zorder=6))
        for dy in (-.18, 0, .18):
            half = np.sqrt(max(.36 ** 2 - dy ** 2, 0))
            ax.plot([x - half * s, x + half * s], [y + dy * s] * 2, color="#1F618D", lw=.6, zorder=6)
        ax.add_patch(Polygon([(x - .12 * s, y + .1 * s), (x + .1 * s, y + .2 * s), (x + .15 * s, y - .05 * s), (x - .05 * s, y - .15 * s)],
                             fc="#52BE80", ec="none", zorder=6))

    def lock(ax, x, y, s):
        ax.add_patch(Arc((x, y + .08 * s), .4 * s, .5 * s, theta1=0, theta2=180, color="#566573", lw=2.4, zorder=4))
        ax.add_patch(FB((x - .3 * s, y - .35 * s), .6 * s, .45 * s, boxstyle="round,pad=0,rounding_size=%f" % (.06 * s),
                        fc="#F4D03F", ec="black", lw=.6, zorder=5))
        ax.add_patch(Circle((x, y - .1 * s), .06 * s, fc="black", zorder=6))
        ax.plot([x, x], [y - .1 * s, y - .25 * s], color="black", lw=1.4, zorder=6)

    return dict(satellite=satellite, weather=weather, city=city, leaf=leaf, mountain=mountain, stations=stations, cube=cube,
                funnel=funnel, tree=tree, lstm=lstm, map=mapicon, target=target, globe=globe, lock=lock)


def fig02():
    """methodology: four coloured lanes, geospatial icons, keywords only (>= 10 pt), connected by coloured arrows"""
    from matplotlib.patches import FancyBboxPatch as FB, FancyArrowPatch
    I = _icons()
    fig = plt.figure(figsize=(W2, 4.3))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 716)
    ax.set_ylim(0, 430)
    ax.set_aspect("equal")
    ax.axis("off")
    lanes = [(4, 186, "Inputs", "#D6EAF8", "#5DADE2"), (198, 150, "Harmonization", "#FDEBD0", "#F5B041"),
             (356, 162, "Modeling", "#E8DAEF", "#AF7AC5"), (526, 186, "Evaluation", "#D5F5E3", "#58D68D")]
    top = 380
    cx = {}
    for x0, w, title, light, strong in lanes:
        ax.add_patch(FB((x0, 6), w, top - 6, boxstyle="round,pad=0,rounding_size=12", fc=light, ec=strong, lw=1.2, zorder=0))
        ax.add_patch(FB((x0 + 12, top + 8), w - 24, 32, boxstyle="round,pad=0,rounding_size=10", fc=strong, ec="black", lw=.6, zorder=1))
        ax.text(x0 + w / 2, top + 24, title, ha="center", va="center", fontsize=11, fontweight="bold", zorder=2)
        cx[title] = (x0, x0 + w / 2, x0 + w)
    T = dict(fontsize=10, va="center", ha="left", zorder=8)
    Cn = dict(fontsize=10, va="center", ha="center", zorder=8)
    # lane 1: six inputs
    items = [("satellite", "TROPOMI NO$_2$", 34), ("weather", "ERA5, ERA5-Land", 42), ("city", "VIIRS, GHSL", 42),
             ("leaf", "MODIS NDVI", 42), ("mountain", "SRTM, land mask", 42), ("stations", "Ground NO$_2$", 42)]
    ys = np.linspace(340, 42, len(items))
    for (ic, lab, sz), yy in zip(items, ys):
        I[ic](ax, 34, yy, sz)
        ax.text(66, yy, lab, **T)
    # lane 2
    x2 = cx["Harmonization"][1]
    I["cube"](ax, x2, 285, 104)
    ax.text(x2, 210, "0.01° data cube", **Cn)
    ax.text(x2, 191, "weekly, 2019–2026", **Cn)
    I["funnel"](ax, x2, 108, 66)
    ax.text(x2, 54, "Quality control", **Cn)
    ax.text(x2, 35, "sealed 2026 set", **Cn)
    # lane 3
    x3 = cx["Modeling"][1]
    I["tree"](ax, x3, 292, 96)
    ax.text(x3, 222, "Stage 1: XGBoost", **Cn)
    ax.text(x3, 203, "column → surface", **Cn)
    I["lstm"](ax, x3, 112, 96)
    ax.text(x3, 54, "Stage 2: LSTM", **Cn)
    ax.text(x3, 35, "estimation, projection", **Cn)
    # lane 4
    x4 = cx["Evaluation"][0]
    out = [("map", "1 km maps", "AOA flag"), ("target", "Validation", "schemes A–E"),
           ("globe", "External tests", "network, transfer"), ("lock", "2026 forward", "scored once")]
    for (ic, l1, l2), yy in zip(out, np.linspace(330, 75, 4)):
        I[ic](ax, x4 + 34, yy, 46)
        ax.text(x4 + 66, yy + 9, l1, **T)
        ax.text(x4 + 66, yy - 10, l2, **T)

    def arr(p0, p1, c, cs="arc3,rad=0", ls="-"):
        ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=cs, arrowstyle="-|>,head_length=5,head_width=3",
                                     color=c, lw=1.8, ls=ls, zorder=7, shrinkA=2, shrinkB=2))
    e1, e2, e3 = cx["Inputs"][2], cx["Harmonization"][2], cx["Modeling"][2]
    arr((e1 - 8, 300), (x2 - 58, 290), "#2E86C1")                       # satellite / reanalysis / proxies -> cube
    arr((e1 - 40, 42), (x2 - 38, 100), "#E74C3C")                       # ground -> QC
    arr((x2 + 58, 292), (x3 - 52, 292), "#F39C12")                      # cube -> Stage 1
    from matplotlib.path import Path
    gap = (cx["Harmonization"][2] + cx["Modeling"][0]) / 2              # QC targets -> Stage 1, routed through the lane gap
    ax.add_patch(FancyArrowPatch(path=Path([(x2 + 40, 112), (gap, 112), (gap, 272), (x3 - 52, 272)],
                                           [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO]),
                                 arrowstyle="-|>,head_length=5,head_width=3", color="#E67E22", lw=1.8, zorder=7))
    arr((x3, 190), (x3, 150), "#8E44AD")                                # Stage 1 -> Stage 2
    arr((x3 + 52, 305), (x4 + 6, 330), "#27AE60")                       # Stage 1 -> maps
    arr((x3 + 52, 285), (x4 + 6, 245), "#27AE60")                       # Stage 1 -> validation
    arr((x3 + 48, 262), (x4 + 6, 160), "#27AE60", "arc3,rad=0.12")      # Stage 1 -> external
    arr((x3 + 56, 112), (x4 + 6, 82), "#8E44AD")                        # Stage 2 -> forward
    arr((x2 + 40, 16), (x4 + 34, 52), "#7F8C8D", "angle,angleA=0,angleB=90,rad=8", "--")  # sealed set -> forward test
    save(fig, "Fig02_workflow")


def _grid(nrows, ncols, w, h, **kw):
    fig, axs = plt.subplots(nrows, ncols, figsize=(w, h), squeeze=False, **kw)
    return fig, axs


def fig03(r):
    A = r.scheme_a()
    T = r.weekly_test()
    Fw = r.forward("weekly")
    M = r.monthly_eval()
    I24 = M[M.indep & (M.month.dt.year == 2024)]
    sets = [(A.y, A.pred), (T.y, T.pred_X1), (Fw.y, Fw.pred_X1), (I24.y, I24.pred_X1)]
    lim = np.nanpercentile(np.concatenate([np.asarray(s[0], float) for s in sets]), 99.5) * 1.1
    lim = float(np.ceil(lim / 10) * 10)
    fig, axs = plt.subplots(2, 2, figsize=(W2, 6.9), sharex=True, sharey=True, gridspec_kw=dict(wspace=0.06, hspace=0.12))
    cols = ["#4E79A7", "#59A14F", "#F28E2B", "#B07AA1"]
    for ax, (y, p), L, c in zip(axs.flat, sets, "abcd", cols):
        st = scatter(ax, np.asarray(y, float), np.asarray(p, float), lim, c)
        say(f"   Fig03({L}) n={st['n']} R2={st['R2']:.2f} RMSE={st['RMSE']:.1f} bias={st['bias']:+.1f}  (for the caption)")
        panel(ax, L)
    for ax in axs[1]:
        ax.set_xlabel(f"Observed NO$_2$ ({UG})")
    for ax in axs[:, 0]:
        ax.set_ylabel(f"Estimated NO$_2$ ({UG})")
    save(fig, "Fig03_validation_scatter")


def fig04(r, g):
    """2 x 2 maps 2022-2025: (a) winter DJF, (b) summer JJA, (c) all months, (d) share of months inside the AOA"""
    from matplotlib.colors import Normalize
    files = sorted(glob.glob(os.path.join(C.BASE_DIR, "maps", "monthly", "DUNE_NO2_M*.tif")))
    files = [f for f in files if os.path.basename(f)[10:14] in ("2022", "2023", "2024", "2025")]
    if not files:
        say("   Fig04 skipped: no monthly maps (run 07c)")
        return
    acc = {k: [None, None] for k in ("all", "djf", "jja", "aoa")}

    def add(k, v):
        ok = np.isfinite(v)
        if acc[k][0] is None:
            acc[k] = [np.zeros_like(v), np.zeros_like(v)]
        acc[k][0][ok] += v[ok]
        acc[k][1][ok] += 1
    for f in files:
        mth = int(os.path.basename(f)[15:17])
        with rasterio.open(f) as src:
            no2, aoa = src.read(1).astype("float64"), src.read(4).astype("float64")
        add("all", no2)
        add("aoa", aoa)
        if mth in (12, 1, 2):
            add("djf", no2)
        if mth in (6, 7, 8):
            add("jja", no2)
    mean = {k: (np.where(v[1] > 0, v[0] / np.maximum(v[1], 1), np.nan) if v[0] is not None else None) for k, v in acc.items()}
    from matplotlib.colors import PowerNorm, LinearSegmentedColormap
    from matplotlib.ticker import MaxNLocator
    no2_cmap = LinearSegmentedColormap.from_list("dune_no2", ["#FFF7D6", "#FDE29A", "#FDB863", "#F4845F", "#D9536F", "#9E3D7A", "#5C2A6E"])
    aoa_cmap = LinearSegmentedColormap.from_list("dune_aoa", ["#F7FCF0", "#CCEBC5", "#7BCCC4", "#2B8CBE", "#084081"])
    cg = countries(g)
    vmin, vmax = (float(np.nanpercentile(mean["all"], q)) for q in (1, 99))
    norm = PowerNorm(0.6, vmin=max(vmin, 0), vmax=vmax)
    b = g.total_bounds
    asp = (b[2] - b[0] + 0.6) / (b[3] - b[1] + 0.6) * np.cos(np.deg2rad((b[1] + b[3]) / 2))
    W = W2
    lm, cg_ = 0.40, 0.06                         # left margin for latitude labels, gap between columns (in)
    pw = (W - lm - cg_ - 0.04) / 2               # panel width (in)
    ph = pw / asp
    cbh, gap, tick = 0.12, 0.06, 0.20            # colour-bar height, row gap, room for longitude labels
    H = 0.04 + 2 * ph + gap + tick + 0.06 + cbh + 0.34
    fig = plt.figure(figsize=(W, H))
    axs = np.empty((2, 2), dtype=object)
    for i in range(2):
        for j in range(2):
            x0 = lm + j * (pw + cg_)
            y0 = H - 0.04 - (i + 1) * ph - i * gap
            axs[i, j] = fig.add_axes([x0 / W, y0 / H, pw / W, ph / H])
    panels = [("djf", "a"), ("jja", "b"), ("all", "c"), ("aoa", "d")]
    ims = {}
    for ax, (k, L) in zip(axs.flat, panels):
        basemap(ax, g, cg, land=True)
        if mean[k] is not None:
            ims[k] = ax.imshow(mean[k], extent=raster_extent(), interpolation="nearest", zorder=1,
                               **(dict(cmap=aoa_cmap, vmin=0, vmax=1) if k == "aoa" else dict(cmap=no2_cmap, norm=norm)))
        borders(ax, g, cg)
        ax.text(0.015, 0.03, f"({L})", transform=ax.transAxes, fontsize=11, fontweight="bold", ha="left", va="bottom", zorder=12)
    for ax in axs[0]:
        ax.tick_params(labelbottom=False)
    for ax in axs[:, 1]:
        ax.tick_params(labelleft=False)
    ybar = (H - 0.04 - 2 * ph - gap - tick - 0.06 - cbh) / H
    no2_im = next((ims[k] for k in ("all", "djf", "jja") if k in ims), None)
    if no2_im is not None:
        x0 = lm + 0.06 * pw
        cax = fig.add_axes([x0 / W, ybar, 0.88 * pw / W, cbh / H])
        cb = fig.colorbar(no2_im, cax=cax, orientation="horizontal", extend="max")
        cb.set_ticks([t for t in MaxNLocator(6).tick_values(norm.vmin, norm.vmax) if norm.vmin <= t <= norm.vmax])
        cb.set_label(f"Surface NO$_2$ ({UG})", labelpad=2)
        cb.outline.set_linewidth(0.5)
    if "aoa" in ims:
        x0 = lm + pw + cg_ + 0.06 * pw
        cax2 = fig.add_axes([x0 / W, ybar, 0.88 * pw / W, cbh / H])
        cb2 = fig.colorbar(ims["aoa"], cax=cax2, orientation="horizontal")
        cb2.set_label("Share of months inside the AOA", labelpad=2)
        cb2.outline.set_linewidth(0.5)
    save(fig, "Fig04_maps_aoa")
    say(f"   Fig04: {len(files)} monthly maps 2022-2025 averaged; colour limit {vmax:.1f} ug/m3 (99th percentile)")


def fig05(r):
    import xgboost as xgb
    S1 = r.S1
    bst = xgb.Booster()
    bst.load_model(os.path.join(R.V2, f"stage1_{r.winner}.json"))
    feats = r.meta1["features"]
    X = r.train[feats].values.astype("float32")
    shap = np.abs(bst.predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]).mean(axis=0)
    s = pd.Series(shap, index=feats).sort_values(ascending=False).head(15)[::-1]
    fam = {"satellite": "#4E79A7", "weather": "#59A14F", "season": "#B07AA1",
           "emission proxies / surface": "#F28E2B", "fixed geography": "#A0A0A0"}
    fig, ax = plt.subplots(figsize=(W1, 4.6))
    ax.barh(range(len(s)), s.values, color=[fam[S1.family(f)] for f in s.index], edgecolor="black", lw=0.3, height=0.72)
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels([f.replace("_", " ").replace("NO2", "NO$_2$") for f in s.index])
    ax.set_xlabel(f"Mean |SHAP| ({UG})")
    ax.set_ylim(-0.6, len(s) - 0.4)
    ax.spines[["top", "right"]].set_visible(False)
    from matplotlib.patches import Patch
    present = [k for k in fam if k in {S1.family(f) for f in s.index}]
    ax.legend(handles=[Patch(fc=fam[k], ec="black", lw=0.3, label=k) for k in present],
              loc="upper center", bbox_to_anchor=(0.35, -0.13), ncol=2, fontsize=10, handlelength=1.2, columnspacing=0.8)
    save(fig, "Fig05_shap")


def fig06(r):
    """month by month on identical rows (as Table V); each panel's legend gives the R2 of that model's monthly values"""
    M = r.monthly_eval()
    Fm = r.forward("monthly")
    groups = [(M[M.own & (M.month.dt.year == 2024)], ["pred_X1", "pred_E", "pred_P", "pred_S1clim", "pred_persist"], "a"),
              (M[M.indep & (M.month.dt.year == 2024)], ["pred_X1", "pred_E", "pred_P", "pred_S1clim", "pred_persist"], "b"),
              (Fm, ["pred_X1", "pred_E", "pred_P", "pred_Proll", "pred_S1clim", "pred_persist"], "c")]
    short = {"pred_X1": "X1", "pred_E": "LSTM-E", "pred_P": "LSTM-P", "pred_Proll": "LSTM-P (rolling)",
             "pred_S1clim": "Stage 1 climatology", "pred_persist": "Persistence"}
    fig, axs = plt.subplots(3, 1, figsize=(W2, 7.4), sharex=True, gridspec_kw=dict(hspace=0.12))
    fig.subplots_adjust(right=0.70)
    MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for ax, (d, models, L) in zip(axs, groups):
        models = [m for m in models if m in d.columns]
        d = d.dropna(subset=["y"] + models)
        panel(ax, L)
        ax.grid(axis="y", color="#E5E5E5", lw=0.5, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylabel(f"NO$_2$ ({UG})")
        if not len(d):
            continue
        g = d.groupby(d.month.dt.month)
        hs, labs = [], []
        h, = ax.plot(g.y.mean().index, g.y.mean().values, color="black", marker="o", ms=4, lw=1.7, zorder=5)
        hs.append(h)
        labs.append(f"Observed (n = {len(d)})")
        for m in models:
            v = g[m].mean()
            h, = ax.plot(v.index, v.values, color=MCOL[m], marker="o", ms=2.6, lw=1.3, ls="--" if m == "pred_persist" else "-", zorder=4)
            r2 = 1 - np.sum((d.y - d[m]) ** 2) / np.sum((d.y - d.y.mean()) ** 2)
            hs.append(h)
            labs.append(f"{short[m]}, $R^2$ = {r2:.2f}".replace("-0", "\u22120"))
            say(f"   Fig06({L}) {short[m]} R2 = {r2:.2f} (n = {len(d)})")
        ax.legend(hs, labs, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=10, handlelength=1.8, borderaxespad=0,
                  labelspacing=0.35)
    axs[-1].set_xticks(range(1, 13))
    axs[-1].set_xticklabels(MON)
    axs[-1].set_xlim(0.6, 12.4)
    save(fig, "Fig06_month_by_month")


def fig07(r):
    S1 = r.S1
    Fw = r.forward("weekly").dropna(subset=["pred_X1"])
    FW = pd.read_csv(os.path.join(C.BASE_DIR, "features", "station_weekly_features.csv"), usecols=["Station", "week_start", S1.VCD,
                                                                                                    "raster_set"], parse_dates=["week_start"])
    FW = FW[FW.raster_set == "main"].rename(columns={"Station": "Anchor_Station"}).drop(columns="raster_set")
    Fw = Fw.merge(FW, on=["Anchor_Station", "week_start"], how="left")
    h = r.TW[(r.TW.pollutant == "NO2") & np.isfinite(r.TW.y) & (r.TW.week_start.dt.year == 2024)]
    h = h.merge(r.points[["Anchor_Station", "week_start", "pred_X1"]], on=["Anchor_Station", "week_start"], how="left")
    rows = []
    for t, a in Fw.assign(m=Fw.week_start.dt.month).groupby("Target_ID"):
        b = h[h.Target_ID == t].assign(m=lambda x: x.week_start.dt.month)
        mo = sorted(set(a.m) & set(b.m))
        if mo:
            a, b = a[a.m.isin(mo)], b[b.m.isin(mo)]
            rows.append(dict(obs=a.y.mean() - b.y.mean(), col=100 * (a[S1.VCD].mean() / b[S1.VCD].mean() - 1),
                             x1=a.pred_X1.mean() - b.pred_X1.mean()))
    D = pd.DataFrame(rows)
    if not len(D):
        say("   Fig07 skipped: no station with 2024 and 2026 data in the same months")
        return
    say(f"   Fig07: {len(D)} stations with common calendar months (for the text)")
    fig, axs = plt.subplots(1, 2, figsize=(W2, 3.3), gridspec_kw=dict(wspace=0.32))
    axs[0].scatter(D.col, D.obs, s=26, c="#4E79A7", edgecolors="black", linewidths=0.4, zorder=3)
    axs[0].axhline(0, color="black", lw=0.5)
    axs[0].axvline(0, color="black", lw=0.5)
    axs[0].set_xlabel("Change in TROPOMI column (%)")
    axs[0].set_ylabel(f"Observed change ({UG})")
    axs[1].scatter(D.x1, D.obs, s=26, c="#F28E2B", edgecolors="black", linewidths=0.4, zorder=3)
    lim = float(np.nanmax(np.abs(np.r_[D.x1, D.obs])) * 1.1)
    axs[1].plot([-lim, lim], [-lim, lim], color="black", lw=0.8)
    axs[1].axhline(0, color="black", lw=0.5)
    axs[1].axvline(0, color="black", lw=0.5)
    axs[1].set_xlim(-lim, lim)
    axs[1].set_ylim(-lim, lim)
    axs[1].set_aspect("equal")
    axs[1].set_xlabel(f"Change in X1 estimate ({UG})")
    axs[1].set_ylabel(f"Observed change ({UG})")
    for ax, L in zip(axs, "ab"):
        panel(ax, L)
    save(fig, "Fig07_forward_change")


def fig08():
    p = os.path.join(C.DIRS["tables"], "District_NO2_monthly_LSTMP.csv")
    if not os.path.exists(p):
        say("   Fig08 skipped: run 07d")
        return
    D = pd.read_csv(p, parse_dates=["month"])
    g = D.groupby("month")[["NO2_stage1", "NO2_LSTM_P", "NO2_LSTM_P_roll"]].mean()
    g = g[g.index >= "2022-01-01"]
    import matplotlib.dates as mdates
    C8 = {"s1": "#4E79A7", "p": "#F28E2B", "pr": "#B07AA1"}
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(W2, 4.9), gridspec_kw=dict(height_ratios=[1.6, 1], hspace=0.40))
    ax.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2024-01-01"), color="#EEF3F8", lw=0)
    ax.axvline(pd.Timestamp("2026-01-01"), color="#555555", lw=0.7, ls=":")
    ax.plot(g.index, g.NO2_stage1, color=C8["s1"], lw=1.8, marker="o", ms=2.5, label="Stage 1 (satellite)")
    ax.plot(g.index, g.NO2_LSTM_P, color=C8["p"], lw=1.5, marker="s", ms=2.3, label="LSTM-P (fixed baseline)")
    ax.plot(g.index, g.NO2_LSTM_P_roll, color=C8["pr"], lw=1.5, ls="--", marker="^", ms=2.5, label="LSTM-P (rolling baseline)")
    ax.set_ylabel(f"District mean NO$_2$ ({UG})")
    ax.set_xlim(g.index.min() - pd.Timedelta(days=15), g.index.max() + pd.Timedelta(days=15))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(4, 7, 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", color="#E5E5E5", lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01), borderaxespad=0)
    panel(ax, "a")
    # (b) annual mean difference LSTM-P minus Stage 1, with the values on the bars
    y = g.assign(yr=g.index.year).groupby("yr").mean()
    d = pd.DataFrame({"fixed": y.NO2_LSTM_P - y.NO2_stage1, "rolling": y.NO2_LSTM_P_roll - y.NO2_stage1})
    xs = np.arange(len(d))
    w = 0.36
    for k, (col, c) in enumerate((("fixed", C8["p"]), ("rolling", C8["pr"]))):
        bars = bx.bar(xs + (k - 0.5) * w, d[col].values, w, color=c, edgecolor="black", lw=0.4,
                      label=f"LSTM-P ({col} baseline) − Stage 1")
        for xb, v in zip(xs + (k - 0.5) * w, d[col].values):
            if np.isfinite(v):
                lab = "0.00" if abs(v) < 0.005 else f"{v:+.2f}".replace("-", "\u2212")
                bx.text(xb, v, lab, ha="center", va="bottom" if v >= 0 else "top", fontsize=10)
    lo, hi = np.nanmin(d.values), np.nanmax(d.values)
    pad = 0.35 * max(hi - lo, 0.05)
    bx.set_ylim(min(lo, 0) - pad, max(hi, 0) + pad)
    bx.axhline(0, color="black", lw=0.6)
    bx.set_xticks(xs)
    bx.set_xticklabels([str(v) for v in d.index])
    bx.set_ylabel(f"Difference ({UG})")
    bx.spines[["top", "right"]].set_visible(False)
    bx.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0), borderaxespad=0)
    panel(bx, "b")
    save(fig, "Fig08_district_projection")


def figS1_gru():
    """Supplementary Fig. S1: GRU minus LSTM (R2) with paired site-bootstrap 95 % intervals, from 07h (post hoc)"""
    p = os.path.join(C.BASE_DIR, "sensitivity", "gru", "gru_vs_lstm.csv")
    if not os.path.exists(p):
        say("   FigS1 skipped: run 07h_gru_sensitivity.py first")
        return
    D = pd.read_csv(p)
    D = D[D.role.isin(["E (estimation)", "P (projection)"])]
    order = ["2024 weekly, training-network sites", "2024 monthly, training-network sites", "2024 monthly, independent stations",
             "2026 weekly, forward (POST-HOC)", "2026 monthly, forward (POST-HOC)"]
    ylab = ["2024 weekly, 23 sites", "2024 monthly, 23 sites", "2024 monthly, independent", "2026 weekly (post hoc)",
            "2026 monthly (post hoc)"]
    sty = {"E (estimation)": ("GRU-E − LSTM-E", MCOL["pred_E"], -0.14), "P (projection)": ("GRU-P − LSTM-P", MCOL["pred_P"], 0.14)}
    fig, ax = plt.subplots(figsize=(W2, 3.4))
    ax.axvspan(-0.05, 0.05, color="#EEF3F8", lw=0, zorder=0)
    ax.axvline(0, color="black", lw=0.8, zorder=1)
    ax.axhline(2.5, color="#9D9D9D", lw=0.6, ls=":", zorder=1)
    for role, (lab, col, off) in sty.items():
        d = D[D.role == role].set_index("test").reindex(order)
        y = np.arange(len(order))[::-1] + off
        post = np.array(["POST-HOC" in t for t in order])
        ax.errorbar(d.dR2_GRU_minus_LSTM, y, xerr=[d.dR2_GRU_minus_LSTM - d.CI_lo, d.CI_hi - d.dR2_GRU_minus_LSTM],
                    fmt="none", ecolor=col, elinewidth=1.4, capsize=2.5, zorder=3)
        ax.scatter(d.dR2_GRU_minus_LSTM[~post], y[~post], s=34, color=col, edgecolors="black", linewidths=0.5, zorder=4, label=lab)
        ax.scatter(d.dR2_GRU_minus_LSTM[post], y[post], s=34, facecolors="white", edgecolors=col, linewidths=1.4, zorder=4)
    ax.set_yticks(np.arange(len(order))[::-1])
    ax.set_yticklabels(ylab)
    ax.set_xlabel("Difference in R$^2$ (GRU − LSTM)")
    lo, hi = float(D.CI_lo.min()), float(D.CI_hi.max())
    ax.set_xlim(min(lo, -0.1) - 0.03, max(hi, 0.1) + 0.03)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.spines[["top", "right"]].set_visible(False)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h, l = ax.get_legend_handles_labels()
    h += [Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor="#555555", markersize=6),
          Patch(fc="#EEF3F8", ec="none")]
    l += ["post hoc (2026)", "±0.05 criterion"]
    ax.legend(h, l, loc="lower center", bbox_to_anchor=(0.45, 1.01), ncol=4, borderaxespad=0, columnspacing=1.2, handletextpad=0.4)
    save(fig, "FigS1_gru_vs_lstm")


def main():
    style()
    say(f"=== Figures (step 08a) | {DPI} dpi, TIFF (LZW) + PNG, no in-figure titles | -> {FIG}")
    r = R.Results()
    g = gpd.read_file(C.GEOJSON).to_crs(4326)
    for name, f in (("Fig01", lambda: fig01(r, g)), ("Fig02", fig02), ("Fig03", lambda: fig03(r)),
                    ("Fig04", lambda: fig04(r, g)), ("Fig05", lambda: fig05(r)), ("Fig06", lambda: fig06(r)),
                    ("Fig07", lambda: fig07(r)), ("Fig08", fig08), ("FigS1", figS1_gru)):
        try:
            f()
        except Exception as ex:
            say(f"   {name} FAILED: {type(ex).__name__}: {ex}")
    p = os.path.join(C.DIRS["logs"], "figures_report.txt")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("\n".join(OUT))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
