"""
DUNE - shared loaders for the figure (08_figures.py) and table (08_tables.py) scripts.
Everything here only READS the frozen outputs of steps 06b-07g and rebuilds the evaluation rows with the
same definitions as the steps that produced the reported numbers (identical rows, same aggregation).
"""
import os, json, importlib.util
import numpy as np, pandas as pd
import dune_config as C

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(C.BASE_DIR, "model")
V2 = os.path.join(MODEL, "v2")
TAB = C.DIRS["tables"]
FWD = os.path.join(TAB, "forward")


def load_module(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(CODE_DIR, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Results:
    """lazy access to every table the paper uses"""

    def __init__(self):
        self.S1 = load_module("stage1", "07b_stage1.py")
        self.meta1 = json.load(open(os.path.join(V2, "stage1_meta.json")))
        self.winner = self.meta1["winner"]
        self._c = {}

    def _get(self, k, f):
        if k not in self._c:
            self._c[k] = f()
        return self._c[k]

    # ------------------------------------------------------------------ base tables
    @property
    def sites(self):
        return self._get("sites", lambda: pd.read_csv(os.path.join(TAB, "ground_sites.csv")))

    @property
    def stations(self):
        return self._get("stations", lambda: pd.read_csv(C.STATIONS_CSV))

    @property
    def TW(self):
        return self._get("TW", lambda: pd.read_csv(os.path.join(MODEL, "weekly_table.csv"), parse_dates=["week_start"]))

    @property
    def TM(self):
        def f():
            T = pd.read_csv(os.path.join(MODEL, "monthly_table.csv"), parse_dates=["month"])
            T = T[np.isfinite(T.y)].copy()
            T["network"] = T.network.astype(str)
            return T
        return self._get("TM", f)

    def _rows(self, period):
        """the Stage 1 row rule: finite target, NO2, valid 3x3 column and PBLH"""
        T = self.TW
        T = T[np.isfinite(T.y) & (T.pollutant == "NO2") & np.isfinite(T[self.S1.VCD]) & np.isfinite(T.PBLH)]
        return T[T.period == period].reset_index(drop=True)

    @property
    def train(self):
        return self._get("train", lambda: self._rows("train"))

    @property
    def test(self):
        return self._get("test", lambda: self._rows("test"))

    @property
    def train_sites(self):
        return set(self.train.Site_ID)

    @property
    def points(self):
        """Station x week: frozen predictions of every model (07d output)"""
        def f():
            P = pd.read_csv(os.path.join(V2, "stage2_point_weekly.csv"), parse_dates=["week_start"])
            return P.rename(columns={"Station": "Anchor_Station"})
        return self._get("points", f)

    @property
    def monthly_points(self):
        """monthly averages of the weekly predictions (same rule as 07b-07e: >= 90 % of days)"""
        def f():
            P = self.points
            out = None
            for c in [c for c in P.columns if c.startswith("pred_")]:
                m = self.S1.weekly_to_monthly(P[["Anchor_Station", "week_start", c]].rename(
                    columns={"Anchor_Station": "Station", c: "pred"})).rename(columns={"pred": c, "Station": "Anchor_Station"})
                out = m if out is None else out.merge(m, on=["Anchor_Station", "month"], how="outer")
            return out
        return self._get("mpoints", f)

    def indep_mask(self, T):
        return (T.pollutant == "NO2") & ~T.Site_ID.isin(self.train_sites) & T.network.str.contains("UAE_monthly", na=False)

    def own_mask(self, T):
        return (T.pollutant == "NO2") & T.Site_ID.isin(self.train_sites)

    # ------------------------------------------------------------------ evaluation frames
    def weekly_test(self):
        k = ["Anchor_Station", "week_start"]
        return self.test[["Target_ID", "Site_ID", "y"] + k].merge(self.points.drop(columns="raster_set"), on=k, how="left")

    def scheme_a(self):
        oof = pd.read_csv(os.path.join(V2, "oof_predictions.csv"), parse_dates=["week_start"])
        return oof[oof.key == f"A new station (leave one site out)|{self.winner}"].copy()

    def monthly_eval(self):
        M = self.TM.merge(self.monthly_points, on=["Anchor_Station", "month"], how="left")
        prev = self.TM[self.TM.month.dt.year == 2023].assign(m=lambda x: x.month.dt.month).groupby(["Target_ID", "m"]).y.mean()
        M["pred_persist"] = prev.reindex(pd.MultiIndex.from_arrays([M.Target_ID, M.month.dt.month])).values
        M["own"], M["indep"] = self.own_mask(M), self.indep_mask(M)
        return M

    def forward(self, res="weekly"):
        tcol = "week_start" if res == "weekly" else "month"
        p = os.path.join(FWD, f"ground_{res}_forward_{C.FORWARD_START[:4]}.csv")
        F = pd.read_csv(p, parse_dates=[tcol])
        F = F[F.pollutant == "NO2"].rename(columns={"NO2": "y"})
        F["Anchor_Station"] = F.Target_ID.map(self.sites.set_index("Target_ID").Anchor_Station)
        P = self.points.drop(columns="raster_set") if res == "weekly" else self.monthly_points
        F = F.merge(P, on=["Anchor_Station", tcol], how="left")
        if res == "monthly":
            hm = self.TM[(self.TM.pollutant == "NO2") & (self.TM.month.dt.year == 2024)]
            pm = hm.assign(m=hm.month.dt.month).groupby(["Target_ID", "m"]).y.mean()
            F["pred_persist"] = pm.reindex(pd.MultiIndex.from_arrays([F.Target_ID, F.month.dt.month])).values
        return F

    def metrics(self, d, c):
        m = self.S1.metrics(d.y, d[c])
        m.update(sites=d.Site_ID.nunique(), anomaly_r=self.S1.anomaly_r(d, c), **self.S1.boot_ci(d, c))
        return m
