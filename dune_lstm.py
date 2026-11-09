"""
DUNE- LSTM regressor in PyTorch (CPU), used by 07d_lstm.py.

Model:  x_1..x_T (T time steps, D features) -> one LSTM layer (H units) -> dropout -> linear -> y
Training: mean squared error, Adam with weight decay, gradient-norm clipping, mini-batches, early stopping
on a validation set of held-out SITES / SERIES (best weights kept; never on test data).
Reproducibility: fixed seeds, torch.use_deterministic_algorithms(True), a fixed number of CPU threads.
Inputs are standardised with the training mean / sd; a missing input becomes 0 (= the training mean) and
the model receives an explicit missing-value flag as a separate feature where that matters.
An Ensemble averages several seeds; the spread between members is kept as an uncertainty indicator.
Saved with torch.save as plain tensors + a JSON string (loads with weights_only=True).
"""
import json
import numpy as np
import torch
from torch import nn

THREADS = 4
torch.set_num_threads(THREADS)
torch.use_deterministic_algorithms(True)


class Net(nn.Module):
    def __init__(self, D, H, dropout):
        super().__init__()
        self.lstm = nn.LSTM(D, H, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(H, 1)

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.head(self.drop(o[:, -1])).squeeze(1)


def _fit_one(X, y, Xv, yv, H, dropout, wd, lr, epochs, batch, patience, seed):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    net = Net(X.shape[-1], H, dropout)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    Xvt, yvt = (torch.from_numpy(Xv), torch.from_numpy(yv)) if Xv is not None and len(yv) else (Xt, yt)
    best, best_state, best_ep, wait = np.inf, None, 0, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(yt), generator=g)
        for s in range(0, len(yt), batch):
            j = perm[s:s + batch]
            opt.zero_grad()
            loss = torch.mean((net(Xt[j]) - yt[j]) ** 2)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(torch.mean((_predict(net, Xvt) - yvt) ** 2))
        if vl < best - 1e-6:
            best, best_ep, wait = vl, ep + 1, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    net.best_epoch, net.best_val = best_ep, best
    return net


def _predict(net, Xt, batch=16384):
    with torch.no_grad():
        return torch.cat([net(Xt[s:s + batch]) for s in range(0, len(Xt), batch)]) if len(Xt) else torch.zeros(0)


class Ensemble:
    def __init__(self, feats, L, H, n_members=5, wd=1e-4, dropout=0.2, seed=0, target="log", step_days=7):
        self.feats, self.L, self.H, self.n, self.wd, self.dropout = list(feats), L, H, n_members, wd, dropout
        self.seed, self.target, self.step_days = seed, target, step_days
        self.members, self.mu, self.sd, self.smear = [], None, None, 1.0

    def scale(self, X):
        Z = (X - self.mu) / self.sd
        return np.where(np.isfinite(Z), Z, 0.0).astype("float32")

    def fit(self, X, y, Xv=None, yv=None, epochs=200, batch=64, lr=3e-3, patience=20):
        if not np.isfinite(y).all() or (yv is not None and not np.isfinite(yv).all()):
            raise ValueError("non-finite training target")
        flat = X.reshape(-1, X.shape[-1])
        mu, sd = np.nanmean(flat, axis=0), np.nanstd(flat, axis=0)
        self.mu = np.where(np.isfinite(mu), mu, 0.0)
        self.sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
        Z, Zv = self.scale(X), (self.scale(Xv) if Xv is not None else None)
        y32, yv32 = y.astype("float32"), (yv.astype("float32") if yv is not None else None)
        self.members = [_fit_one(Z, y32, Zv, yv32, self.H, self.dropout, self.wd, lr, epochs, batch, patience,
                                 self.seed + 101 * k) for k in range(self.n)]
        res = y - self.predict_raw(X)[0]
        self.smear = float(np.mean(np.exp(res))) if self.target == "log" else 1.0     # Duan's smearing
        return self

    def predict_raw(self, X):
        Zt = torch.from_numpy(self.scale(X))
        P = np.stack([_predict(m, Zt).numpy().astype(float) for m in self.members])
        return P.mean(axis=0), P.std(axis=0)

    def best_epochs(self):
        return [m.best_epoch for m in self.members]

    def save(self, path, extra=None):
        cfg = dict(feats=self.feats, L=self.L, H=self.H, n=self.n, wd=self.wd, dropout=self.dropout, seed=self.seed,
                   target=self.target, step_days=self.step_days, smear=self.smear, extra=extra or {},
                   torch=torch.__version__)
        torch.save({"cfg": json.dumps(cfg), "mu": torch.from_numpy(self.mu), "sd": torch.from_numpy(self.sd),
                    "members": [m.state_dict() for m in self.members]}, path)

    @classmethod
    def load(cls, path):
        d = torch.load(path, weights_only=True)
        c = json.loads(d["cfg"])
        e = cls(c["feats"], c["L"], c["H"], c["n"], c["wd"], c["dropout"], c["seed"], c["target"], c["step_days"])
        e.mu, e.sd, e.smear, e.extra = d["mu"].numpy(), d["sd"].numpy(), c["smear"], c["extra"]
        for sd_ in d["members"]:
            m = Net(len(e.feats), e.H, e.dropout)
            m.load_state_dict(sd_)
            m.eval()
            e.members.append(m)
        return e
