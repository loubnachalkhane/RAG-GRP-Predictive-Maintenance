"""
CalibRAG - controlled experimental pipeline (v2, C-MAPSS FD001-FD004 and N-CMAPSS DS01)
=====================================================================
Built to answer the reviewers' main objections:

  1. ONE pipeline for every configuration (same splits, preprocessing, seeds, cost
     function, decision rule). Only the component under study changes:
        A   deterministic GRU, intervene iff predicted RUL <= H
        A+m deterministic GRU + safety margin tuned on validation engines (strong baseline)
        B   quantile GRU, raw quantiles -> failure probability F
        C   B + temperature scaling + split conformal
        D   C + RAG with ONE fixed fusion weight, no gating      (weight tuned on validation)
        E   C + RAG, confidence-conditioned fusion w(s) with gates s_min, w_max, band
                                                                  (params tuned on validation)
        MCD / MCD+CP   MC-Dropout baseline, raw and with the same split conformal
        Ens / Ens+CP   Deep-Ensemble baseline, raw and with the same split conformal
     B, C, D, E share the SAME trained quantile network; A, A+m, MCD, Ens[0] share the
     SAME deterministic network.
  2. No test labels are used for training, early stopping, calibration or tuning.
     Training engines are split BY ENGINE into train / val / cal_ts / cal_cp.
  3. Only ONE fusion formulation exists: F_fused = (1-w) F_sensor + w F_rag with
     w = w_max * clip((s - s_min)/(1 - s_min), 0, 1) inside the gates. (No Bayesian/PoE variant.)
  4. "When does RAG help?": every activation is logged (F_sensor, distance to threshold,
     uncertainty, retrieval similarity, w, F_fused, decision before/after, outcome) and
     classified unchanged / beneficial / harmful; plus an (s_min, w_max) sensitivity grid.
  5. Robustness: sensor noise is applied to the final pipeline (same linear fusion) and the
     degradation of sensor-only (C) vs CalibRAG (E) is measured separately.
  6. Cost sensitivity over Cf/Cm replaces any single-cost headline claim.

IMPORTANT caveat about RAG: the query text is built from the sensor model's own q50
(+ operating condition), and F_rag is read from knowledge-base entries. RAG can therefore only
add information that is stored in the knowledge base. Document how the KB values
(f_prob_lo / f_prob_hi) were obtained; if they were written or tuned using the same
dataset's test engines this is leakage.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# CONFIG
# ============================================================

SENSORS = ["s_2", "s_3", "s_4", "s_7", "s_8", "s_9", "s_11",
           "s_12", "s_13", "s_14", "s_15", "s_17", "s_20", "s_21"]
OS_COLS = ["os_1", "os_2", "os_3"]
COLS = ["unit", "cycle"] + OS_COLS + [f"s_{i}" for i in range(1, 22)]
PIPELINE_VERSION = "v4.1 (v4 + three arms: sensor-only / retrieval-only / sensor+retrieval)"
DEFAULT_H = {"FD001": 30, "FD002": 30, "FD003": 30, "FD004": 25, "NCMAPSS": 50, "CAC": 48}
Z80 = float(norm.ppf(0.90))  # q10..q90 spans +/- Z80 sigma


@dataclass
class Cfg:
    dataset: str = "FD001"
    data_dir: str = "/data/cmapss"
    out_dir: str = "/data/results"
    kb_path: Optional[str] = "/data/kb/kb_{dataset}.json"   # None -> skip configs D/E
    ncmapss_path: str = "/data/ncmapss/N-CMAPSS_DS01-005.h5"  # used when dataset == "NCMAPSS"
    nc_downsample: int = 50            # keep every 50th second (1 Hz -> 0.02 Hz), as in the final notebook
    seed: int = 0

    # data
    window: int = 30
    max_rul: int = 125
    cap_test_rul: bool = True

    # decision problem
    H: Optional[int] = None
    Cm: float = 1.0
    Cf: float = 10.0
    cost_ratios: Tuple[float, ...] = (5.0, 10.0, 20.0, 50.0)   # Cf values for the cost sweep (Cm = 1)

    # engine-level split of the TRAINING engines
    frac_val: float = 0.10
    frac_cal_ts: float = 0.10
    frac_cal_cp: float = 0.20

    # model / training
    hidden: int = 64
    layers: int = 2
    dropout: float = 0.2
    epochs: int = 50
    batch: int = 256
    lr: float = 1e-3
    target_scale: float = 1.0          # train on RUL / target_scale (e.g. 720 for the CAC); outputs are rescaled to hours
    patience: int = 0                  # early stopping on the validation loss (epochs without improvement); 0 = off
    weight_decay: float = 0.0          # Adam L2 penalty
    lr_step: int = 20                  # StepLR period (epochs); the LR is halved every lr_step epochs
    quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9)

    # calibration
    T_max: float = 1.5
    alpha: float = 0.20
    crc_deltas: Tuple[float, ...] = (0.05, 0.10, 0.20)   # target missed-failure rates for the risk-controlled threshold (all reported)
    floor_qhat: bool = False           # allow conformal to SHRINK intervals when the model over-covers

    # RAG
    embed_model: str = "all-MiniLM-L6-v2"
    top_k: int = 3
    use_knn: bool = True               # independent historical-evidence retrieval over TRAIN windows
    knn_k: int = 50
    local_k: int = 200                 # neighbours used by the retrieval-based local conformal (C-local)
    tag: str = ""                      # appended to output file names (used by the CAC folds)

    # baselines
    run_baselines: bool = True
    ens_size: int = 5
    mc_passes: int = 30

    # robustness (noise std in MinMax-scaled feature units, i.e. fraction of the sensor range)
    run_robustness: bool = True
    noise_levels: Tuple[float, ...] = (0.0, 0.05, 0.10, 0.20, 0.30)
    noise_reps: int = 5

    device: str = "cpu"

    def horizon(self) -> int:
        return self.H if self.H is not None else DEFAULT_H[self.dataset]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# DATA
# ============================================================

def load_cmapss(data_dir: str, dataset: str):
    def rd(fname, names):
        return pd.read_csv(os.path.join(data_dir, fname), sep=r"\s+", header=None, names=names)
    return (rd(f"train_{dataset}.txt", COLS), rd(f"test_{dataset}.txt", COLS),
            rd(f"RUL_{dataset}.txt", ["RUL"]))


def add_rul(df: pd.DataFrame, max_rul: int) -> pd.DataFrame:
    df = df.copy()
    df["RUL"] = (df.groupby("unit")["cycle"].transform("max") - df["cycle"]).clip(upper=max_rul)
    return df


def split_units(units, seed: int, cfg: Cfg) -> Dict[str, np.ndarray]:
    u = np.array(sorted(units))
    np.random.RandomState(seed).shuffle(u)
    n = len(u)
    n_val = max(1, int(round(cfg.frac_val * n)))
    n_ts = max(1, int(round(cfg.frac_cal_ts * n)))
    n_cp = max(1, int(round(cfg.frac_cal_cp * n)))
    n_tr = n - n_val - n_ts - n_cp
    assert n_tr > 0, "not enough engines for the requested split"
    return {"train": u[:n_tr], "val": u[n_tr:n_tr + n_val],
            "cal_ts": u[n_tr + n_val:n_tr + n_val + n_ts],
            "cal_cp": u[n_tr + n_val + n_ts:]}


class Preprocessor:
    """MinMax scaling fitted on TRAIN-split engines only (per operating condition for FD002/FD004).
    transform() also adds a 'cond' column (k-means operating condition, 0 for single-condition sets)."""

    def __init__(self, dataset: str):
        self.multi = dataset in ("FD002", "FD004")
        self.km = None
        self.scalers: Dict[int, MinMaxScaler] = {}

    def fit(self, df: pd.DataFrame):
        if self.multi:
            self.km = KMeans(n_clusters=6, random_state=42, n_init=10).fit(df[OS_COLS].values)
            cond = self.km.predict(df[OS_COLS].values)
            for c in range(6):
                m = cond == c
                if m.any():
                    self.scalers[c] = MinMaxScaler().fit(df.loc[m, SENSORS].values)
        else:
            self.scalers[0] = MinMaxScaler().fit(df[SENSORS].values)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if not self.multi:
            df[SENSORS] = self.scalers[0].transform(df[SENSORS].values)
            df["cond"] = 0
            return df
        cond = self.km.predict(df[OS_COLS].values)
        out = df[SENSORS].values.astype(np.float64).copy()
        for c, sc in self.scalers.items():
            m = cond == c
            if m.any():
                out[m] = sc.transform(df.loc[m, SENSORS].values)
        df[SENSORS] = out
        df["cond"] = cond
        return df


def make_windows(df: pd.DataFrame, w: int):
    """All full windows (stride 1). Returns X, y(RUL), unit ids, operating condition at window end."""
    X, y, u, c = [], [], [], []
    for unit, g in df.groupby("unit"):
        g = g.sort_values("cycle")
        s = g[SENSORS].values.astype(np.float32)
        if len(s) < w:
            continue
        win = np.lib.stride_tricks.sliding_window_view(s, (w, s.shape[1]))[:, 0]
        X.append(np.ascontiguousarray(win))
        y.append(g["RUL"].values.astype(np.float32)[w - 1:])
        u.append(np.full(len(win), unit))
        c.append(g["cond"].values[w - 1:])
    return np.concatenate(X), np.concatenate(y), np.concatenate(u), np.concatenate(c)


def make_test_windows(df: pd.DataFrame, w: int):
    """Last window of every test engine (left zero-padded if shorter than w)."""
    X, units, conds = [], [], []
    for unit, g in df.groupby("unit"):
        g = g.sort_values("cycle")
        s = g[SENSORS].values.astype(np.float32)
        if len(s) < w:
            s = np.vstack([np.zeros((w - len(s), s.shape[1]), np.float32), s])
        X.append(s[-w:])
        units.append(unit)
        conds.append(int(g["cond"].values[-1]))
    return np.stack(X), np.array(units), np.array(conds)


def load_ncmapss_bundle(cfg: Cfg):
    """N-CMAPSS DS01-005 (flat HDF5 keys X_s_dev, A_dev, Y_dev, X_s_test, A_test, Y_test).

    Protocol (6 dev units, 3 test units):
      * dev units are shuffled with the seed: 4 train, 1 validation, 1 calibration unit
      * the calibration unit's flight cycles are split by parity: even cycles -> temperature
        scaling, odd cycles -> split conformal (no window is shared between the two)
      * StandardScaler is fitted on the TRAIN units only
      * rows are down-sampled per unit (every nc_downsample-th sample), then windowed (stride 1)
      * test samples: ONE window per flight cycle of each test unit (the last window of that
        cycle), i.e. a decision after every flight - not thousands of near-duplicate windows
    The windows inside one unit are strongly correlated, so the effective sample size of the
    calibration and test sets is about the number of flight cycles, not the number of windows.
    """
    import h5py
    from sklearn.preprocessing import StandardScaler

    with h5py.File(cfg.ncmapss_path, "r") as f:
        Xd, Ad, Yd = f["X_s_dev"][:], f["A_dev"][:], f["Y_dev"][:].ravel()
        Xt, At, Yt = f["X_s_test"][:], f["A_test"][:], f["Y_test"][:].ravel()

    def per_unit(X, A, Y):
        units, cyc = A[:, 0].astype(int), A[:, 1].astype(int)
        out = {}
        for u in np.unique(units):
            idx = np.where(units == u)[0][::cfg.nc_downsample]
            out[int(u)] = (X[idx].astype(np.float32),
                           np.minimum(Y[idx], cfg.max_rul).astype(np.float32), cyc[idx])
        return out

    dev, tst = per_unit(Xd, Ad, Yd), per_unit(Xt, At, Yt)
    del Xd, Xt
    order = [int(u) for u in np.random.RandomState(cfg.seed).permutation(sorted(dev))]
    val_u, cal_u, train_u = order[:1], order[1:2], order[2:]
    scaler = StandardScaler().fit(np.concatenate([dev[u][0] for u in train_u]))
    w = cfg.window

    def windows(seq):
        X, y, c = seq
        if len(X) < w:
            return None
        X = scaler.transform(X).astype(np.float32)
        win = np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(X, (w, X.shape[1]))[:, 0])
        return win, y[w - 1:], c[w - 1:]

    def collect(us, src, keep=None):
        Xs, ys, ids, cs = [], [], [], []
        for u in us:
            r = windows(src[u])
            if r is None:
                continue
            X, y, c = r
            if keep is not None:
                m = keep(c)
                X, y, c = X[m], y[m], c[m]
            Xs.append(X); ys.append(y); ids.append(np.full(len(y), u)); cs.append(c)
        return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ids), np.concatenate(cs)

    def pack(t):  # (X, y, unit, cycle) -> (X, y, unit, cond=0), same layout as the C-MAPSS windows
        return t[0], t[1], t[2], np.zeros(len(t[1]), dtype=int)

    win = {"train": pack(collect(train_u, dev)), "val": pack(collect(val_u, dev)),
           "cal_ts": pack(collect(cal_u, dev, lambda c: c % 2 == 0)),
           "cal_cp": pack(collect(cal_u, dev, lambda c: c % 2 == 1))}

    Xs, ys, us_, cs = [], [], [], []
    for u in sorted(tst):
        r = windows(tst[u])
        if r is None:
            continue
        X, y, c = r
        for cv in np.unique(c):
            i = np.where(c == cv)[0][-1]
            Xs.append(X[i]); ys.append(y[i]); us_.append(u); cs.append(cv)
    X_te, y_te = np.stack(Xs), np.asarray(ys, dtype=np.float32)
    splits = {"train": np.array(train_u), "val": np.array(val_u),
              "cal_ts": np.array(cal_u), "cal_cp": np.array(cal_u)}
    return win, X_te, y_te, np.array(us_), np.array(cs), np.zeros(len(y_te), dtype=int), splits


# ============================================================
# MODELS
# ============================================================

class QuantileGRU(nn.Module):
    """GRU + one small MLP head per quantile, made monotone with cumulative softplus gaps."""

    def __init__(self, input_size, hidden=64, layers=2, dropout=0.2, quantiles=(0.1, 0.5, 0.9)):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1)) for _ in quantiles])

    def forward(self, x):
        out, _ = self.gru(x)
        h = self.drop(out[:, -1, :])
        p = torch.cat([head(h) for head in self.heads], dim=1)
        base = p[:, 0:1]
        return torch.cat([base, base + torch.nn.functional.softplus(p[:, 1:])], dim=1)


class DeterministicGRU(nn.Module):
    """Same backbone as QuantileGRU, single point-estimate head."""

    def __init__(self, input_size, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(self.drop(out[:, -1, :]))


def pinball_loss(pred, target, quantiles):
    total = 0.0
    for i, q in enumerate(quantiles):
        e = target - pred[:, i]
        total = total + torch.max(q * e, (q - 1) * e).mean()
    return total


def mse_loss(pred, target):
    return torch.nn.functional.mse_loss(pred[:, 0], target)


class ScaledOut(nn.Module):
    """Wraps a network trained on targets divided by `scale`; its output is back in the original units (hours)."""

    def __init__(self, inner, scale: float):
        super().__init__()
        self.m = inner
        self.register_buffer("scale", torch.tensor(float(scale)))

    def forward(self, x):
        return self.m(x) * self.scale


def fit_model(model, loss_fn, Xtr, ytr, Xva, yva, cfg: Cfg, seed: int, tag: str):
    dev = cfg.device
    model.to(dev)
    sc = float(cfg.target_scale)
    ytr_s, yva_s = (ytr / sc).astype(np.float32), (yva / sc).astype(np.float32)
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr_s)),
                        batch_size=cfg.batch, shuffle=True, generator=g)
    Xv, yv = torch.from_numpy(Xva).to(dev), torch.from_numpy(yva_s).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, cfg.lr_step), gamma=0.5)
    best, best_state, bad = float("inf"), None, 0
    for ep in range(cfg.epochs):
        model.train()
        run, nb = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run, nb = run + float(loss.item()), nb + 1
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xv), yv).item()
        if vl < best:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if (ep + 1) % 10 == 0:
            print(f"  [{tag}] epoch {ep + 1}/{cfg.epochs} train={run / max(nb, 1):.4f} val_loss={vl:.4f} "
                  f"best={best:.4f}", flush=True)
        if cfg.patience > 0 and bad >= cfg.patience:
            print(f"  [{tag}] early stop at epoch {ep + 1} (no val improvement for {cfg.patience} epochs; "
                  f"best={best:.4f})", flush=True)
            break
    model.load_state_dict(best_state)
    model.eval()
    return ScaledOut(model, sc).to(dev).eval() if sc != 1.0 else model


@torch.no_grad()
def predict(model, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    model.eval()
    return np.concatenate([model(torch.from_numpy(X[i:i + 4096]).to(device)).cpu().numpy()
                           for i in range(0, len(X), 4096)])


@torch.no_grad()
def mc_predict(model, X: np.ndarray, passes: int, device: str, seed: int):
    """MC-Dropout: dropout stays active at inference; returns predictive mean and std."""
    torch.manual_seed(seed)
    model.train()
    runs = [np.concatenate([model(torch.from_numpy(X[i:i + 4096]).to(device)).cpu().numpy()[:, 0]
                            for i in range(0, len(X), 4096)]) for _ in range(passes)]
    model.eval()
    A = np.stack(runs)
    return A.mean(0), A.std(0)


def ens_predict(models, X: np.ndarray, device: str):
    A = np.stack([predict(m, X, device)[:, 0] for m in models])
    return A.mean(0), A.std(0)


# ============================================================
# CALIBRATION (fitted on held-out TRAIN engines)
# ============================================================

def fit_temperature(q10, q50, q90, y, T_max: float, target=0.80, lr=0.05, steps=500) -> float:
    q10, q50, q90, y = [torch.tensor(a, dtype=torch.float32) for a in (q10, q50, q90, y)]
    log_T = nn.Parameter(torch.zeros(1))
    opt = torch.optim.Adam([log_T], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        T = torch.exp(log_T)
        lo, hi = q50 - T * (q50 - q10), q50 + T * (q90 - q50)
        soft = (torch.sigmoid((y - lo) / 0.5) * torch.sigmoid((hi - y) / 0.5)).mean()
        ((soft - target) ** 2).backward()
        opt.step()
    return float(min(torch.exp(log_T).item(), T_max))


def apply_temperature(q10, q50, q90, T):
    return q50 - T * (q50 - q10), q50 + T * (q90 - q50)


def fit_conformal(lo, hi, y, alpha: float, floor: bool) -> float:
    scores = np.maximum(lo - y, y - hi)
    n = len(scores)
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    q_hat = float(np.quantile(scores, level))
    return max(q_hat, 0.0) if floor else q_hat


# ============================================================
# DECISION RULE + METRICS (identical for every configuration)
# ============================================================

def failure_prob(q10, q50, q90, H) -> np.ndarray:
    """P(RUL <= H) under a Gaussian whose q10..q90 span matches the interval (closed form)."""
    sigma = np.maximum((q90 - q10) / (2 * Z80), 1e-6)
    return norm.cdf((H - q50) / sigma)


def decide(F, Cm, Cf) -> np.ndarray:
    return F * Cf > Cm


def crc_threshold(F: np.ndarray, critical: np.ndarray, delta: float) -> float:
    """Conformal risk control for the alarm threshold (alarm iff F > lam).
    Loss on a critical calibration window (RUL <= H) is 1[no alarm] = 1[F <= lam]; it is non-increasing as lam
    decreases. Returns the LARGEST lam (fewest alarms) with  n/(n+1) * mean loss + 1/(n+1) <= delta,
    n = number of critical calibration windows. If delta is unattainable with n windows, returns -1.0 (always alarm)."""
    fc = np.sort(np.asarray(F, dtype=float)[np.asarray(critical, dtype=bool)])
    n = len(fc)
    k = int(np.floor(delta * (n + 1) - 1e-9)) - 1          # number of misses we may afford on the calibration set
    if n == 0 or k < 0:
        return -1.0
    return float(np.nextafter(fc[min(k, n - 1)], -np.inf))    # alarm iff F >= fc[k]: at most k calibration misses


def skill_gate(P_list, y_list, u_list, quantiles, y_train, n_boot: int = 2000, seed: int = 0) -> dict:
    """Skill gate: does the sensor model beat a trivial climatological forecast on held-out engines?
    Both forecasts are the 3 quantiles (q10, q50, q90); score = mean pinball loss. The climatological forecast uses the
    quantiles of the TRAINING RUL. skill = 1 - L_model / L_clim, with a bootstrap over held-out engines (clusters).
    The gate passes iff the lower 95% bound of the skill is above 0. Uses no test data."""
    P, y, u = np.concatenate(P_list), np.concatenate(y_list), np.concatenate(u_list)
    qs = np.asarray(quantiles, dtype=float)
    clim = np.quantile(np.asarray(y_train, dtype=float), qs)

    def pin(pred):
        e = y[:, None] - pred
        return np.maximum(qs * e, (qs - 1) * e).mean(axis=1)

    lm, lc = pin(P), pin(np.tile(clim, (len(y), 1)))
    ids = np.unique(u)
    sm = np.array([lm[u == i].sum() for i in ids])
    sc = np.array([lc[u == i].sum() for i in ids])
    skill = 1.0 - sm.sum() / max(sc.sum(), 1e-12)
    rng = np.random.RandomState(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.randint(0, len(ids), len(ids))
        boots.append(1.0 - sm[pick].sum() / max(sc[pick].sum(), 1e-12))
    lo = float(np.percentile(boots, 2.5))
    return {"skill": float(skill), "skill_lo": lo, "passed": bool(lo > 0.0), "n_engines": int(len(ids))}


def cost_vector(interv, y, H, Cm, Cf) -> np.ndarray:
    return np.where(interv, Cm, np.where(y <= H, Cf, 0.0))


def cost_stats(intervene, y, H, Cm, Cf) -> dict:
    fail = (~intervene) & (y <= H)
    return {"Cost": float(Cm * intervene.sum() + Cf * fail.sum()),
            "Failures": int(fail.sum()), "Interventions": int(intervene.sum()),
            "Unnecessary": int((intervene & (y > H)).sum())}


def gaussian_crps(y, mu, sigma) -> float:
    z = (y - mu) / sigma
    return float(np.mean(sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))))


def point_metrics(y, pred) -> dict:
    return {"RMSE": float(np.sqrt(np.mean((y - pred) ** 2))), "MAE": float(np.mean(np.abs(y - pred)))}


def prob_metrics(y, lo, mid, hi, H, quantiles=(0.1, 0.5, 0.9)) -> dict:
    F = failure_prob(lo, mid, hi, H)
    sigma = np.maximum((hi - lo) / (2 * Z80), 1e-6)
    qce = np.mean([abs(t - np.mean(y <= q)) for t, q in zip(quantiles, (lo, mid, hi))])
    return {"Coverage80": float(np.mean((y >= lo) & (y <= hi))) * 100, "QCE": float(qce),
            "PI_Width": float(np.mean(hi - lo)),
            "Brier": float(np.mean((F - (y <= H).astype(float)) ** 2)),
            "CRPS": gaussian_crps(y, mid, sigma)}


# ============================================================
# RAG: knowledge base, retrieval, fusion (single linear formulation)
# ============================================================

def build_query(q50: float, os_cond: Optional[int], dataset: str) -> str:
    """Query text, identical to the original notebook."""
    if q50 > 80:
        state = "healthy RUL above 80 cycles early degradation low failure risk"
    elif q50 > 60:
        state = "moderate degradation RUL 60 to 80 cycles low to moderate failure risk"
    elif q50 > 40:
        state = "advanced degradation RUL 40 to 60 cycles moderate failure probability"
    elif q50 > 20:
        state = "critical degradation RUL 20 to 40 cycles high failure probability urgent"
    else:
        state = "near failure RUL below 20 cycles very high failure probability imminent"
    q = f"Turbofan {dataset} predicted RUL {int(q50)} cycles. {state}."
    if os_cond is not None and os_cond >= 0 and dataset in ("FD002", "FD004"):
        cm = {0: "sea level", 1: "high altitude cruise", 2: "high power takeoff",
              3: "low power descent", 4: "intermediate climb", 5: "high speed cruise"}
        q += f" Condition: {cm.get(int(os_cond), 'unknown')}."
    if dataset == "FD003":
        q += " Single condition dual fault HPC fan blade simultaneous degradation."
    if dataset == "FD004":
        q += " Dual fault HPC fan blade simultaneous degradation six conditions."
    return q


class Retriever:
    """Sentence-embedding retrieval over a JSON knowledge base
    (list of dicts with 'text', 'f_prob_lo', 'f_prob_hi'). Retrieval is computed once per
    distinct query; gating is applied later in fuse(), so D/E/sensitivity share the same retrieval."""

    def __init__(self, kb: List[dict], dataset: str, model_name: str, k: int = 3, encoder=None, query_fn=None):
        if encoder is None:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer(model_name)
        self.enc, self.dataset, self.k, self.query_fn = encoder, dataset, k, query_fn
        self.E = self._norm(self.enc.encode([d["text"] for d in kb], batch_size=32,
                                            show_progress_bar=False))
        self.lo = np.array([d["f_prob_lo"] for d in kb], dtype=np.float64)
        self.hi = np.array([d["f_prob_hi"] for d in kb], dtype=np.float64)
        self.cache: Dict[str, Tuple[float, float]] = {}

    @staticmethod
    def _norm(a):
        a = np.asarray(a, dtype=np.float64)
        return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)

    def retrieve(self, q50, cond):
        multi = self.dataset in ("FD002", "FD004")
        fn = self.query_fn
        texts = [fn(float(q), int(c)) if fn is not None else
                 build_query(float(q), int(c) if (multi and c >= 0) else None, self.dataset)
                 for q, c in zip(np.asarray(q50), np.asarray(cond))]
        new = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if new:
            S = self._norm(self.enc.encode(new, batch_size=64, show_progress_bar=False)) @ self.E.T
            for t, s in zip(new, S):
                idx = np.argsort(s)[::-1][:self.k]
                sims = s[idx]
                w = sims / (sims.sum() + 1e-8)
                self.cache[t] = (float(0.5 * (np.dot(w, self.lo[idx]) + np.dot(w, self.hi[idx]))),
                                 float(sims.mean()))
        res = np.array([self.cache[t] for t in texts], dtype=np.float64)
        return res[:, 0], res[:, 1]


def window_features(X: np.ndarray) -> np.ndarray:
    """Compact descriptor of a window: last time step + window mean (2 x n_sensors)."""
    return np.concatenate([X[:, -1, :], X.mean(axis=1)], axis=1).astype(np.float32)


class KNNRetriever:
    """Independent historical evidence: nearest TRAIN-split windows in sensor space.
    F_rag = fraction of the k neighbours whose RUL <= H (an empirical failure frequency);
    sim = 1 / (1 + mean neighbour distance). Every neighbour is traceable to a training window
    (index returned by retrieve()). Unlike the text knowledge base, this evidence does not depend
    on the sensor model's own prediction."""

    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, H: float, k: int = 50):
        from sklearn.neighbors import NearestNeighbors
        self.y, self.H, self.k = np.asarray(y_train), H, k
        self.nn = NearestNeighbors(n_neighbors=min(k, len(X_train)), n_jobs=-1).fit(window_features(X_train))

    def retrieve(self, X: np.ndarray):
        d, idx = self.nn.kneighbors(window_features(X))
        return (self.y[idx] <= self.H).mean(axis=1), 1.0 / (1.0 + d.mean(axis=1)), idx


def fuse(F_s, f_rag, sim, mode: str, p: dict):
    """F_fused = (1-w) F_sensor + w F_rag.
    mode 'D': w = p['w'] for every sample (no gating).
    mode 'E': w = w_max * clip((sim-s_min)/(1-s_min),0,1) if F_sensor in [band_lo, band_hi]
              and sim >= s_min, else 0."""
    if mode == "D":
        w = np.full(F_s.shape, float(p["w"]))
    else:
        gate = (F_s >= p["band_lo"]) & (F_s <= p["band_hi"]) & (sim >= p["s_min"])
        scale = np.clip((sim - p["s_min"]) / (1.0 - p["s_min"] + 1e-8), 0.0, 1.0)
        w = np.where(gate, p["w_max"] * scale, 0.0)
    return (1.0 - w) * F_s + w * f_rag, w


def _total_cost(F, y, H, Cm, Cf) -> float:
    return float(cost_vector(decide(F, Cm, Cf), y, H, Cm, Cf).sum())


W_GRID = (0.1, 0.2, 0.3, 0.5, 0.7)


def tune_D(F_s, f_rag, sim, y, H, Cm, Cf) -> dict:
    """Pick the single fixed weight on VALIDATION cost (w=0 means RAG does not help)."""
    best_w, best_c = 0.0, _total_cost(F_s, y, H, Cm, Cf)
    for w in W_GRID:
        c = _total_cost(fuse(F_s, f_rag, sim, "D", {"w": w})[0], y, H, Cm, Cf)
        if c < best_c:
            best_w, best_c = w, c
    return {"w": best_w}


def tune_E(F_s, f_rag, sim, y, H, Cm, Cf, s_grid) -> dict:
    """Grid-search the gates on the tuning windows (val + cal_ts engines, never the test set).
    s_min values come from the quantiles of the observed similarity (s_grid), so the grid is
    meaningful whatever the similarity scale. If nothing beats sensor-only, w_max = 0 (RAG off)."""
    base = _total_cost(F_s, y, H, Cm, Cf)
    best = {"w_max": 0.0, "s_min": float(s_grid[0]), "band_lo": 0.05, "band_hi": 0.95}
    best_c = base
    for w_max in W_GRID:
        for s_min in s_grid:
            for lo in (0.02, 0.05, 0.10):
                for hi in (0.5, 0.8, 0.95):
                    p = {"w_max": w_max, "s_min": float(s_min), "band_lo": lo, "band_hi": hi}
                    c = _total_cost(fuse(F_s, f_rag, sim, "E", p)[0], y, H, Cm, Cf)
                    if c < best_c:
                        best, best_c = p, c
    return best



# ============================================================
# LEARNED FUSION INSIDE THE CALIBRATION (new in v3)
# ============================================================

def _logit(p, eps: float = 0.02):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


class LearnedFusion:
    """P(RUL <= H) = sigmoid(a + b * logit(F_sensor) [+ c * logit(f_evidence)]), fitted on held-out engines.
    One linear fusion that is ITSELF the calibration step: it replaces the hand-set weight w and the gates.
    With no evidence column it is a Platt recalibration of F_sensor (control: 'C+platt'); the difference
    L-kNN vs C+platt is the value added by the retrieved evidence. c near 0 means RAG adds nothing.

    SAFEGUARD (new): the fit is cross-validated over blocks of held-out engines / flight cycles. If the
    out-of-fold Brier score is not better than that of the uncorrected F_sensor, the fusion is NOT used and
    the config returns F_sensor unchanged (used=False). This prevents a recalibration learned on a few
    held-out units from degrading a model that was already calibrated (seen on N-CMAPSS)."""

    def __init__(self, use_evidence: bool, use_sensor: bool = True):
        # use_sensor=False -> RETRIEVAL-ONLY arm: P = sigmoid(a + c*logit(f_evidence)); the sensor probability is
        # ignored. The CV safeguard is switched off for that arm (falling back to F_sensor would defeat its purpose).
        self.use_evidence, self.use_sensor, self.model, self.used = use_evidence, use_sensor, None, True
        self.cv = {"cv_brier_identity": float("nan"), "cv_brier_model": float("nan")}

    def _x(self, F, f=None):
        cols = [_logit(F, 1e-4)] if self.use_sensor else []
        if self.use_evidence:
            cols.append(_logit(f))
        return np.stack(cols, axis=1)

    def _new(self):
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(C=1.0, max_iter=1000)

    def fit(self, F, f, y_bin, groups=None, min_gain: float = 0.0):
        if len(np.unique(y_bin)) < 2:
            return None                       # nothing to learn from (no positives or no negatives)
        X = self._x(F, f)
        self.model = self._new().fit(X, y_bin)
        if groups is not None and len(np.unique(groups)) >= 3:
            from sklearn.model_selection import GroupKFold
            oof = np.full(len(y_bin), np.nan)
            for tr, te in GroupKFold(n_splits=min(5, len(np.unique(groups)))).split(X, y_bin, groups):
                if len(np.unique(y_bin[tr])) < 2:
                    oof[te] = np.clip(F[te], 0, 1)
                    continue
                oof[te] = self._new().fit(X[tr], y_bin[tr]).predict_proba(X[te])[:, 1]
            b_id = float(np.mean((np.clip(F, 0, 1) - y_bin) ** 2))
            b_md = float(np.mean((oof - y_bin) ** 2))
            self.cv = {"cv_brier_identity": b_id, "cv_brier_model": b_md}
            self.used = (b_md < b_id - min_gain) if self.use_sensor else True
        return self

    def predict(self, F, f=None):
        if not self.used:
            return np.clip(np.asarray(F, dtype=np.float64), 0.0, 1.0)
        return self.model.predict_proba(self._x(F, f))[:, 1]

    def coefs(self) -> dict:
        c = self.model.coef_[0]
        i_ev = 1 if self.use_sensor else 0
        return {"intercept": float(self.model.intercept_[0]),
                "coef_sensor": float(c[0]) if self.use_sensor else float("nan"),
                "coef_evidence": float(c[i_ev]) if self.use_evidence else float("nan"),
                "used": bool(self.used), **self.cv}


def block_groups(units, n_block: int = 5, many_units: int = 10) -> np.ndarray:
    """Groups for cross-validation of the held-out set. Few held-out units (N-CMAPSS): split every unit into
    contiguous blocks of flight cycles; many engines (C-MAPSS, CAC episodes): the unit itself is the group."""
    units = np.asarray(units)
    u_all = np.unique(units)
    if len(u_all) >= many_units:
        return units.astype(np.int64)
    g = np.zeros(len(units), dtype=np.int64)
    for j, u in enumerate(u_all):
        idx = np.where(units == u)[0]
        g[idx] = j * 100 + np.minimum((np.arange(len(idx)) * n_block) // max(len(idx), 1), n_block - 1)
    return g


class LocalConformal:
    """Retrieval-based (localised) split conformal: the width correction of a test window is the conformal
    quantile of the nonconformity scores of its k nearest HELD-OUT calibration windows (retrieved in sensor
    space), instead of one global q_hat. Retrieval evidence therefore enters the calibration itself."""

    def __init__(self, X_cal, lo_cal, hi_cal, y_cal, alpha: float, k: int = 200):
        from sklearn.neighbors import NearestNeighbors
        self.scores = np.maximum(lo_cal - y_cal, y_cal - hi_cal)
        self.k = int(max(20, min(k, len(self.scores) // 5)))
        self.alpha = alpha
        self.nn = NearestNeighbors(n_neighbors=self.k, n_jobs=-1).fit(window_features(X_cal))

    def qhat(self, X) -> np.ndarray:
        _, idx = self.nn.kneighbors(window_features(X))
        sc = self.scores[idx]
        level = min(np.ceil((self.k + 1) * (1 - self.alpha)) / self.k, 1.0)
        return np.quantile(sc, level, axis=1)


# (name, evidence key, use_evidence, use_sensor):  S-only = C+platt | R-only = R-kNN, R-KB | S+R = L-kNN, L-KB
LEARNED = (("C+platt", None, False, True), ("L-kNN", "f_knn", True, True), ("L-KB", "f_rag", True, True),
           ("R-kNN", "f_knn", True, False), ("R-KB", "f_rag", True, False))
DIAG_W = 0.3        # DIAG-oracle: evidence = the true label (diagnostic upper bound on RAG headroom)

# ============================================================
# CONFIGURATIONS
# ============================================================

# (ungated name, gated name, evidence key in T, similarity key in T, retriever key)
SOURCES = (("D", "E", "f_rag", "sim", "kb"), ("D-kNN", "E-kNN", "f_knn", "sim_knn", "knn"))
SRC_OF = {n: (fk, sk, key) for nD, nE, fk, sk, key in SOURCES for n in (nD, nE)}
RAG_NAMES = [n for src in SOURCES for n in src[:2]]
S_LEVELS = (0.0, 0.25, 0.5, 0.75)    # s_min is taken at these quantiles of the observed similarity


def _ie(lo, mid, hi, H, Cm, Cf) -> dict:
    F = failure_prob(lo, mid, hi, H)
    return {"interv": decide(F, Cm, Cf), "point": mid, "lo": lo, "hi": hi, "F": F}


def build_configs(T: dict, ctx: dict, Cm: float, Cf: float) -> Dict[str, dict]:
    """Decision of every configuration from the prediction bundle T (clean or noisy)."""
    H = ctx["H"]
    out: Dict[str, dict] = {}
    out["A"] = {"interv": T["d"] <= H, "point": T["d"]}
    out["A+margin"] = {"interv": T["d"] <= H + ctx["margin"], "point": T["d"], "margin": ctx["margin"]}
    out["B"] = _ie(T["q10B"], T["q50B"], T["q90B"], H, Cm, Cf)
    out["C"] = _ie(T["q10C"], T["q50B"], T["q90C"], H, Cm, Cf)
    F_C = out["C"]["F"]
    for d_, lam_ in (ctx.get("crc") or {}).items():       # risk-controlled threshold, chosen on held-out engines only
        out[f"C-CRC{int(round(d_ * 100)):02d}"] = {"interv": F_C > lam_, "point": T["q50B"]}
    n_ = len(F_C)
    out["Always-alarm"] = {"interv": np.ones(n_, dtype=bool), "point": T["q50B"]}        # trivial policies for reference
    out["Never-alarm"] = {"interv": np.zeros(n_, dtype=bool), "point": T["q50B"]}
    g_ = ctx.get("gate")
    if g_ is not None and "C-CRC10" in out:              # skill gate: risk-controlled rule if the model has skill, else fail safe
        out["C-GATE"] = {"interv": out["C-CRC10"]["interv"] if g_["passed"] else out["Always-alarm"]["interv"],
                         "point": T["q50B"]}
    if ctx.get("iso") is not None:
        F_i = ctx["iso"].predict(F_C)
        out["C+iso"] = {"interv": decide(F_i, Cm, Cf), "point": T["q50B"], "F": F_i}
    for nD, nE, fk, sk, key in SOURCES:
        tuned = ctx.get("tuned", {}).get(key)
        if tuned is not None and fk in T:
            for name, mode in ((nD, "D"), (nE, "E")):
                Ff, w = fuse(F_C, T[fk], T[sk], mode, tuned[mode])
                out[name] = {"interv": decide(Ff, Cm, Cf), "point": T["q50B"], "F": Ff, "w": w}
    if "q10L" in T:
        out["C-local"] = _ie(T["q10L"], T["q50B"], T["q90L"], H, Cm, Cf)
    w_p = ctx.get("tuned", {}).get("knn", {}).get("D", {}).get("w") if ctx.get("base_rate") is not None else None
    if w_p is not None:                                # control: same weight as D-kNN, but a constant prior
        Fp = (1.0 - w_p) * F_C + w_p * ctx["base_rate"]
        out["D-prior"] = {"interv": decide(Fp, Cm, Cf), "point": T["q50B"], "F": Fp}
    for name, fk, use_ev, use_s in LEARNED:
        lf = ctx.get("lf", {}).get(name)
        if lf is not None and (fk is None or fk in T):
            Fl = lf.predict(F_C, T[fk] if fk else None)
            out[name] = {"interv": decide(Fl, Cm, Cf), "point": T["q50B"], "F": Fl}
    if ctx.get("diag_oracle") and "y" in T:            # diagnostic only: uses the TEST labels, never a method
        Fo = (1.0 - DIAG_W) * F_C + DIAG_W * (T["y"] <= H).astype(float)
        out["DIAG-oracle"] = {"interv": decide(Fo, Cm, Cf), "point": T["q50B"], "F": Fo}
    for key, name in (("mcd", "MCD"), ("mcd_cp", "MCD+CP"), ("ens", "Ens"), ("ens_cp", "Ens+CP")):
        if key in T:
            lo, mid, hi = T[key]
            out[name] = _ie(lo, mid, hi, H, Cm, Cf)
    return out


def make_rows(T: dict, ctx: dict, Cm: float, Cf: float) -> List[dict]:
    cfg, H, y = ctx["cfg"], ctx["H"], T["y"]
    oracle = float(Cm * (y <= H).sum())           # one intervention per truly critical sample, nothing else
    rows = []
    for name, e in build_configs(T, ctx, Cm, Cf).items():
        row = {"dataset": cfg.dataset, "seed": cfg.seed, "tag": cfg.tag, "config": name, "H": H,
               "Cm": Cm, "Cf": Cf, "n_eval": len(y)}
        row.update(cost_stats(e["interv"], y, H, Cm, Cf))
        row["Oracle"], row["Regret"] = oracle, row["Cost"] - oracle
        row["Critical"] = int((y <= H).sum())
        row["MissRate"] = row["Failures"] / max(row["Critical"], 1)
        row.update(point_metrics(y, e["point"]))
        if "lo" in e:
            row.update(prob_metrics(y, e["lo"], e["point"], e["hi"], H, cfg.quantiles))
        elif "F" in e:
            row["Brier"] = float(np.mean((e["F"] - (y <= H)) ** 2))
        if "margin" in e:
            row["margin"] = e["margin"]
        if "w" in e:
            row["n_activated"] = int((e["w"] > 0).sum())
        if name == "C":
            row["T"], row["q_hat"] = ctx["T"], ctx["q_hat"]
            if ctx.get("gate") is not None:
                row["Skill"], row["SkillLo"], row["GatePass"] = (ctx["gate"]["skill"], ctx["gate"]["skill_lo"],
                                                                 int(ctx["gate"]["passed"]))
        rows.append(row)
    return rows


def predict_core(X: np.ndarray, cond: np.ndarray, ctx: dict) -> dict:
    """Sensor-model predictions (+ retrieval evidence) for a block of test windows."""
    dev = ctx["cfg"].device
    P = predict(ctx["qm"], X, dev)
    lo, hi = apply_temperature(P[:, 0], P[:, 1], P[:, 2], ctx["T"])
    out = {"q10B": P[:, 0], "q50B": P[:, 1], "q90B": P[:, 2],
           "q10C": lo - ctx["q_hat"], "q90C": hi + ctx["q_hat"],
           "d": predict(ctx["dm"], X, dev)[:, 0]}
    if ctx.get("local") is not None:
        ql = ctx["local"].qhat(X)
        out["q10L"], out["q90L"] = lo - ql, hi + ql
    if ctx.get("kb") is not None:
        out["f_rag"], out["sim"] = ctx["kb"].retrieve(P[:, 1], cond)
    if ctx.get("knn") is not None:
        out["f_knn"], out["sim_knn"], _ = ctx["knn"].retrieve(X)
    return out


# ============================================================
# "WHEN DOES RAG HELP?" ANALYSIS
# ============================================================

def activation_table(T: dict, ctx: dict, name: str, Cm: float, Cf: float) -> pd.DataFrame:
    H, y = ctx["H"], T["y"]
    fk, sk, _ = SRC_OF[name]
    cfgs = build_configs(T, ctx, Cm, Cf)
    base, e = cfgs["C"], cfgs[name]
    before, after = base["interv"], e["interv"]
    c0, c1 = cost_vector(before, y, H, Cm, Cf), cost_vector(after, y, H, Cm, Cf)
    act = e["w"] > 0
    changed = act & (before != after)
    cat = np.where(~act, "not_activated",
                   np.where(~changed, "no_effect", np.where(c1 < c0, "beneficial", "harmful")))
    extra = {"cycle": T["cycle"]} if "cycle" in T else {}
    return pd.DataFrame({
        "unit": T["unit"], **extra, "y": y, "F_sensor": base["F"],
        "dist_to_threshold": base["F"] - Cm / Cf,
        "sigma": (base["hi"] - base["lo"]) / (2 * Z80),
        "sim": T[sk], "f_rag": T[fk], "w": e["w"], "F_fused": e["F"],
        "dec_before": before, "dec_after": after, "failure_by_H": y <= H,
        "cost_before": c0, "cost_after": c1, "category": cat})


def activation_summary(tab: pd.DataFrame, ctx: dict, name: str) -> dict:
    cfg = ctx["cfg"]
    c = tab["category"]
    return {"dataset": cfg.dataset, "seed": cfg.seed, "tag": cfg.tag, "config": name, "n_total": len(tab),
            "n_activated": int((c != "not_activated").sum()),
            "n_decision_changed": int(((c == "beneficial") | (c == "harmful")).sum()),
            "n_beneficial": int((c == "beneficial").sum()), "n_harmful": int((c == "harmful").sum()),
            "n_no_effect": int((c == "no_effect").sum()),
            "n_near_boundary": int((tab["dist_to_threshold"].abs() <= NEAR).sum()),
            "net_cost_change": float((tab["cost_after"] - tab["cost_before"]).sum())}


NEAR = 0.05      # a decision is "near the boundary" when |F_sensor - Cm/Cf| <= NEAR


def change_rows(T: dict, ctx: dict, Cm: float, Cf: float) -> List[dict]:
    """Same columns as activation_summary, for the configs that touch every sample (learned fusion, controls,
    diagnostic oracle): how many decisions differ from C, and whether they were beneficial or harmful."""
    cfg, H, y = ctx["cfg"], ctx["H"], T["y"]
    cfgs = build_configs(T, ctx, Cm, Cf)
    base = cfgs["C"]
    c0 = cost_vector(base["interv"], y, H, Cm, Cf)
    rows = []
    for name in ("C+platt", "L-kNN", "L-KB", "R-kNN", "R-KB", "C-local", "D-prior", "DIAG-oracle", "C+iso"):
        if name not in cfgs:
            continue
        c1 = cost_vector(cfgs[name]["interv"], y, H, Cm, Cf)
        ch = cfgs[name]["interv"] != base["interv"]
        rows.append({"dataset": cfg.dataset, "seed": cfg.seed, "tag": cfg.tag, "config": name,
                     "n_total": len(y), "n_activated": len(y), "n_decision_changed": int(ch.sum()),
                     "n_beneficial": int((ch & (c1 < c0)).sum()), "n_harmful": int((ch & (c1 > c0)).sum()),
                     "n_no_effect": int((~ch).sum()),
                     "n_near_boundary": int((np.abs(base["F"] - Cm / Cf) <= NEAR).sum()),
                     "net_cost_change": float((c1 - c0).sum())})
    return rows


def sensitivity_rows(T: dict, ctx: dict, Cm: float, Cf: float) -> List[dict]:
    """(s_min, w_max) grid for the gated configs on the TEST samples. Reporting only - never used to select.
    s_min is expressed as a quantile of the similarity distribution (s_quantile), so the axis is
    comparable across datasets, seeds and retrievers."""
    cfg, H, y = ctx["cfg"], ctx["H"], T["y"]
    F_C = build_configs(T, ctx, Cm, Cf)["C"]["F"]
    rows = []
    for nD, nE, fk, sk, key in SOURCES:
        tuned = ctx["tuned"].get(key)
        if tuned is None or fk not in T:
            continue
        for q, s_min in zip(S_LEVELS, ctx["s_grid"][key]):
            for w_max in W_GRID:
                p = dict(tuned["E"], s_min=float(s_min), w_max=w_max)
                Ff, w = fuse(F_C, T[fk], T[sk], "E", p)
                row = {"dataset": cfg.dataset, "seed": cfg.seed, "config": nE, "s_quantile": q,
                       "s_min": float(s_min), "w_max": w_max, "band_lo": p["band_lo"],
                       "band_hi": p["band_hi"], "n_activated": int((w > 0).sum())}
                row.update(cost_stats(decide(Ff, Cm, Cf), y, H, Cm, Cf))
                rows.append(row)
    return rows


# ============================================================
# EXPERIMENT
# ============================================================

def run_experiment(cfg: Cfg) -> Dict[str, List[dict]]:
    """C-MAPSS (FD001-FD004) and N-CMAPSS: load the data bundle, then run the shared controlled pipeline."""
    set_seed(cfg.seed)
    H, Cm, Cf = cfg.horizon(), cfg.Cm, cfg.Cf
    print(f"=== pipeline {PIPELINE_VERSION} | {cfg.dataset}/seed{cfg.seed}  H={H} Cm={Cm} Cf={Cf} ===", flush=True)
    cycle_te = None
    if cfg.dataset == "NCMAPSS":
        win, X_te, y_te, test_units, cycle_te, cond_te, splits = load_ncmapss_bundle(cfg)
    else:
        train_raw, test_raw, rul_df = load_cmapss(cfg.data_dir, cfg.dataset)
        train_raw = add_rul(train_raw, cfg.max_rul)
        splits = split_units(train_raw["unit"].unique(), cfg.seed, cfg)
        part = {k: train_raw[train_raw["unit"].isin(v)] for k, v in splits.items()}
        prep = Preprocessor(cfg.dataset).fit(part["train"])
        win = {k: make_windows(prep.transform(df), cfg.window) for k, df in part.items()}
        X_te, test_units, cond_te = make_test_windows(prep.transform(test_raw), cfg.window)
        y_te = rul_df["RUL"].values.astype(np.float32)
        assert len(y_te) == len(X_te), "RUL file and test engines do not match"
        if cfg.cap_test_rul:
            y_te = np.minimum(y_te, cfg.max_rul)
    print(f"  units train/val/cal_ts/cal_cp = {[len(splits[k]) for k in ('train', 'val', 'cal_ts', 'cal_cp')]}"
          f"  windows train/val/cal_ts/cal_cp = {[len(win[k][1]) for k in ('train', 'val', 'cal_ts', 'cal_cp')]}"
          f"  test={len(y_te)}", flush=True)
    return run_from_bundle(cfg, win, X_te, y_te, test_units, cond_te, splits, cycle_te)


def run_from_bundle(cfg: Cfg, win: dict, X_te, y_te, test_units, cond_te, splits,
                    cycle_te=None, query_fn=None, kb_builder=None) -> Dict[str, List[dict]]:
    """The controlled pipeline. `win[k] = (X, y, unit, cond)` for k in train/val/cal_ts/cal_cp.
    Any dataset (C-MAPSS, N-CMAPSS, CAC folds) enters here, so every config is treated identically.

    Data roles (no test labels are ever used):
      train           -> network weights
      val             -> early stopping, A+margin, isotonic fit (with cal_cp)
      cal_ts          -> temperature scaling, and (with val) tuning of the RAG gates
      cal_cp          -> split conformal, isotonic fit (with val)
    """
    set_seed(cfg.seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    H, Cm, Cf = cfg.horizon(), cfg.Cm, cfg.Cf
    dev = cfg.device
    n_in = X_te.shape[2]
    Xtr, ytr = win["train"][0], win["train"][1]
    Xva, yva, _, cond_va = win["val"]

    # ---- models (trained once per seed) ----
    set_seed(cfg.seed + 7)
    qm = QuantileGRU(n_in, cfg.hidden, cfg.layers, cfg.dropout, cfg.quantiles)
    qm = fit_model(qm, lambda p, t: pinball_loss(p, t, cfg.quantiles), Xtr, ytr, Xva, yva, cfg,
                   cfg.seed + 7, "quantile")
    det_models = []
    for m in range(cfg.ens_size if cfg.run_baselines else 1):
        s_m = cfg.seed + 100 * m
        set_seed(s_m)
        dm_m = DeterministicGRU(n_in, cfg.hidden, cfg.layers, cfg.dropout)
        det_models.append(fit_model(dm_m, mse_loss, Xtr, ytr, Xva, yva, cfg, s_m, f"determ.{m}"))
    dm = det_models[0]  # Config A, A+margin, MC-Dropout and ensemble member 0

    # ---- calibration on held-out engines ----
    P_ts = predict(qm, win["cal_ts"][0], dev)
    T_val = fit_temperature(P_ts[:, 0], P_ts[:, 1], P_ts[:, 2], win["cal_ts"][1], cfg.T_max)
    P_cp = predict(qm, win["cal_cp"][0], dev)
    lo_cp, hi_cp = apply_temperature(P_cp[:, 0], P_cp[:, 1], P_cp[:, 2], T_val)
    q_hat = fit_conformal(lo_cp, hi_cp, win["cal_cp"][1], cfg.alpha, cfg.floor_qhat)
    print(f"  T={T_val:.3f}  q_hat={q_hat:.3f}", flush=True)
    lo_ts, hi_ts = apply_temperature(P_ts[:, 0], P_ts[:, 1], P_ts[:, 2], T_val)
    F_cal = np.concatenate([failure_prob(lo_ts - q_hat, P_ts[:, 1], hi_ts + q_hat, H),
                            failure_prob(lo_cp - q_hat, P_cp[:, 1], hi_cp + q_hat, H)])
    y_cal = np.concatenate([win["cal_ts"][1], win["cal_cp"][1]])
    crc = {float(d_): crc_threshold(F_cal, y_cal <= H, float(d_)) for d_ in cfg.crc_deltas}
    print("  CRC thresholds (alarm iff F > lam): " + ", ".join(f"delta={d_:g}: lam={l_:.4f}" for d_, l_ in crc.items())
          + f"   (cost-rule threshold Cm/Cf = {Cm / Cf:.4f}; {int((y_cal <= H).sum())} critical calibration windows)", flush=True)

    gate = skill_gate([P_ts, P_cp], [win["cal_ts"][1], win["cal_cp"][1]], [win["cal_ts"][2], win["cal_cp"][2]],
                      cfg.quantiles, ytr)
    print(f"  SKILL GATE: skill vs climatology = {gate['skill']:+.3f} (95% lower bound {gate['skill_lo']:+.3f}, "
          f"{gate['n_engines']} held-out engines) -> {'PASS' if gate['passed'] else 'FAIL (fail-safe rule used by C-GATE)'}", flush=True)

    ctx = {"cfg": cfg, "H": H, "qm": qm, "dm": dm, "T": T_val, "q_hat": q_hat, "margin": 0,
           "iso": None, "kb": None, "knn": None, "tuned": {}, "s_grid": {}, "crc": crc, "gate": gate}

    def calibrated(X):
        P = predict(qm, X, dev)
        lo, hi = apply_temperature(P[:, 0], P[:, 1], P[:, 2], T_val)
        return failure_prob(lo - q_hat, P[:, 1], hi + q_hat, H), P[:, 1]

    # ---- Config A margin (validation engines, same cost function) ----
    d_val = predict(dm, Xva, dev)[:, 0]
    ctx["margin"] = min((cost_stats(d_val <= H + m, yva, H, Cm, Cf)["Cost"], m) for m in range(0, 61))[1]

    # ---- C+iso: isotonic recalibration of P(RUL <= H) (tests whether RAG adds more than recalibration) ----
    from sklearn.isotonic import IsotonicRegression
    F_cp, _ = calibrated(win["cal_cp"][0])
    F_va, _ = calibrated(Xva)
    ctx["iso"] = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(
        np.concatenate([F_cp, F_va]), (np.concatenate([win["cal_cp"][1], yva]) <= H).astype(float))

    # ---- RAG evidence sources + tuning on val + cal_ts windows ----
    X_tn = np.concatenate([Xva, win["cal_ts"][0]])
    y_tn = np.concatenate([yva, win["cal_ts"][1]])
    c_tn = np.concatenate([cond_va, win["cal_ts"][3]])
    F_tn, q50_tn = calibrated(X_tn)

    def tune(key, f, s):
        grid = [float(v) for v in np.quantile(s, S_LEVELS)]
        ctx["s_grid"][key] = grid
        ctx["tuned"][key] = {"D": tune_D(F_tn, f, s, y_tn, H, Cm, Cf),
                             "E": tune_E(F_tn, f, s, y_tn, H, Cm, Cf, grid)}
        print(f"  RAG[{key}] tuned: D={ctx['tuned'][key]['D']}  E={ctx['tuned'][key]['E']}  "
              f"(mean sim={s.mean():.3f})", flush=True)

    kb_file = cfg.kb_path.format(dataset=cfg.dataset) if cfg.kb_path else None
    if kb_builder is not None or (kb_file and os.path.exists(kb_file)):
        try:
            if kb_builder is not None:
                _, q50_cp = calibrated(win["cal_cp"][0])
                kb = kb_builder(q50_cp, win["cal_cp"][1], H)       # built from held-out cal_cp predictions only
                print(f"  RAG[kb]: data-derived knowledge base, {len(kb)} entries (cal_cp windows)", flush=True)
            else:
                with open(kb_file, encoding="utf-8") as f:
                    kb = json.load(f)
            ctx["kb"] = Retriever(kb, cfg.dataset, cfg.embed_model, cfg.top_k, query_fn=query_fn)
            f_tn, s_tn = ctx["kb"].retrieve(q50_tn, c_tn)
            tune("kb", f_tn, s_tn)
        except ImportError:
            print("  [warn] sentence-transformers is not installed; text-KB configs D/E are skipped", flush=True)
            ctx["kb"] = None
    else:
        print(f"  [warn] knowledge base not found ({kb_file}); text-KB configs D/E are skipped", flush=True)
    if cfg.use_knn:
        ctx["knn"] = KNNRetriever(Xtr, ytr, H, cfg.knn_k)
        f_tn, s_tn, _ = ctx["knn"].retrieve(X_tn)
        tune("knn", f_tn, s_tn)

    # ---- retrieval-based local conformal (calibration windows = cal_cp, held out) and constant-prior control ----
    P_l = predict(qm, win["cal_cp"][0], dev)
    lo_l, hi_l = apply_temperature(P_l[:, 0], P_l[:, 1], P_l[:, 2], T_val)
    ctx["local"] = LocalConformal(win["cal_cp"][0], lo_l, hi_l, win["cal_cp"][1], cfg.alpha, cfg.local_k)
    ctx["base_rate"] = float((ytr <= H).mean())

    # ---- learned fusion inside the calibration: fitted on ALL held-out engines (val + cal_ts + cal_cp) ----
    X_h = np.concatenate([Xva, win["cal_ts"][0], win["cal_cp"][0]])
    y_h = np.concatenate([yva, win["cal_ts"][1], win["cal_cp"][1]])
    c_h = np.concatenate([cond_va, win["cal_ts"][3], win["cal_cp"][3]])
    F_h, q50_h = calibrated(X_h)
    yb_h = (y_h <= H).astype(int)
    g_h = block_groups(np.concatenate([win["val"][2], win["cal_ts"][2], win["cal_cp"][2]]))
    ev_h = {}
    if ctx["knn"] is not None:
        ev_h["f_knn"] = ctx["knn"].retrieve(X_h)[0]
    if ctx["kb"] is not None:
        ev_h["f_rag"] = ctx["kb"].retrieve(q50_h, c_h)[0]
    ctx["lf"], ctx["fusion_rows"] = {}, []
    for name, fk, use_ev, use_s in LEARNED:
        if use_ev and fk not in ev_h:
            continue
        lf = LearnedFusion(use_ev, use_s).fit(F_h, ev_h.get(fk), yb_h, groups=g_h)
        if lf is None:
            print(f"  [warn] {name}: held-out set has a single class; skipped", flush=True)
            continue
        ctx["lf"][name] = lf
        ctx["fusion_rows"].append({"dataset": cfg.dataset, "seed": cfg.seed, "config": name,
                                   "n_fit": len(yb_h), "n_pos": int(yb_h.sum()), **lf.coefs()})
        print(f"  fusion[{name}]: {lf.coefs()}", flush=True)
    ctx["diag_oracle"] = True

    # ---- test predictions ----
    T = predict_core(X_te, cond_te, ctx)
    T["y"], T["unit"] = y_te, np.asarray(test_units)
    if cycle_te is not None:
        T["cycle"] = np.asarray(cycle_te)

    # ---- uncertainty-aware baselines (same data, same conformal procedure) ----
    if cfg.run_baselines:
        def raw_and_cp(mu_c, sd_c, mu_t, sd_t):
            sd_c, sd_t = np.maximum(sd_c, 1e-3), np.maximum(sd_t, 1e-3)
            qh = fit_conformal(mu_c - Z80 * sd_c, mu_c + Z80 * sd_c, win["cal_cp"][1],
                               cfg.alpha, cfg.floor_qhat)
            raw = (mu_t - Z80 * sd_t, mu_t, mu_t + Z80 * sd_t)
            return raw, (raw[0] - qh, mu_t, raw[2] + qh)
        mu_c, sd_c = mc_predict(dm, win["cal_cp"][0], cfg.mc_passes, dev, cfg.seed)
        mu_t, sd_t = mc_predict(dm, X_te, cfg.mc_passes, dev, cfg.seed + 1)
        T["mcd"], T["mcd_cp"] = raw_and_cp(mu_c, sd_c, mu_t, sd_t)
        mu_c, sd_c = ens_predict(det_models, win["cal_cp"][0], dev)
        mu_t, sd_t = ens_predict(det_models, X_te, dev)
        T["ens"], T["ens_cp"] = raw_and_cp(mu_c, sd_c, mu_t, sd_t)

    # ---- tables ----
    out: Dict[str, List[dict]] = {"main": make_rows(T, ctx, Cm, Cf), "sweep": [], "robust": [],
                                  "activation": [], "sensitivity": [], "fusion": list(ctx["fusion_rows"])}
    out["activation"] += change_rows(T, ctx, Cm, Cf)
    for cf in cfg.cost_ratios:
        out["sweep"] += make_rows(T, ctx, 1.0, float(cf))

    od = os.path.join(cfg.out_dir, cfg.dataset)
    os.makedirs(od, exist_ok=True)
    pre = f"seed{cfg.seed}{cfg.tag}"
    for nD, nE, fk, sk, key in SOURCES:
        if key in ctx["tuned"]:
            for name in (nD, nE):
                tab = activation_table(T, ctx, name, Cm, Cf)
                tab.to_csv(os.path.join(od, f"{pre}_activations_{name}.csv"), index=False)
                out["activation"].append(activation_summary(tab, ctx, name))
    if ctx["tuned"]:
        out["sensitivity"] = sensitivity_rows(T, ctx, Cm, Cf)

    # ---- robustness: same trained models, same calibration, same linear fusion ----
    if cfg.run_robustness:
        rng = np.random.RandomState(cfg.seed + 1000)
        keep = {"A", "C", "C-GATE", "C-CRC05", "C-CRC10", "C-CRC20", "C+iso", "C+platt", "L-kNN", "L-KB", "R-kNN", "R-KB", "C-local", "D-prior"} | set(RAG_NAMES)
        for level in cfg.noise_levels:
            for rep in range(1 if level == 0 else cfg.noise_reps):
                Xn = X_te if level == 0 else (X_te + rng.normal(0, level, X_te.shape)).astype(np.float32)
                Tn = predict_core(Xn, cond_te, ctx)
                for name, e in build_configs(Tn, ctx, Cm, Cf).items():
                    if name in keep:
                        row = {"dataset": cfg.dataset, "seed": cfg.seed, "noise": level, "rep": rep,
                               "config": name}
                        row.update(cost_stats(e["interv"], y_te, H, Cm, Cf))
                        out["robust"].append(row)

    # ---- persist: per-sample decisions and costs (for engine-level bootstrap and the CAC episode analysis) ----
    arrays = {k: v for k, v in T.items() if isinstance(v, np.ndarray)}
    arrays.update({"q50": T["q50B"], "q10_raw": T["q10B"], "q90_raw": T["q90B"], "H": np.array(H)})
    for name, e in build_configs(T, ctx, Cm, Cf).items():
        arrays[f"interv_{name}"] = e["interv"]
        arrays[f"cost_{name}"] = cost_vector(e["interv"], y_te, H, Cm, Cf)
        if "F" in e:
            arrays[f"F_{name}"] = e["F"]
    np.savez(os.path.join(od, f"{pre}_preds.npz"), **arrays)
    with open(os.path.join(od, f"{pre}_meta.json"), "w") as f:
        json.dump({"cfg": asdict(cfg), "splits": {k: np.asarray(v).tolist() for k, v in splits.items()},
                   "T": T_val, "q_hat": q_hat, "crc": {str(k_): v_ for k_, v_ in crc.items()}, "gate": gate, "margin": ctx["margin"], "tuned": ctx["tuned"], "fusion": ctx["fusion_rows"],
                   "s_grid": ctx["s_grid"]}, f, indent=1, default=str)
    for k, rows in out.items():
        if rows:
            pd.DataFrame(rows).to_csv(os.path.join(od, f"{pre}_{k}.csv"), index=False)
    torch.save({"quantile": qm.state_dict(), "deterministic": [m.state_dict() for m in det_models]},
               os.path.join(od, f"{pre}_models.pt"))
    return out


# ============================================================
# AGGREGATION ACROSS SEEDS
# ============================================================

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """mean ± std over seeds, default cost ratio only."""
    keys = [c for c in ("RMSE", "Coverage80", "QCE", "Brier", "CRPS", "Failures", "Interventions", "Cost", "Regret")
            if c in df.columns]
    g = df.groupby(["dataset", "config"], sort=False)[keys]
    return g.mean().round(3).astype(str) + " ± " + g.std().round(3).astype(str)


def paired_delta(df: pd.DataFrame, a: str, b: str, metric: str = "Cost") -> pd.DataFrame:
    """Per dataset: mean ± std over seeds of (metric[b] - metric[a]); negative = b better."""
    wide = df.pivot_table(index=["dataset", "seed"], columns="config", values=metric)
    d = (wide[b] - wide[a]).groupby("dataset")
    return pd.DataFrame({"mean_diff": d.mean(), "std_diff": d.std(), "n_seeds": d.count(),
                         "b_better_in": (wide[b] < wide[a]).groupby("dataset").sum()})


def robustness_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Cost degradation relative to the noise-free run, per config; delta_E_minus_C < 0 means RAG
    reduces the degradation."""
    m = df.groupby(["dataset", "config", "noise"])["Cost"].mean().reset_index()
    base = m[m["noise"] == 0].set_index(["dataset", "config"])["Cost"]
    m["degradation"] = m.apply(lambda r: r["Cost"] - base[(r["dataset"], r["config"])], axis=1)
    wide = m.pivot_table(index=["dataset", "noise"], columns="config", values="degradation")
    if {"E", "C"} <= set(wide.columns):
        wide["delta_E_minus_C"] = wide["E"] - wide["C"]
    return wide.round(2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run the pipeline locally (no Modal).")
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--out_dir", default="./results")
    ap.add_argument("--kb_path", default="./kb/kb_{dataset}.json")
    ap.add_argument("--dataset", default="FD001", help="FD001..FD004 or NCMAPSS")
    ap.add_argument("--ncmapss_path", default="./ncmapss/N-CMAPSS_DS01-005.h5")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--no_baselines", action="store_true")
    ap.add_argument("--no_robustness", action="store_true")
    a, _unknown = ap.parse_known_args()  # tolerate arguments injected by Jupyter / Modal notebooks
    tables: Dict[str, List[dict]] = {}
    for s in a.seeds.split(","):
        res = run_experiment(Cfg(dataset=a.dataset, data_dir=a.data_dir, out_dir=a.out_dir,
                                 kb_path=a.kb_path, ncmapss_path=a.ncmapss_path, seed=int(s), epochs=a.epochs,
                                 run_baselines=not a.no_baselines, run_robustness=not a.no_robustness))
        for k, v in res.items():
            tables.setdefault(k, []).extend(v)
    main_df = pd.DataFrame(tables["main"])
    print(summarize(main_df).to_string())
