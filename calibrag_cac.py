"""
calibrag_cac.py - CAC industrial validation, SEPARATE from the C-MAPSS / N-CMAPSS runs.

Run it LOCALLY (CPU is enough): the CAC data is confidential, so it should not be uploaded to
Modal or any other cloud service unless your confidentiality agreement allows it.

It uses exactly the same controlled pipeline as the benchmarks (calibrag_pipeline.run_from_bundle),
so configs A, A+margin, B, C, C+iso, D, E, D-kNN, E-kNN, MC-Dropout and Deep-Ensemble are treated
identically. What differs is how 18 stoppage events are handled honestly:
  * episodes: every run-up to a stoppage is one episode (18 episodes); windows never cross a stoppage
  * leave-episodes-out cross-validation: no episode is ever in train and test at the same time
  * episode-level evaluation: one decision outcome per episode (timely / premature / missed), so
    the sample size is 18, not thousands of correlated hourly windows
  * 95% bootstrap confidence intervals over episodes for cost, coverage and paired differences

INPUT FILES (edit load_cac() if yours differ)
  cac_series.csv : one row per measurement. Columns: `timestamp` (ISO date-time) + numeric sensor columns.
                   Any sampling rate; it is resampled to hourly means.
  cac_events.csv : one row per stoppage. Column: `stop_time` (ISO date-time).
  kb/kb_CAC.json : optional text knowledge base (list of dicts with text, f_prob_lo, f_prob_hi).

RUN
  python calibrag_cac.py --series cac\\cac_series.csv --events cac\\cac_events.csv --kb kb\\kb_CAC.json --seeds 0,1,2
  python calibrag_cac.py --analyze_only                       # redo the analysis from saved folds
"""
import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import calibrag_pipeline as cp

SCHED_HOURS = (96, 168, 336, 720)       # fixed-interval preventive rule: intervene T hours after the previous stoppage
BASELINES = ["Always-alarm"] + [f"Sched-{t}h" for t in SCHED_HOURS]
CONFIGS = ["A", "A+margin", "B", "C", "C-GATE", "C-CRC05", "C-CRC10", "C-CRC20", "C+iso", "C+platt", "L-kNN", "L-KB", "C-local", "D-prior", "D", "E", "D-kNN", "E-kNN", "MCD+CP", "Ens+CP"] + BASELINES


@dataclass
class CacCfg:
    series_csv: str = "./cac/cac_series.csv"
    events_csv: str = "./cac/cac_events.csv"
    kb_path: Optional[str] = "./kb/kb_CAC.json"
    out_dir: str = "./results_cac"
    time_col: str = "timestamp"
    stop_col: str = "stop_time"
    resample: str = "1h"
    window: int = 48                  # hours of history per window
    max_rul_h: int = 720              # RUL label cap (hours)
    H: int = 48                       # decision horizon (hours)
    Cm: float = 1.0
    Cf: float = 10.0
    waste_per_horizon: float = 0.0    # optional extra cost per H hours of life wasted by a premature intervention
    n_folds: int = 9                  # 18 episodes -> 2 test episodes per fold
    n_val: int = 2
    n_ts: int = 2
    n_cp: int = 2
    seeds: str = "0"
    epochs: int = 50
    ens_size: int = 3
    target_scale: float = 1.0         # e.g. 720 = train on RUL/720 (same cap as max_rul_h); outputs stay in hours
    patience: int = 0                 # early stopping on val loss (epochs); 0 = off
    weight_decay: float = 0.0
    lr_step: int = 20
    T_max: float = 1.5                # upper bound of the temperature-scaling search
    max_missing: float = 0.30         # drop sensor columns with more missing values than this
    ffill_limit: int = 6              # hours of gap that may be forward-filled
    device: str = "cpu"


# ---------------------------------------------------------------- data
def load_cac(cfg: CacCfg) -> Dict[int, dict]:
    """Episodes: {episode id: {X (n, f), rul (hours, capped), t (hour index)}}."""
    ser = pd.read_csv(cfg.series_csv, parse_dates=[cfg.time_col]).set_index(cfg.time_col).sort_index()
    ser = ser.select_dtypes("number").resample(cfg.resample).mean()
    ser = ser.loc[:, ser.isna().mean() <= cfg.max_missing].ffill(limit=cfg.ffill_limit).dropna()
    stops = pd.to_datetime(pd.read_csv(cfg.events_csv)[cfg.stop_col]).sort_values().reset_index(drop=True)
    eps: Dict[int, dict] = {}
    for k, stop in enumerate(stops):
        start = stops[k - 1] if k > 0 else ser.index[0] - pd.Timedelta(1, "s")
        ep = ser[(ser.index > start) & (ser.index <= stop)]
        if len(ep) <= cfg.window:
            print(f"  [warn] episode {k} has only {len(ep)} hourly rows (< window {cfg.window}); skipped")
            continue
        rul = (stop - ep.index).total_seconds() / 3600.0
        eps[k] = {"X": ep.values.astype(np.float32),
                  "rul": np.minimum(rul, cfg.max_rul_h).astype(np.float32),
                  "t": np.arange(len(ep))}
    print(f"  {len(ser.columns)} sensors, {len(eps)} usable episodes out of {len(stops)} stoppages, "
          f"{sum(len(e['t']) for e in eps.values())} hourly rows")
    return eps


def make_folds(ids, seed: int, cfg: CacCfg):
    ids = np.array(sorted(ids))
    np.random.RandomState(seed).shuffle(ids)
    n_folds = min(cfg.n_folds, len(ids))
    for k, test in enumerate(np.array_split(ids, n_folds)):
        rest = np.array([i for i in ids if i not in set(test.tolist())])
        np.random.RandomState(seed * 100 + k).shuffle(rest)
        nv, nt, nc = cfg.n_val, cfg.n_ts, cfg.n_cp
        if len(rest) - nv - nt - nc < 3:
            raise ValueError("too few training episodes: lower n_val / n_ts / n_cp or n_folds")
        yield k, rest[nv + nt + nc:], rest[:nv], rest[nv:nv + nt], rest[nv + nt:nv + nt + nc], test


def make_bundle(eps, train, val, ts, cpi, test, cfg: CacCfg):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(np.concatenate([eps[i]["X"] for i in train]))
    w = cfg.window

    def windows(i):
        X = sc.transform(eps[i]["X"]).astype(np.float32)
        win = np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(X, (w, X.shape[1]))[:, 0])
        return win, eps[i]["rul"][w - 1:], eps[i]["t"][w - 1:]

    def collect(ids):
        parts = [(windows(i), i) for i in ids]
        X = np.concatenate([p[0][0] for p in parts])
        y = np.concatenate([p[0][1] for p in parts])
        u = np.concatenate([np.full(len(p[0][1]), p[1]) for p in parts])
        t = np.concatenate([p[0][2] for p in parts])
        return X, y, u, t

    win = {}
    for name, ids in (("train", train), ("val", val), ("cal_ts", ts), ("cal_cp", cpi)):
        X, y, u, _ = collect(ids)
        win[name] = (X, y, u, np.zeros(len(y), dtype=int))
    X_te, y_te, u_te, t_te = collect(sorted(test))
    splits = {"train": train, "val": val, "cal_ts": ts, "cal_cp": cpi}
    return win, X_te, y_te.astype(np.float32), u_te, t_te, np.zeros(len(y_te), dtype=int), splits


KB_EDGES = (0, 24, 48, 96, 168, 336, 500, 10 ** 9)          # predicted-RUL buckets (hours)


def _wilson(k: int, n: int, z: float = 1.2816):
    p = k / n
    den = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(max(c - h, 0.0)), float(min(c + h, 1.0))


def make_kb_builder(past_causes, min_n: int = 8):
    """Knowledge base for one fold.  Each entry = a predicted-RUL bucket; its failure-probability range is the
    80% Wilson interval of P(RUL <= H) observed on the held-out cal_cp windows of THIS fold (never on test
    episodes).  'past_causes' lists the recorded causes of the stoppages of the TRAINING episodes only, kept in
    every entry for traceability (they do not influence the numbers)."""
    def build(q50, y, H):
        q50, y = np.asarray(q50, dtype=float), np.asarray(y, dtype=float)
        kb = []
        for lo, hi in zip(KB_EDGES[:-1], KB_EDGES[1:]):
            m = (q50 >= lo) & (q50 < hi)
            n = int(m.sum())
            if n < min_n:
                continue
            k = int((y[m] <= H).sum())
            plo, phi = _wilson(k, n)
            kb.append({"text": cac_query(float(np.median(q50[m]))), "rul_min": lo, "rul_max": min(hi, 720),
                       "f_prob_lo": plo, "f_prob_hi": phi, "n_windows": n, "dataset": "CAC",
                       "source": "data-derived (cal_cp)", "past_causes": list(past_causes)[:10]})
        if not kb:
            k, n = int((y <= H).sum()), max(len(y), 1)
            plo, phi = _wilson(k, n)
            kb.append({"text": cac_query(float(np.median(q50))), "f_prob_lo": plo, "f_prob_hi": phi,
                       "n_windows": n, "dataset": "CAC", "source": "data-derived (all cal_cp)",
                       "past_causes": list(past_causes)[:10]})
        return kb
    return build


def cac_query(rul_h: float, cond: int = 0) -> str:
    """Text query for the CAC knowledge base (RUL in hours). Adapt the wording to your documents."""
    if rul_h <= 48:
        state = "critical degradation failure expected within 48 hours imminent stoppage"
    elif rul_h <= 168:
        state = "advanced degradation failure expected within one week elevated risk"
    elif rul_h <= 500:
        state = "moderate degradation weeks of remaining life monitor"
    else:
        state = "healthy operation long remaining life low failure risk"
    return f"Combustion chamber asset predicted remaining useful life {int(rul_h)} hours. {state}."


# ---------------------------------------------------------------- run
def run_all(cfg: CacCfg):
    eps = load_cac(cfg)
    ev_meta = pd.read_csv(cfg.events_csv).sort_values(cfg.stop_col).reset_index(drop=True)
    cause_of = {i: str(ev_meta.loc[i, "cause"])[:120] for i in range(len(ev_meta))} if "cause" in ev_meta.columns else {}
    for seed in [int(s) for s in cfg.seeds.split(",")]:
        for k, tr, va, ts, cpi, te in make_folds(list(eps), seed, cfg):
            print(f"=== CAC seed {seed} fold {k}: test episodes {te.tolist()} ===", flush=True)
            bundle = make_bundle(eps, tr, va, ts, cpi, te, cfg)
            win, X_te, y_te, u_te, t_te, cond_te, splits = bundle
            pc = cp.Cfg(dataset="CAC", H=cfg.H, Cm=cfg.Cm, Cf=cfg.Cf, seed=seed, tag=f"_fold{k}",
                        out_dir=cfg.out_dir, kb_path=cfg.kb_path, epochs=cfg.epochs, window=cfg.window,
                        max_rul=cfg.max_rul_h, device=cfg.device, ens_size=cfg.ens_size,
                        target_scale=cfg.target_scale, patience=cfg.patience, weight_decay=cfg.weight_decay,
                        lr_step=cfg.lr_step, T_max=cfg.T_max,
                        run_robustness=False, cost_ratios=(5.0, 10.0, 20.0, 50.0))
            builder = make_kb_builder([cause_of[i] for i in tr if i in cause_of])
            cp.run_from_bundle(pc, win, X_te, y_te, u_te, cond_te, splits, cycle_te=t_te, query_fn=cac_query,
                               kb_builder=builder)


# ---------------------------------------------------------------- episode-level analysis
def load_folds(cfg: CacCfg) -> Dict[int, List[dict]]:
    """seed -> list of fold arrays (as dicts)."""
    out: Dict[int, List[dict]] = {}
    for f in sorted(glob.glob(os.path.join(cfg.out_dir, "CAC", "seed*_fold*_preds.npz"))):
        seed = int(os.path.basename(f).split("_")[0][4:])
        z = np.load(f)
        out.setdefault(seed, []).append({k: z[k] for k in z.files})
    return out


def episode_rows(folds: Dict[int, List[dict]], cfg: CacCfg) -> pd.DataFrame:
    rows = []
    for seed, fl in folds.items():
        for z in fl:
            for config in CONFIGS:
                key = f"interv_{config}"
                if key not in z:
                    continue
                for u in np.unique(z["unit"]):
                    m = z["unit"] == u
                    order = np.argsort(z["cycle"][m])
                    y, iv = z["y"][m][order], z[key][m][order]
                    hit = np.where(iv)[0]
                    if len(hit) == 0:
                        outcome, cost, lead, waste = "missed", cfg.Cf, np.nan, 0.0
                    else:
                        lead = float(y[hit[0]])                      # hours before the stoppage at the first alarm
                        outcome = "timely" if lead <= cfg.H else "premature"
                        waste = max(lead - cfg.H, 0.0)
                        cost = cfg.Cm + cfg.waste_per_horizon * waste / cfg.H
                    rows.append({"seed": seed, "episode": int(u), "config": config, "outcome": outcome,
                                 "cost": cost, "lead_h": lead, "wasted_h": waste})
            # model-free baselines (never use the sensor model): alarm at the first window, or T hours after the
            # previous stoppage (= episode start). A schedule that fires after the stoppage is a missed failure.
            for u in np.unique(z["unit"]):
                m = z["unit"] == u
                order = np.argsort(z["cycle"][m])
                y, t = z["y"][m][order], z["cycle"][m][order]
                for config in BASELINES:
                    T_ = 0 if config == "Always-alarm" else int(config.split("-")[1][:-1])
                    hit = np.where(t >= T_)[0]
                    if len(hit) == 0:
                        outcome, cost, lead, waste = "missed", cfg.Cf, np.nan, 0.0
                    else:
                        lead = float(y[hit[0]])
                        outcome = "timely" if lead <= cfg.H else "premature"
                        waste = max(lead - cfg.H, 0.0)
                        cost = cfg.Cm + cfg.waste_per_horizon * waste / cfg.H
                    rows.append({"seed": seed, "episode": int(u), "config": config, "outcome": outcome,
                                 "cost": cost, "lead_h": lead, "wasted_h": waste})
    return pd.DataFrame(rows)


def boot_ci(per_episode: np.ndarray, n_boot=5000, seed=0, stat=np.mean):
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(per_episode), size=(n_boot, len(per_episode)))
    b = np.array([stat(per_episode[i]) for i in idx])
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def analyze(cfg: CacCfg):
    folds = load_folds(cfg)
    if not folds:
        print("no saved folds in", cfg.out_dir)
        return
    od = os.path.join(cfg.out_dir, "analysis")
    os.makedirs(od, exist_ok=True)
    ep = episode_rows(folds, cfg)
    ep.to_csv(os.path.join(od, "cac_episode_results.csv"), index=False)
    n_ep = ep["episode"].nunique()

    # per-config episode-level summary (cost per episode averaged over seeds, bootstrap over episodes)
    per_ep = ep.groupby(["config", "episode"])["cost"].mean().unstack("config")
    rows = []
    for c in [c for c in CONFIGS if c in per_ep.columns]:
        v = per_ep[c].values
        lo, hi = boot_ci(v, stat=np.sum)
        e = ep[ep["config"] == c]
        per_seed = e.groupby("seed")["outcome"].value_counts().unstack(fill_value=0)
        rows.append({"config": c, "episodes": n_ep, "total_cost": v.sum(), "ci_low": lo, "ci_high": hi,
                     "timely": per_seed.get("timely", pd.Series(0)).mean(),
                     "premature": per_seed.get("premature", pd.Series(0)).mean(),
                     "missed": per_seed.get("missed", pd.Series(0)).mean(),
                     "mean_lead_h": e["lead_h"].mean(), "mean_wasted_h": e["wasted_h"].mean()})
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(od, "cac_episode_summary.csv"), index=False)

    # Premature alarms: with waste_per_horizon = 0 an alarm hundreds of hours early costs the same as a timely
    # one, so "alarm immediately" can never lose. Show the total cost under several prices for wasted life
    # (w = extra cost, in units of Cm, per H hours of useful life thrown away). No single w is chosen.
    grid = (0.0, 0.1, 0.25, 0.5, 1.0)
    piv = {}
    for w in grid:
        cst = np.where(ep["outcome"] == "missed", cfg.Cf, cfg.Cm + w * ep["wasted_h"] / cfg.H)
        t = ep.assign(c=cst).groupby(["config", "episode"])["c"].mean().unstack("config")
        piv[f"w={w}"] = {c: float(t[c].sum()) for c in t.columns}
    pv = pd.DataFrame(piv).reindex([c for c in CONFIGS if c in per_ep.columns])
    pv.to_csv(os.path.join(od, "cac_cost_by_waste_price.csv"))

    # paired differences with CI over episodes
    pairs = [("C", "C-GATE"), ("C", "C-CRC10"), ("C", "C-CRC05"), ("C", "C-CRC20"), ("Always-alarm", "C-CRC10"), ("Always-alarm", "C"), ("Sched-168h", "C"), ("Sched-336h", "C"), ("A+margin", "C"), ("B", "C"), ("C", "C+iso"), ("C", "E"), ("C", "E-kNN"), ("C", "Ens+CP"), ("C", "MCD+CP")]
    prow = []
    for a, b in pairs:
        if a in per_ep.columns and b in per_ep.columns:
            d = (per_ep[b] - per_ep[a]).values
            lo, hi = boot_ci(d, stat=np.sum)
            prow.append({"from": a, "to": b, "total_cost_diff": d.sum(), "ci_low": lo, "ci_high": hi,
                         "significant": bool(lo > 0 or hi < 0), "episodes": n_ep})
    paired = pd.DataFrame(prow)
    paired.to_csv(os.path.join(od, "cac_paired.csv"), index=False)

    # window-level coverage of the 80% interval, bootstrap over episodes (B = raw, C = calibrated)
    cov_rows = []
    for name, lo_k, hi_k in (("B (uncalibrated)", "q10_raw", "q90_raw"), ("C (calibrated)", "q10C", "q90C")):
        per = {}
        for fl in folds.values():
            for z in fl:
                inside = (z["y"] >= z[lo_k]) & (z["y"] <= z[hi_k])
                for u in np.unique(z["unit"]):
                    m = z["unit"] == u
                    s_, n_ = per.get(int(u), (0.0, 0))
                    per[int(u)] = (s_ + inside[m].sum(), n_ + m.sum())
        s_arr = np.array([v[0] for v in per.values()])
        n_arr = np.array([v[1] for v in per.values()])
        rng = np.random.RandomState(0)
        idx = rng.randint(0, len(s_arr), size=(5000, len(s_arr)))
        boots = 100 * s_arr[idx].sum(1) / n_arr[idx].sum(1)
        cov_rows.append({"config": name, "coverage80_%": 100 * s_arr.sum() / n_arr.sum(),
                         "ci_low": np.percentile(boots, 2.5), "ci_high": np.percentile(boots, 97.5),
                         "episodes": len(per)})
    cov = pd.DataFrame(cov_rows)
    cov.to_csv(os.path.join(od, "cac_coverage.csv"), index=False)

    # RAG activation log (retrieval traceability): all folds, text-KB and kNN
    logs = []
    for f in sorted(glob.glob(os.path.join(cfg.out_dir, "CAC", "seed*_fold*_activations_E*.csv"))):
        d = pd.read_csv(f)
        d["file"] = os.path.basename(f)
        logs.append(d)
    if logs:
        pd.concat(logs).to_csv(os.path.join(od, "cac_activation_log.csv"), index=False)

    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(f"\n=== CAC episode-level results ({n_ep} episodes; cost = Cm per intervention, Cf per missed stoppage) ===")
    print(summ.round(2).to_string(index=False))
    print("\n=== Total cost when premature alarms also cost w x (wasted horizons), w = 0 is the unfair default ===")
    print(pv.round(1).to_string())
    print("\n=== Paired differences (95% bootstrap CI over episodes; negative = 'to' is better) ===")
    print(paired.round(2).to_string(index=False))
    print("\n=== Interval coverage (80% target) ===")
    print(cov.round(1).to_string(index=False))
    print(f"\nWith only {n_ep} events these are exploratory results: report them as feasibility and "
          f"calibration evidence, not as proof of economic superiority.\nSaved to {od}")


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default="./cac/cac_series.csv")
    ap.add_argument("--events", default="./cac/cac_events.csv")
    ap.add_argument("--kb", default="./kb/kb_CAC.json")
    ap.add_argument("--out", default="./results_cac")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--window", type=int, default=48)
    ap.add_argument("--H", type=int, default=48)
    ap.add_argument("--folds", type=int, default=9)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--n_val", type=int, default=2, help="episodes for early stopping / validation")
    ap.add_argument("--n_ts", type=int, default=2, help="episodes for temperature scaling")
    ap.add_argument("--n_cp", type=int, default=2, help="episodes for conformal calibration")
    ap.add_argument("--target_scale", type=float, default=1.0, help="train on RUL/target_scale (720 recommended)")
    ap.add_argument("--patience", type=int, default=0, help="early-stopping patience in epochs (0 = off)")
    ap.add_argument("--wd", type=float, default=0.0, help="weight decay")
    ap.add_argument("--lr_step", type=int, default=20, help="halve the learning rate every N epochs")
    ap.add_argument("--T_max", type=float, default=1.5, help="upper bound of the temperature search")
    ap.add_argument("--analyze_only", action="store_true")
    a, _ = ap.parse_known_args()
    cfg = CacCfg(series_csv=a.series, events_csv=a.events, kb_path=a.kb, out_dir=a.out, seeds=a.seeds,
                 epochs=a.epochs, window=a.window, H=a.H, n_folds=a.folds, n_val=a.n_val, n_ts=a.n_ts, n_cp=a.n_cp,
                 target_scale=a.target_scale, patience=a.patience, weight_decay=a.wd, lr_step=a.lr_step, T_max=a.T_max,
                 device=("cuda" if (a.device == "auto" and _cuda()) else ("cpu" if a.device == "auto" else a.device)))
    if not a.analyze_only:
        run_all(cfg)
    analyze(cfg)


if __name__ == "__main__":
    main()
