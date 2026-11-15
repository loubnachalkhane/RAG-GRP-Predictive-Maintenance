"""Public-benchmark figures for the revision: decision validation, risk-controlled threshold, RAG analysis, robustness.
Usage:  python make_figures_public2.py --tables . --results C:\\calibrag_results\\run_XXXX --out figures_public
Reads <tables>\\{main,sweep,robust,activation}_results.csv and <results>\\<DATASET>\\seed*_preds.npz.
  P1 reliability of the failure probability (with bin counts, Brier)   P2 risk-controlled threshold: achieved miss rate
  P3 cost of the risk-controlled threshold vs C and trivial policies   P4 three-arm test (sensor / retrieval / both)
  P5 RAG activation breakdown                                          P6 robustness to the cost ratio
  P7 robustness to sensor noise
Error bars across seeds are seed-to-seed variation; the primary inference is the engine-level bootstrap in paired_bootstrap.csv.
"""
import argparse, glob, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--tables", default=".")
ap.add_argument("--results", default="results")
ap.add_argument("--out", default="figures_public")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
INK, MUTED, GRID = "#1f2933", "#6b7785", "#e4e7eb"
DS = ["FD001", "FD002", "FD003", "FD004", "NCMAPSS"]
COL = {"A": "#9aa5b1", "A+margin": "#d97706", "B": "#c2410c", "C": "#1d4ed8", "C-CRC05": "#0f766e",
       "C-CRC10": "#14b8a6", "C-CRC20": "#99f6e4", "Always-alarm": "#111827", "Never-alarm": "#7c3aed", "L-kNN": "#0f766e", "R-kNN": "#d97706"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": .7, "axes.axisbelow": True})


def rd(name):
    p = os.path.join(a.tables, f"{name}_results.csv")
    if os.path.exists(p):
        return pd.read_csv(p)
    print("  [skip] not found:", p)


def save(fig, name, note=""):
    fig.tight_layout(rect=(0, .03, 1, 1))
    if note:
        fig.text(.01, .005, note, fontsize=7.5, color=MUTED)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.out, f"{name}.{ext}"), dpi=200)
    plt.close(fig)
    print("saved", name)


def dsets(df):
    return [d for d in DS if d in set(df["dataset"])] + sorted(set(df["dataset"]) - set(DS))


def mci(x):
    x = np.asarray(x, float)
    return x.mean(), (1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0)


main, sweep, rob, act = rd("main"), rd("sweep"), rd("robust"), rd("activation")

# ---------- P1 reliability with bin counts
files = {d: sorted(glob.glob(os.path.join(a.results, d, "seed*_preds.npz"))) for d in DS}
files = {d: f for d, f in files.items() if f}
if files:
    fig, axs = plt.subplots(2, len(files), figsize=(3.3 * len(files) + .6, 5.2), squeeze=False,
                            gridspec_kw={"height_ratios": [3, 1]})
    edges = np.linspace(0, 1, 11)
    for j, (d, fl) in enumerate(files.items()):
        F, Y = [], []
        for f in fl:
            z = np.load(f)
            F.append(z["F_C"]); Y.append((z["y"] <= float(z["H"])).astype(float))
        F, Y = np.concatenate(F), np.concatenate(Y)
        idx = np.clip(np.digitize(F, edges) - 1, 0, 9)
        cnt = np.array([(idx == b).sum() for b in range(10)])
        obs = np.array([Y[idx == b].mean() if (idx == b).sum() >= 20 else np.nan for b in range(10)])
        mid = (edges[:-1] + edges[1:]) / 2
        ax, bx = axs[0, j], axs[1, j]
        ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1)
        ax.plot(mid, obs, "o-", color=COL["C"], lw=2, ms=4)
        ax.set_title(f"{d}  (Brier {np.mean((F - Y) ** 2):.3f})", loc="left", fontsize=9.5, fontweight="bold")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticklabels([])
        if j == 0: ax.set_ylabel("Observed frequency of RUL <= H")
        bx.bar(mid, cnt, width=.085, color="#93c5fd"); bx.set_yscale("log"); bx.minorticks_off(); bx.set_xlim(0, 1)
        bx.set_xlabel("Predicted failure probability F")
        if j == 0: bx.set_ylabel("windows")
    save(fig, "P1_reliability_decision_probability",
         "Config C, pooled over seeds. Bins with fewer than 20 windows are not plotted. Lower panels: windows per bin (log scale).")

if main is not None:
    ds = dsets(main)
    piv = main.pivot_table(index=["dataset", "seed"], columns="config", values="Cost")
    # ---------- P2 achieved miss rate vs target
    if "MissRate" in main.columns and "C-CRC10" in set(main["config"]):
        fig, ax = plt.subplots(figsize=(7.2, 3.9))
        cfgs = [c for c in ["C", "C-CRC05", "C-CRC10", "C-CRC20"] if c in set(main["config"])]
        wd = .8 / len(cfgs)
        for k, c in enumerate(cfgs):
            vals = []
            for d in ds:
                v = main[(main["dataset"] == d) & (main["config"] == c)]["MissRate"].values
                vals.append(mci(v) if len(v) else (np.nan, 0))
            x = np.arange(len(ds)) + (k - (len(cfgs) - 1) / 2) * wd
            ax.bar(x, [v[0] * 100 for v in vals], wd, yerr=[v[1] * 100 for v in vals], capsize=2, color=COL[c], label=c)
        for dl, c in ((5, "C-CRC05"), (10, "C-CRC10"), (20, "C-CRC20")):
            if c in cfgs: ax.axhline(dl, color=COL[c], ls=":", lw=1.3)
        ax.set_xticks(range(len(ds))); ax.set_xticklabels(ds)
        ax.set_ylabel("Missed critical windows (%)"); ax.legend(frameon=False, fontsize=8, ncol=4)
        ax.set_title("Risk-controlled threshold: achieved miss rate (dotted: targets 5/10/20%)", loc="left", fontweight="bold")
        save(fig, "P2_crc_miss_rate", "Window-level miss rate on test engines; error bars: 95% across seeds. Windows of one engine are dependent, so this is an empirical check of the target, not a proof.")
    # ---------- P3 cost relative to C
    cmp_ = [c for c in ["A+margin", "B", "C-CRC05", "C-CRC10", "C-CRC20", "Never-alarm", "Always-alarm"] if c in piv.columns]
    if "C" in piv.columns and cmp_:
        fig, axs = plt.subplots(1, len(ds), figsize=(2.9 * len(ds) + .6, 3.8), squeeze=False)
        for ax, d in zip(axs[0], ds):
            sub = piv.loc[d]
            vals = [mci(100 * (sub[c] - sub["C"]) / sub["C"]) for c in cmp_]
            ax.barh(range(len(cmp_)), [v[0] for v in vals], xerr=[v[1] for v in vals], color=[COL[c] for c in cmp_], capsize=2)
            lim = 60
            for i, (v, _) in enumerate(vals):
                if v > lim:
                    ax.text(lim - 2, i, f"+{v:.0f}% ->", ha="right", va="center", color="white", fontsize=8, fontweight="bold")
            ax.set_xlim(-30, lim)
            ax.axvline(0, color=INK, lw=1); ax.set_yticks(range(len(cmp_)))
            ax.set_yticklabels(cmp_ if d == ds[0] else []); ax.invert_yaxis(); ax.set_title(d, loc="left", fontweight="bold")
            ax.grid(axis="y", visible=False); ax.set_xlabel("cost vs C (%)")
        save(fig, "P3_cost_vs_C", "Positive = costlier than C. Window-level, Cf/Cm=10; 95% CI across seeds. Bars beyond +60% clipped.")
    # ---------- P4 three-arm
    arms = [c for c in ["C+platt", "R-kNN", "L-kNN"] if c in piv.columns]
    if len(arms) == 3:
        names = {"C+platt": "sensor only", "R-kNN": "retrieval only", "L-kNN": "sensor + retrieval"}
        fig, ax = plt.subplots(figsize=(7.2, 3.9)); wd = .26
        for k, c in enumerate(arms):
            vals = [mci(100 * (piv.loc[d][c] - piv.loc[d]["C+platt"]) / piv.loc[d]["C+platt"]) for d in ds]
            ax.bar(np.arange(len(ds)) + (k - 1) * wd, [v[0] for v in vals], wd, yerr=[v[1] for v in vals], capsize=2,
                   color=["#9aa5b1", "#d97706", "#1d4ed8"][k], label=names[c])
        ax.axhline(0, color=INK, lw=1); ax.set_xticks(range(len(ds))); ax.set_xticklabels(ds)
        ax.set_ylabel("cost relative to sensor-only (%)"); ax.legend(frameon=False, fontsize=8.5)
        ax.set_title("Three-arm test: does retrieval add decision value?", loc="left", fontweight="bold")
        save(fig, "P4_three_arm", "All arms recalibrated by the same logistic fit on the same held-out engines; mean and 95% CI across seeds.")

# ---------- P5 activation breakdown
if act is not None and "n_total" in act.columns:
    g = act.groupby(["dataset", "config"])[["n_total", "n_activated", "n_beneficial", "n_harmful", "n_no_effect"]].sum().reset_index()
    cfgs = [c for c in ["D", "E", "D-kNN", "E-kNN"] if c in set(g["config"])]
    if cfgs:
        fig, axs = plt.subplots(1, len(cfgs), figsize=(3.4 * len(cfgs) + .6, 3.8), squeeze=False, sharey=True)
        cats = [("not activated", "#cbd2d9"), ("no effect", "#7c8da0"), ("beneficial", "#15803d"), ("harmful", "#b91c1c")]
        for ax, c in zip(axs[0], cfgs):
            sub = g[g["config"] == c].set_index("dataset").reindex([d for d in DS if d in set(g["dataset"])])
            tot = sub["n_total"].replace(0, np.nan)
            parts = [(tot - sub["n_activated"]) / tot, (sub["n_activated"] - sub["n_beneficial"] - sub["n_harmful"]) / tot,
                     sub["n_beneficial"] / tot, sub["n_harmful"] / tot]
            left = np.zeros(len(sub))
            for (lab, colr), v in zip(cats, parts):
                v = (100 * v).fillna(0).values
                ax.bar(range(len(sub)), v, bottom=left, color=colr, label=lab); left += v
            ax.set_xticks(range(len(sub))); ax.set_xticklabels(sub.index, rotation=30); ax.set_title(c, loc="left", fontweight="bold")
        axs[0][0].set_ylabel("share of test windows (%)"); axs[0][-1].legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1, .5))
        save(fig, "P5_rag_activation", "Summed over seeds. 'not activated': gate closed; 'no effect': activated, same decision as C.")

# ---------- P6 cost-ratio robustness
if sweep is not None:
    ds = dsets(sweep); show = [c for c in ["A", "A+margin", "B", "C", "Always-alarm"] if c in set(sweep["config"])]
    fig, axs = plt.subplots(1, len(ds), figsize=(3.0 * len(ds) + .6, 3.8), squeeze=False)
    for ax, d in zip(axs[0], ds):
        for c in show:
            s = sweep[(sweep["dataset"] == d) & (sweep["config"] == c)].groupby("Cf")["Cost"].mean()
            ax.plot(s.index, s.values, "o-", ms=3, lw=2.2 if c == "C" else 1.5, color=COL.get(c, MUTED), label=c)
        ax.set_xscale("log"); ax.minorticks_off(); ax.set_xticks(sorted(sweep["Cf"].unique())); ax.set_xticklabels([f"{x:g}" for x in sorted(sweep["Cf"].unique())])
        ax.set_title(d, loc="left", fontweight="bold"); ax.set_xlabel("Cf / Cm")
    axs[0][0].set_ylabel("Total cost"); axs[0][0].legend(frameon=False, fontsize=7.5)
    save(fig, "P6_robust_cost_ratio", "Mean over seeds. Cost ratio C_f/C_m on a logarithmic axis; C_m = 1.")

# ---------- P7 noise robustness
if rob is not None:
    ds = dsets(rob); show = [c for c in ["A", "C", "L-kNN", "R-kNN"] if c in set(rob["config"])]
    fig, axs = plt.subplots(1, len(ds), figsize=(3.0 * len(ds) + .6, 3.8), squeeze=False)
    for ax, d in zip(axs[0], ds):
        for c in show:
            s = rob[(rob["dataset"] == d) & (rob["config"] == c)].groupby("noise")["Cost"].mean()
            ax.plot(s.index, s.values, "o-", ms=3, lw=2.2 if c == "C" else 1.5, color=COL.get(c, "#0f766e"), label=c)
        ax.set_title(d, loc="left", fontweight="bold"); ax.set_xlabel("noise (fraction of sensor range)")
    axs[0][0].set_ylabel("Total cost"); axs[0][0].legend(frameon=False, fontsize=7.5)
    save(fig, "P7_robust_noise", "Mean over seeds and noise repetitions; same trained models, same calibration.")
