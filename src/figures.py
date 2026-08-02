"""Build the result figures from aggregated metrics + GPU logs.

    python src/figures.py      # after the sweep + aggregate.py

Reads results/all_metrics.tsv, results/ablation_splice.tsv, logs/util_<model>_<prot>.csv.
Writes results/figures/{model_comparison,gpu_dashboard,acc_vs_compute,splice_ablation}.png
"""
import glob, re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
RES, LOGS, FIGS = REPO / "results", REPO / "logs", REPO / "results/figures"
ORDER = ["CNN", "RNABERT", "SpliceBERT", "RNA-FM (LoRA)"]
COLOR = {"CNN": "#2a78d6", "RNABERT": "#eb6834", "SpliceBERT": "#1baf7a", "RNA-FM (LoRA)": "#8a5cf6"}
KEY2LABEL = {"cnn": "CNN", "rnabert": "RNABERT", "splicebert": "SpliceBERT", "rnafm": "RNA-FM (LoRA)"}
INK, MUTED, GRID, SURF = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"

mpl.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.edgecolor": "#c3c2b7", "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                     "text.color": INK, "axes.labelcolor": "#52514e"})


def parse_util():
    rows = []
    for p in glob.glob(str(LOGS / "util_*.csv")):
        m = re.match(r"util_([a-z]+)_(.+)\.csv", Path(p).name)
        if not m:
            continue
        try:
            d = pd.read_csv(p)
        except Exception:
            continue
        if d.empty:
            continue
        ucol = next((c for c in d.columns if "utilization.gpu" in c), None)
        mcol = next((c for c in d.columns if "memory.used" in c), None)
        util = d[ucol].astype(str).str.replace(r"[^0-9.]", "", regex=True).replace("", np.nan).astype(float) if ucol else pd.Series([np.nan])
        mem = d[mcol].astype(str).str.replace(r"[^0-9.]", "", regex=True).replace("", np.nan).astype(float) if mcol else pd.Series([np.nan])
        rows.append(dict(model=KEY2LABEL.get(m.group(1), m.group(1)), protein=m.group(2),
                         seconds=len(d) * 5, gpu_util_mean=util.mean(), gpu_mem_peak_gb=mem.max() / 1024))
    return pd.DataFrame(rows)


def fig_model_comparison(met):
    piv = met.pivot_table(index="protein", columns="model", values="test_auroc")
    cols = [c for c in ORDER if c in piv.columns]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.5, 1]})
    x = np.arange(len(cols))
    means = [piv[c].mean() for c in cols]
    for i, c in enumerate(cols):                       # per-protein points + mean bar
        ax.bar(i, means[i], 0.6, color=COLOR[c], alpha=0.85, zorder=1)
        ax.scatter(np.full(piv[c].notna().sum(), i) + (np.random.default_rng(0).random(piv[c].notna().sum()) - .5) * .3,
                   piv[c].dropna(), s=14, color=INK, alpha=0.35, zorder=2)
        ax.text(i, means[i] + 0.008, f"{means[i]:.3f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(cols, rotation=15); ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("test AUROC"); ax.set_title("Mean test AUROC across 16 proteins (dots = proteins)", loc="left", fontsize=11)

    best = piv[cols].idxmax(axis=1).value_counts()     # wins per model
    for i, c in enumerate(cols):
        ax2.barh(i, best.get(c, 0), color=COLOR[c])
        ax2.text(best.get(c, 0) + 0.1, i, str(best.get(c, 0)), va="center", fontsize=9)
    ax2.set_yticks(range(len(cols))); ax2.set_yticklabels(cols)
    ax2.set_xlabel("# proteins where model wins"); ax2.set_title("Per-protein winner", loc="left", fontsize=11)
    fig.suptitle("Model comparison — all models tuned to peak", x=0.01, ha="left", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "model_comparison.png", dpi=150); plt.close(fig)


def fig_gpu_dashboard(util):
    g = util.groupby("model").agg(seconds=("seconds", "mean"), util=("gpu_util_mean", "mean"),
                                  mem=("gpu_mem_peak_gb", "max")).reindex([m for m in ORDER if m != "CNN"]).dropna(how="all")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, col, title, unit in zip(axes, ["seconds", "util", "mem"],
                                    ["Mean train time / protein", "Mean GPU utilisation", "Peak GPU memory"],
                                    ["seconds", "%", "GB"]):
        for i, m in enumerate(g.index):
            ax.bar(i, g[col][m], color=COLOR[m])
            ax.text(i, g[col][m], f"{g[col][m]:.0f}" if unit != "GB" else f"{g[col][m]:.1f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(len(g.index))); ax.set_xticklabels(g.index, rotation=15)
        ax.set_ylabel(unit); ax.set_title(title, loc="left", fontsize=11)
    fig.suptitle("GPU performance — the LM fine-tunes", x=0.01, ha="left", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "gpu_dashboard.png", dpi=150); plt.close(fig)


def fig_acc_vs_compute(met, util):
    m = met.merge(util[["model", "protein", "seconds"]], on=["model", "protein"], how="left")
    m.loc[m.model == "CNN", "seconds"] = m.loc[m.model == "CNN", "seconds"].fillna(30)   # CPU baseline
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for c in ORDER:
        d = m[m.model == c]
        if d.empty:
            continue
        ax.scatter(d.seconds, d.test_auroc, s=28, color=COLOR[c], alpha=0.7, label=c, edgecolors="none")
    ax.set_xscale("log"); ax.set_xlabel("training time per protein (s, log)"); ax.set_ylabel("test AUROC")
    ax.set_title("Accuracy vs compute cost", loc="left", fontsize=11.5)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "acc_vs_compute.png", dpi=150); plt.close(fig)


def fig_splice_ablation(abl):
    """AUROC lost when negatives are splice-distance-matched = the splice-proximity share."""
    prots = sorted(abl.protein.unique())
    models = [m for m in ORDER if m in abl.model.unique()]
    x = np.arange(len(prots)); w = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    for j, m in enumerate(models):
        d = abl[abl.model == m].set_index("protein").reindex(prots)
        ax.bar(x + j * w - 0.4 + w / 2, d["drop"].values, w, color=COLOR[m], label=m)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(prots)
    ax.set_ylabel("AUROC drop (primary − splice-matched)")
    ax.set_title("Splice ablation — how much of the signal was splice proximity", loc="left", fontsize=11.5)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    fig.tight_layout(); FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "splice_ablation.png", dpi=150); plt.close(fig)


def main():
    met = pd.read_csv(RES / "all_metrics.tsv", sep="\t")
    util = parse_util()
    fig_model_comparison(met)
    if not util.empty:
        fig_gpu_dashboard(util)
        fig_acc_vs_compute(met, util)
        util.to_csv(RES / "gpu_perf.tsv", sep="\t", index=False)
    abl_path = RES / "ablation_splice.tsv"
    if abl_path.exists():
        fig_splice_ablation(pd.read_csv(abl_path, sep="\t"))
    print(f"wrote figures to {FIGS}" + ("" if not util.empty else "  (no GPU logs — comparison only)"))


if __name__ == "__main__":
    main()
