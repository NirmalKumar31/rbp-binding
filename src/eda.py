"""Intense EDA + assumption stress-test for the rbp-binding datasets.

Reads data/processed/<P>/dataset.tsv (+ raw eCLIP peaks) and produces figures and a
red-flag report that checks every modelling assumption BEFORE we train anything:

  split balance    do all 16 proteins have enough pairs in val/test under the new split?
  class balance    positives vs negatives per protein
  GC match         positives vs matched negatives (matching worked?)
  composition      nucleotide freq, region, peak width
  motif signal     top enriched 6-mer per protein + how centred it is
  complexity       low-complexity window fraction (U-rich / homopolymer sites)
  redundancy       exact-duplicate windows + cross-split identical sequences (leakage)

    python src/eda.py                                  # uses data/processed
    python src/eda.py --processed ../rbp-v2/data/processed   # preview on existing windows

The new split (teammate standard): test = chr1/chr2/chr20, val = chr19/16/13/18, rest train.
"""
import argparse, gzip
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
TEST_CHR = {"chr1", "chr2", "chr20"}
VAL_CHR = {"chr19", "chr16", "chr13", "chr18"}
MIN_EVAL_PAIRS = 100          # below this a val/test set is too small to trust
KNOWN = {"TARDBP": "UGUGUG", "FUS": "GGUG", "RBFOX2": "UGCAUG", "PUM2": "UGUAUA",
         "PUM1": "UGUAUA", "IGF2BP1": "ACACAC", "TIA1": "UUUUU", "PTBP1": "UCUCU", "QKI": "ACUAAC"}

INK, MUTED, GRID, SURF = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
BLUE, ORANGE, GREEN, PURPLE, GOLD, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cf6", "#e0b020", "#9aa0a6"
mpl.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": "#c3c2b7", "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "text.color": INK, "axes.labelcolor": "#52514e"})


def split_of(chrom):
    return "test" if chrom in TEST_CHR else "val" if chrom in VAL_CHR else "train"


def kmers(seqs, k=6):
    c = Counter()
    for s in seqs:
        for i in range(len(s) - k + 1):
            c[s[i:i + k]] += 1
    return c


def low_complexity_frac(seqs):
    """fraction of windows dominated by one nucleotide (>60%) or a long homopolymer (>=8)."""
    lc = 0
    for s in seqs:
        n = len(s)
        top = max(s.count(b) for b in "ACGU") / n
        run = mx = 1
        for i in range(1, n):
            run = run + 1 if s[i] == s[i - 1] else 1
            mx = max(mx, run)
        if top > 0.60 or mx >= 8:
            lc += 1
    return lc / max(len(seqs), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default=str(REPO / "data/processed"))
    ap.add_argument("--encode", default=str(REPO / "data/raw/encode"))
    ap.add_argument("--out", default=str(REPO / "eda"))
    a = ap.parse_args()
    PROC, ENC, OUT = Path(a.processed), Path(a.encode), Path(a.out)
    FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)

    reg = pd.read_csv(REPO / "config/proteins.tsv", sep="\t")
    proteins = list(reg.protein)
    acc = dict(zip(reg.protein, reg.accession))

    rows, region_frac, split_counts, motif_pos = [], {}, {}, {}
    gc_pos_all, gc_neg_all = [], []
    nuc_pos, nuc_neg = Counter(), Counter()

    for p in proteins:
        df = pd.read_csv(PROC / p / "dataset.tsv", sep="\t")
        df["split2"] = df.chrom.map(split_of)
        pos, neg = df[df.label == 1], df[df.label == 0]

        # split balance on positives (pairs)
        sc = pos.split2.value_counts()
        split_counts[p] = {s: int(sc.get(s, 0)) for s in ("train", "val", "test")}

        # gc
        gc_pos_all.append(pos.gc.to_numpy()); gc_neg_all.append(neg.gc.to_numpy())
        pg, ng = pos.gc.to_numpy(), neg.gc.to_numpy()
        m = min(len(pg), len(ng))
        pct_relaxed = float((np.abs(pg[:m] - ng[:m]) > 0.05).mean()) * 100

        # nucleotide composition
        for s in pos.seq_rna:
            nuc_pos.update(s)
        for s in neg.seq_rna:
            nuc_neg.update(s)

        # region
        rc = pos.region.value_counts(normalize=True); region_frac[p] = rc

        # motif enrichment
        kp, kn = kmers(pos.seq_rna), kmers(neg.seq_rna)
        tp, tn = sum(kp.values()), sum(kn.values())
        enr = {km: np.log2((cp / tp) / ((kn.get(km, 0) + 1) / (tn + 1)))
               for km, cp in kp.items() if cp >= 30}
        best = max(enr, key=enr.get) if enr else "-"

        # how centred is the top motif (positions of its occurrences across the 101-window)
        centred = np.nan; pos_hist = np.zeros(101 - 6 + 1)
        if best != "-":
            for s in pos.seq_rna:
                i = s.find(best)
                while i != -1:
                    if i < len(pos_hist):
                        pos_hist[i] += 1
                    i = s.find(best, i + 1)
            if pos_hist.sum():
                centred = float(pos_hist[30:66].sum() / pos_hist.sum()) * 100   # central 36 of 96 starts
                motif_pos[p] = pos_hist / pos_hist.sum()

        # complexity + redundancy
        lc = low_complexity_frac(pos.seq_rna)
        dup = 1 - pos.seq_rna.nunique() / len(pos)

        rows.append(dict(protein=p, pairs=len(pos),
            train=split_counts[p]["train"], val=split_counts[p]["val"], test=split_counts[p]["test"],
            gc_pos=round(pos.gc.mean(), 3), gc_neg=round(neg.gc.mean(), 3),
            gc_gap=round(abs(pos.gc.mean() - neg.gc.mean()), 3), pct_gc_relaxed=round(pct_relaxed, 1),
            top_region=rc.index[0], top_region_pct=round(rc.iloc[0] * 100),
            top_6mer=best, enrich_log2=round(enr[best], 2) if enr else 0.0, known_motif=KNOWN.get(p, ""),
            motif_centred_pct=round(centred, 1) if centred == centred else np.nan,
            low_complexity_pct=round(lc * 100, 1), dup_pct=round(dup * 100, 2)))
        print(f"{p:9} pairs={len(pos):5} split(tr/va/te)={split_counts[p]['train']:5}/"
              f"{split_counts[p]['val']:4}/{split_counts[p]['test']:4} gcgap={rows[-1]['gc_gap']:.3f} "
              f"6mer={best}({rows[-1]['enrich_log2']}) lc={lc*100:4.1f}% dup={dup*100:.2f}%")

    S = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    S.to_csv(OUT / "eda_summary.tsv", sep="\t", index=False)

    # ---- figures ----
    order = S.sort_values("pairs").protein.tolist()
    idx = {p: i for i, p in enumerate(order)}

    # 1) dataset sizes
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(order, S.set_index("protein").loc[order, "pairs"], color=BLUE)
    ax.set_xlabel("matched pairs"); ax.set_title("Dataset size per protein", loc="left")
    fig.tight_layout(); fig.savefig(FIG / "01_dataset_sizes.png", dpi=150); plt.close(fig)

    # 2) split balance (the decisive one)
    fig, ax = plt.subplots(figsize=(9, 6)); left = np.zeros(len(order))
    for s, c in [("train", GREY), ("val", GOLD), ("test", ORANGE)]:
        vals = np.array([split_counts[p][s] for p in order])
        ax.barh(order, vals, left=left, color=c, label=s); left += vals
    ax.axvline(MIN_EVAL_PAIRS, color=INK, lw=1, ls="--")
    ax.set_xlabel("positive pairs"); ax.legend(frameon=False, ncol=3)
    ax.set_title(f"Split balance under new split (dashed = {MIN_EVAL_PAIRS}-pair floor)", loc="left")
    fig.tight_layout(); fig.savefig(FIG / "02_split_balance.png", dpi=150); plt.close(fig)

    # 3) chromosome distribution heatmap (positive fraction per protein x chrom)
    chroms = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
    M = np.zeros((len(proteins), len(chroms)))
    for r, p in enumerate(proteins):
        df = pd.read_csv(PROC / p / "dataset.tsv", sep="\t")
        vc = df[df.label == 1].chrom.value_counts(normalize=True)
        for cix, c in enumerate(chroms):
            M[r, cix] = vc.get(c, 0) * 100
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(M, aspect="auto", cmap="magma")
    ax.set_xticks(range(len(chroms))); ax.set_xticklabels([c.replace("chr", "") for c in chroms])
    ax.set_yticks(range(len(proteins))); ax.set_yticklabels(proteins)
    for cix, c in enumerate(chroms):
        col = ORANGE if c in TEST_CHR else GOLD if c in VAL_CHR else None
        if col:
            ax.add_patch(plt.Rectangle((cix - .5, -.5), 1, len(proteins), fill=False, edgecolor=col, lw=2))
    ax.set_title("Positives per chromosome (%) — orange=test, gold=val", loc="left")
    fig.colorbar(im, ax=ax, label="% of positives"); fig.tight_layout()
    fig.savefig(FIG / "03_chrom_distribution.png", dpi=150); plt.close(fig)

    # 4) GC match
    fig, ax = plt.subplots(figsize=(9, 5)); x = np.arange(len(S))
    ax.scatter(x, S.gc_pos, s=34, color=ORANGE, label="positives")
    ax.scatter(x, S.gc_neg, s=34, color=BLUE, marker="x", label="matched negatives")
    ax.set_xticks(x); ax.set_xticklabels(S.protein, rotation=90); ax.set_ylabel("mean GC")
    ax.legend(frameon=False); ax.set_title("GC match: positives vs negatives", loc="left")
    fig.tight_layout(); fig.savefig(FIG / "04_gc_match.png", dpi=150); plt.close(fig)

    # 5) GC distribution (aggregate)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(np.concatenate(gc_pos_all), bins=50, color=ORANGE, alpha=.6, density=True, label="positives")
    ax.hist(np.concatenate(gc_neg_all), bins=50, color=BLUE, alpha=.6, density=True, label="negatives")
    ax.set_xlabel("GC content"); ax.set_ylabel("density"); ax.legend(frameon=False)
    ax.set_title("GC distribution, all proteins pooled", loc="left")
    fig.tight_layout(); fig.savefig(FIG / "05_gc_distribution.png", dpi=150); plt.close(fig)

    # 6) nucleotide composition
    fig, ax = plt.subplots(figsize=(7, 5)); bases = list("ACGU"); w = .38
    xp = np.arange(4)
    tp, tn = sum(nuc_pos[b] for b in bases), sum(nuc_neg[b] for b in bases)
    ax.bar(xp - w/2, [nuc_pos[b]/tp*100 for b in bases], w, color=ORANGE, label="positives")
    ax.bar(xp + w/2, [nuc_neg[b]/tn*100 for b in bases], w, color=BLUE, label="negatives")
    ax.set_xticks(xp); ax.set_xticklabels(bases); ax.set_ylabel("% of bases"); ax.legend(frameon=False)
    ax.set_title("Nucleotide composition, positives vs negatives", loc="left")
    fig.tight_layout(); fig.savefig(FIG / "06_nucleotide_composition.png", dpi=150); plt.close(fig)

    # 7) region composition
    regions = ["5UTR", "3UTR", "CDS", "exon", "ncRNA_exon", "intron"]
    cols = [BLUE, ORANGE, GREEN, PURPLE, GOLD, GREY]
    fig, ax = plt.subplots(figsize=(9, 6)); bottom = np.zeros(len(proteins))
    for r, c in zip(regions, cols):
        vals = np.array([region_frac[p].get(r, 0) * 100 for p in proteins])
        ax.barh(proteins, vals, left=bottom, color=c, label=r); bottom += vals
    ax.set_xlabel("% of positives"); ax.legend(frameon=False, ncol=3, fontsize=8)
    ax.set_title("Region composition per protein", loc="left")
    fig.tight_layout(); fig.savefig(FIG / "07_region_composition.png", dpi=150); plt.close(fig)

    # 8) peak widths (raw eCLIP)
    data, labels = [], []
    for p in proteins:
        try:
            w = []
            with gzip.open(ENC / f"{p}.{acc[p]}.bed.gz", "rt") as f:
                for line in f:
                    c = line.split("\t"); w.append(int(c[2]) - int(c[1]))
            data.append(w); labels.append(p)
        except Exception:
            pass
    if data:
        fig, ax = plt.subplots(figsize=(9, 5)); ax.boxplot(data, showfliers=False)
        ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels, rotation=90)
        ax.set_ylabel("peak width (nt)"); ax.set_title("eCLIP peak width per protein", loc="left")
        fig.tight_layout(); fig.savefig(FIG / "08_peak_widths.png", dpi=150); plt.close(fig)

    # 9) motif enrichment + known-motif recovery
    ss = S.sort_values("enrich_log2")
    fig, ax = plt.subplots(figsize=(8, 6)); ax.barh(ss.protein, ss.enrich_log2, color=GREEN)
    for i, (km, kn) in enumerate(zip(ss.top_6mer, ss.known_motif)):
        tag = km + ("  ✓" if kn and kn in km else (f"  (lit:{kn})" if kn else ""))
        ax.text(ss.enrich_log2.iloc[i], i, " " + tag, va="center", fontsize=8)
    ax.set_xlabel("log2 enrichment (positives vs negatives)")
    ax.set_title("Top enriched 6-mer per protein (✓ = matches literature motif)", loc="left")
    fig.tight_layout(); fig.savefig(FIG / "09_kmer_enrichment.png", dpi=150); plt.close(fig)

    # 10) positional signal of top motif (are sites centred in the window?)
    fig, ax = plt.subplots(figsize=(9, 5))
    for p in sorted(motif_pos, key=lambda q: -S.set_index("protein").loc[q, "enrich_log2"])[:8]:
        ax.plot(np.arange(len(motif_pos[p])), motif_pos[p], lw=1.4, label=p, alpha=.85)
    ax.axvline(50, color=INK, lw=1, ls="--")
    ax.set_xlabel("start position of motif in 101-nt window"); ax.set_ylabel("density")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    ax.set_title("Where the motif sits (dashed = window centre) — signal should peak at centre", loc="left")
    fig.tight_layout(); fig.savefig(FIG / "10_motif_position.png", dpi=150); plt.close(fig)

    # 11) complexity vs redundancy
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(S.low_complexity_pct, S.dup_pct, s=40, color=PURPLE)
    for _, r in S.iterrows():
        ax.text(r.low_complexity_pct, r.dup_pct, " " + r.protein, fontsize=7, va="center")
    ax.set_xlabel("low-complexity windows (%)"); ax.set_ylabel("exact-duplicate windows (%)")
    ax.set_title("Sequence complexity vs redundancy per protein", loc="left")
    fig.tight_layout(); fig.savefig(FIG / "11_complexity_redundancy.png", dpi=150); plt.close(fig)

    # ---- red-flag stress test ----
    print("\n=== RED-FLAG STRESS TEST ===")
    tot = S[["train", "val", "test"]].sum()
    ratio = (tot / tot.sum() * 100).round(1)
    print(f"overall split  train/val/test = {tot.train}/{tot.val}/{tot.test}  "
          f"({ratio.train}/{ratio.val}/{ratio.test} %)")
    flags = []
    for _, r in S.iterrows():
        if r.val < MIN_EVAL_PAIRS:
            flags.append(f"{r.protein}: only {r.val} val pairs (< {MIN_EVAL_PAIRS}) — unstable model selection")
        if r.test < MIN_EVAL_PAIRS:
            flags.append(f"{r.protein}: only {r.test} test pairs (< {MIN_EVAL_PAIRS}) — noisy final metric")
        if r.gc_gap > 0.03:
            flags.append(f"{r.protein}: mean-GC gap {r.gc_gap} > 0.03 — GC mismatch")
        if r.enrich_log2 < 1.0:
            flags.append(f"{r.protein}: weak top-6mer enrichment ({r.enrich_log2}) — faint motif")
        if r.dup_pct > 5:
            flags.append(f"{r.protein}: {r.dup_pct}% exact-duplicate windows — redundancy")

    # cross-split identical sequences (true leakage via repeats)
    for p in proteins:
        df = pd.read_csv(PROC / p / "dataset.tsv", sep="\t")
        df["split2"] = df.chrom.map(split_of)
        seen = df.groupby("seq_rna").split2.nunique()
        n_leak = int((seen > 1).sum())
        if n_leak:
            flags.append(f"{p}: {n_leak} sequences appear in >1 split (repeat-driven leakage)")

    if flags:
        for f in flags:
            print("  ⚠️ " + f)
    else:
        print("  ✅ NO red flags.")
    print(f"\nwrote {OUT/'eda_summary.tsv'} + {len(list(FIG.glob('*.png')))} figures to {FIG}")


if __name__ == "__main__":
    main()
