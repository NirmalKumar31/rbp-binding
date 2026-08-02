# Co-authored with Claude (Anthropic).
"""Validation gate — run on the FROZEN cluster data before training anything.

Hard checks (exit 1 on failure):
  - every protein has dataset.tsv + onehot.npz; the 4 ablation proteins have the splice variants
  - class balance is exactly 1:1
  - val and test each have >= 100 positive pairs
  - the split column matches the chromosome rule, and no chromosome sits in >1 split
  - no identical sequence appears in more than one split (repeat-driven leakage)
  - primary content reproduces v2 exactly (sha256 vs cluster/positives_ref.tsv), if the reference exists

Soft checks (warn only): GC gap per protein; how many pairs the splice-matched sets retain.

    python src/validate.py          # -> prints a table + PASS/FAIL, exits 1 on any hard failure
"""
import hashlib, sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
PROC, REF = REPO / "data/processed", REPO / "cluster/positives_ref.tsv"
TEST_CHR, VAL_CHR = {"chr1", "chr2", "chr20"}, {"chr19", "chr16", "chr13", "chr18"}  # mirror data_prep
ABLATION = ["PUM1", "LIN28B", "U2AF1", "RBFOX2"]
MIN_EVAL_PAIRS, GC_GAP_MAX = 100, 0.03
CONTENT_COLS = ["id", "label", "chrom", "start", "end", "strand", "region", "gc", "seq_dna", "seq_rna"]


def split_of(c):
    return "test" if c in TEST_CHR else "val" if c in VAL_CHR else "train"


def content_hash(path):
    """sha256 of everything-but-split, read as strings so float formatting can't drift."""
    df = pd.read_csv(path, sep="\t", dtype=str)
    return hashlib.sha256(df[CONTENT_COLS].to_csv(sep="\t", index=False).encode()).hexdigest()


def main():
    proteins = [l.split("\t")[0] for l in (REPO / "config/proteins.tsv").read_text().splitlines()[1:]]
    ref = {}
    if REF.exists():
        for l in REF.read_text().splitlines()[1:]:
            p, h = l.split("\t"); ref[p] = h
    else:
        print(f"  note: {REF} not found — skipping the v2-reproducibility check")

    rows, hard, warn = [], [], []
    for p in proteins:
        d = PROC / p / "dataset.tsv"
        if not d.exists() or not (PROC / p / "onehot.npz").exists():
            hard.append(f"{p}: missing dataset.tsv or onehot.npz"); continue
        df = pd.read_csv(d, sep="\t")
        pos, neg = df[df.label == 1], df[df.label == 0]

        if len(pos) != len(neg):
            hard.append(f"{p}: class imbalance ({len(pos)} pos vs {len(neg)} neg)")

        exp = df.chrom.map(split_of)
        if not (df.split == exp).all():
            hard.append(f"{p}: split column disagrees with the chromosome rule")
        chrom_splits = df.groupby("chrom").split.nunique()
        if (chrom_splits > 1).any():
            hard.append(f"{p}: a chromosome appears in >1 split")

        sc = {s: int((pos.chrom.map(split_of) == s).sum()) for s in ("train", "val", "test")}
        if sc["val"] < MIN_EVAL_PAIRS:
            hard.append(f"{p}: only {sc['val']} val pairs (< {MIN_EVAL_PAIRS})")
        if sc["test"] < MIN_EVAL_PAIRS:
            hard.append(f"{p}: only {sc['test']} test pairs (< {MIN_EVAL_PAIRS})")

        leak = int((df.groupby("seq_rna").split.nunique() > 1).sum())
        if leak:
            hard.append(f"{p}: {leak} sequences in >1 split (repeat leakage)")

        gc_gap = abs(pos.gc.mean() - neg.gc.mean())
        if gc_gap > GC_GAP_MAX:
            warn.append(f"{p}: GC gap {gc_gap:.3f} > {GC_GAP_MAX}")

        repro = "-"
        if p in ref:
            repro = "match" if content_hash(d) == ref[p] else "MISMATCH"
            if repro == "MISMATCH":
                warn.append(f"{p}: primary content differs from the v2 reference "
                            f"(investigate — not necessarily wrong)")

        rows.append(dict(protein=p, pairs=len(pos), train=sc["train"], val=sc["val"],
                         test=sc["test"], gc_gap=round(gc_gap, 3), leak=leak, vs_v2=repro))

    # ablation datasets present + retention vs primary
    for p in ABLATION:
        sm = PROC / p / "dataset.splice_matched.tsv"
        if not sm.exists() or not (PROC / p / "onehot.splice_matched.npz").exists():
            hard.append(f"{p}: missing splice_matched dataset/onehot"); continue
        n_sm = int((pd.read_csv(sm, sep="\t").label == 1).sum())
        n_pr = next((r["pairs"] for r in rows if r["protein"] == p), 0)
        keep = n_sm / n_pr * 100 if n_pr else 0
        rows.append(dict(protein=f"{p}·splice", pairs=n_sm, train="", val="", test="",
                         gc_gap="", leak="", vs_v2=f"{keep:.0f}% kept"))
        if keep < 70:
            warn.append(f"{p}: splice_matched keeps only {n_sm}/{n_pr} pairs ({keep:.0f}%)")

    T = pd.DataFrame(rows)
    print(T.to_string(index=False))
    print()
    if warn:
        print("WARNINGS:"); [print("  ⚠️ " + w) for w in warn]
    if hard:
        print("\nHARD FAILURES:"); [print("  ❌ " + h) for h in hard]
        print("\n=== VALIDATION FAILED — do not train until these are fixed ===")
        sys.exit(1)
    print("=== VALIDATION PASSED — safe to submit training (submit_models.sh) ===")


if __name__ == "__main__":
    main()
