# Co-authored with Claude (Anthropic).
"""Build the binding dataset for ONE protein (array-friendly).

Positives  = 101-nt windows centred on reproducible eCLIP peaks (strand-corrected).
Negatives  = one per positive, matched on region type + GC (±5%), ≥500 nt from any
             peak, drawn from the same or a pooled bound transcript.
Split by chromosome (teammate standard): test={chr1,chr2,chr20}, val={chr19,16,13,18}, rest=train.

Two negative modes:
  primary         region + GC matched            (frozen set for all 16, teammate-comparable)
  splice_matched  also matches distance-to-nearest-splice-site bucket   (ablation control)

    python src/data_prep.py TARDBP                       # primary, by name
    python src/data_prep.py PUM1 --negatives splice_matched
    python src/data_prep.py                              # primary, by $SLURM_ARRAY_TASK_ID

Output → data/processed/<PROTEIN>/{dataset[.splice_matched].tsv, onehot[.splice_matched].npz}
"""
import argparse, gzip, os, sys, time
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from pyfaidx import Fasta

REPO   = Path(__file__).resolve().parent.parent
CONFIG = REPO / "config/proteins.tsv"
GTF    = REPO / "data/reference/gencode.v45.primary_assembly.annotation.gtf.gz"
FASTA  = REPO / "data/reference/GRCh38.primary_assembly.genome.fa"
ENC    = REPO / "data/raw/encode"
OUTDIR = REPO / "data/processed"

STD = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"}
TEST_CHR = {"chr1", "chr2", "chr20"}
VAL_CHR  = {"chr19", "chr16", "chr13", "chr18"}
WIN, HALF, GC_BAND, MIN_DIST, BIN, SEED = 101, 50, 0.05, 500, 1 << 17, 7
SPLICE_EDGES = [50, 150, 500, 1500]        # distance-to-splice buckets for the ablation
rng = np.random.default_rng(SEED)

COMP = str.maketrans("ACGTacgt", "TGCAtgca")
revcomp = lambda s: s.translate(COMP)[::-1]
fa = Fasta(str(FASTA))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("protein", nargs="?")
    ap.add_argument("--negatives", choices=["primary", "splice_matched"], default="primary")
    a = ap.parse_args()
    reg = pd.read_csv(CONFIG, sep="\t")
    if a.protein:
        row = reg[reg.protein == a.protein]
        if row.empty:
            sys.exit(f"unknown protein: {a.protein}")
        row = row.iloc[0]
    else:
        row = reg.iloc[int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))]
    return row.protein, row.accession, a.negatives


def tx_id(attr):
    i = attr.find('transcript_id "')
    return attr[i + 15: attr.find('"', i + 15)] if i >= 0 else None

def get_seq(chrom, a, b, strand):
    s = str(fa[chrom][a:b]).upper()
    return revcomp(s) if strand == "-" else s

def gc_frac(seq):
    return (seq.count("G") + seq.count("C")) / len(seq) if seq else 0.0

def load_peaks(path):
    out = []
    with gzip.open(path, "rt") as f:
        for line in f:
            c = line.split("\t")
            if c[0] in STD:
                s, e = int(c[1]), int(c[2])
                out.append((c[0], s, e, c[5].strip(), (s + e) // 2))
    return out


def load_gtf():
    """Pass 1: transcript spans + a coarse bin index for point lookups."""
    tx_span, tx_bins = {}, defaultdict(list)
    with gzip.open(GTF, "rt") as f:
        for line in f:
            if line[0] == "#":
                continue
            c = line.split("\t")
            if c[2] != "transcript" or c[0] not in STD:
                continue
            s0, e = int(c[3]) - 1, int(c[4])
            tid = tx_id(c[8])
            ttype = "protein_coding" if 'transcript_type "protein_coding"' in c[8] else "other"
            tx_span[tid] = (c[0], s0, e, c[6], ttype)
            for b in range(s0 // BIN, e // BIN + 1):
                tx_bins[(c[0], b)].append(tid)
    return tx_span, tx_bins

def load_gtf_features(keep):
    """Pass 2: exon/CDS/UTR intervals for the bound transcripts only."""
    feat = defaultdict(lambda: {"exon": [], "CDS": [], "UTR": []})
    with gzip.open(GTF, "rt") as f:
        for line in f:
            if line[0] == "#":
                continue
            c = line.split("\t")
            if c[2] in ("exon", "CDS", "UTR") and c[0] in STD:
                tid = tx_id(c[8])
                if tid in keep:
                    feat[tid][c[2]].append((int(c[3]) - 1, int(c[4])))
    return feat


def find_tx(tx_span, tx_bins, chrom, p):
    """Longest transcript overlapping point p, preferring protein_coding."""
    best, best_score = None, -1
    for tid in tx_bins.get((chrom, p // BIN), ()):
        _, s0, e, _, ttype = tx_span[tid]
        if s0 <= p < e:
            score = (e - s0) + (10**12 if ttype == "protein_coding" else 0)
            if score > best_score:
                best, best_score = tid, score
    return best

def merge(ivs):
    out = []
    for a, b in sorted(ivs):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out

def region_of(tx_span, feat, tid, p):
    _, s0, e, strand, _ = tx_span[tid]
    cds, utr, exon = feat[tid]["CDS"], feat[tid]["UTR"], feat[tid]["exon"]
    if any(a <= p < b for a, b in cds):
        return "CDS"
    if any(a <= p < b for a, b in utr):
        if not cds:
            return "ncRNA_exon"
        lo, hi = min(a for a, _ in cds), max(b for _, b in cds)
        if strand == "+":
            return "5UTR" if p < lo else ("3UTR" if p >= hi else "CDS")
        return "5UTR" if p >= hi else ("3UTR" if p < lo else "CDS")
    if any(a <= p < b for a, b in exon):
        return "ncRNA_exon" if not cds else "exon"
    return "intron"

def region_intervals(tx_span, feat, tid, region):
    """Genomic intervals of `region` within transcript tid (negative-sampling space)."""
    _, s0, e, strand, _ = tx_span[tid]
    cds, utr, exon = feat[tid]["CDS"], feat[tid]["UTR"], feat[tid]["exon"]
    if region == "CDS":
        return merge(cds)
    if region in ("ncRNA_exon", "exon"):
        return merge(exon)
    if region in ("5UTR", "3UTR"):
        if not cds:
            return merge(utr)
        lo, hi = min(a for a, _ in cds), max(b for _, b in cds)
        want_5 = region == "5UTR"
        keep = []
        for a, b in utr:
            is5 = (b <= lo) if strand == "+" else (a >= hi)
            if is5 == want_5:
                keep.append((a, b))
        return merge(keep)
    if region == "intron":
        introns, prev = [], s0
        for a, b in merge(exon):
            if a > prev:
                introns.append((prev, a))
            prev = max(prev, b)
        if prev < e:
            introns.append((prev, e))
        return introns
    return []


def splice_sites(feat, tid):
    """Internal exon boundaries (donors + acceptors) of a transcript, sorted."""
    ex = merge(feat[tid]["exon"])
    if len(ex) < 2:
        return np.empty(0, dtype=np.int64)
    sites = [b for _, b in ex[:-1]] + [a for a, _ in ex[1:]]
    return np.array(sorted(set(sites)), dtype=np.int64)

def splice_bucket(pos, sites):
    """Which distance-to-nearest-splice bucket does `pos` fall in? (-1 if no sites)."""
    if sites.size == 0:
        return -1
    i = np.searchsorted(sites, pos)
    d = min(abs(pos - sites[i - 1]) if i > 0 else 10**9,
            abs(pos - sites[i]) if i < sites.size else 10**9)
    return int(np.searchsorted(SPLICE_EDGES, d))


def build_forbidden(peaks):
    """Bin index of peak intervals expanded by MIN_DIST (negatives must avoid these)."""
    idx = defaultdict(list)
    for chrom, s, e, strand, mid in peaks:
        a, b = s - MIN_DIST, e + MIN_DIST
        for bn in range(a // BIN, b // BIN + 1):
            idx[(chrom, bn)].append((a, b))
    return idx

def forbidden(idx, chrom, w0, w1):
    for bn in range(w0 // BIN, w1 // BIN + 1):
        for a, b in idx.get((chrom, bn), ()):
            if w0 < b and a < w1:
                return True
    return False


def candidate_starts(chrom, a, b, target_gc, fidx, band, sites=None, target_bucket=None):
    """Valid negative-window starts in [a,b): fits, no N, not forbidden, GC in band,
    and (splice_matched mode) whose window centre is in the same splice-distance bucket."""
    if b - a < WIN:
        return []
    seq = str(fa[chrom][a:b]).upper()
    if "N" in seq:
        isN = np.frombuffer(seq.replace("N", "\x01").encode("latin1"), dtype=np.uint8)
        nN = np.concatenate([[0], np.cumsum(isN == 1)])
    else:
        nN = None
    gcv = np.frombuffer(seq.translate(str.maketrans("GCgc", "\x01\x01\x01\x01")).encode("latin1"),
                        dtype=np.uint8)
    pref = np.concatenate([[0], np.cumsum((gcv == 1).astype(np.int32))])
    starts = []
    for s in range((b - a) - WIN + 1):
        if nN is not None and nN[s + WIN] - nN[s] > 0:
            continue
        if abs((pref[s + WIN] - pref[s]) / WIN - target_gc) > band:
            continue
        g0 = a + s
        if forbidden(fidx, chrom, g0, g0 + WIN):
            continue
        if target_bucket is not None and splice_bucket(g0 + HALF, sites) != target_bucket:
            continue
        starts.append(g0)
    return starts

def make_negative(tx_span, feat, tid, region, target_gc, chrom, strand, fidx, pool,
                  target_bucket=None):
    """Same transcript first, then a sample of pooled transcripts; relax GC band last.
    If target_bucket is set, negatives must also match the positive's splice-distance bucket."""
    for band in (GC_BAND, 0.10, 1.0):
        sites = splice_sites(feat, tid) if target_bucket is not None else None
        for a, b in region_intervals(tx_span, feat, tid, region):
            cs = candidate_starts(chrom, a, b, target_gc, fidx, band, sites, target_bucket)
            if cs:
                g0 = int(rng.choice(cs))
                return chrom, g0, g0 + WIN, strand, band
        cand = [t for t in pool if t != tid]
        rng.shuffle(cand)
        for t in cand[:60]:
            ch2, _, _, st2, _ = tx_span[t]
            sites2 = splice_sites(feat, t) if target_bucket is not None else None
            for a, b in region_intervals(tx_span, feat, t, region):
                cs = candidate_starts(ch2, a, b, target_gc, fidx, band, sites2, target_bucket)
                if cs:
                    g0 = int(rng.choice(cs))
                    return ch2, g0, g0 + WIN, st2, band
    return None


BASE_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}
def one_hot(seq):
    x = np.zeros((4, WIN), dtype=np.float32)
    for i, ch in enumerate(seq):
        j = BASE_IDX.get(ch)
        if j is not None:
            x[j, i] = 1.0
    return x

def split_of(chrom):
    return "test" if chrom in TEST_CHR else "val" if chrom in VAL_CHR else "train"


def main():
    protein, acc, neg_mode = parse_args()
    matched = neg_mode == "splice_matched"
    suffix = ".splice_matched" if matched else ""
    t0 = time.time()
    print(f"[{protein}] {acc}  negatives={neg_mode}")

    tx_span, tx_bins = load_gtf()
    print(f"  GTF: {len(tx_span)} transcripts ({time.time()-t0:.0f}s)")

    peaks = load_peaks(ENC / f"{protein}.{acc}.bed.gz")
    tids = [find_tx(tx_span, tx_bins, ch, mid) for ch, s, e, st, mid in peaks]
    bound = {t for t in tids if t}
    feat = load_gtf_features(bound)
    regions = [region_of(tx_span, feat, t, mid) if t else "intergenic"
               for (ch, s, e, st, mid), t in zip(peaks, tids)]
    print(f"  {len(peaks)} peaks, {len(bound)} bound transcripts ({time.time()-t0:.0f}s)")

    pool_by_region = defaultdict(list)
    for t, r in zip(tids, regions):
        if t:
            pool_by_region[r].append(t)
    fidx = build_forbidden(peaks)

    rows, relax, dropped = [], Counter(), 0
    for (ch, s, e, st, mid), tid, reg in zip(peaks, tids, regions):
        if tid is None or reg == "intergenic":
            continue
        w0, w1 = mid - HALF, mid + HALF + 1
        pos_seq = get_seq(ch, w0, w1, st)
        if len(pos_seq) != WIN or "N" in pos_seq:
            continue
        pgc = gc_frac(pos_seq)
        bucket = splice_bucket(mid, splice_sites(feat, tid)) if matched else None
        neg = make_negative(tx_span, feat, tid, reg, pgc, ch, st, fidx,
                            pool_by_region.get(reg, []), target_bucket=bucket)
        if neg is None:
            dropped += 1
            continue
        nch, n0, n1, nst, band = neg
        relax[band] += 1
        neg_seq = get_seq(nch, n0, n1, nst)
        rows.append((f"{protein}_pos_{len(rows)}", 1, ch, w0, w1, st, reg, round(pgc, 4),
                     split_of(ch), pos_seq, pos_seq.replace("T", "U")))
        rows.append((f"{protein}_neg_{len(rows)}", 0, nch, n0, n1, nst, reg,
                     round(gc_frac(neg_seq), 4), split_of(nch),
                     neg_seq, neg_seq.replace("T", "U")))

    cols = ["id", "label", "chrom", "start", "end", "strand", "region",
            "gc", "split", "seq_dna", "seq_rna"]
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        sys.exit(f"[{protein}] no usable examples")

    pdir = OUTDIR / protein
    pdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(pdir / f"dataset{suffix}.tsv", sep="\t", index=False)
    X = np.stack([one_hot(s) for s in df.seq_dna]).astype(np.float32)
    np.savez_compressed(pdir / f"onehot{suffix}.npz", X=X, y=df.label.to_numpy(np.int8),
                        split=df.split.to_numpy())

    by = {s: int((df.split == s).sum()) for s in ("train", "val", "test")}
    print(f"  pairs={int((df.label==1).sum())} total={len(df)} dropped={dropped} "
          f"train/val/test={by['train']}/{by['val']}/{by['test']} "
          f"GCband={dict(relax)} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
