# Co-authored with Claude (Anthropic).
"""v2 ClinVar variant-effect analysis at real binding sites, all 4 tuned models.

Improvements over v1's coin-flip:
  (a) domain — only Pathogenic/Benign SNVs in/near a real eCLIP peak; report
      noncoding separately (where a binding model has jurisdiction).
  (b) sharper score — slide the variant across window offsets and take the MAX
      disruption  delta = max_shift |p(ref) - p(alt)|  (motif may be off-centre).
  (c) coverage — 16 proteins, so more variants land in some protein's site.

Scores each variant with CNN + RNA-FM(LoRA) + RNABERT + SpliceBERT.

    python src/clinvar.py        # needs a GPU (for the LM models)
Output: results/clinvar_scores.tsv, results/clinvar_summary.tsv
"""
import gzip, sys, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_prep import load_peaks, fa, revcomp, one_hot, STD, BIN, ENC
from models.cnn import DeepBindCNN, predict_probs
from train_lm import LMClassifier, load_encoder, MODELS

REPO = Path(__file__).resolve().parent.parent
CONFIG = REPO / "config/proteins.tsv"
CLINVAR = REPO / "data/raw/clinvar/clinvar.vcf.gz"
CNN_DIR, LM_DIR = REPO / "models/cnn", REPO / "models/lm"
WIN, HALF, MARGIN = 101, 50, 25
SHIFTS = list(range(-40, 41, 20))          # variant at 5 offsets within the window
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_KEYS = [("CNN", "cnn"), ("RNA-FM (LoRA)", "rnafm"), ("RNABERT", "rnabert"), ("SpliceBERT", "splicebert")]

CODING = ("missense", "synonymous", "stop_gained", "stop_lost", "start_lost",
          "frameshift", "inframe", "protein_altering", "coding_sequence")
NONCODING = ("intron", "utr", "splice", "non_coding", "upstream", "downstream", "intergenic")


def info_get(info, key):
    i = info.find(key + "=")
    if i < 0:
        return None
    j = info.find(";", i)
    return info[i + len(key) + 1: j if j > 0 else None]

def classify_mc(info):
    mc = info_get(info, "MC")
    if not mc:
        return "other"
    s = mc.lower()
    if any(k in s for k in CODING):
        return "coding"
    if any(k in s for k in NONCODING):
        return "noncoding"
    return "other"


def proteins():
    return [l.split("\t")[:2] for l in CONFIG.read_text().splitlines()[1:]]

def peak_index(prot_acc):
    idx = defaultdict(list)
    for prot, acc in prot_acc:
        for ch, s, e, strand, mid in load_peaks(ENC / f"{prot}.{acc}.bed.gz"):
            a, b = s - MARGIN, e + MARGIN
            for bn in range(a // BIN, b // BIN + 1):
                idx[(ch, bn)].append((a, b, strand, prot))
    return idx

def hits_at(idx, chrom, pos0):
    out = {}
    for a, b, strand, prot in idx.get((chrom, pos0 // BIN), ()):
        if a <= pos0 < b:
            out.setdefault(prot, strand)
    return out

def windows(chrom, pos0, ref, alt, strand):
    """ref/alt 101-nt DNA windows with the variant at each SHIFT offset (model-oriented)."""
    out = []
    for sh in SHIFTS:
        a = pos0 + sh - HALF
        b = a + WIN
        if a < 0 or b > len(fa[chrom]):
            continue
        g = str(fa[chrom][a:b]).upper()
        vi = HALF - sh                       # variant index within the window
        if len(g) != WIN or "N" in g or g[vi] != ref:
            continue
        galt = g[:vi] + alt + g[vi + 1:]
        out.append((revcomp(g), revcomp(galt)) if strand == "-" else (g, galt))
    return out


def load_cnn(prot):
    m = DeepBindCNN()
    m.load_state_dict(torch.load(CNN_DIR / f"{prot}.pt", map_location=DEVICE))
    return m.to(DEVICE).eval()

def load_lm(prot, key):
    tok, enc = load_encoder(key)
    m = LMClassifier(tok, enc)
    if MODELS[key][2] == "lora":
        m.apply_lora()
    m.load_state_dict(torch.load(LM_DIR / f"{prot}.{key}.pt", map_location=DEVICE), strict=False)
    return tok, m.to(DEVICE).eval()

@torch.no_grad()
def lm_probs(seqs, tok, model, bs=64):
    out = []
    for i in range(0, len(seqs), bs):
        e = tok(seqs[i:i + bs], return_tensors="pt", padding=True)
        out.append(torch.sigmoid(model(e["input_ids"].to(DEVICE), e["attention_mask"].to(DEVICE))).cpu().numpy())
    return np.concatenate(out) if out else np.array([])

def reduce_max(vids, pr, pa):
    """max |p(ref)-p(alt)| per variant id across its shifted windows."""
    d = np.abs(np.asarray(pr) - np.asarray(pa))
    best = defaultdict(float)
    for v, x in zip(vids, d):
        if x > best[v]:
            best[v] = float(x)
    return best


def main():
    t0 = time.time()
    pa = proteins()
    idx = peak_index(pa)
    print(f"indexed peaks for {len(pa)} proteins ({time.time()-t0:.0f}s)", flush=True)

    # scan ClinVar: per protein collect shifted windows tagged with a variant id
    per = defaultdict(lambda: {"vids": [], "ref": [], "alt": []})
    meta = {}
    with gzip.open(CLINVAR, "rt") as f:
        for line in f:
            if line[0] == "#":
                continue
            c = line.rstrip("\n").split("\t")
            chrom, pos, ref, alt, info = c[0], c[1], c[3], c[4], c[7]
            if len(ref) != 1 or len(alt) != 1 or ref not in "ACGT" or alt not in "ACGT":
                continue
            sig = info_get(info, "CLNSIG")
            if sig not in ("Pathogenic", "Benign"):
                continue
            cstd = "chr" + chrom
            if cstd not in STD:
                continue
            pos0 = int(pos) - 1
            h = hits_at(idx, cstd, pos0)
            if not h:
                continue
            vid, mc = f"{cstd}:{pos}:{ref}:{alt}", classify_mc(info)
            for prot, strand in h.items():
                ws = windows(cstd, pos0, ref, alt, strand)
                if not ws:
                    continue
                meta[vid] = dict(vid=vid, clnsig=sig, mc=mc)
                for g, galt in ws:
                    per[prot]["vids"].append(vid)
                    per[prot]["ref"].append(g)
                    per[prot]["alt"].append(galt)
    print(f"variants at binding sites: {len(meta)} ({time.time()-t0:.0f}s)", flush=True)

    # score: per protein, per model -> max delta per variant, keep max across proteins
    delta = defaultdict(lambda: defaultdict(float))
    for prot, P in per.items():
        if not P["vids"]:
            continue
        vids, ref_dna, alt_dna = P["vids"], P["ref"], P["alt"]
        ref_rna = [s.replace("T", "U") for s in ref_dna]
        alt_rna = [s.replace("T", "U") for s in alt_dna]
        for label, key in MODEL_KEYS:
            if key == "cnn":
                cnn = load_cnn(prot)
                pr = predict_probs(cnn, np.stack([one_hot(s) for s in ref_dna]).astype(np.float32), DEVICE)
                paa = predict_probs(cnn, np.stack([one_hot(s) for s in alt_dna]).astype(np.float32), DEVICE)
                del cnn
            else:
                tok, m = load_lm(prot, key)
                pr, paa = lm_probs(ref_rna, tok, m), lm_probs(alt_rna, tok, m)
                del m
            for vid, x in reduce_max(vids, pr, paa).items():
                if x > delta[vid][label]:
                    delta[vid][label] = x
        print(f"  scored {prot} ({time.time()-t0:.0f}s)", flush=True)

    sc = pd.DataFrame([{**meta[v], **delta[v]} for v in meta])
    (REPO / "results").mkdir(exist_ok=True)
    sc.to_csv(REPO / "results/clinvar_scores.tsv", sep="\t", index=False)

    labels = [l for l, _ in MODEL_KEYS]
    def auroc_row(df, name):
        y = (df.clnsig == "Pathogenic").astype(int)
        row = dict(stratum=name, n_path=int(y.sum()), n_benign=int(len(y) - y.sum()))
        if row["n_path"] >= 8 and row["n_benign"] >= 8:
            for l in labels:
                if l in df:
                    row[l] = round(roc_auc_score(y, df[l].fillna(0.0)), 3)
        return row

    summ = pd.DataFrame([auroc_row(sc, "all @ sites"),
                         auroc_row(sc[sc.mc == "noncoding"], "noncoding @ sites"),
                         auroc_row(sc[sc.mc == "coding"], "coding @ sites")])
    summ.to_csv(REPO / "results/clinvar_summary.tsv", sep="\t", index=False)
    print("\n=== variant-effect AUROC (max-shift delta) at binding sites ===")
    print(summ.to_string(index=False))
    print(f"\nwrote clinvar_scores.tsv + clinvar_summary.tsv ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
