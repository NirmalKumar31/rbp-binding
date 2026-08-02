# Co-authored with Claude (Anthropic).
"""Train the CNN baseline for ONE protein, picking hyperparameters by validation AUROC.

    python src/train_cnn.py TARDBP                       # primary, or omit -> $SLURM_ARRAY_TASK_ID
    python src/train_cnn.py PUM1 --negatives splice_matched   # ablation variant

Writes best weights to models/cnn/<PROTEIN>[.splice_matched].pt and metrics to
results/metrics/<PROTEIN>.cnn[.splice_matched].json (one file per run -> array-friendly).
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.cnn import DeepBindCNN, load_split_arrays, make_loader, predict_probs

REPO = Path(__file__).resolve().parent.parent
PROC, MODELS, METRICS = REPO / "data/processed", REPO / "models/cnn", REPO / "results/metrics"

MAX_EPOCHS, PATIENCE, BATCH, WD, SEED = 40, 6, 64, 1e-4, 7
GRID = [(1e-3, 0.5), (1e-3, 0.3), (5e-4, 0.5), (5e-4, 0.3)]   # (lr, dropout)
DEVICE = "cpu"   # tiny model; CPU is fastest and matches the cluster CPU array


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("protein", nargs="?")
    ap.add_argument("--negatives", choices=["primary", "splice_matched"], default="primary")
    a = ap.parse_args()
    names = [l.split("\t")[0] for l in (REPO / "config/proteins.tsv").read_text().splitlines()[1:]]
    prot = a.protein or names[int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))]
    return prot, a.negatives


def fit(Xtr, ytr, Xva, yva, lr, dropout):
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = DeepBindCNN(dropout).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WD)
    lossfn = nn.BCEWithLogitsLoss()
    loader = make_loader(Xtr, ytr, BATCH, shuffle=True, seed=SEED)
    best, best_state, best_ep = -1.0, None, -1
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            lossfn(model(xb.to(DEVICE)), yb.to(DEVICE)).backward()
            opt.step()
        auroc = roc_auc_score(yva, predict_probs(model, Xva, DEVICE))
        if auroc > best:
            best, best_ep = auroc, ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep - best_ep >= PATIENCE:
            break
    return best, best_state


def main():
    prot, neg = parse_args()
    suffix = ".splice_matched" if neg == "splice_matched" else ""
    t0 = time.time()
    arr = load_split_arrays(PROC / prot / f"onehot{suffix}.npz")
    Xtr, ytr = arr["train"]; Xva, yva = arr["val"]; Xte, yte = arr["test"]

    best_auroc, best_state, best_hp = -1.0, None, None
    for lr, dp in GRID:
        auroc, state = fit(Xtr, ytr, Xva, yva, lr, dp)
        if auroc > best_auroc:
            best_auroc, best_state, best_hp = auroc, state, (lr, dp)

    model = DeepBindCNN(best_hp[1]).to(DEVICE)
    model.load_state_dict(best_state)
    p = predict_probs(model, Xte, DEVICE)
    test_auroc = roc_auc_score(yte, p)
    test_auprc = average_precision_score(yte, p)

    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, MODELS / f"{prot}{suffix}.pt")
    METRICS.mkdir(parents=True, exist_ok=True)
    (METRICS / f"{prot}.cnn{suffix}.json").write_text(json.dumps(dict(
        protein=prot, model="CNN", negatives=neg, val_auroc=round(float(best_auroc), 4),
        test_auroc=round(float(test_auroc), 4), test_auprc=round(float(test_auprc), 4),
        lr=best_hp[0], dropout=best_hp[1], n_train=int(len(ytr)), n_test=int(len(yte))), indent=2))
    print(f"[{prot}] CNN({neg}) val={best_auroc:.4f} test={test_auroc:.4f} auprc={test_auprc:.4f} "
          f"hp(lr={best_hp[0]}, dp={best_hp[1]}) ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
