"""Fine-tune an RNA language model for ONE protein; write per-protein metrics + weights.

    python src/train_lm.py rnafm TARDBP                  # model + protein (primary)
    python src/train_lm.py rnabert                       # protein from $SLURM_ARRAY_TASK_ID
    python src/train_lm.py splicebert PUM1 --negatives splice_matched   # ablation variant

    rnafm       -> LoRA (r8, a16 on q/v)      ~100M, adapters + head train
    rnabert     -> full fine-tune             ~0.5M, tiny enough to tune end-to-end
    splicebert  -> full fine-tune             ~20M, the pre-mRNA specialist

Needs a GPU. Output: results/metrics/<PROTEIN>.<model>[.splice_matched].json,
models/lm/<PROTEIN>.<model>[.splice_matched].pt
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

REPO = Path(__file__).resolve().parent.parent
PROC, METRICS, CKPT = REPO / "data/processed", REPO / "results/metrics", REPO / "models/lm"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED, EPOCHS, PATIENCE, BATCH = 7, 12, 4, 32

# key -> (repo id, multimolecule model class, mode, report label)
MODELS = {
    "rnafm":      ("multimolecule/rnafm",       "RnaFmModel",      "lora", "RNA-FM (LoRA)"),
    "rnabert":    ("multimolecule/rnabert",     "RnaBertModel",    "full", "RNABERT"),
    "splicebert": ("multimolecule/splicebert",  "SpliceBertModel", "full", "SpliceBERT"),
}


def masked_mean_pool(h, ids, tok):
    """Mean over real nucleotide tokens (exclude cls/eos/pad)."""
    mask = torch.ones_like(ids, dtype=torch.bool)
    for sid in (tok.cls_token_id, tok.eos_token_id, tok.pad_token_id):
        if sid is not None:
            mask &= ids != sid
    m = mask.unsqueeze(-1).float()
    return (h * m).sum(1) / m.sum(1).clamp(min=1.0)


class LMClassifier(nn.Module):
    def __init__(self, tok, encoder, dropout=0.3):
        super().__init__()
        self.tok, self.encoder = tok, encoder
        self.head = nn.Sequential(
            nn.Linear(encoder.config.hidden_size, 128), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, ids, am):
        out = self.encoder(input_ids=ids, attention_mask=am)
        return self.head(masked_mean_pool(out.last_hidden_state, ids, self.tok)).squeeze(-1)

    def apply_lora(self, r=8, alpha=16, dropout=0.05):
        from peft import LoraConfig, get_peft_model
        self.encoder = get_peft_model(self.encoder, LoraConfig(
            r=r, lora_alpha=alpha, lora_dropout=dropout,
            target_modules=["query", "value"], bias="none"))


class SeqDS(Dataset):
    def __init__(self, s, y): self.s, self.y = list(s), list(y)
    def __len__(self): return len(self.s)
    def __getitem__(self, i): return self.s[i], self.y[i]


def collate(tok):
    def c(b):
        e = tok([x[0] for x in b], return_tensors="pt", padding=True)
        return e["input_ids"], e["attention_mask"], torch.tensor([x[1] for x in b], dtype=torch.float32)
    return c


@torch.no_grad()
def predict(model, loader):
    model.eval()
    P, Y = [], []
    for ids, am, y in loader:
        P.append(torch.sigmoid(model(ids.to(DEVICE), am.to(DEVICE))).cpu().numpy())
        Y.append(y.numpy())
    return np.concatenate(Y), np.concatenate(P)


def load_encoder(key):
    import multimolecule as mm
    from multimolecule import RnaTokenizer
    repo_id, cls = MODELS[key][0], MODELS[key][1]
    return RnaTokenizer.from_pretrained(repo_id), getattr(mm, cls).from_pretrained(repo_id)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=list(MODELS))
    ap.add_argument("protein", nargs="?")
    ap.add_argument("--negatives", choices=["primary", "splice_matched"], default="primary")
    a = ap.parse_args()
    names = [l.split("\t")[0] for l in (REPO / "config/proteins.tsv").read_text().splitlines()[1:]]
    prot = a.protein or names[int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))]
    return a.model, prot, a.negatives


def main():
    key, prot, neg = parse_args()
    assert DEVICE == "cuda", "No GPU — run on a GPU node (sbatch --gres=gpu:1)."
    suffix = ".splice_matched" if neg == "splice_matched" else ""
    _, _, mode, label = MODELS[key]
    torch.manual_seed(SEED)
    t0 = time.time()

    df = pd.read_csv(PROC / prot / f"dataset{suffix}.tsv", sep="\t")
    tok, enc = load_encoder(key)
    model = LMClassifier(tok, enc)
    if mode == "lora":
        model.apply_lora()
    model.to(DEVICE)

    if mode == "lora":
        adapters = [p for _, p in model.encoder.named_parameters() if p.requires_grad]
        opt = torch.optim.AdamW([{"params": adapters, "lr": 1e-4},
                                 {"params": model.head.parameters(), "lr": 1e-3}], weight_decay=1e-2)
    else:
        opt = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": 3e-5},
                                 {"params": model.head.parameters(), "lr": 1e-3}], weight_decay=1e-2)

    coll = collate(tok)
    def loader(sp, shuffle):
        d = df[df.split == sp]
        return DataLoader(SeqDS(d.seq_rna, d.label), batch_size=BATCH, shuffle=shuffle, collate_fn=coll)
    tr, va, te = loader("train", True), loader("val", False), loader("test", False)
    lossfn = nn.BCEWithLogitsLoss()

    best, best_state, best_ep = -1.0, None, -1
    for ep in range(1, EPOCHS + 1):
        model.train()
        for ids, am, y in tr:
            opt.zero_grad()
            lossfn(model(ids.to(DEVICE), am.to(DEVICE)), y.to(DEVICE)).backward()
            opt.step()
        yv, pv = predict(model, va)
        au = roc_auc_score(yv, pv)
        print(f"  {prot} {key}({neg}) ep{ep} val={au:.3f}", flush=True)
        if au > best:
            best, best_ep = au, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep - best_ep >= PATIENCE:
            break

    model.load_state_dict(best_state)
    yt, pt = predict(model, te)
    test_auroc, test_auprc = roc_auc_score(yt, pt), average_precision_score(yt, pt)

    CKPT.mkdir(parents=True, exist_ok=True)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    save_state = {k: v for k, v in best_state.items() if k in trainable} if mode == "lora" else best_state
    torch.save(save_state, CKPT / f"{prot}.{key}{suffix}.pt")

    METRICS.mkdir(parents=True, exist_ok=True)
    (METRICS / f"{prot}.{key}{suffix}.json").write_text(json.dumps(dict(
        protein=prot, model=label, mode=mode, negatives=neg, val_auroc=round(float(best), 4),
        test_auroc=round(float(test_auroc), 4), test_auprc=round(float(test_auprc), 4)), indent=2))
    print(f"[{prot}] {label}({neg}) val={best:.4f} test={test_auroc:.4f} auprc={test_auprc:.4f} "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
