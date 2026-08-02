"""Merge per-protein/-model metric JSONs into comparison tables + the splice ablation.

    python src/aggregate.py
      -> results/model_comparison.tsv   (primary: protein x model test AUROC)
      -> results/all_metrics.tsv        (every primary run, long form)
      -> results/ablation_splice.tsv    (primary vs splice-matched AUROC, the 4 ablation proteins)
"""
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
METRICS, OUT = REPO / "results/metrics", REPO / "results"
ORDER = ["CNN", "RNABERT", "SpliceBERT", "RNA-FM (LoRA)"]

rows = [json.loads(p.read_text()) for p in sorted(METRICS.glob("*.json"))]
df = pd.DataFrame(rows)
if "negatives" not in df:
    df["negatives"] = "primary"

primary = df[df.negatives == "primary"]
ablation = df[df.negatives == "splice_matched"]

# --- primary comparison table ---
wide = primary.pivot_table(index="protein", columns="model", values="test_auroc")
wide = wide[[m for m in ORDER if m in wide.columns]]
wide.loc["MEAN"] = wide.mean()
OUT.mkdir(exist_ok=True)
wide.round(4).to_csv(OUT / "model_comparison.tsv", sep="\t")
primary.to_csv(OUT / "all_metrics.tsv", sep="\t", index=False)
print("=== primary model comparison (test AUROC) ===")
print(wide.round(3).to_string())

# --- splice ablation: how much AUROC is splice-proximity? ---
if not ablation.empty:
    a = ablation.rename(columns={"test_auroc": "splice_auroc"})[["protein", "model", "splice_auroc"]]
    p = primary.rename(columns={"test_auroc": "primary_auroc"})[["protein", "model", "primary_auroc"]]
    abl = a.merge(p, on=["protein", "model"], how="left")
    abl["drop"] = (abl.primary_auroc - abl.splice_auroc).round(4)
    abl = abl.sort_values(["protein", "model"])
    abl.to_csv(OUT / "ablation_splice.tsv", sep="\t", index=False)
    print("\n=== splice ablation (primary vs splice-matched negatives) ===")
    print(abl.round(3).to_string(index=False))
    print("\ninterpretation: larger `drop` = more of that protein/model's AUROC was splice proximity.")
else:
    print("\n(no splice-matched runs found yet — ablation table skipped)")
