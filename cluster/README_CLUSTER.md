# Cluster runbook (NEU Explorer)

The whole project runs as Slurm job arrays, gated so **nothing trains until the data passes
validation**. Two phases: data + validate (Phase A), then train + aggregate (Phase B).

## 0 · fresh start
```bash
bash cluster/clean_start.sh            # dry run — shows what it would cancel/remove
bash cluster/clean_start.sh --force    # cancel all jobs + wipe old trees (edit OLD_DIRS first)
```

## 1 · one-time setup
```bash
srun --partition=short --cpus-per-task=8 --mem=16G --time=02:00:00 --pty /bin/bash
bash cluster/setup_env.sh              # venv + pinned LM stack + cu118 torch
bash cluster/download_data.sh          # 16 eCLIP peaks + genome + GTF + ClinVar (~3 GB)
```

## 2 · Phase A — data + validation gate
```bash
bash cluster/submit_data.sh
```
- `prep`     `short · --array=0-19%8` — 16 primary + 4 splice-matched datasets (manifest-driven)
- `validate` `afterok(prep)` — re-runs EDA on the frozen data, then `src/validate.py`

**Read `logs/validate_<id>.out` before going on.** It must print `VALIDATION PASSED`. Hard checks:
class balance 1:1, val/test ≥ 100 pairs, split matches the chromosome rule, no chromosome in two
splits, no cross-split sequence leakage, and (if `positives_ref.tsv` is present) primary content
reproduces v2.

## 3 · Phase B — train + aggregate (only after PASSED)
```bash
bash cluster/submit_models.sh
```
- `cnn` `short · --array=0-19%8` — tuned CNN per dataset
- `lm`  `gpu · --array=0-7%4`   — 60 LM fine-tunes (48 primary + 12 ablation), 8 tasks × 8 lines,
   ≤ 4 GPUs at once to fit the gpu QOS (8 submit / 4 run). Primary runs are GPU-profiled.
- `aggregate` `afterok(cnn,lm)` — comparison table + splice-ablation table + figures

Then, once aggregate is done (the LM array frees its gpu-QOS slots), run ClinVar separately:
```bash
sbatch cluster/clinvar.sbatch          # gpu · variant-effect scoring + figures
```
(kept separate on purpose: the gpu QOS caps submitted jobs at 8 and the LM array uses all 8.)

## what lands where
```
data/processed/<P>/dataset.tsv, onehot.npz                  primary
data/processed/<P>/dataset.splice_matched.tsv, ...          the 4 ablation proteins
results/model_comparison.tsv                                primary AUROC, protein x model
results/ablation_splice.tsv                                 primary vs splice-matched (the confound)
results/clinvar_summary.tsv                                 variant-effect AUROC by stratum
results/figures/*.png                                       model_comparison, gpu_dashboard,
                                                            acc_vs_compute, splice_ablation
eda/figures/*.png, eda/eda_summary.tsv                      EDA on the frozen data
```

## manifests (single source of truth for the arrays)
`make_manifests.sh` regenerates them from `config/proteins.tsv` + the ablation list:
`manifest_prep.txt`/`manifest_cnn.txt` (20 lines), `manifest_lm.txt` (60 lines). If you change the
protein panel, the array ranges in the `.sbatch` files must match the new line counts.

## the ablation (4 splice-matched runs)
`PUM1, LIN28B, U2AF1` bind near splice sites (EDA §9: top 6-mer = `GGUAAG`, the 5′ donor), so a
model can win partly by finding splice sites. We regenerate their negatives **also matched on
distance-to-nearest-splice-site**, retrain, and compare — `RBFOX2` (clean `UGCAUG` motif) is the
control. `results/ablation_splice.tsv` reports the AUROC drop = the splice-proximity share.
