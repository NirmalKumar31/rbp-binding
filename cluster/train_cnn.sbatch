#!/bin/bash
# Co-authored with Claude (Anthropic).
#SBATCH --job-name=rbp-cnn
#SBATCH --partition=short
#SBATCH --array=0-19%8
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=logs/cnn_%A_%a.out
#SBATCH --error=logs/cnn_%A_%a.out
# One tuned CNN per manifest line (16 primary + 4 splice_matched = 20). CPU is fastest here.
set -euo pipefail
module load python/3.13.5
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
read prot neg <<< "$(sed -n "$(( SLURM_ARRAY_TASK_ID + 1 ))p" cluster/manifest_cnn.txt)"
[ -z "${prot:-}" ] && { echo "no manifest line for task $SLURM_ARRAY_TASK_ID"; exit 0; }
python src/train_cnn.py "$prot" --negatives "$neg"
