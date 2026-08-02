#!/bin/bash
# Co-authored with Claude (Anthropic).
#SBATCH --job-name=rbp-prep
#SBATCH --partition=short
#SBATCH --array=0-19%8
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/prep_%A_%a.out
#SBATCH --error=logs/prep_%A_%a.out
# One data_prep run per manifest line (16 primary + 4 splice_matched = 20).
set -euo pipefail
module load python/3.13.5
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
read prot neg <<< "$(sed -n "$(( SLURM_ARRAY_TASK_ID + 1 ))p" cluster/manifest_prep.txt)"
[ -z "${prot:-}" ] && { echo "no manifest line for task $SLURM_ARRAY_TASK_ID"; exit 0; }
echo "prep $prot ($neg)"
python src/data_prep.py "$prot" --negatives "$neg"
