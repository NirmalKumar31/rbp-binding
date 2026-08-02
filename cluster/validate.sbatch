#!/bin/bash
#SBATCH --job-name=rbp-validate
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/validate_%j.out
#SBATCH --error=logs/validate_%j.out
# Runs after prep. Regenerates EDA on the FROZEN data, then the integrity gate.
# Exits non-zero if any hard check fails -> read this log before submit_models.sh.
set -euo pipefail
module load python/3.13.5
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
python src/eda.py
python src/validate.py
