#!/bin/bash
#SBATCH --job-name=rbp-agg
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/agg_%j.out
#SBATCH --error=logs/agg_%j.out
set -euo pipefail
module load python/3.13.5
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
python src/aggregate.py
python src/figures.py
