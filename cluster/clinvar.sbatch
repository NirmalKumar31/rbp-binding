#!/bin/bash
#SBATCH --job-name=rbp-clinvar
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=logs/clinvar_%j.out
#SBATCH --error=logs/clinvar_%j.out
# Run after the sweep finishes (needs the trained CNN + LM checkpoints).
set -euo pipefail
module load python/3.13.5
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
python src/clinvar.py
python src/figures.py
