#!/bin/bash
# Co-authored with Claude (Anthropic).
#SBATCH --job-name=rbp-lm
#SBATCH --partition=gpu
#SBATCH --array=0-7%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=logs/lm_%A_%a.out
#SBATCH --error=logs/lm_%A_%a.out
# gpu QOS caps submit at 8 / run at 4 -> 8 array tasks, each walks PER lines of manifest_lm.txt
# (60 LM fine-tunes = 48 primary + 12 ablation). Only primary runs are GPU-profiled.
set -euo pipefail
module load python/3.13.5
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

PER=8
start=$(( SLURM_ARRAY_TASK_ID * PER + 1 ))
for i in $(seq 0 $(( PER - 1 ))); do
  line=$(( start + i ))
  entry=$(sed -n "${line}p" cluster/manifest_lm.txt || true)
  [ -z "$entry" ] && continue
  read model prot neg <<< "$entry"
  [ -z "${model:-}" ] && continue
  echo ">>> $model $prot ($neg)  [line $line]"
  if [ "$neg" = "primary" ]; then
    nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv -l 5 \
      > "logs/util_${model}_${prot}.csv" &
    nvpid=$!
    python src/train_lm.py "$model" "$prot" --negatives "$neg"
    kill "$nvpid" 2>/dev/null || true
  else
    python src/train_lm.py "$model" "$prot" --negatives "$neg"
  fi
done
