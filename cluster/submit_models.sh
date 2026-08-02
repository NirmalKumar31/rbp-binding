#!/bin/bash
# PHASE B — training + aggregation. Run ONLY after validate said PASSED.
# CNN (short) + LM (gpu) run in parallel; aggregation waits on both.
# ClinVar runs SEPARATELY after the sweep: the gpu QOS caps submitted jobs at 8, the LM array
# already uses all 8, so a 9th held gpu job would be rejected. Fire it once aggregate is done.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
cnn=$(sbatch --parsable cluster/train_cnn.sbatch)
lm=$(sbatch --parsable cluster/train_lm.sbatch)
agg=$(sbatch --parsable --dependency=afterok:$cnn:$lm cluster/aggregate.sbatch)
echo "submitted  cnn=$cnn  lm=$lm  agg=$agg"
echo "after aggregate finishes, run:  sbatch cluster/clinvar.sbatch"
