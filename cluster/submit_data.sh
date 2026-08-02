#!/bin/bash
# PHASE A — data + validation gate. Stops here on purpose: review the validate log before training.
# Run setup_env.sh + download_data.sh once first.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
bash cluster/make_manifests.sh
prep=$(sbatch --parsable cluster/prep.sbatch)
val=$(sbatch --parsable --dependency=afterok:$prep cluster/validate.sbatch)
echo "submitted  prep=$prep  validate=$val"
echo "when validate finishes: read logs/validate_${val}.out  (must say VALIDATION PASSED)"
echo "then launch training:   bash cluster/submit_models.sh"
