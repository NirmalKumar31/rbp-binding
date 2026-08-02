#!/bin/bash
# Fresh start on Explorer. DRY-RUN by default — prints what it would do. Add --force to act.
#   bash cluster/clean_start.sh            # show what would be cancelled/removed
#   bash cluster/clean_start.sh --force    # actually cancel jobs + delete the trees below
#
# Edit OLD_DIRS to match your cluster layout before using --force.
set -euo pipefail
cd "$(dirname "$0")/.."
FORCE=${1:-}
PROJ=$(pwd)

# regenerable trees to wipe for a clean rebuild of THIS project + the old v1/v2 project trees
OLD_DIRS=(
  "$PROJ/.venv" "$PROJ/data" "$PROJ/models" "$PROJ/results" "$PROJ/logs"
  "$HOME/rbp-v2" "$HOME/rbp-prediction"
)

echo "== jobs currently queued for $USER =="
squeue -u "$USER" || true
echo
echo "== would remove =="
for d in "${OLD_DIRS[@]}"; do
  if [ -e "$d" ]; then du -sh "$d" 2>/dev/null || echo "  ?  $d"; else echo "  (absent)  $d"; fi
done
echo

if [ "$FORCE" = "--force" ]; then
  echo "cancelling all your jobs..."; scancel -u "$USER" 2>/dev/null || true
  for d in "${OLD_DIRS[@]}"; do rm -rf "$d"; done
  echo "done — clean slate. Next: setup_env.sh -> download_data.sh -> submit_data.sh"
else
  echo "DRY RUN. Review the list above, edit OLD_DIRS if needed, then re-run with --force."
fi
