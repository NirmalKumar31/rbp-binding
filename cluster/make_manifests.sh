#!/bin/bash
# Generate the job manifests (single source of truth for the arrays) from config + ablation list.
#   prep/cnn: <protein>\t<negatives>          16 primary + 4 splice_matched = 20 lines
#   lm:       <model>\t<protein>\t<negatives> (16+4) x 3 models             = 60 lines
set -euo pipefail
cd "$(dirname "$0")/.."
ABLATION="PUM1 LIN28B U2AF1 RBFOX2"
LMS="rnafm rnabert splicebert"

: > cluster/manifest_prep.txt
: > cluster/manifest_cnn.txt
: > cluster/manifest_lm.txt

emit () {   # $1=protein  $2=negatives
  printf '%s\t%s\n' "$1" "$2" >> cluster/manifest_prep.txt
  printf '%s\t%s\n' "$1" "$2" >> cluster/manifest_cnn.txt
  for m in $LMS; do printf '%s\t%s\t%s\n' "$m" "$1" "$2" >> cluster/manifest_lm.txt; done
}

while IFS=$'\t' read -r prot acc cell; do emit "$prot" primary; done < <(tail -n +2 config/proteins.tsv)
for p in $ABLATION; do emit "$p" splice_matched; done

wc -l cluster/manifest_prep.txt cluster/manifest_cnn.txt cluster/manifest_lm.txt
