#!/bin/bash
# Co-authored with Claude (Anthropic).
# Download all v2 datasets: eCLIP peaks (from config/proteins.tsv) + genome + GTF + ClinVar.
#   bash cluster/download_data.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw/encode data/raw/clinvar data/reference

enc=https://www.encodeproject.org/files
gencode=https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_45

while IFS=$'\t' read -r prot acc cell; do
  curl -sL "$enc/$acc/@@download/$acc.bed.gz" -o "data/raw/encode/$prot.$acc.bed.gz"
  echo "  peaks $prot $acc ($cell)"
done < <(tail -n +2 config/proteins.tsv)

curl -sL -o data/reference/GRCh38.primary_assembly.genome.fa.gz "$gencode/GRCh38.primary_assembly.genome.fa.gz"
curl -sL -o data/reference/gencode.v45.primary_assembly.annotation.gtf.gz "$gencode/gencode.v45.primary_assembly.annotation.gtf.gz"
gunzip -kf data/reference/GRCh38.primary_assembly.genome.fa.gz
curl -sL -o data/raw/clinvar/clinvar.vcf.gz https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz

echo "done -> data/"
