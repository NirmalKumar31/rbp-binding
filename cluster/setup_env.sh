#!/bin/bash
# One-time environment setup on Explorer. Run on a compute node from the project root:
#   srun --partition=short --cpus-per-task=8 --mem=16G --time=02:00:00 --pty /bin/bash
#   bash cluster/setup_env.sh
set -euo pipefail
cd "$(dirname "$0")/.."
module load python/3.13.5
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install "transformers==5.14.1" "tokenizers==0.22.2" "huggingface-hub==1.26.0" \
            "multimolecule==0.2.1" "peft==0.20.0"
pip uninstall -y torchao || true

# Explorer GPU nodes cap at CUDA 12.3; default torch is built newer and can't init CUDA.
# The cu118 build runs on any modern driver and has py3.13 wheels.
pip install --force-reinstall --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu118

mkdir -p logs
python -c "import torch; print('torch', torch.__version__, '| cuda_ok', torch.cuda.is_available())"
