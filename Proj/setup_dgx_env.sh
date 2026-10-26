#!/usr/bin/env bash
# ARM environment only: never clone or modify the existing LLM environments.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
if [[ "$(uname -m)" != "aarch64" ]]; then
  echo 'Run this installer on DGX Spark (aarch64).' >&2
  exit 1
fi
if [[ ! -x "$HOME/miniconda3/envs/aiaa3201/bin/python" ]]; then
  conda create -n aiaa3201 python=3.13.16 pip -y ||
    conda create -n aiaa3201 python=3.13.17 pip -y
fi
conda activate aiaa3201
export CUDA_HOME=/usr/local/cuda
export PATH="$CONDA_PREFIX/bin:$CUDA_HOME/bin:$PATH"
python -m pip install 'torch==2.12.0+cu130' 'torchvision==0.27.0+cu130' \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r "$project_dir/requirements-arm64-cache.txt" \
  -c "$project_dir/constraints-arm64-cache.txt"
python -m pip install --no-deps -e "$project_dir/third_party/ml-depth-pro"
wheel_url='https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.22/flash_attn-2.8.3%2Bcu130torch2.12-cp313-cp313-manylinux_2_34_aarch64.whl'
python -m pip install --no-deps "$wheel_url"
python -m pip freeze > "$project_dir/requirements-arm64-cache.lock.txt"
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
python -m explore_a.cache.environment --output "$project_dir/storage/dgx-environment.json"
