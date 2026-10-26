#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export XFORMERS_DISABLED=1
runtime="${AIAA3201_PYTHON:-$HOME/miniconda3/envs/aiaa3201/bin/python}"
exec "$runtime" -m explore_a.cache.cli "$@"
