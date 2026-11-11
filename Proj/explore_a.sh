#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
# GPU training uses CPU readers concurrently; avoid a full OpenMP team per tensor copy.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
exec python -m explore_a "$@"
