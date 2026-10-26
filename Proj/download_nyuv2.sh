#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$script_dir/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$script_dir/.."
case "${1:-}" in
  plan|download|verify|extract)
    exec conda run --no-capture-output -n aiaa3201 \
      python -m explore_a.download.nyu "$@"
    ;;
esac
# Keep the earlier --mode / --plan interface available for existing scripts.
exec conda run --no-capture-output -n aiaa3201 \
  python -m depth_project.storage.nyuv2_download "$@"
