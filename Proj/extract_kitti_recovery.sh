#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/.."
exec conda run --no-capture-output -n aiaa3201 \
  python -m depth_project.storage.kitti_recovery "$@"
