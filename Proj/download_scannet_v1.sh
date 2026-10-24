#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
action="${1:-download}"
if [ "$#" -gt 0 ]; then shift; fi
manifest="$script_dir/storage/scannet/v1-full.json"
if [ "$action" = plan ]; then
  exec python3 "$script_dir/src/explore_a/download_scannet.py" plan \
    --v1-full --include-v1-task-data --manifest "$manifest" "$@"
else
  exec python3 "$script_dir/src/explore_a/download_scannet.py" "$action" \
    --manifest "$manifest" "$@"
fi
