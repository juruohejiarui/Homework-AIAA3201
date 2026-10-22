#!/usr/bin/env bash
set -euo pipefail

# Reuse interrupted downloads without touching the running Raw downloader.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
destination="$HOME/Documents/ResourcesLow/Datasets/KITTI/archives/recovery"

mkdir -p "$destination"

# Wait for a previous recovery download to finish saving its partial files.
if pgrep -f "^aria2c .* -d $destination " > /dev/null; then
  printf '等待此前的补下载进程退出，不影响 downloader.sh……\n'
  while pgrep -f "^aria2c .* -d $destination " > /dev/null; do
    sleep 2
  done
fi

aria2c -c -j 3 -x 8 -s 8 \
  --file-allocation=none \
  --auto-file-renaming=false \
  --summary-interval=10 \
  --download-result=full \
  -d "$destination" \
  -i "$script_dir/storage/kitti_repair_urls.txt"

printf '\n下载完成，清单中的 ZIP 位于：%s\n' "$destination"
printf '此脚本不解压、不删除文件，也不覆盖已有 Raw 目录。\n'
