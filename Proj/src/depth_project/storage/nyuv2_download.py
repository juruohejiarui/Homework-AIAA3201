"""Download full NYUv2 Raw, or a frozen HF training subset, plus labeled assets."""
import argparse
import subprocess

from depth_project.io import read_json
from depth_project.paths import DATASETS, PROJECT
from depth_project.storage.mount import require_dataset_volume


def download_entries(mode):
    sources = read_json(PROJECT / 'storage/nyuv2_download_sources.json')
    entries = [sources[name] for name in ['labeled', 'split', 'toolbox']]
    if mode == 'full':
        raw = read_json(PROJECT / 'storage/nyuv2_full_download_manifest.json')
        for item in raw['archives']:
            entries.append({
                'path': f'archives/raw/{item["name"]}',
                'bytes': item['bytes'],
                'md5': item['md5'],
                'urls': [item['url']],
            })
    if mode == 'prepared':
        prepared = sources['prepared']
        for shard in prepared['shards']:
            name = shard['name']
            primary = f'https://huggingface.co/datasets/{prepared["repo"]}'
            fallback = f'https://huggingface.co/datasets/{prepared["fallback_repo"]}'
            entries.append({
                'path': f'prepared_fastdepth/archives/{name}',
                'bytes': shard['bytes'],
                'sha256': shard['sha256'],
                'urls': [
                    f'{primary}/resolve/{prepared["revision"]}/data/{name}',
                    f'{fallback}/resolve/{prepared["fallback_revision"]}/data/{name}',
                ],
            })
    return entries


def existing_download(target, entry):
    if not target.is_file() or target.with_name(target.name + '.aria2').exists():
        return False
    expected = entry.get('bytes')
    size = target.stat().st_size
    return size == expected if expected is not None else size > 0


def queue_text(entries, root):
    sections = []
    for entry in entries:
        target = root / entry['path']
        if existing_download(target, entry):
            print(f'复用现有文件，不重复哈希：{target}', flush=True)
            continue
        section = ['\t'.join(entry['urls']), f'  dir={target.parent}', f'  out={target.name}']
        if 'sha256' in entry:
            section.append(f'  checksum=sha-256={entry["sha256"]}')
        elif 'md5' in entry:
            section.append(f'  checksum=md5={entry["md5"]}')
        sections.append('\n'.join(section))
    return '\n'.join(sections) + '\n' if sections else ''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['full', 'prepared', 'labeled'], default='full')
    parser.add_argument('--plan', action='store_true', help='只显示下载清单，不写文件、不下载')
    args = parser.parse_args()
    root = (DATASETS / 'nyu_depth_v2').resolve()
    entries = download_entries(args.mode)
    total = sum(entry.get('bytes', 0) for entry in entries)
    print(f'模式：{args.mode}；目标：{root}', flush=True)
    print(f'清单：{len(entries)}个文件，已知体积 {total / 1e9:.2f} GB', flush=True)
    if args.plan:
        for entry in entries:
            state = '已有' if existing_download(root / entry['path'], entry) else '下载/续传'
            print(f'[{state}] {entry["path"]}')
        return

    require_dataset_volume()
    queue = queue_text(entries, root)
    if not queue:
        print('所选清单中的文件均已存在。', flush=True)
        return
    requests = root / 'archives/download_requests'
    requests.mkdir(parents=True, exist_ok=True)
    queue_path = requests / f'nyuv2_{args.mode}.aria2.txt'
    queue_path.write_text(queue)
    subprocess.run([
        'aria2c', '-c', '-j', '3', '-x', '8', '-s', '8',
        '--file-allocation=none', '--auto-file-renaming=false',
        '--max-tries=20', '--retry-wait=10', '--summary-interval=10',
        '-i', str(queue_path),
    ], check=True)
    print('下载完成；原包保留。解压请使用系统 unzip / tar，脚本不会启动训练。', flush=True)


if __name__ == '__main__':
    main()
