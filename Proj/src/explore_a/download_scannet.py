#!/usr/bin/env python3
"""Plan, resume and verify ScanNet v1 full release or an RGB-D subset."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from urllib.parse import urlparse

BASE = 'https://kaldir.vc.in.tum.de/scannet/'
SPLITS = 'https://raw.githubusercontent.com/ScanNet/ScanNet/master/Tasks/Benchmark/'
SCENE = re.compile(r'scene\d{4}_\d{2}\Z')
HERE = Path(__file__).resolve().parents[2]
DEFAULT_PLAN = HERE / 'storage/scannet/plan.json'
DEFAULT_OUTPUT = Path.home() / 'Documents/ResourcesLow/Datasets/ScanNet'
HOSTS = {'kaldir.vc.in.tum.de', 'kaldir.vc.cit.tum.de'}
V1_FILETYPES = ['.aggregation.json', '.sens', '.txt', '_vh_clean.ply',
                '_vh_clean_2.0.010000.segs.json', '_vh_clean_2.ply',
                '_vh_clean.segs.json', '_vh_clean.aggregation.json',
                '_vh_clean_2.labels.ply', '_2d-instance.zip',
                '_2d-instance-filt.zip', '_2d-label.zip', '_2d-label-filt.zip']


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def get_text(url):
    with urllib.request.urlopen(url, timeout=45) as response:
        data = response.read(16 * 1024 * 1024 + 1)
    if len(data) > 16 * 1024 * 1024:
        raise ValueError('Metadata exceeds 16 MiB: ' + url)
    return data.decode('utf-8-sig')


def scene_ids(text):
    ids = [line.strip() for line in text.splitlines() if line.strip()]
    if not ids or any(not SCENE.fullmatch(s) for s in ids) or len(set(ids)) != len(ids):
        raise ValueError('Invalid or duplicate scene IDs in scene list')
    return ids


def checksum_entries(text):
    """Read GNU md5sum/sha256sum lines; do not treat ETag as a checksum."""
    entries = {}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        match = re.fullmatch(r'([0-9a-fA-F]{32}|[0-9a-fA-F]{64})\s+\*?(.+)', line.strip())
        if not match:
            raise ValueError('Unsupported checksum line: ' + line[:120])
        digest, name = match.groups()
        name = name.removeprefix('./')
        p = PurePosixPath(name)
        if p.is_absolute() or '..' in p.parts:
            raise ValueError('Unsafe checksum path: ' + name)
        entry = {'algorithm': 'sha256' if len(digest) == 64 else 'md5',
                 'digest': digest.lower()}
        if name in entries and entries[name] != entry:
            raise ValueError('Conflicting checksums: ' + name)
        entries[name] = entry
    if not entries:
        raise ValueError('Empty checksum list')
    return entries


def lookup_checksum(entries, remote, local):
    candidates = [entries[k] for k in (remote, local, PurePosixPath(local).name) if k in entries]
    if candidates and any(c != candidates[0] for c in candidates):
        raise ValueError('Ambiguous checksum: ' + local)
    return candidates[0] if candidates else None


def head_file(item):
    request = urllib.request.Request(item['url'], method='HEAD')
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                length = response.headers.get('Content-Length')
                if response.status != 200 or length is None or int(length) <= 0:
                    raise ValueError('Missing valid Content-Length: ' + item['url'])
                return {**item, 'bytes': int(length),
                        'etag': response.headers.get('ETag'),
                        'last_modified': response.headers.get('Last-Modified')}
        except Exception:
            if attempt == 2:
                raise


def make_plan(args):
    if args.manifest.exists():
        raise ValueError('Plan already exists. Reuse it or choose a new --manifest path.')
    source_lists = {}
    if args.v1_full:
        source_lists['scannet_v1_scans.txt'] = get_text(BASE + 'v1/scans.txt')
        selected = sorted(scene_ids(source_lists['scannet_v1_scans.txt']))
    else:
        source_lists['scannetv2_train.txt'] = get_text(SPLITS + 'scannetv2_train.txt')
        source_lists['scannetv2_val.txt'] = get_text(SPLITS + 'scannetv2_val.txt')
        train = scene_ids(source_lists['scannetv2_train.txt'])
        val = scene_ids(source_lists['scannetv2_val.txt'])
        val_spaces = {s.split('_')[0] for s in val}
        eligible = [s for s in train if s.split('_')[0] not in val_spaces]
        if args.scene_list:
            selected = scene_ids(args.scene_list.read_text())
            if not set(selected) <= set(eligible):
                raise ValueError('Requested scenes must be in v2 train and outside validation spaces')
        elif args.all_train:
            selected = sorted(eligible)
        else:
            # Prefer diverse physical spaces over repeat scans of the same room.
            by_space = {}
            for scene in sorted(eligible):
                by_space.setdefault(scene.split('_')[0], []).append(scene)
            rng = random.Random(args.seed)
            spaces = sorted(by_space)
            rng.shuffle(spaces)
            selected = sorted(rng.choice(by_space[s]) for s in spaces[:args.max_scenes])
    hashes = {}
    checksum_source = None
    checksum_text = None
    if args.official_checksums_url:
        parsed = urlparse(args.official_checksums_url)
        if parsed.scheme != 'https' or parsed.hostname not in HOSTS:
            raise ValueError('Official checksum URL must use HTTPS on the ScanNet publisher host')
        checksum_source = args.official_checksums_url
        checksum_text = get_text(checksum_source)
        hashes = checksum_entries(checksum_text)
    items = []
    for scene in selected:
        types = [(extension, 'v1') for extension in V1_FILETYPES] if args.v1_full else [('.sens', 'v1'), ('.txt', 'v2')]
        for extension, release in types:
            remote = f'{release}/scans/{scene}/{scene}{extension}'
            local = f'scans/{scene}/{scene}{extension}'
            items.append({'scene': scene, 'path': local, 'url': BASE + remote,
                          'official_checksum': lookup_checksum(hashes, remote, local)})
    if args.v1_full:
        tasks = ['scannet-labels.combined.tsv']
        if args.include_v1_task_data:
            tasks.extend(['obj_classification/data.zip', 'obj_classification/trained_models.zip',
                          'voxel_labeling/data.zip', 'voxel_labeling/trained_models.zip'])
        for task in tasks:
            remote, local = 'v1/tasks/' + task, 'tasks/' + task
            items.append({'scene': None, 'path': local, 'url': BASE + remote,
                          'official_checksum': lookup_checksum(hashes, remote, local)})
    if args.require_official_checksums and any(not f['official_checksum'] for f in items):
        raise ValueError('Official checksums are absent or do not cover every selected file')
    # Append each completed lookup so interrupted planning can reuse its work.
    cache_path = args.manifest.with_suffix('.heads.jsonl')
    cached = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            try:
                entry = json.loads(line)
                if isinstance(entry.get('bytes'), int) and entry['bytes'] > 0:
                    cached[entry['url']] = entry
            except (ValueError, KeyError, TypeError, AttributeError):
                continue  # An interrupted final write is safe to query again.
    resolved = {}
    pending = []
    for item in items:
        entry = cached.get(item['url'])
        if entry:
            resolved[item['url']] = {**item, **{k: entry.get(k) for k in
                                               ('bytes', 'etag', 'last_modified')}}
        else:
            pending.append(item)
    print(f'Checking sizes for {len(items)} files (HEAD only); '
          f'cached: {len(resolved)}; pending: {len(pending)}; '
          f'workers: {args.metadata_workers}', flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.metadata_workers) as pool:
        with cache_path.open('a', buffering=1) as cache_file:
            # Separate a potentially truncated final record from new records.
            cache_file.write('\n')
            futures = [pool.submit(head_file, item) for item in pending]
            try:
                for completed, future in enumerate(as_completed(futures), 1):
                    item = future.result()
                    resolved[item['url']] = item
                    cache_file.write(json.dumps({k: item[k] for k in
                                                ('url', 'bytes', 'etag', 'last_modified')}) + '\n')
                    if completed % 50 == 0 or completed == len(pending):
                        elapsed = max(time.monotonic() - started, .001)
                        rate = completed / elapsed
                        eta = (len(pending) - completed) / rate
                        print(f'Metadata: {len(resolved)}/{len(items)}; '
                              f'{rate:.1f} files/s; ETA {eta / 60:.1f} min', flush=True)
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
    items = [resolved[item['url']] for item in items]
    output = args.output.expanduser().resolve()
    if args.v1_full and args.output == DEFAULT_OUTPUT:
        output = output / 'v1'
    manifest = {'schema': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
                'output': str(output), 'seed': args.seed,
                'split': 'scannet_v1_full' if args.v1_full else 'scannetv2_train',
                'split_source': BASE + 'v1/scans.txt' if args.v1_full else SPLITS,
                'include_v1_task_data': args.include_v1_task_data,
                'scenes': selected, 'files': items,
                'official_checksums_url': checksum_source,
                'official_checksum_count': sum(bool(f['official_checksum']) for f in items),
                'total_bytes': sum(f['bytes'] for f in items)}
    save_json(args.manifest, manifest)
    metadata = args.manifest.with_suffix('.metadata')
    metadata.mkdir(parents=True, exist_ok=True)
    for name, text in source_lists.items():
        (metadata / name).write_text(text)
    (metadata / 'selected_scans.txt').write_text('\n'.join(selected) + '\n')
    if checksum_text is not None:
        (metadata / 'official_checksums.txt').write_text(checksum_text)
    print(f'Plan: {args.manifest}\nScenes: {len(selected)}; files: {len(items)}; '
          f'total: {manifest["total_bytes"] / 1e9:.3f} GB / '
          f'{manifest["total_bytes"] / 2**30:.3f} GiB\nOutput: {manifest["output"]}')
    print(f'Official checksums: {manifest["official_checksum_count"]}/{len(items)} files. '
          'No dataset payload downloaded.')


def load_plan(path):
    plan = json.loads(path.read_text())
    if plan.get('schema') != 1 or not plan.get('files'):
        raise ValueError('Invalid plan')
    root = Path(plan['output']).resolve()
    paths = set()
    for item in plan['files']:
        p = PurePosixPath(item['path'])
        if p.is_absolute() or '..' in p.parts or item['path'] in paths:
            raise ValueError('Invalid or duplicate local path')
        paths.add(item['path'])
        if not (root / item['path']).resolve().is_relative_to(root):
            raise ValueError('Output path escapes dataset root')
        u = urlparse(item['url'])
        if u.scheme != 'https' or u.hostname not in HOSTS:
            raise ValueError('Unexpected dataset URL')
    return plan, root


def inspect_file(path, item, old=None):
    if not path.is_file():
        raise ValueError('Missing file: ' + str(path))
    if path.stat().st_size != item['bytes']:
        raise ValueError('Size mismatch: ' + str(path))
    official = item.get('official_checksum')
    sha = hashlib.sha256()
    other = hashlib.new(official['algorithm']) if official and official['algorithm'] != 'sha256' else None
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            sha.update(block)
            if other is not None:
                other.update(block)
    digest = sha.hexdigest()
    if official:
        actual = other.hexdigest() if other is not None else digest
        if actual != official['digest']:
            raise ValueError('OFFICIAL checksum mismatch: ' + str(path))
    if old and old['sha256'] != digest:
        raise ValueError('LOCAL baseline SHA256 mismatch: ' + str(path))
    return {'bytes': item['bytes'], 'sha256': digest,
            'official_checksum_verified': bool(official), 'url': item['url']}


def process_files(args):
    plan, root = load_plan(args.manifest)
    if args.require_official_checksums and any(not i.get('official_checksum') for i in plan['files']):
        raise ValueError('Plan lacks complete official checksums; refusing requested strict mode')
    receipt_path = args.manifest.with_suffix('.verified.json')
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {'files': {}}
    if args.command == 'verify':
        failures = []
        for item in plan['files']:
            try:
                if not item.get('official_checksum') and item['path'] not in receipt['files']:
                    raise ValueError('No official checksum or previously established local baseline')
                inspect_file(root / item['path'], item, receipt['files'].get(item['path']))
                print('OK ' + item['path'], flush=True)
            except (OSError, ValueError) as e:
                failures.append(str(e))
                print('FAIL ' + str(e), flush=True)
        if failures:
            raise ValueError(f'{len(failures)} file(s) failed verification')
        print('Verified sizes and available digests. Local-only hashes are NOT publisher checksums.')
        return
    if not shutil.which('aria2c'):
        raise ValueError('aria2c is required for resumable downloads')
    pending = []
    for item in plan['files']:
        target = root / item['path']
        if target.exists():
            result = inspect_file(target, item, receipt['files'].get(item['path']))
            if not item.get('official_checksum') and item['path'] not in receipt['files']:
                print('Existing file: size checked; establishing LOCAL baseline only: ' + item['path'])
            receipt['files'][item['path']] = result
            save_json(receipt_path, receipt)
        else:
            pending.append(item)
    lines = []
    for item in pending:
        target = root / item['path']
        lines.extend([item['url'], '  dir=' + str(target.parent), '  out=' + target.name + '.part'])
        if item.get('official_checksum'):
            h = item['official_checksum']
            name = 'sha-256' if h['algorithm'] == 'sha256' else 'md5'
            lines.append(f'  checksum={name}={h["digest"]}')
    input_path = args.manifest.with_suffix('.aria2.txt')
    input_path.write_text('\n'.join(lines) + '\n')
    status = 0
    if pending:
        status = subprocess.run(['aria2c', '--input-file=' + str(input_path),
                                 '--continue=true', '--auto-file-renaming=false',
                                 '--allow-overwrite=true', '--check-certificate=true',
                                 '--check-integrity=true', '--file-allocation=none',
                                 '--max-tries=8', '--retry-wait=5', '--connect-timeout=30',
                                 '--timeout=60', '--max-concurrent-downloads=' + str(args.jobs),
                                 '--split=2', '--max-connection-per-server=2',
                                 '--summary-interval=30'], check=False).returncode
    failed = []
    for item in pending:
        target = root / item['path']
        part = target.with_name(target.name + '.part')
        try:
            if part.with_name(part.name + '.aria2').exists():
                raise ValueError('Incomplete transfer (resume by rerunning): ' + str(part))
            result = inspect_file(part, item)
            part.replace(target)
            receipt['files'][item['path']] = result
            save_json(receipt_path, receipt)
            print('Completed; ' + ('OFFICIAL digest checked: ' if result['official_checksum_verified']
                                  else 'size checked, LOCAL SHA256 recorded: ') + item['path'], flush=True)
        except (OSError, ValueError) as e:
            failed.append(str(e))
            print('FAIL ' + str(e), flush=True)
    if status or failed:
        raise ValueError(f'aria2 exit={status}; incomplete/failed files={len(failed)}; rerun download to resume')
    print('Done. Receipt: ' + str(receipt_path))
    print('Official digests and LOCAL integrity baselines are recorded separately.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'download', 'verify'])
    parser.add_argument('--manifest', type=Path, default=DEFAULT_PLAN)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--max-scenes', type=int, default=20)
    selection.add_argument('--all-train', action='store_true')
    selection.add_argument('--scene-list', type=Path)
    selection.add_argument('--v1-full', action='store_true', help='all official v1 scans and all 13 scan file types')
    parser.add_argument('--include-v1-task-data', action='store_true', help='also include legacy v1 task data and pretrained models')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--metadata-workers', type=int, default=32,
                        help='parallel HEAD requests during planning (1-64; default 32)')
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--official-checksums-url')
    parser.add_argument('--require-official-checksums', action='store_true')
    args = parser.parse_args()
    if args.max_scenes < 1 or not 1 <= args.metadata_workers <= 64 or not 1 <= args.jobs <= 8:
        parser.error('Scene count must be positive; metadata workers: 1-64; download jobs: 1-8')
    args.manifest = args.manifest.expanduser().resolve()
    if args.include_v1_task_data and not args.v1_full:
        parser.error('--include-v1-task-data requires --v1-full during planning')
    if args.command != 'plan' and (args.official_checksums_url or args.scene_list or args.all_train or args.v1_full
                                   or args.output != DEFAULT_OUTPUT or args.max_scenes != 20 or args.seed != 2026):
        parser.error('Dataset selection/output/checksum URL are plan options; use the saved manifest afterwards')
    try:
        make_plan(args) if args.command == 'plan' else process_files(args)
    except (OSError, ValueError, KeyError) as e:
        print('ERROR: ' + str(e), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
