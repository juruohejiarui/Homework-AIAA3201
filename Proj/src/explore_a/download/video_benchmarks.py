"""Plan first; explicitly download only the selected video benchmark assets."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
import re
import subprocess
import time
from urllib.parse import urlparse

from depth_project.io import read_json, write_json
from explore_a.download import tum
from explore_a.download.video_sources import spring_files, scannet_files, ddad_files, sized

DEFAULT_ROOT = Path.home() / 'Documents/ResourcesLow/Datasets'
FOLDERS = {'tum-validation': 'TUM-RGBD/official_validation', 'spring': 'Spring',
           'scannet': 'ScanNet/video_benchmark', 'ddad': 'DDAD'}
HOSTS = {'cvg.cit.tum.de', 'vision.in.tum.de', 'darus.uni-stuttgart.de',
         'kaldir.vc.in.tum.de', 'tri-ml-public.s3.amazonaws.com'}


def plan(args):
    if args.manifest.exists():
        raise ValueError('Plan exists; reuse it or choose another manifest')
    if args.dataset == 'tum-validation':
        entries = tum.select_archives(tum.crawl(), 'validation')
        with ThreadPoolExecutor(max_workers=4) as pool:
            files = list(pool.map(sized, [
                {**e, 'path': 'archives/' + e['name']} for e in entries]))
        protocol = {'source': tum.PAGE, 'selection': 'All official validation RGB-D recordings; '
                    'hidden GT is trajectory, not sensor depth; no automatic submission.'}
    elif args.dataset == 'spring':
        files, protocol = spring_files()
    elif args.dataset == 'scannet':
        if args.scannet_scenes is None:
            raise ValueError('ScanNet requires --scannet-scenes; no invented paper scene list')
        files, protocol = scannet_files(args.scannet_scenes)
    else:
        files, protocol = ddad_files()
    result = {'schema': 1, 'dataset': args.dataset,
              'output': str((args.root / FOLDERS[args.dataset]).expanduser().resolve()),
              'files': files, 'total_bytes': sum(f['bytes'] for f in files),
              'protocol': protocol}
    validate(result)
    write_json(args.manifest, result)
    print(args.dataset, len(files), 'files;', result['total_bytes']/1e9,
          'GB; no payload downloaded. Plan:', args.manifest)


def validate(value):
    if value.get('schema') != 1 or value.get('dataset') not in FOLDERS or not value.get('files'):
        raise ValueError('Unsupported download plan')
    root, seen = Path(value['output']).resolve(), set()
    for entry in value['files']:
        path, url = PurePosixPath(entry['path']), urlparse(entry['url'])
        if (path.is_absolute() or '..' in path.parts or not path.parts
                or '\n' in str(path) or '\r' in str(path) or str(path) in seen
                or not (root / str(path)).resolve().is_relative_to(root)):
            raise ValueError('Unsafe/duplicate output path')
        seen.add(str(path))
        if (url.scheme != 'https' or url.hostname not in HOSTS or url.username or url.password
                or '\n' in entry['url'] or '\r' in entry['url']):
            raise ValueError('Unexpected publisher URL')
        if not isinstance(entry['bytes'], int) or entry['bytes'] <= 0:
            raise ValueError('Invalid expected file size')
        digest = entry.get('checksum')
        if digest:
            kind = digest['type'].lower()
            size = {'md5': 32, 'sha256': 64}.get(kind)
            if size is None or not re.fullmatch('[0-9a-fA-F]{'+str(size)+'}', digest['value']):
                raise ValueError('Invalid publisher checksum')
    return root


def complete(root, entry, receipts):
    path = root / entry['path']
    saved = receipts.get(entry['path'], {})
    return (path.is_file() and path.stat().st_size == entry['bytes']
            and saved.get('bytes') == entry['bytes']
            and saved.get('mtime_ns') == path.stat().st_mtime_ns)


def download(path, value, root, jobs, allow_ddad):
    if value['dataset'] == 'ddad' and not allow_ddad:
        raise ValueError('DDAD publisher bundles train+val (~275GB / 256GiB); '
                         'pass --allow-ddad-full-bundle only if that download is intended')
    receipt_path = path.with_suffix('.receipts.json')
    receipts = read_json(receipt_path) if receipt_path.exists() else {}
    pending, queue = [], []
    for entry in value['files']:
        if complete(root, entry, receipts):
            continue
        target = root / entry['path']
        if target.exists():
            raise ValueError(f'Existing file has no matching receipt; refusing overwrite: {target}')
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.name + '.part'
        url = entry['url']
        if urlparse(url).hostname == 'darus.uni-stuttgart.de':
            # Dataverse redirects to expiring signed URLs; request a fresh redirect per run.
            separator = '&' if '?' in url else '?'
            url += separator + 'format=original&download_at=' + str(time.time_ns())
        text = f"{url}\n  dir={target.parent}\n  out={partial}\n"
        if entry.get('checksum'):
            digest = entry['checksum']
            text += f"  checksum={digest['type'].lower()}={digest['value']}\n"
        queue.append(text)
        pending.append(entry)
    if pending:
        request = path.with_suffix('.aria2.txt')
        request.write_text(''.join(queue))
        subprocess.run(['aria2c', '--continue=true', '--auto-file-renaming=false',
                        '--allow-overwrite=false', '--check-integrity=true',
                        '--check-certificate=true', '--file-allocation=none',
                        '--split=2', '--max-connection-per-server=2', '--max-tries=8',
                        '--retry-wait=5', '--connect-timeout=30', '--timeout=60',
                        f'--max-concurrent-downloads={jobs}', '--input-file='+str(request)],
                       check=True)
    for entry in pending:
        target = root / entry['path']
        part = target.with_name(target.name + '.part')
        if (not part.is_file() or part.stat().st_size != entry['bytes']
                or part.with_name(part.name+'.aria2').exists()):
            raise ValueError('Incomplete download; rerun the same command: ' + str(part))
        # A successful aria2 invocation checked the publisher digest when one was supplied.
        receipts[entry['path']] = {'bytes': entry['bytes'], 'mtime_ns': part.stat().st_mtime_ns,
                                   'official_checksum_verified': bool(entry.get('checksum'))}
        # Commit metadata before rename: either interruption point remains resumable.
        write_json(receipt_path, receipts)
        if target.exists():
            raise ValueError('Destination appeared during download: ' + str(target))
        part.rename(target)
    print('Complete:', value['dataset'], root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'download', 'status', 'extract-tum'))
    parser.add_argument('--dataset', choices=FOLDERS)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--scannet-scenes', type=Path)
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--allow-ddad-full-bundle', action='store_true')
    args = parser.parse_args()
    args.manifest = args.manifest.expanduser().resolve()
    if not 1 <= args.jobs <= 8:
        parser.error('jobs must be 1–8')
    if args.command == 'plan':
        if args.dataset is None:
            parser.error('plan requires --dataset')
        plan(args)
        return
    if args.dataset or args.scannet_scenes or args.root != DEFAULT_ROOT:
        parser.error('Selection/root apply only to plan; reuse its fixed paths')
    value = read_json(args.manifest)
    root = validate(value)
    if args.command == 'download':
        download(args.manifest, value, root, args.jobs, args.allow_ddad_full_bundle)
    elif args.command == 'extract-tum':
        if value['dataset'] != 'tum-validation':
            raise ValueError('extract-tum supports only TUM RGB-D archives')
        for entry in value['files']:
            target = root / entry['path']
            if target.stat().st_size != entry['bytes']:
                raise ValueError('Archive size mismatch: '+str(target))
            from explore_a.download.tum_extract import extract_archive
            (root / 'sequences').mkdir(parents=True, exist_ok=True)
            extract_archive(target, root / 'sequences', entry)
    else:
        receipts = args.manifest.with_suffix('.receipts.json')
        saved = read_json(receipts) if receipts.exists() else {}
        done = sum(complete(root, e, saved) for e in value['files'])
        print(f"{done}/{len(value['files'])} registered complete; "
              f"{value['total_bytes']/1e9:.3f} GB planned; {root}")


if __name__ == '__main__':
    main()
