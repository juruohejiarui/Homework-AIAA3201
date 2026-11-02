"""Acquire the VPP4DC front-camera validation export, not the 275GB raw DDAD bundle."""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import zipfile
import zlib

from depth_project.io import read_json, write_json
from depth_project.paths import DATASETS, OUTPUTS

URL = ('https://drive.usercontent.google.com/download?'
       'id=1y8Rt3Hld8zVTSKxx9d9yYXSzr5niKN7i&confirm=t')
BYTES = 10654343330
ARCHIVE = 'ddad_vpp4dc_val.zip'
KINDS = {'rgb': '.png', 'gt': '.png', 'intrinsics': '.txt'}
SOURCE = {'provider': 'VPP4DC export; linked by OGNI-DC', 'url': URL, 'bytes': BYTES,
          'reference': 'https://github.com/princeton-vl/OGNI-DC#ddad',
          'converter': 'https://github.com/bartn8/vppdc/blob/main/convert_ddad.py',
          'scope': '3950 front-camera validation images; single-image comparison only',
          'gt': 'LiDAR projection filtered by VPP4DC heuristic; uint16 / 256 meters',
          'temporal_mapping': 'not present; never pair consecutive numeric filenames as video'}


def download(root):
    target = root / 'archives' / ARCHIVE
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + '.part')
    if target.exists():
        if target.stat().st_size != BYTES:
            raise ValueError('Existing archive has wrong size; refusing overwrite')
    else:
        subprocess.run(['aria2c', '--continue=true', '--auto-file-renaming=false',
                        '--allow-overwrite=false', '--file-allocation=none', '--split=4',
                        '--max-connection-per-server=4', '--max-tries=10', '--retry-wait=10',
                        '--connect-timeout=30', '--timeout=60', '--summary-interval=30',
                        '--dir=' + str(target.parent), '--out=' + part.name, URL], check=True)
        if part.stat().st_size != BYTES or part.with_name(part.name + '.aria2').exists():
            raise ValueError('Incomplete archive; rerun the download command')
        part.replace(target)
    write_json(root / 'source.json', SOURCE)
    print('Downloaded validation-only export:', target, flush=True)


def selected_members(source):
    entries, ids = [], {k: set() for k in KINDS}
    for member in source.infolist():
        path = PurePosixPath(member.filename)
        if path.parts and path.parts[0] == '__MACOSX':
            continue
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Unsafe ZIP member: ' + member.filename)
        if member.is_dir():
            continue
        if len(path.parts) != 4 or path.parts[:2] != ('pregenerated', 'val'):
            raise ValueError('Unexpected ZIP layout: ' + member.filename)
        kind = path.parts[2]
        if kind == 'hints':
            continue  # RGB-only depth estimation must not consume sparse GT hints.
        if kind not in KINDS or path.suffix != KINDS[kind] or not path.stem.isdigit():
            raise ValueError('Unexpected validation asset: ' + member.filename)
        if path.stem in ids[kind]:
            raise ValueError('Duplicate validation ID')
        ids[kind].add(path.stem)
        entries.append(member)
    expected = {f'{i:010d}' for i in range(3950)}
    if any(value != expected for value in ids.values()):
        raise ValueError('Validation export does not contain the expected 3950 matched IDs')
    return entries


def extract(root):
    archive = root / 'archives' / ARCHIVE
    if archive.stat().st_size != BYTES:
        raise ValueError('Download the complete validation export first')
    receipts_path = root / 'extraction.jsonl'
    receipts = {}
    if receipts_path.exists():
        for line in receipts_path.read_text().splitlines():
            try:
                record = json.loads(line)
                receipts[record['name']] = record
            except json.JSONDecodeError:
                pass  # Only an interrupted last receipt can be incomplete; recheck that file.
    with zipfile.ZipFile(archive) as source, receipts_path.open('a') as log:
        entries = selected_members(source)
        for number, member in enumerate(entries):
            target = root / member.filename
            saved = receipts.get(member.filename, {})
            if target.is_file() and target.stat().st_size == member.file_size:
                if (saved.get('crc32') == member.CRC and
                        saved.get('mtime_ns') == target.stat().st_mtime_ns):
                    continue
                # Recover a rename completed immediately before its receipt was committed.
                digest = 0
                with target.open('rb') as existing:
                    for chunk in iter(lambda: existing.read(1024*1024), b''):
                        digest = zlib.crc32(chunk, digest)
                if digest == member.CRC:
                    record = {'name': member.filename, 'crc32': member.CRC,
                              'mtime_ns': target.stat().st_mtime_ns}
                    log.write(json.dumps(record) + '\n')
                    log.flush()
                    continue
            if target.exists():
                raise ValueError('Unregistered/conflicting extraction: ' + str(target))
            target.parent.mkdir(parents=True, exist_ok=True)
            pending = target.with_name(target.name + '.part')
            with source.open(member) as reader, pending.open('wb') as writer:
                shutil.copyfileobj(reader, writer, length=1024*1024)
            # ZipExtFile checks CRC at EOF; no extra full reread is necessary.
            pending.replace(target)
            log.write(json.dumps({'name': member.filename, 'crc32': member.CRC,
                                  'mtime_ns': target.stat().st_mtime_ns}) + '\n')
            log.flush()
            if (number + 1) % 500 == 0:
                print('EXTRACT', number + 1, '/', len(entries), flush=True)
    write_json(root / 'prepared.json', {**SOURCE, 'extracted': ['rgb', 'gt', 'intrinsics'],
                                      'hints': 'not extracted or supplied to models'})
    print('Prepared:', root / 'pregenerated/val', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('download', 'extract', 'status', 'background', 'fetch'))
    parser.add_argument('--root', type=Path, default=DATASETS / 'DDAD/vpp4dc_validation')
    parser.add_argument('--state-root', type=Path,
                        default=OUTPUTS / 'explore_a/ddad_validation_download_20261102')
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if args.command == 'background':
        state = args.state_root.expanduser().resolve()
        state.mkdir(parents=True, exist_ok=True)
        with (state / 'background.log').open('ab') as log:
            process = subprocess.Popen([
                sys.executable, '-u', '-m', 'explore_a.download.ddad_validation', 'fetch',
                '--root', str(root), '--state-root', str(state)], stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(state / 'process.json', {'pid': process.pid, 'root': str(root)})
        print('Background acquisition PID:', process.pid, '; log:', state / 'background.log')
    elif args.command == 'fetch':
        state = args.state_root.expanduser().resolve()
        def status(value, **extra):
            write_json(state / 'status.json', {'status': value, 'pid': os.getpid(),
                                              'root': str(root), **extra})
        try:
            status('downloading')
            download(root)
            status('extracting')
            extract(root)
            status('complete')
        except BaseException as error:
            status('failed', error=str(error))
            raise
    elif args.command == 'download':
        download(root)
    elif args.command == 'extract':
        extract(root)
    else:
        path = root / 'archives' / ARCHIVE
        part = path.with_name(path.name + '.part')
        actual = path if path.exists() else part
        print('Archive:', path)
        print('Logical bytes:', actual.stat().st_size if actual.exists() else 0, '/', BYTES,
              '(an aria2 partial file can be sparse; logical size is not downloaded progress)')
        print('Extraction complete:', (root / 'prepared.json').is_file())


if __name__ == '__main__':
    main()
