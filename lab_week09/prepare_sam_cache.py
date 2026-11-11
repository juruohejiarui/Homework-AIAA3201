"""Fetch an official, hash-verified SAM snapshot when old Hub metadata parsing fails."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time

import requests


def main():
    assert Path(sys.prefix).name == 'aiaa3201-lab'
    repo = 'facebook/sam-vit-base'
    response = requests.get(f'https://huggingface.co/api/models/{repo}?blobs=true', timeout=60)
    response.raise_for_status()
    metadata = response.json()
    revision = metadata['sha']
    root = Path.home()/'.cache/huggingface-aiaa3201-lab/hub/models--facebook--sam-vit-base'
    snapshot = root/'snapshots'/revision
    snapshot.mkdir(parents=True, exist_ok=True)
    records = []
    for name in ('config.json', 'preprocessor_config.json', 'model.safetensors'):
        entry = next(x for x in metadata['siblings'] if x['rfilename'] == name)
        target = snapshot/name
        expected = entry.get('lfs', {}).get('sha256')
        def verify():
            if not target.exists() or target.stat().st_size != entry['size']:
                return False
            if expected:
                digest = hashlib.sha256(target.read_bytes()).hexdigest()
                return digest == expected
            data = target.read_bytes()
            return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest() == entry['blobId']
        if not verify():
            url = f'https://huggingface.co/{repo}/resolve/{revision}/{name}'
            response = requests.get(url, params={'download': 'true', 'lab_fetch': str(int(time.time()))},
                                    stream=True, timeout=(30, 120))
            response.raise_for_status()
            pending = target.with_suffix(target.suffix+'.part')
            total, last = 0, time.monotonic()
            with pending.open('wb') as stream:
                for chunk in response.iter_content(1024*1024):
                    if chunk:
                        stream.write(chunk)
                        total += len(chunk)
                        if time.monotonic()-last > 10:
                            print(name, total, '/', entry['size'], flush=True)
                            last = time.monotonic()
            pending.replace(target)
            assert verify(), f'Official hash/size verification failed: {name}'
        records.append({'file': name, 'size': target.stat().st_size,
                        'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                        'official_lfs_sha256': expected})
        print('Verified official file:', name, flush=True)
    (root/'refs').mkdir(exist_ok=True)
    (root/'refs/main').write_text(revision)
    Path(__file__).with_name('logs').joinpath('sam_download.json').write_text(
        json.dumps({'repo': repo, 'revision': revision, 'files': records}, indent=2)+'\n')
    print('SAM official snapshot ready:', snapshot, flush=True)


if __name__ == '__main__':
    main()
