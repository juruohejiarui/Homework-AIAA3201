"""Publisher metadata for video-only benchmark acquisition, never evaluation split invention."""
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen

SPRING = 'https://darus.uni-stuttgart.de'
SPRING_METADATA = SPRING + '/api/datasets/:persistentId/?persistentId=doi:10.18419/darus-3376'
SCANNET = 'https://kaldir.vc.in.tum.de/scannet/'
SCANNET_VAL = ('https://raw.githubusercontent.com/ScanNet/ScanNet/master/'
               'Tasks/Benchmark/scannetv2_val.txt')
DDAD = 'https://tri-ml-public.s3.amazonaws.com/github/DDAD/datasets/DDAD.tar'


def metadata(url):
    with urlopen(url, timeout=45) as response:
        data = response.read(8 * 1024 * 1024 + 1)
    if len(data) > 8 * 1024 * 1024:
        raise ValueError('Metadata response exceeds 8 MiB')
    return data.decode('utf-8')


def sized(item):
    with urlopen(Request(item['url'], method='HEAD'), timeout=45) as response:
        size = int(response.headers['Content-Length'])
    if size <= 0:
        raise ValueError('Missing download size: ' + item['url'])
    return {**item, 'bytes': size}


def spring_files():
    info = json.loads(metadata(SPRING_METADATA))
    version = info['data']['latestVersion']
    wanted = {'train_cam_data.zip', 'train_frame_left.zip', 'train_disp1_left.zip'}
    result = []
    for row in version['files']:
        file = row['dataFile']
        if file['filename'] in wanted:
            digest = file['checksum']
            result.append({'name': file['filename'], 'path': 'archives/' + file['filename'],
                           'url': SPRING + '/api/access/datafile/' + str(file['id']),
                           'bytes': file['filesize'], 'checksum': digest})
    if {f['name'] for f in result} != wanted:
        raise ValueError('Spring publisher files changed; inspect metadata')
    return result, {'source': SPRING_METADATA,
                    'publisher_version': [version['versionNumber'], version['versionMinorNumber']],
                    'selection': 'Public-GT train release, left RGB + disp1 + camera metadata. '
                                 'Selected video benchmark protocol.'}


def scannet_files(list_path):
    scenes = list_path.read_text().split()
    if (not scenes or len(scenes) != len(set(scenes))
            or any(not re.fullmatch(r'scene\d{4}_\d{2}', s) for s in scenes)):
        raise ValueError('Invalid/duplicate ScanNet scene IDs')
    official = set(metadata(SCANNET_VAL).split())
    if not set(scenes) <= official:
        raise ValueError('Use explicit scenes from official v2 validation, outside fusion training')
    files = []
    for scene in sorted(scenes):
        relative = f'v1/scans/{scene}/{scene}.sens'
        files.append(sized({'name': scene + '.sens', 'path': relative,
                            'url': SCANNET + relative}))
    return files, {'source': SCANNET_VAL, 'scenes': sorted(scenes),
                   'selection': 'User-selected official-val sequence list; VDA code does not '
                                'publish the exact scene-root list, so not claimed identical.'}


def ddad_files():
    file = sized({'name': 'DDAD.tar', 'path': 'archives/DDAD.tar', 'url': DDAD,
                  'checksum': {'type': 'MD5', 'value': 'c0da97967f76da80f86d6f97d0d98904'}})
    return [file], {'source': 'https://github.com/TRI-ML/DDAD',
                    'selection': 'Official download bundles train+val (~275 GB / 256 GiB). '
                                 'Evaluation should select val only; no official val-only RGB '
                                 'archive verified. Explicit opt-in required to download bundle.'}
