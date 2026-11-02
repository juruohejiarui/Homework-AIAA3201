"""Normalize publisher validation _secret roots without overwriting existing sequences."""
import copy
from pathlib import Path, PurePosixPath
import tarfile

from depth_project.io import read_json, write_json


def extract_archive(archive, root, entry):
    name = entry['name'].removesuffix('.tgz')
    destination = root / name
    marker = destination / '.explore_a_extracted.json'
    if marker.exists():
        receipt = read_json(marker)
        if receipt.get('archive') != entry['name'] or receipt.get('bytes') != entry['bytes']:
            raise ValueError(f'Extraction receipt conflict: {destination}')
        return
    if destination.exists():
        raise ValueError(f'Unregistered extraction exists; refusing overwrite: {destination}')
    allowed = {name}
    if name.endswith('_validation'):
        allowed.add(name.removesuffix('_validation') + '_secret')
    staging = root / ('.extracting-' + name)
    with tarfile.open(archive, 'r:gz') as source:
        members, observed = [], set()
        for member in source.getmembers():
            path = PurePosixPath(member.name)
            if (path.is_absolute() or '..' in path.parts or not path.parts
                    or path.parts[0] not in allowed or not (member.isfile() or member.isdir())):
                raise ValueError(f'Unsafe/unexpected tar member: {member.name}')
            observed.add(path.parts[0])
            item = copy.copy(member)
            item.name = str(PurePosixPath(name, *path.parts[1:]))
            members.append(item)
        if len(observed) != 1:
            raise ValueError('Expected one archive root')
        staging.mkdir(parents=True, exist_ok=True)
        source.extractall(staging, members=members, filter='data')
    write_json(staging / name / marker.name,
               {'archive': entry['name'], 'bytes': entry['bytes'],
                'publisher_root': next(iter(observed))})
    (staging / name).rename(destination)
    staging.rmdir()
    print('Extracted', entry['name'], flush=True)
