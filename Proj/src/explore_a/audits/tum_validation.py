"""Post-download frame timestamp census; scene independence is a separate question."""
import argparse
from collections import defaultdict
from pathlib import Path

from depth_project.io import read_json, write_json


def audit(validation_root, ordinary_root, manifest):
    sequences = sorted((validation_root / 'sequences').glob('rgbd_dataset_*_validation'))
    if not sequences:
        raise ValueError('Extract validation RGB-D archives first; no empty-audit success')
    ordinary = defaultdict(set)
    for sequence in ordinary_root.glob('rgbd_dataset_*'):
        for image in (sequence / 'rgb').glob('*.png'):
            ordinary[image.name].add(sequence.name)
    endpoints = defaultdict(set)
    for row in read_json(manifest)['samples']:
        if row['dataset'] == 'tum':
            for key in ('previous_rgb', 'current_rgb'):
                endpoints[Path(row[key]).name].add(row['split'])
    results = []
    for sequence in sequences:
        images = sorted((sequence / 'rgb').glob('*.png'))
        if not images or not list((sequence / 'depth').glob('*.png')):
            raise ValueError('Missing RGB/depth: ' + str(sequence))
        matches, used = {}, {}
        for image in images:
            if image.name in ordinary:
                matches[image.name] = sorted(ordinary[image.name])
            if image.name in endpoints:
                used[image.name] = sorted(endpoints[image.name])
        stamps = [float(p.stem) for p in images]
        results.append({'sequence': sequence.name, 'rgb_frames': len(images),
                        'rgb_time_range': [min(stamps), max(stamps)],
                        'ordinary_timestamp_matches': matches, 'manifest_timestamp_matches': used})
    return {'sequences': results,
            'scope': 'All extracted validation RGB timestamps vs local ordinary frames and '
                     'current manifest endpoints; no full-image hashing.',
            'scene_note': 'Publisher secondary recordings repeat ordinary scenes. '
                          'Distinct timestamps do not establish physical-room independence.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('validation-root', 'ordinary-root', 'manifest', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.validation_root, args.ordinary_root, args.manifest)
    write_json(args.output, result)
    print('Audited validation sequences:', len(result['sequences']))


if __name__ == '__main__':
    main()
