"""Explicit incremental native prediction production; inference consumers never write NPZ."""
import argparse
import copy
from pathlib import Path

from depth_project.io import read_json, write_json
from explore_a.cache.storage import write_npz
from explore_a.config import load_config
from explore_a.frontend import field
from explore_a.online.cache_components import native_binding
from explore_a.online.factory import make_runtime


def samples(config, scope, split, benchmark_root, setting, datasets, nyu_root):
    if scope == 'manifest':
        from explore_a.data.prepare import load_manifest
        from depth_project.io import read_rgb
        from explore_a.evaluation.part2 import raw_input
        for index, row in enumerate(load_manifest(config)['samples']):
            if split != 'all' and row['split'] != split:
                continue
            images = {}
            for old, prefix in ((True, 'previous'), (False, 'current')):
                if row['depth_encoding'] == 'nyu_h5':
                    from explore_a.data.nyu_h5 import read_h5
                    images[old] = read_h5(row[prefix+'_rgb'], depth=False)[0]
                elif row['depth_encoding'] in ('sens', 'nyu_raw'):
                    from explore_a.data.reading import read_frame
                    images[old] = read_frame(row, old)[0]
                else:
                    images[old] = read_rgb(row[prefix+'_rgb'])
            raw = raw_input(images[True], images[False], row['dataset'], row['sequence'],
                            row['previous_id'], row['current_id'], config)
            raw['row'] = row
            yield index, raw
    elif scope == 'nyu':
        yield from labeled_raw(config, nyu_root)
    else:
        from depth_project.data.protocol import select_sequences
        from explore_a.evaluation.part2 import MANIFESTS, benchmark_path, benchmark_root_for
        from explore_a.online.video import video_raw
        runtime = make_runtime(config, ['farneback'])
        try:
            for dataset in datasets:
                dataset_root = benchmark_root_for(dataset, benchmark_root)
                manifest = read_json(dataset_root/MANIFESTS[dataset])
                selected_setting = "full" if dataset == "tum-validation" else setting
                for sequence in select_sequences(manifest, selected_setting):
                    for frame in sequence['frames']:
                        frame['image'] = benchmark_path(frame['image'], dataset_root)
                    for index, raw, _ in video_raw(sequence, dataset, setting,
                                                   manifest['config']['crop'], config,
                                                   runtime, submit=False):
                        yield index, raw
        finally:
            runtime.close()


def labeled_raw(config, root):
    import torch
    from depth_project.io import read_rgb
    from explore_a.evaluation.part2 import raw_input
    root = Path(root)
    for index, image_id in enumerate(read_json(root/'test/manifest.json')['test_ids']):
        path = root/'test/rgb'/f'{image_id:04d}.png'
        rgb = read_rgb(path)
        raw = raw_input(rgb, rgb, 'nyuv2', 'official_labeled', str(image_id), str(image_id), config)
        raw['row'].update(current_rgb=str(path), previous_rgb=str(path),
                          input_transform={'scope':'nyu-labeled', 'rgb':'original-v1'})
        raw['data']['previous_image_valid'] = torch.zeros_like(raw['data']['image_valid'])
        yield index, raw


def _produce(config, scope, split='all', benchmark_root=None, setting='long',
            datasets=('kitti', 'sintel', 'bonn'), nyu_root=None, device='cuda'):
    config = copy.deepcopy(config)
    runtime = make_runtime(config, ['farneback'], False, device)
    written, reused = 0, 0
    try:
        stream = samples(config, scope, split, benchmark_root, setting, datasets, nyu_root)
        for index, raw in stream:
            # Production requests depth only, never GT, optical flow or fusion heads.
            raw.pop('frozen_jobs', None)
            raw['frozen_ticket'] = runtime.prepare(
                raw['row'], index, {}, phase='native', native=True)
            for old, prefix in ((True, 'previous'), (False, 'current')):
                path, identity = native_binding(config, raw, old)
                sidecar = path.with_suffix('.json')
                if sidecar.exists() and read_json(sidecar) != identity:
                    raise ValueError('Conflicting native production identity: '+str(path))
                if path.exists() and sidecar.exists():
                    reused += 1
                    continue
                job = runtime.submit_component(raw, 'prediction', old, native=True)
                value = runtime.resolve(job)['prediction']
                if not job.native:
                    raise ValueError('Native producer received a canvas prediction')
                if path.exists():
                    raise ValueError('Uncommitted native cache: '+str(path))
                write_json(sidecar, identity)
                write_npz(path, **{field(config):value[None]})
                written += 1
            if (index+1) % 100 == 0:
                print('NATIVE', index+1, 'written', written, 'reused', reused, flush=True)
    finally:
        runtime.close()
    return {'written':written, 'reused':reused}



def produce(config, scope, split='all', benchmark_root=None, setting='long',
            datasets=('kitti', 'sintel', 'bonn'), nyu_root=None, device='cuda'):
    from explore_a.cache.locks import process_lock
    from explore_a.cache.reader import configured_root
    from explore_a.frontend import shared_root, name
    directory = shared_root(configured_root(config))/'frontends'/name(config)/'native'
    with process_lock(directory/'producer.lock', 'Native prediction producer'):
        return _produce(config, scope, split, benchmark_root, setting, datasets, nyu_root, device)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--scope', choices=('manifest', 'benchmark', 'nyu'), default='manifest')
    parser.add_argument('--split', choices=('all', 'train', 'dev', 'test'), default='all')
    parser.add_argument('--benchmark-root', type=Path,
                        default=Path.home()/'Documents/ResourcesLow/Datasets')
    parser.add_argument('--setting', choices=('short', 'long'), default='long')
    choices = ('kitti', 'sintel', 'bonn', 'tum-validation')
    parser.add_argument('--datasets', nargs='+', choices=choices,
                        default=('kitti', 'sintel', 'bonn'))
    parser.add_argument('--nyu-root', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    if args.scope == 'nyu' and args.nyu_root is None:
        parser.error('--scope nyu requires --nyu-root')
    print(produce(load_config(args.config), args.scope, args.split, args.benchmark_root,
                  args.setting, args.datasets, args.nyu_root, args.device))


if __name__ == '__main__':
    main()
