"""Bounded video decoding; explicit source and RGB-transform identities shared with producers."""
from pathlib import Path

from explore_a.execution import prefetch, Timings


def video_raw(sequence, dataset, setting, crop, config, runtime, *, native=True, submit=True,
              flow="farneback", teacher=False):
    from explore_a.evaluation.part2 import input_rgb, raw_input
    frames = sequence['frames']

    def load(index):
        now = frames[index]
        before_index = index if index == 0 or now.get('reset_history') else index-1
        before = frames[before_index]
        current = input_rgb(now, dataset, setting, crop)
        previous = current if before_index == index else input_rgb(before, dataset, setting, crop)
        raw = raw_input(previous, current, dataset, sequence['name'],
                        str(before['source_index']), str(now['source_index']), config)
        raw['row'].update(current_rgb=str(Path(now['image']).resolve()),
                          previous_rgb=str(Path(before['image']).resolve()),
                          input_transform={'scope':'part2', 'dataset':dataset, 'crop':list(crop),
                                           'rgb':'official-crop-even-v1'})
        raw['frozen_ticket'] = runtime.prepare(raw['row'], index,
            {} if before_index == index else {flow:teacher}, phase='benchmark', native=native)
        if submit:
            runtime.submit_decoded(raw)
        return index, raw, before_index != index

    options = config['online']
    with runtime.input_phase('benchmark'):
        yield from prefetch(range(len(frames)), load, options['read_workers'], options['prefetch'],
                            Timings('video RGB input'))
