"""Fixed performance cohorts, unrelated to model selection or test accuracy tuning."""
from collections import defaultdict
from hashlib import sha256
import math

from explore_a.training.dev_subset import select_dev, quotas, clips
from explore_a.training.sampling import StepBatches


def evaluation_rows(rows, split, pairs, batch, seed, dev_seed=2026):
    selected = [r for r in rows if r['split'] == split]
    if split == 'dev':
        selected = select_dev(selected, .1, dev_seed)
    groups = defaultdict(lambda: defaultdict(list))
    for row in selected:
        groups[row['dataset']][row['sequence']].append(row)
    sizes = {s: sum(map(len, g.values())) for s, g in groups.items()}
    weights = {'kitti':3, 'nyuv2':3, 'tum':2} if split == 'dev' else sizes
    units = quotas({s: max(1, n//batch) for s, n in sizes.items()}, pairs//batch,
                   {s: weights[s] for s in sizes}, 1)
    output = []
    for source, videos in sorted(groups.items()):
        budget = units[source]*batch
        # Several complete runs, not isolated pairs, preserve evaluation endpoint reuse.
        ordered = sorted(videos, key=lambda k: sha256(f'{seed}:{source}:{k}'.encode()).hexdigest())
        picked, count = {}, 0
        for name in ordered:
            if len(videos[name]) < min(8, budget):
                continue
            picked[name] = videos[name]
            count += len(videos[name])
            if len(picked) >= min(3, max(1, budget//8)) and count >= budget:
                break
        if count < budget:
            picked = videos
        capacities = {k:len(v) for k, v in picked.items()}
        allocation = quotas(capacities, budget, {k:math.sqrt(v) for k,v in capacities.items()})
        for name, group in sorted(picked.items()):
            output.extend(clips(group, allocation[name], seed))
    return output


def cohorts(config, manifest, steps, balanced=False):
    rows = manifest['samples']
    batch, seed = config['training']['effective_batch'], config['training']['seed']
    train = [r for r in rows if r['split'] == 'train']
    by_source = defaultdict(list)
    for i, row in enumerate(train):
        by_source[row['dataset']].append(i)
    warmup = [indices[:batch] for source, indices in sorted(by_source.items())]
    selected = list(StepBatches(train, batch, batch, steps, 0, seed,
                               config['training'].get('source_weights'),
                               config['training'].get('sampling_mode', 'fix')))
    if balanced:
        capacities = {s:steps for s in by_source}
        target = quotas(capacities, steps, config['training']['source_weights'], 1)
        selected, counts = [], defaultdict(int)
        stream = StepBatches(train, batch, batch, steps*8, 0, seed,
                             config['training']['source_weights'],
                             config['training'].get('sampling_mode', 'fix'))
        for indices in stream:
            source = train[indices[0]]['dataset']
            if counts[source] < target[source]:
                selected.append(indices)
                counts[source] += 1
            if len(selected) == steps:
                break
        if len(selected) != steps:
            raise ValueError('Insufficient balanced performance batches')
    return train, warmup, selected, {
        split:evaluation_rows(rows, split, steps*batch, batch, seed, config['data']['seed'])
        for split in ('dev', 'test')}
