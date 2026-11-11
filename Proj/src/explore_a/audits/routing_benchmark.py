"""Paired throughput sweep with scratch heads, bounded inputs, no formal run/cache writes."""
import argparse
from collections import defaultdict
import copy
import gc
import json
from pathlib import Path
import random
from statistics import median
import time

import torch

from depth_project.io import write_json, read_json
from explore_a.audits.routing_samples import cohorts
from explore_a.config import load_config
from explore_a.data.prepare import load_manifest
from explore_a.data.reading import depth_range
from explore_a.evaluation.alignment import RelativeScores
from explore_a.evaluation.depth import MetricSummary, field_scores
from explore_a.online.factory import make_runtime
from explore_a.online.loading import OnlinePairs, read_pair
from explore_a.online.statistics import snapshot
from explore_a.training.checkpoint import seed_everything
from explore_a.training.correction_metrics import CorrectionMetrics
from explore_a.training.metrics import add_depth_scores
from explore_a.training.prediction import make_fusion, predict
from explore_a.training.two_frame import supervised

CANDIDATES = ((.2, .28), (0., 0.), (0., .28), (0., .5), (.2, 0.), (.4, .28))


def synchronize(device):
    if device == 'cuda':
        torch.cuda.synchronize()


def batch_indices(rows, size):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row['dataset']].append(i)
    return [group[i:i+size] for group in groups.values() for i in range(0,len(group),size)]


def exercise(model, optimizer, scaler, data, rows, config, phase, scorer, device, step):
    amp = device == 'cuda' and config['training']['amp']
    if phase == 'train':
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, enabled=amp):
            output, terms = supervised(model, data, config, 'rule', step+500)
        if not torch.isfinite(terms.total):
            raise FloatingPointError('Nonfinite scratch tuning loss')
        # Match production metric/diagnostic work without saving test accuracy or checkpoints.
        add_depth_scores(MetricSummary(), output, data, rows, config)
        CorrectionMetrics(config['training']['correction_fraction']).add(output, rows)
        scaler.scale(terms.total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
        scaler.step(optimizer)
        scaler.update()
    else:
        with torch.no_grad(), torch.autocast(device_type=device, enabled=amp):
            output, _ = predict(model, data, config, 'rule')
        CorrectionMetrics(config['training']['correction_fraction']).add(output, rows)
        for i, row in enumerate(rows):
            field_scores(output['depth'][i], data['gt'][i], data['gt_valid'][i],
                         config, row['dataset'])
            if scorer is not None:
                scorer.add(row, output['depth'][i], data['gt'][i], data['gt_valid'][i],
                           depth_range(row['dataset'], config['data']))


def trial(config, train, warmup, batches, eval_rows, candidate, phase, fusion, flow, device,
          frozen_only=False):
    config = copy.deepcopy(config)
    config['execution']['backend'] = 'hybrid'
    cache, remote = candidate
    config['execution']['routing'] = {
        p:{'cache':cache, 'remote':remote} for p in ('train','evaluation')}
    config['training']['microbatch'] = config['training']['effective_batch']
    seed_everything(config['training']['seed'])
    runtime = make_runtime(config, [flow], False, device)
    model = None if frozen_only else make_fusion(fusion, config).to(device)
    if model is not None:
        model.train(phase == 'train')
    optimizer = (torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'],
                                  weight_decay=config['training']['weight_decay'])
                 if model is not None else None)
    scaler = torch.amp.GradScaler('cuda', enabled=device == 'cuda' and config['training']['amp'])
    iterator, scorer = None, None
    try:
        # Both shapes, local libraries and this trial's persistent SSH warm up outside timing.
        warm = OnlinePairs(config, train, runtime, flow, phase=phase).iter_batches(warmup)
        try:
            for step, (indices, data) in enumerate(warm):
                if not frozen_only:
                    exercise(model, optimizer, scaler, data, [train[i] for i in indices],
                             config, phase, None, device, step)
        finally:
            warm.close()
        selected = train if phase == 'train' else eval_rows
        schedule = batches if phase == 'train' else batch_indices(
            selected, config['training']['effective_batch'])
        dataset = OnlinePairs(config, selected, runtime, flow, phase=phase)
        scorer = RelativeScores() if phase != 'train' and not frozen_only else None
        iterator = dataset.iter_batches(schedule)
        synchronize(device)
        before = snapshot(runtime)
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        began, durations, count, head_seconds = time.perf_counter(), [], 0, 0.
        for step in range(len(schedule)):
            start = time.perf_counter()
            indices, data = next(iterator)
            synchronize(device)
            head_start = time.perf_counter()
            if not frozen_only:
                exercise(model, optimizer, scaler, data, [selected[i] for i in indices],
                         config, phase, scorer, device, step)
            synchronize(device)
            head_seconds += time.perf_counter()-head_start
            synchronize(device)
            durations.append(time.perf_counter()-start)
            count += len(indices)
        if scorer is not None:
            score_start = time.perf_counter()
            for _ in scorer.finish():
                pass
            head_seconds += time.perf_counter()-score_start
        synchronize(device)
        wall = time.perf_counter()-began
        stats = {k:(v if k.endswith('_peak') else v-before.get(k,0))
                 for k,v in snapshot(runtime).items()}
        return {'phase':phase, 'cache':cache, 'remote':remote, 'pairs':count, 'wall_s':wall,
                'pairs_s':count/wall, 'step_s_median':median(durations), 'counts':stats,
                'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30 if device == 'cuda' else 0,
                'local':1-cache-remote, 'steps':len(durations),
                'head_s':head_seconds, 'frozen_only':frozen_only}
    finally:
        if iterator is not None:
            iterator.close()
        if scorer is not None:
            scorer.close()
        runtime.close()
        del optimizer, scaler, model, runtime
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()


def choose(records):
    grouped = defaultdict(list)
    for row in records:
        group = 'train' if row['phase'] == 'train' else 'evaluation'
        grouped[group, row['cache'], row['remote']].append(row)
    results = {}
    for group in ('train','evaluation'):
        candidates = []
        for (phase, cache, remote), rows in grouped.items():
            if phase != group:
                continue
            # Equal per-frame dev/test budgets; pooled time represents both evaluation workloads.
            rounds = defaultdict(list)
            for row in rows:
                rounds[row['round']].append(row)
            rates = [sum(r['pairs'] for r in rs)/sum(r['wall_s'] for r in rs)
                     for rs in rounds.values()]
            conservative = group == 'train' and not all(r.get('frozen_only') for r in rows)
            score = min(rates) if conservative else median(rates)
            candidates.append({'cache':cache, 'remote':remote, 'pairs_s':score,
                               'criterion':'slower round' if conservative else 'median',
                               'round_pairs_s':rates})
        if candidates:
            candidates.sort(key=lambda r:r['pairs_s'], reverse=True)
            # Within 3% of the best, prefer the existing allocation to avoid noise-driven churn.
            baseline = next((c for c in candidates if (c['cache'],c['remote']) == (.2,.28)), None)
            selected = candidates[0]
            if baseline and baseline['pairs_s'] >= .97*selected['pairs_s']:
                selected = baseline
            results[group] = {'selected':selected, 'candidates':candidates}
    return results


def run(config, output, steps=16, rounds=2, fusion='final_flow', flow='farneback', device='cuda',
        candidates=CANDIDATES, phases=('train','dev','test'), balanced=False,
        frozen_only=False, available_only=False, sample_seed=None):
    output = Path(output)
    sample_config = copy.deepcopy(config)
    if sample_seed is not None:
        sample_config['training']['seed'] = sample_seed
    train, warmup, batches, eval_rows = cohorts(
        sample_config, load_manifest(config), steps, balanced)
    availability = {}
    if available_only:
        from explore_a.online.cache_components import CacheComponents
        cache_reader = CacheComponents(config)
        ready = {}
        def present(row):
            if row['id'] not in ready:
                raw = read_pair(row, config['data'])
                values = [cache_reader.discover(raw, 'prediction', old)
                          for old in (True, False)]
                values.append(cache_reader.discover(raw, 'flow', flow=flow))
                ready[row['id']] = all(values)
            return ready[row['id']]
        proposed = sum(map(len, batches))
        batches = [[i for i in batch if present(train[i])] for batch in batches]
        batches = [batch for batch in batches if batch]
        availability['train'] = sum(map(len, batches))/proposed
        for phase, rows in eval_rows.items():
            kept = [r for r in rows if present(r)]
            availability[phase] = len(kept)/len(rows)
            eval_rows[phase] = kept
        if not batches or any(not rows for rows in eval_rows.values()):
            raise ValueError('No fully cached performance cohort')
    identity = {'steps':steps, 'rounds':rounds, 'fusion':fusion, 'flow':flow,
                'seed':config['training']['seed'],
                'config':config, 'candidates':candidates, 'phases':phases,
                'train_ids':[[train[i]['id'] for i in batch] for batch in batches],
                'evaluation_ids':{s:[r['id'] for r in rows] for s,rows in eval_rows.items()},
                'note':'scratch training, no formal checkpoint/log/cache writes; test accuracy '
                       'discarded; startup/warmup and formal visualization/checkpoint I/O excluded'}
    if sample_seed is not None:
        identity['sample_seed'] = sample_seed
    if balanced:
        identity['balanced_sources'] = True
    if frozen_only:
        identity['frozen_only'] = True
    if available_only:
        identity['cache_availability'] = availability
    path = output/'protocol.json'
    # JSON canonicalization normalizes tuples before resume comparison.
    identity = json.loads(json.dumps(identity))
    if path.exists() and read_json(path) != identity:
        raise ValueError('Tuning output belongs to another protocol')
    write_json(path, identity)
    records = []
    order = list(range(len(candidates)))
    random.Random(2026).shuffle(order)
    for repetition in range(rounds):
        for index in (order if repetition % 2 == 0 else order[::-1]):
            for phase in phases:
                receipt = output/'trials'/f'{repetition}-{index}-{phase}.json'
                if receipt.exists():
                    result = read_json(receipt)
                else:
                    print('TUNING', repetition, candidates[index], phase, flush=True)
                    result = trial(config, train, warmup, batches, eval_rows.get(phase),
                                   candidates[index], phase, fusion, flow, device, frozen_only)
                    result.update(round=repetition)
                    write_json(receipt, result)
                    print('RESULT', result, flush=True)
                records.append(result)
                write_json(output/'progress.json', {'completed_trials':len(records),
                           'total_trials':rounds*len(candidates)*len(phases)})
    result = {'protocol':identity, 'records':records, 'selection':choose(records)}
    write_json(output/'summary.json', result)
    print('SELECTED', result['selection'], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=16)
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--fusion', choices=('unet_guided','flow_refine','final_flow'),
                        default='final_flow')
    parser.add_argument('--flow', choices=('farneback','raft'), default='farneback')
    parser.add_argument('--sample-seed', type=int)
    parser.add_argument('--frozen-only', action='store_true')
    parser.add_argument('--cache-available-only', action='store_true')
    parser.add_argument('--balanced-sources', action='store_true')
    parser.add_argument('--phases', nargs='+', choices=('train','dev','test'),
                        default=('train','dev','test'))
    parser.add_argument('--candidates', nargs='+', help='absolute cache:remote probabilities')
    args = parser.parse_args()
    if args.steps < 3 or args.rounds < 1:
        parser.error('steps must be >=3 and rounds >=1')
    candidates = CANDIDATES
    if args.candidates:
        try:
            candidates = tuple(tuple(map(float, value.split(':'))) for value in args.candidates)
            if any(len(v) != 2 or min(v) < 0 or sum(v) > 1 for v in candidates):
                raise ValueError()
        except ValueError:
            parser.error('candidates must be cache:remote, nonnegative, sum <=1')
    run(load_config(args.config), args.output, args.steps, args.rounds, args.fusion, args.flow,
        candidates=candidates, phases=tuple(args.phases), balanced=args.balanced_sources,
        frozen_only=args.frozen_only, available_only=args.cache_available_only,
        sample_seed=args.sample_seed)


if __name__ == '__main__':
    main()
