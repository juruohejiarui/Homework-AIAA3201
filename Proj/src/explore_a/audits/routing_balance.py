"""Solve the measured three-stage capacity model, including cache availability."""
import argparse
import math
from pathlib import Path
from statistics import median

from depth_project.io import read_json, write_json


def solve(local, cache, remote, overhead, availability=1.):
    """Find minimal t with host/head serialization and concurrent cache/RPC stages.

    Costs are seconds per consumed pair at the path's existing concurrency, not summed
    individual RPC latencies. Availability caps actual cache use; attempted cache routing
    is converted back after optimization. A 5% non-cache margin avoids a degenerate p=1.
    """
    if any(not math.isfinite(v) or v <= 0 for v in (local, cache, remote)):
        raise ValueError('Path costs must be positive finite seconds per pair')
    if not math.isfinite(overhead) or overhead < 0 or not 0 <= availability <= 1:
        raise ValueError('Invalid head cost or cache availability')
    cap = .95*availability

    def capacity(t):
        return max(0., (t-overhead)/local), min(cap, t/cache), t/remote

    lo, hi = overhead, max(overhead+local, cache, remote)
    for _ in range(80):
        middle = (lo+hi)/2
        if sum(capacity(middle)) >= 1:
            hi = middle
        else:
            lo = middle
    h, c, r = capacity(hi)
    total = h+c+r
    h, c, r = h/total, c/total, r/total
    attempted = c/availability if availability else 0.
    # Missing cached components are preallocated in the remote:local ratio.
    direct_remote = r*(1-attempted)/(1-c) if c < 1 else 0.
    profile = {'cache':round(attempted, 2), 'remote':round(direct_remote, 2)}
    profile['remote'] = min(profile['remote'], 1-profile['cache'])
    return {'predicted_pair_s':hi, 'predicted_pairs_s':1/hi,
            'actual_fractions':{'local':h, 'cache':c, 'remote':r}, 'profile':profile,
            'costs':{'local':local, 'cache':cache, 'remote':remote, 'overhead':overhead,
                     'cache_availability':availability}}


def derive(coarse, paths):
    result = {}
    for group, phases in (('train',('train',)), ('evaluation',('dev','test'))):
        costs = []
        for phase in phases:
            full_host = [r for r in coarse['records'] if r['phase'] == phase
                         and r['cache'] == 0 and r['remote'] == 0]
            overhead = median(max(0., r['wall_s']-r['counts']['local_compute_s'])/r['pairs']
                              for r in full_host)
            isolated = [r for r in paths['records'] if r['phase'] == phase]
            host = [r['counts']['local_compute_s']/r['pairs'] for r in isolated
                    if r['cache'] == 0 and r['remote'] == 0]
            cache = [r['wall_s']/r['pairs'] for r in isolated if r['cache'] == 1]
            remote = [r['wall_s']/r['pairs'] for r in isolated if r['remote'] == 1]
            # Random training sees many cold files. Use the slower cache round conservatively.
            c = max(cache) if group == 'train' else median(cache)
            costs.append({'local':median(host), 'cache':c, 'remote':median(remote),
                          'overhead':overhead, 'availability':paths['protocol'][
                              'cache_availability'][phase]})
        pooled = {k:sum(c[k] for c in costs)/len(costs) for k in costs[0]}
        result[group] = {**solve(**pooled), 'phase_costs':dict(zip(phases,costs))}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coarse', type=Path, required=True)
    parser.add_argument('--paths', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = derive(read_json(args.coarse), read_json(args.paths))
    write_json(args.output, result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
