"""Three representative training pairs, real frozen inference and disposable head backward."""
import argparse
import copy
from pathlib import Path

import torch

from depth_project.io import write_json
from explore_a.config import load_config
from explore_a.data.prepare import load_manifest
from explore_a.frontend import metadata
from explore_a.online.factory import make_runtime
from explore_a.online.loading import read_pair
from explore_a.training.prediction import make_fusion
from explore_a.training.two_frame import supervised


def run(config, output, route="local", flow="farneback"):
    if not torch.cuda.is_available():
        raise RuntimeError('This limited integration check requires CUDA')
    config = copy.deepcopy(config)
    config['execution'] = {'backend':'online' if route == 'local' else 'hybrid'}
    if route == 'remote':
        config['cache_read'] = {'mode':'blocking', 'probability':0., 'max_inflight':4}
        config['remote_online'].update(enabled=True, probability=1.)
    rows = load_manifest(config)['samples']
    runtime = make_runtime(config, [flow], False, 'cuda')
    records = []
    try:
        for source in config['data']['datasets']:
            row = next(r for r in rows if r['dataset'] == source and r['split'] == 'train')
            raw = read_pair(row, config['data'])
            values = runtime.compute(raw, flow)
            data = {k:v.unsqueeze(0) for k,v in values.items()}
            record = {'id':row['id'], 'source':source, 'heads':{}}
            for architecture in ('unet_guided', 'flow_refine', 'final_flow'):
                model = make_fusion(architecture, config).cuda().train()
                with torch.autocast('cuda', dtype=torch.float16):
                    predicted, terms = supervised(model, data, config, 'rule', 500)
                if not torch.isfinite(terms.total):
                    raise AssertionError('Nonfinite loss')
                terms.total.backward()
                if not all(p.grad is None or torch.isfinite(p.grad).all()
                           for p in model.parameters()):
                    raise AssertionError('Nonfinite head gradient')
                record['heads'][architecture] = {'loss':float(terms.total.detach()),
                    'terms':terms.scalars(), 'kind':predicted['prediction_kind']}
                del model, predicted, terms
            for frozen in runtime.local.frozen.models.values():
                if getattr(frozen, "model", None) is not None:
                    assert all(p.grad is None and not p.requires_grad
                               for p in frozen.model.parameters())
            records.append(record)
            print(source, 'real frozen inference + three head backwards OK', flush=True)
    finally:
        statistics = runtime.statistics()
        runtime.close()
    write_json(output, {'frontend':metadata(config), 'route':route, 'flow':flow,
                        'input_stats':statistics, 'pairs':records,
                        'notice':'No optimizer, checkpoint, prediction NPZ or test evaluation'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--route', choices=('local', 'remote'), default='local')
    parser.add_argument('--flow', choices=('farneback', 'raft'), default='farneback')
    args = parser.parse_args()
    run(load_config(args.config), args.output, args.route, args.flow)


if __name__ == '__main__':
    main()
