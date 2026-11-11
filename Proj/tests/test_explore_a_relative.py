"""Relative-field semantics, conditional bounds and loss-only gauges; CPU synthetic data."""
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from explore_a.config import load_config
from explore_a.geometry.relative import affine, statistics
from explore_a.models.inputs import FusionInputs
from explore_a.training.prediction import fusion_geometry, make_fusion, predict
from explore_a.training.relative import gauge, main_loss
from explore_a.training.two_frame import supervised, reverse_pair
from explore_a.evaluation.relative_oracle import candidates
from test_explore_a_geometry import example
from test_explore_a_distributed import prepared as prepared
from test_explore_a_pipeline import fake_backbones as fake_backbones


@pytest.fixture
def relative_pair():
    torch.set_num_threads(1)
    data = example(32, 48)
    value = torch.linspace(.1, 5, 32*48).reshape(1, 1, 32, 48)
    data.update(current_depth=value, previous_depth=(value-.05)/2,
                gt=1/(.15*value+.2), previous_gt=1/(.1*value+.2),
                gt_valid=torch.ones_like(value, dtype=torch.bool),
                previous_gt_valid=torch.ones_like(value, dtype=torch.bool),
                previous_image_valid=data['image_valid'].clone())
    config = load_config('Proj/configs/explore_a/mixed_h5_da2.yaml')
    return config, data


def test_positive_affine_fit_and_degenerate_rejection(relative_pair):
    _, data = relative_pair
    x, y, mask = data['previous_depth'], data['current_depth'], data['image_valid']
    a, b, good = affine(x, y, mask)
    torch.testing.assert_close(a, torch.full_like(a, 2.))
    torch.testing.assert_close(b, torch.full_like(b, .05), atol=1e-6, rtol=0)
    assert good.all()
    assert not affine(x, -y, mask)[2].any()
    assert not affine(x*0, y, mask)[2].any()
    mask = mask.clone()
    mask.flatten()[63:] = False
    assert not affine(x, y, mask)[2].any()
    center, scale = statistics(y, data['image_valid'])
    other_center, other_scale = statistics(y*3+7, data['image_valid'])
    torch.testing.assert_close(other_center, center*3+7)
    torch.testing.assert_close(other_scale, scale*3)


@pytest.mark.parametrize('architecture', ['unet_guided', 'flow_refine', 'final_flow'])
def test_forward_loss_backward_and_exact_fallback(relative_pair, architecture):
    config, data = relative_pair
    model = make_fusion(architecture, config)
    output, terms = supervised(model, data, config, 'rule', 500)
    assert output['prediction_kind'] == 'relative_inverse'
    assert torch.isfinite(terms.total)
    terms.total.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    geometry = fusion_geometry(data, config)
    inputs = FusionInputs.from_pair(data, geometry, torch.ones_like(data['gt']))
    absent = replace(inputs, availability=inputs.availability*0,
                     current_depth=inputs.current_depth*0)
    result = model(absent)
    torch.testing.assert_close(result['depth'], absent.current_depth, atol=0, rtol=0)
    if architecture == 'unet_guided':
        low = torch.minimum(inputs.current_depth, inputs.historical)
        high = torch.maximum(inputs.current_depth, inputs.historical)
        assert (output['depth'] >= low-1e-6).all() and (output['depth'] <= high+1e-6).all()
    else:
        with torch.no_grad():
            local = model.local_head if architecture == 'flow_refine' else model.spatial_local
            local.bias.fill_(8)
        changed = model(inputs)
        assert (changed['depth']-changed['mixed_depth']).abs().max() <= inputs.scale.max()*.100001
        for mode in (() if architecture == 'flow_refine' else
                     ('no-temporal', 'global-only', 'fusion-only', 'shuffled-history')):
            assert torch.isfinite(model.ablate(changed, inputs, mode)['depth']).all()
    assert set(reverse_pair(reverse_pair(data))) == set(reverse_pair(data))


def test_fixed_gauge_oracle_nesting_and_no_gt_in_forward(relative_pair):
    config, data = relative_pair
    fixed = gauge(data)
    assert main_loss(fixed['target'], fixed) < 1e-10
    geometry = fusion_geometry(data, config)
    branches = candidates(data, geometry, fixed)
    assert len(branches) == 8
    values = {k: main_loss(v, fixed) for k, v in branches.items()}
    assert values['oracle_additive10'] <= values['oracle_aligned']+1e-7
    assert values['oracle_aligned'] <= values['raw']+1e-7
    model = make_fusion('final_flow', config).eval()
    out = predict(model, data, config, 'rule')[0]
    changed = predict(model, {**data, 'gt': data['gt']*13}, config, 'rule')[0]
    torch.testing.assert_close(out['depth'], changed['depth'], atol=0, rtol=0)
    empty = {**data, 'gt_valid': data['gt_valid'] & False, 'gt': data['gt']*float('nan'),
             'previous_gt_valid': data['previous_gt_valid'] & False,
             'previous_gt': data['previous_gt']*float('nan')}
    assert supervised(model, empty, config, 'rule', 500)[1].total == 0


def test_namespace_workers_and_checkpoint_task(tmp_path):
    from explore_a.cache.settings import root_path, ready_files
    from explore_a.cache.paths import artifact
    from explore_a.training.identity import assert_same_task
    root = root_path(tmp_path, 'glob-da2')
    row = {'dataset':'tum', 'sequence':'seq', 'camera':'rgb',
           'current_id':'2', 'previous_id':'1'}
    assert artifact(root, 'frames', row).is_relative_to(root)
    assert artifact(root, 'flow/farneback', row).is_relative_to(root_path(tmp_path))
    flag = root/'jobs/seq/local.ready.json'
    flag.parent.mkdir(parents=True)
    flag.touch()
    assert ready_files(tmp_path, flag.name) == [flag]
    with pytest.raises(ValueError, match='frontend'):
        assert_same_task({'experiment':{}}, {'experiment':{}, 'frontend':{'name':'da2'}})




def test_da_engine_online_training_resume_and_frozen_gradients(tmp_path, monkeypatch,
                                                               fake_backbones):
    import numpy as np
    from pathlib import Path
    from explore_a.training.engine import run_training
    from test_explore_a_pipeline import online_experiment
    from depth_project.io import read_json
    config = online_experiment(tmp_path)
    config['frontend'] = {'name':'da2'}
    config['frontend'] = {'name':'da2'}
    config['training'].update(steps=1, temporal_warmup_steps=1)
    class Frozen:
        model = torch.nn.Conv2d(1,1,1).requires_grad_(False)
        def predict(self, rgb):
            return np.linspace(.1, 2, rgb.shape[0]*rgb.shape[1], dtype=np.float32).reshape(
                rgb.shape[:2])
    frozen = Frozen()
    monkeypatch.setattr('explore_a.frontend.make', lambda config: frozen)
    run_training(config, 'farneback', fusion='final_flow', device='cpu')
    config['training']['steps'] = 2
    run_training(config, 'farneback', fusion='final_flow', device='cpu')
    root = Path(config['paths']['runs'])/'farneback-rule-final_flow'
    assert read_json(root/'complete.json')['selection'] == 'SSI AbsRel'
    assert torch.load(root/'last.pt', weights_only=False)['step'] == 2
    assert all(p.grad is None for p in frozen.model.parameters())
    assert not list(tmp_path.rglob('*.npz'))


@pytest.mark.parametrize('flow', ['farneback', 'raft'])
def test_complete_dev_oracle_resume_without_frozen_recompute(tmp_path, monkeypatch,
                                                            fake_backbones, flow):
    import numpy as np
    from test_explore_a_pipeline import online_experiment
    from explore_a.evaluation.relative_oracle import run
    config = online_experiment(tmp_path)
    config['frontend'] = {'name':'da2'}
    config['training']['dev_sample_fraction'] = .1
    class Frozen:
        model = torch.nn.Identity()
        calls = 0
        def predict(self, rgb):
            self.calls += 1
            return np.linspace(.1, 2, rgb.shape[0]*rgb.shape[1], dtype=np.float32).reshape(
                rgb.shape[:2])
    frozen = Frozen()
    monkeypatch.setattr('explore_a.frontend.make', lambda config: frozen)
    from explore_a.online.factory import make_runtime
    requested = []
    def runtime(config, flows, teacher, device):
        requested.append(flows)
        return make_runtime(config, flows, teacher, device)
    monkeypatch.setattr('explore_a.evaluation.relative_oracle.make_runtime', runtime)
    output = tmp_path/'oracle'
    first = run(config, output, 'cpu', flow=flow)
    calls = frozen.calls
    second = run(config, output, 'cpu', flow=flow)
    assert requested == [[flow]]
    assert first['protocol']['flow'] == flow
    assert first['branches'] == second['branches']
    assert frozen.calls == calls
    assert len(first['protocol']['sample_ids']) == 1
    assert len(first['branches']) == 32
    assert not list(tmp_path.rglob('*.npz'))
    with pytest.raises(ValueError, match='another input/score protocol'):
        run(config, output, 'cpu', flow='raft' if flow == 'farneback' else 'farneback')
    if flow == 'raft':
        Path(config['weights']['raft']).write_bytes(b'changed RAFT weights')
        with pytest.raises(ValueError, match='another input/score protocol'):
            run(config, output, 'cpu', flow=flow)
    assert requested == [[flow]]
    assert frozen.calls == calls


def test_shared_affine_change_preserves_normalized_model_inputs(relative_pair):
    config, data = relative_pair
    original = fusion_geometry(data, config)
    other = {**data, 'current_depth':3*data['current_depth']+7,
             'previous_depth':3*data['previous_depth']+7}
    geometry = fusion_geometry(other, config)
    first = FusionInputs.from_pair(data, original, torch.ones_like(data['gt']))
    second = FusionInputs.from_pair(other, geometry, torch.ones_like(data['gt']))
    torch.testing.assert_close(first.features(), second.features(), atol=1e-5, rtol=1e-4)
    model = make_fusion('final_flow', config).eval()
    a, b = model(first), model(second)
    torch.testing.assert_close(b['depth'], 3*a['depth']+7, atol=1e-4, rtol=1e-4)


def test_short_fixed_sample_learning_checks_main_loss(relative_pair):
    config, data = relative_pair
    data['gt'] = data['gt']*(1+.04*torch.sin(data['current_depth']*3))
    torch.manual_seed(2026)
    model = make_fusion('final_flow', config)
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    losses = []
    for step in range(12):
        optimizer.zero_grad()
        _, terms = supervised(model, data, config, 'rule', 500)
        losses.append(float(terms.depth.detach()))
        terms.total.backward()
        optimizer.step()
    assert min(losses[-4:]) < losses[0], losses


def test_video_score_reuses_whole_sequence_baseline_gauge(tmp_path):
    import numpy as np
    from explore_a.evaluation.part2_protocol import baseline_sequence_scores
    frames, raw = [], []
    for i in range(2):
        gt = np.linspace(1, 4, 48, dtype=np.float32).reshape(6,8)+i*.1
        path = tmp_path/f'gt{i}.npy'
        np.save(path, gt)
        frames.append({'gt':str(path)})
        raw.append(2/gt+.1)
    config = {'crop':[0,6,0,8], 'max_depth':10, 'factor':1}
    values = baseline_sequence_scores({'raw_canvas':raw, 'changed':[v+.2 for v in raw]},
                                       frames, config)
    assert values['gauge']['scale'] == pytest.approx(.5, rel=1e-5)
    assert values['branches']['raw_canvas']['abs_rel'] < 1e-5
    assert values['branches']['changed']['abs_rel'] > .01


def test_da_producer_outputs_only_native_inverse_and_component_identities(prepared, monkeypatch,
                                                                         fake_backbones):
    from pathlib import Path
    from depth_project.io import read_json
    from explore_a.cache.plan import prepare_plan
    from explore_a.cache.producer import Producer
    from explore_a.cache.storage import read_npz
    from explore_a.cache.reader import cache_spec, check_component
    config, base, shared, name = prepared
    config['frontend'] = {'name':'da2'}
    config['frontend'] = {'name':'da2'}
    config['cache'] = {'root':str(base), 'recipe':'glob-da2'}
    own = prepare_plan(config, 'glob-da2', base)
    job = read_json(own/'jobs'/name/'job.json')
    depth_class = fake_backbones[0]
    monkeypatch.setattr('explore_a.frontend.make', lambda config: depth_class())
    producer = Producer(config, 'cpu')
    assert not producer.runtime.frozen.models
    producer.generate(own, job, sources=read_json(own/'jobs'/name/'sources.json'),
                      minimum_gib=0, host='local')
    assert depth_class.calls == 3
    for path in own.rglob('*.npz'):
        assert set(read_npz(path)) == {'inverse_depth'}
    assert all('frames/' in value for value in job['outputs'])
    # RAFT changes cannot invalidate unrelated DA predictions or Farneback fields.
    changed = Path(base)/'different_raft.pth'
    changed.write_bytes(b'different')
    config['weights']['raft'] = str(changed)
    spec = cache_spec(config)
    check_component(spec, 'depth')
    check_component(spec, 'flow_farneback')
    with pytest.raises(ValueError, match='identity'):
        check_component(spec, 'flow_raft')
