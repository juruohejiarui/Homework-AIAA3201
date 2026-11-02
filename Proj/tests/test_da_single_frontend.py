"""DA-only cleanup safety, affine units, model boundaries and six training commands."""
from dataclasses import replace
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from depth_project.io import read_json, write_json
from explore_a.config import FUSIONS, load_config
from explore_a.frontend import name
from explore_a.training.prediction import make_fusion, predict
from test_explore_a_relative import relative_pair


def test_retired_frontend_and_model_are_rejected():
    assert FUSIONS == ('unet_guided', 'flow_refine', 'final_flow')
    with pytest.raises(ValueError, match='frontend'):
        name({'frontend':{'name':'depthpro'}})
    with pytest.raises(ValueError, match='Unknown fusion'):
        make_fusion('unet', load_config())
    from depth_project.models import load_model
    with pytest.raises(ValueError, match='Unknown model'):
        load_model('depthpro', '/unused')


@pytest.mark.parametrize('model_name', FUSIONS)
def test_inverse_affine_units_and_gt_isolation(relative_pair, model_name):
    config, data = relative_pair
    model = make_fusion(model_name, config).eval()
    if model_name == 'flow_refine':
        with torch.no_grad():
            model.local_head.bias.fill_(.2)
    first = predict(model, data, config, 'rule')[0]
    other = {**data, 'current_depth':3*data['current_depth']+7,
             'previous_depth':3*data['previous_depth']+7, 'gt':data['gt']*13}
    transformed = predict(model, other, config, 'rule')[0]
    torch.testing.assert_close(transformed['depth'], 3*first['depth']+7, atol=2e-4, rtol=2e-5)
    torch.testing.assert_close(first['depth'], predict(model, {**data, 'gt':data['gt']*13},
                                                      config, 'rule')[0]['depth'])
    assert first['prediction_kind'] == 'relative_inverse'
    assert (first['depth'] >= 0).all()
    if model_name != 'unet_guided':
        assert ((first['depth']-first['mixed_depth']).abs()
                <= .10001*first['field_scale']).all()


def test_shared_cache_cleanup_preserves_numeric_artifacts(tmp_path):
    source = Path(__file__).parents[1]/'maintenance/remove_depthpro.py'
    spec = importlib.util.spec_from_file_location('retirement', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    names = ('flow/raft/one.npz', 'teacher/raft/one.npz',
             'frontends/da2/frames/one.npz', 'frames/one.npz')
    for name in names:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'unchanged component')
    kept = {name:(tmp_path/name).stat().st_mtime_ns for name in names[:-1]}
    write_json(tmp_path/'jobs/one/job.json', {'outputs':list(names[:2])+[names[-1]]})
    write_json(tmp_path/'jobs/one/local.complete.json',
               {'sizes':{name:19 for name in names[:2]+names[-1:]}, 'bytes':57})
    module.clean_cache(tmp_path, True)
    assert not (tmp_path/'frames').exists()
    for name, stamp in kept.items():
        assert (tmp_path/name).read_bytes() == b'unchanged component'
        assert (tmp_path/name).stat().st_mtime_ns == stamp
    assert read_json(tmp_path/'jobs/one/local.complete.json')['bytes'] == 38
    assert read_json(tmp_path/'jobs/one/job.json')['outputs'] == list(names[:2])


def test_separate_tum_root_and_all_six_training_tasks(tmp_path, monkeypatch):
    from explore_a.evaluation.part2 import benchmark_root_for
    assert benchmark_root_for('tum-validation', tmp_path/'Low', tmp_path/'TUM-RGBD') == tmp_path
    assert benchmark_root_for('kitti', tmp_path/'Low', tmp_path/'TUM-RGBD') == tmp_path/'Low'
    from explore_a import matrix
    calls = []
    monkeypatch.setattr(matrix, 'command', lambda cfg, *args:calls.append(args))
    matrix.run_matrix(load_config())
    assert len(calls) == 6
    assert {(args[2], args[6]) for args in calls} == {
        (flow, model) for flow in ('raft','farneback') for model in FUSIONS}


@pytest.mark.parametrize('flow', ['farneback', 'raft'])
@pytest.mark.parametrize('model_name,filename', [
    ('unet_guided','mixed_h5_da2.yaml'), ('flow_refine','mixed_h5_da2_refine.yaml'),
    ('final_flow','mixed_h5_da2_final.yaml')])
def test_training_command_parameters(monkeypatch, flow, model_name, filename):
    from explore_a.__main__ import main
    calls = []
    monkeypatch.setattr('explore_a.training.engine.run_training', lambda *args:calls.append(args))
    cfg = Path(__file__).parents[1]/'configs/explore_a'/filename
    monkeypatch.setattr('sys.argv', ['explore_a','--config',str(cfg),'train',
                                    '--flow',flow,'--occlusion','rule','--fusion',model_name])
    main()
    assert calls[0][1:] == (flow, 'rule', model_name)
    selected = calls[0][0]
    assert selected['training']['source_weights'] == {'kitti':3, 'nyuv2':3, 'tum':2}
    assert selected['cache']['recipe'] == 'glob-da2'


def test_tum_commands_keep_new_target_and_do_not_download(tmp_path, monkeypatch):
    from explore_a.download import tum_validation
    calls = []
    manifest = Path(__file__).parents[1]/'storage/tum/validation_plan.json'
    frozen = tum_validation.validation_plan(manifest)
    assert len(frozen['files']) == 33
    assert frozen['output'] == str(tum_validation.DEFAULT_OUTPUT)
    for command in ('download', 'extract'):
        monkeypatch.setattr(tum_validation.tum, command, lambda *args:calls.append(args))
        monkeypatch.setattr('sys.argv',['download-tum', command,'--manifest',str(manifest)])
        tum_validation.main()
    assert len(calls) == 2
    monkeypatch.setattr(tum_validation.tum,'plan',lambda *args,**kwargs:calls.append(args))
    monkeypatch.setattr('sys.argv',['download-tum','plan','--manifest',str(tmp_path/'plan.json'),
                                   '--output',str(tum_validation.DEFAULT_OUTPUT)])
    tum_validation.main()
    assert calls[-1][1] == tum_validation.DEFAULT_OUTPUT
