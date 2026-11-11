"""Paper frame selection and full TUM coverage; no model inference or benchmark run."""
from types import SimpleNamespace

import numpy as np
import pytest

from depth_project.data.protocol import select_sequences
from depth_project.evaluation.official import load_evaluator
from explore_a.evaluation import comparison
from explore_a.evaluation.part2_protocol import official_score


def manifest(dataset, length):
    return {'dataset': dataset, 'sequences': [
        {'name': 'sequence', 'frames': [{'source_index': i} for i in range(length)]}]}


def test_main_long_and_appendix_short_are_different_cohorts():
    for dataset in ['kitti', 'bonn']:
        long = select_sequences(manifest(dataset, 700), 'long')[0]['frames']
        short = select_sequences(manifest(dataset, 700), 'short')[0]['frames']
        assert len(long) == 500 and len(short) == 110
        assert short[0]['source_index'] == (30 if dataset == 'bonn' else 0)
    assert len(select_sequences(manifest('sintel', 50), 'long')[0]['frames']) == 50


def test_full_tum_keeps_frames_beyond_110_and_500():
    selected = select_sequences(manifest('tum-validation', 731), 'full')
    assert len(selected[0]['frames']) == 731
    for setting in ['short', 'long']:
        with pytest.raises(ValueError, match='use full'):
            select_sequences(manifest('tum-validation', 731), setting)


def test_scorer_does_not_silently_drop_the_501st_frame(tmp_path):
    depth = np.linspace(1, 4, 6).reshape(2, 3).astype(np.float32)
    path = tmp_path / 'gt.npy'
    np.save(path, depth)
    frames = [{'gt': str(path)}] * 501
    predictions = [1 / depth] * 500 + [1 / depth + .2]
    result = official_score(predictions, frames,
                            {'crop': [0, 2, 0, 3], 'max_depth': 10, 'factor': 1})
    evaluator = load_evaluator(device='cpu')
    paths = []
    for name, value in [('normal', predictions[0]), ('last', predictions[-1])]:
        np.save(tmp_path / (name + '.npy'), value)
    paths = [str(tmp_path / 'normal.npy')] * 500 + [str(tmp_path / 'last.npy')]
    args = SimpleNamespace(a=0, b=2, c=0, d=3, max_eval_len=501, max_depth_eval=10)
    expected = evaluator.eval_depthcrafter(paths, [str(path)] * 501, [1] * 501, args)
    np.testing.assert_allclose(list(result.values()), expected, rtol=1e-6)
    assert result['abs_rel'] > 1e-4


@pytest.mark.parametrize('scope,setting', [('benchmark', 'long'), ('tum-validation', 'full')])
def test_cli_defaults_and_output_namespaces(monkeypatch, tmp_path, scope, setting):
    import sys

    args = ['comparison', '--config', 'c.yaml', '--checkpoint', 'best.pt',
            '--output', str(tmp_path), '--scope', scope, '--model', 'unidepthv2']
    monkeypatch.setattr(sys, 'argv', args)
    from explore_a.evaluation import session
    observed = []
    monkeypatch.setattr(session, 'run', lambda *a, **kw: observed.append((a,kw)))
    comparison.main()
    args, options = observed[0]
    assert args[2] == [scope]
    assert options['setting'] == ('long' if scope == 'tum-validation' else setting)
    import json
    models = json.loads((tmp_path/'requested_models.json').read_text())
    assert models[0]['adapter'] == 'unidepthv2'
    assert models[0]['kind'] == 'frozen'
