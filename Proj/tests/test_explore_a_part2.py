"""Part 2 adapter preserves frozen frame cohorts and the upstream evaluation formula."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from depth_project.evaluation.official import load_evaluator
from explore_a.evaluation.part2_protocol import benchmark_path, official_score, restore_canvas
from explore_a.data.reading import fit_image


def test_memory_predictions_match_unmodified_file_evaluator(tmp_path):
    frames, predictions, paths = [], [], []
    for i in range(3):
        depth = np.linspace(1, 8, 48).reshape(6, 8).astype(np.float32) + i * .1
        path = tmp_path / f'gt{i}.npy'
        np.save(path, depth)
        frames.append({'gt': str(path)})
        inverse = .8 / depth + .03 + np.sin(depth) * .002
        predictions.append(inverse)
        path = tmp_path / f'pred{i}.npy'
        np.save(path, inverse)
        paths.append(str(path))
    config = {'crop': [0, 6, 0, 8], 'max_depth': 10, 'factor': 1}
    result = official_score(predictions, frames, config)
    evaluator = load_evaluator(device='cpu')
    args = SimpleNamespace(a=0, b=6, c=0, d=8, max_eval_len=500, max_depth_eval=10)
    expected = evaluator.eval_depthcrafter(paths, [r['gt'] for r in frames], [1]*3, args)
    np.testing.assert_allclose(list(result.values()), expected, rtol=1e-6)
    assert evaluator.os.path.exists(paths[0])


def test_canvas_restoration_discards_padding():
    image = np.ones((20, 60), np.float32) * 3
    fitted, _ = fit_image(image, (32, 48))
    restored = restore_canvas(fitted, image.shape, (32, 48))
    np.testing.assert_allclose(restored, 3)


def test_old_manifest_paths_resolve_only_under_benchmark_root(tmp_path):
    path = tmp_path / 'Bonn-RGBD/raw/rgb.png'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'RGB')
    old = '/media/old/Resouces/Datasets/Bonn-RGBD/raw/rgb.png'
    assert benchmark_path(old, tmp_path) == str(path)
    with pytest.raises(FileNotFoundError):
        benchmark_path(old.replace('rgb.png', 'missing.png'), tmp_path)


def test_missing_benchmark_never_falls_back_to_training_copy(tmp_path):
    training = tmp_path / 'Resources/Datasets/KITTI/rgb.png'
    training.parent.mkdir(parents=True)
    training.write_bytes(b'RGB')
    with pytest.raises(FileNotFoundError):
        benchmark_path(str(training), tmp_path / 'ResourcesLow/Datasets')


@pytest.mark.parametrize("frontend, fusion, ablations", [
    ("da2", "flow_refine", False), ("da2", "final_flow", False),
    ("da2", "final_flow", True), ("da2", "unet_guided", False),
    ("da2", "final_flow", False), ("da2", "final_flow", True)])
def test_part2_pins_checkpoint_and_first_frame_has_no_history(tmp_path, monkeypatch,
                                                            fusion, ablations, frontend):
    import json
    import cv2
    from explore_a.evaluation import part2

    root = tmp_path / 'Datasets'
    frames = []
    for i in range(2):
        rgb, gt = root / f'KITTI/rgb{i}.png', root / f'KITTI/gt{i}.npy'
        rgb.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(rgb), np.ones((8, 12, 3), np.uint8) * 100)
        np.save(gt, np.ones((8, 12), np.float32) * 2)
        frames.append({'image': str(rgb), 'gt': str(gt), 'source_index': i})
    manifest = root / part2.MANIFESTS['kitti']
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({'dataset': 'kitti',
        'config': {'crop': [0, 8, 0, 12], 'factor': 1, 'max_depth': 80},
        'sequences': [{'name': 'heldout-drive', 'frames': frames}]}))

    class Model(torch.nn.Module):
        def cuda(self):
            return self

        def ablate(self, predicted, inputs, mode):
            return {"depth": predicted["depth"] * .95}

    class Runtime:
        def __init__(self):
            self.depth = SimpleNamespace(predict=lambda rgb: np.ones(rgb.shape[:2]) * 2)
            self.flow_calls = 0

        def reset(self):
            pass

        def prepare(self, *args, **kwargs):
            return {}

        def submit_decoded(self, raw):
            pass

        def input_phase(self, phase):
            from contextlib import nullcontext
            return nullcontext()

        def close(self):
            pass

        def depth_pair(self, raw, ordered):
            raw['native_predictions'] = {'current':self.depth.predict(raw['images'][False])}
            return {**raw['data'], 'current_depth': torch.ones(1, 8, 12) * 2,
                    'previous_depth': torch.ones(1, 8, 12) * 2}

        def add_flow(self, raw, base, flow):
            self.flow_calls += 1
            return base

    monkeypatch.setattr(part2, 'make_fusion', lambda *args: Model())
    monkeypatch.setattr(part2, 'FinalFlow', Model)
    monkeypatch.setattr(part2.FusionInputs, 'from_pair', lambda *args: None)
    monkeypatch.setattr(part2, 'predict', lambda model, data, config, strategy: (
        {'depth': data['current_depth'] * .9}, {}))
    captured = []

    def score(predictions, *args):
        captured.append(predictions)
        return {'abs_rel': .1, 'rmse': .2, 'delta1': .9}

    monkeypatch.setattr(part2, 'official_score', score)
    monkeypatch.setattr(part2, 'metric_absrel', lambda *args: .1)
    checkpoint = tmp_path / 'best.pt'
    state = {'step': 8000, 'best': .1, 'model': {}, 'identity': {
        'experiment': {'flow': 'farneback', 'occlusion': 'rule', 'fusion': fusion},
        'training': {'amp': False}}}
    if frontend == 'da2':
        state['identity']['frontend'] = {'name':'da2', 'kind':'relative_inverse'}
    torch.save(state, checkpoint)
    config = {'data': {'canvas_kitti': [8, 12]}, 'training': {'amp': False},
              'online':{'read_workers':2, 'prefetch':4}}
    if frontend == 'da2':
        config['frontend'] = {'name':'da2'}
    runtime = Runtime()
    result = part2.evaluate_part2(config, checkpoint, tmp_path / 'out', root,
                                 datasets=['kitti'], runtime=runtime, ablations=ablations)
    assert result['kitti']['checkpoint_step'] == 8000
    assert runtime.flow_calls == 1
    np.testing.assert_allclose(captured[2][0], 2 if frontend == 'da2' else .5)
    np.testing.assert_allclose(captured[2][1], 1.8 if frontend == 'da2' else 1 / 1.8)
    if frontend == 'da2':
        assert 'baseline_fixed_ssi_mean' in result['kitti']
    if ablations:
        assert len(captured) == 7
        for values in captured[3:]:
            np.testing.assert_allclose(values[0], 2 if frontend == 'da2' else .5)
            np.testing.assert_allclose(values[1], 1.8 * .95 if frontend == 'da2'
                                       else 1 / (1.8 * .95))
    state['step'] = 10000
    torch.save(state, checkpoint)
    resumed = part2.evaluate_part2(config, checkpoint, tmp_path / 'out', root,
                                  datasets=['kitti'], runtime=runtime, ablations=ablations)
    assert resumed['kitti']['checkpoint_step'] == 8000
    assert runtime.flow_calls == 1
