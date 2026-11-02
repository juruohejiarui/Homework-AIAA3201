"""Sensor-depth validation cohort preparation and independent metric scoring, CPU only."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from depth_project.data.protocol import select_sequences
from depth_project.io import write_json
from explore_a.data.tum_validation import prepare
from explore_a.download.ddad_validation import selected_members
from explore_a.evaluation.comparison_report import report
from explore_a.evaluation.part2_protocol import metric_scores


def test_validation_pairs_units_and_gap_reset(tmp_path):
    name = 'rgbd_dataset_freiburg2_xyz_validation'
    scene = tmp_path / 'sequences' / name
    (scene / 'rgb').mkdir(parents=True)
    (scene / 'depth').mkdir()
    times = ['100.000', '100.033', '100.500']
    for stamp in times:
        cv2.imwrite(str(scene / 'rgb' / (stamp + '.png')), np.zeros((8, 12, 3), np.uint8))
        cv2.imwrite(str(scene / 'depth' / (stamp + '.png')),
                    np.full((8, 12), 10000, np.uint16))
    (scene / 'rgb.txt').write_text(''.join(f'{t} rgb/{t}.png\n' for t in times))
    (scene / 'depth.txt').write_text(''.join(f'{t} depth/{t}.png\n' for t in times))
    plan = tmp_path / 'plan.json'
    write_json(plan, {'subset': 'validation', 'output': str(tmp_path), 'files': [
        {'name': name + '.tgz', 'category': 'validation'}]})
    value = prepare(plan, tmp_path / 'manifest.json')
    frames = value['sequences'][0]['frames']
    assert [f['reset_history'] for f in frames] == [True, False, True]
    assert value['config']['factor'] == 5000
    assert len(select_sequences(value, 'full')[0]['frames']) == 3
    with pytest.raises(ValueError, match='use full'):
        select_sequences(value, 'short')
    prepare(plan, tmp_path / 'manifest.json')  # Same preparation can resume without rewriting.
    (scene / 'depth/100.033.png').unlink()
    with pytest.raises(FileNotFoundError):
        prepare(plan, tmp_path / 'other.json')


def test_metric_scores_do_not_align_or_count_invalid_depth(tmp_path):
    target = np.full((8, 12), 10000, np.uint16)  # 2 meters.
    target[0] = 0
    path = tmp_path / 'gt.png'
    cv2.imwrite(str(path), target)
    values = metric_scores([np.full((8, 12), .25, np.float32)], [{'gt': str(path)}],
                           {'factor': 5000, 'crop': [0, 8, 0, 12], 'max_depth': 10})
    assert values['abs_rel'] == 1.
    assert values['rmse'] == 2.
    assert values['d1'] == 0.


def test_tum_report_separates_metric_from_clip_ssi(tmp_path):
    metrics = {'abs_rel': .01, 'rmse': .02, 'delta1': 1.,
               'metric_abs_rel': .2, 'metric_rmse': .3, 'metric_d1': .8}
    write_json(tmp_path / 'tum-validation/fusion/summary.json', {
        'tum-validation': {'mean': {'fusion': metrics}}})
    rows = report(tmp_path)
    assert len(rows) == 2
    assert rows[0]['alignment'] == 'clip inverse SSI'
    assert rows[1]['alignment'] == 'none' and rows[1]['abs_rel'] == .2


def test_ddad_export_rejects_video_assumptions_and_unmatched_ids():
    from types import SimpleNamespace

    invalid = SimpleNamespace(filename='pregenerated/val/rgb/0000000000.png',
                              is_dir=lambda: False)
    with pytest.raises(ValueError, match='3950 matched'):
        selected_members(SimpleNamespace(infolist=lambda: [invalid]))
    invalid.filename = '../escape.png'
    with pytest.raises(ValueError, match='Unsafe'):
        selected_members(SimpleNamespace(infolist=lambda: [invalid]))
