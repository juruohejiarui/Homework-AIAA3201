"""Three experiment entries share settings without changing numeric or resume identities."""
import copy
from pathlib import Path

import pytest
import yaml

from explore_a.config import DEFAULT_CONFIG, load_config, read_config
from explore_a.matrix import experiments


def test_three_small_configs_have_identical_non_output_settings():
    configs = experiments()
    assert len(list(DEFAULT_CONFIG.parent.glob('*.yaml'))) == 3
    reference = load_config()
    for config, fusion in configs:
        actual = copy.deepcopy(config)
        expected = copy.deepcopy(reference)
        for value in (actual, expected):
            value.pop('config_path')
            value['paths'].pop('runs')
        assert actual == expected
        assert Path(config['paths']['runs']).name == {
            'unet_guided':'runs_da2_guided_30k', 'flow_refine':'runs_da2_refine_30k',
            'final_flow':'runs_da2_final_30k'}[fusion]
        assert 'extends' not in config


def test_nested_override_list_replacement_and_independent_loads(tmp_path):
    (tmp_path/'base.yaml').write_text(yaml.safe_dump({'paths':{'runs':'a','data':'b'},
                                                    'data':{'datasets':['kitti','tum']}}))
    child = tmp_path/'child.yaml'
    child.write_text(yaml.safe_dump({'extends':'base.yaml', 'paths':{'runs':'c'},
                                    'data':{'datasets':['nyuv2']}}))
    result = read_config(child)
    assert result == {'paths':{'runs':'c','data':'b'}, 'data':{'datasets':['nyuv2']}}
    result['paths']['data'] = 'changed'
    assert read_config(child)['paths']['data'] == 'b'
    (tmp_path/'base.yaml').write_text('extends: child.yaml\n')
    with pytest.raises(ValueError, match='Cyclic'):
        read_config(child)
