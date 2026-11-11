"""Phase-owned routing is independent of mutable global phase and training RNG."""
import copy
import random

import pytest

from explore_a.online.factory import make_runtime
from explore_a.online.loading import read_pair
from explore_a.online.routing import profiles, for_phase
from test_frozen_pipeline import configured, MemoryClient
from test_explore_a_distributed import prepared as prepared
from test_explore_a_pipeline import fake_backbones as fake_backbones


def test_ticket_phase_selects_profile_even_inside_another_phase(prepared, monkeypatch):
    config, _, _, rows = configured(prepared, 0, .28)
    config['execution']['routing'] = {'train': {'cache': 0., 'remote': 0.},
                                      'evaluation': {'cache': 0., 'remote': 1.}}
    monkeypatch.setattr('explore_a.remote_online.client.Client', MemoryClient)
    MemoryClient.gate, MemoryClient.calls = None, []
    runtime = make_runtime(config, ['farneback'], device='cpu')
    state = random.getstate()
    train = read_pair(rows[0], config['data'])
    train['frozen_ticket'] = runtime.prepare(rows[0], 0, phase='train')
    try:
        with runtime.input_phase('dev'):
            runtime.compute(train, 'farneback')
        assert runtime.statistics()['train_local_assigned'] == 3
        test = read_pair(rows[1], config['data'])
        test['frozen_ticket'] = runtime.prepare(rows[1], 8, phase='test')
        with runtime.input_phase('test'):
            runtime.compute(test, 'farneback')
        assert runtime.statistics()['evaluation_remote_assigned'] > 0
        assert random.getstate() == state
    finally:
        runtime.close()


def test_profile_validation_and_online_override():
    config = {'execution': {'backend': 'hybrid'},
              'cache_read': {'mode': 'blocking', 'probability': .2},
              'remote_online': {'enabled': True, 'probability': .28}}
    for bad in ({'train': {'cache': .8, 'remote': .4}},
                {'evaluation': {'cache': float('nan')}}, {'dev': {}},
                {'train': {'remote': True}}, {'train': {'local': .5}}):
        c = copy.deepcopy(config)
        c['execution']['routing'] = bad
        with pytest.raises(ValueError):
            profiles(c)
    original = profiles(config)
    assert for_phase(original, 'train') == {'cache': .2, 'remote': .28}
    for name in ('dev', 'test', 'nyu', 'benchmark', 'oracle'):
        assert for_phase(original, name) == original['evaluation']
    config['execution'].update(backend='online', routing={'train': {'cache': .2, 'remote': .7}})
    assert profiles(config)['train'] == {'cache': 0., 'remote': 0.}


def test_performance_selection_uses_pooled_dev_test_time_only():
    from explore_a.audits.routing_benchmark import choose
    rows = []
    for repetition in range(2):
        for phase in ('train','dev','test'):
            for cache, remote, seconds in ((.2,.28,10.),(0.,.28,8.)):
                rows.append({'phase':phase, 'cache':cache, 'remote':remote, 'pairs':80,
                             'wall_s':seconds, 'round':repetition})
    result = choose(rows)
    assert result['train']['selected']['remote'] == .28
    assert result['train']['selected']['cache'] == 0
    assert result['evaluation']['selected']['cache'] == 0
    # Small apparent gains are insufficient to override the existing profile.
    for row in rows:
        if row['cache'] == 0:
            row['wall_s'] = 9.9
    assert choose(rows)['train']['selected']['cache'] == .2


def test_capacity_formula_and_availability_conversion():
    from explore_a.audits.routing_balance import solve
    result = solve(.1, .5, .2, .05)
    assert result['predicted_pair_s'] == pytest.approx(1.5/17)
    actual = result['actual_fractions']
    assert sum(actual.values()) == pytest.approx(1)
    bounded = solve(.1, .5, .2, .05, .05)
    assert bounded['actual_fractions']['cache'] <= .95*.05+1e-10
    assert sum(bounded['profile'].values()) <= 1
    # When head work is the bottleneck, no local frozen work should be introduced.
    head = solve(.1, .02, .02, 1.)
    assert head['predicted_pair_s'] == pytest.approx(1.)
    assert head['actual_fractions']['local'] == pytest.approx(0, abs=1e-10)
    with pytest.raises(ValueError):
        solve(0, .1, .1, .1)


def test_decoding_frontier_prevents_later_results_filling_all_leases(prepared, monkeypatch):
    config, _, _, rows = configured(prepared, 0, 1)
    config['remote_online']['max_inflight'] = 1
    monkeypatch.setattr('explore_a.remote_online.client.Client', MemoryClient)
    MemoryClient.gate, MemoryClient.calls = None, []
    runtime = make_runtime(config, ['farneback'], device='cpu')
    scope = object()
    early, later = [read_pair(row, config['data']) for row in rows[:2]]
    # A later history prediction shares the early current frame, despite decoder inversion.
    later['row'] = {**later['row'], 'previous_rgb':early['row']['current_rgb']}
    later['images'][True] = early['images'][False]
    early['frozen_ticket'] = runtime.prepare(rows[0], 0, phase='dev', scope=scope)
    later['frozen_ticket'] = runtime.prepare(rows[1], 1, phase='dev', scope=scope)
    try:
        runtime.submit_decoded(later)  # Simulate decoding out of order.
        with runtime.condition:
            runtime._pump('dev')
            assert not any(j.leased for j in runtime.jobs.values())
        runtime.submit_decoded(early)
        assert early['frozen_jobs']['previous_depth'].position == 0
        assert early['frozen_jobs']['current_depth'].position == 1
        runtime.compute(early, 'farneback')
        runtime.compute(later, 'farneback')
        assert runtime.statistics()['remote_inflight_peak'] == 1
        assert not runtime.pending
        runtime.prepare(rows[0], 0, phase='dev', scope=scope)
        runtime.cancel_prepared(scope)
        assert not runtime.pending
    finally:
        runtime.close()


def test_generated_hybrid_consumer_retains_profiles(prepared):
    import yaml
    from explore_a.cache.plan import write_hybrid_config
    config, root, _, _ = configured(prepared, .2, .28)
    config['execution']['routing'] = {'train': {'cache': .38, 'remote': .49},
                                     'evaluation': {'cache': .42, 'remote': .46}}
    write_hybrid_config(root, config)
    generated = yaml.safe_load((root/'consumer_hybrid.yaml').read_text())
    assert generated['execution']['routing'] == config['execution']['routing']


def test_failed_decode_unregisters_prepared_tickets(prepared, monkeypatch):
    from explore_a.online.loading import OnlinePairs
    config, _, _, rows = configured(prepared, 0, 0)
    runtime = make_runtime(config, ['farneback'], device='cpu')
    def fail(*args):
        raise OSError('synthetic decode failure')
    monkeypatch.setattr('explore_a.online.loading.read_pair', fail)
    iterator = OnlinePairs(config, rows, runtime, 'farneback', phase='dev').iter_batches()
    try:
        with pytest.raises(OSError, match='synthetic decode failure'):
            next(iterator)
        assert not runtime.pending
    finally:
        iterator.close()
        runtime.close()
