"""Unified strict component dispatch, native provenance, phase isolation and memory RPC."""
import copy
from pathlib import Path
from threading import Event, Timer, get_ident
import sys

import numpy as np
import pytest
import torch

from depth_project.io import read_json, write_json
from explore_a.cache.paths import artifact
from explore_a.cache.plan import load_job
from explore_a.cache.producer import Producer
from explore_a.cache.storage import write_npz
from explore_a.online.factory import make_runtime
from explore_a.online.loading import OnlinePairs, read_pair
from explore_a.remote_online.policy import DEFAULTS
from test_explore_a_distributed import prepared as prepared
from test_explore_a_pipeline import fake_backbones as fake_backbones


def configured(prepared, cache=.2, remote=0):
    config, base, root, name = prepared
    config['execution'] = {'backend':'hybrid'}
    config['cache'] = {'root':str(base), 'recipe':'glob-da2'}
    config['cache_read'] = {'mode':'blocking', 'probability':cache, 'max_inflight':4}
    config['remote_online'] = {**DEFAULTS, 'enabled':remote > 0, 'probability':remote}
    rows = read_json(config['paths']['manifest'])['samples']
    from explore_a.cache.plan import prepare_plan
    da_root = prepare_plan(config, recipe='glob-da2', base=base)
    return config, da_root, name, rows


class MemoryClient:
    gate = None
    calls = []
    def __init__(self, config, flows, teacher, policy):
        self.closed = False
    def infer(self, payload):
        assert 'data' not in payload and 'gt' not in payload
        self.calls.append(payload)
        if self.gate is not None:
            self.gate.wait()
        operation, canvas = payload['operation'], payload['canvas']
        if operation == 'prediction':
            shape = payload['images'][0].shape[:2] if payload['native'] else (1,*canvas)
            values = {'prediction':np.full(shape, 2, np.float32)}
        elif operation == 'flow':
            values = {k:np.zeros((2,*canvas), np.float32) for k in ('forward','backward')}
        else:
            values = {'visibility':np.ones((1,*canvas), np.float32)}
        return values, {'remote_compute_s':.001}
    def close(self):
        self.closed = True


@pytest.mark.parametrize('phase', ['train','dev','test','benchmark','oracle','nyu'])
def test_every_phase_uses_remote_and_no_deadline(prepared, monkeypatch, phase):
    config, _, _, rows = configured(prepared, 0, 1)
    monkeypatch.setattr('explore_a.remote_online.client.Client', MemoryClient)
    runtime = make_runtime(config, ['farneback'], device='cpu')
    MemoryClient.gate, MemoryClient.calls = Event(), []
    Timer(.08, MemoryClient.gate.set).start()
    try:
        raw = read_pair(rows[0], config['data'])
        raw['frozen_ticket'] = runtime.prepare(rows[0], 0, phase=phase)
        with runtime.input_phase(phase):
            result = runtime.compute(raw, 'farneback')
        assert result['current_depth'].shape == (1,16,24)
        assert runtime.statistics()['remote_consumed'] == 3
        assert not runtime.local.frozen.models
        assert not any('timeout' in key for key in runtime.statistics())
    finally:
        runtime.close()
        MemoryClient.gate = None


def test_cache_presence_preallocation_corruption_and_no_write(prepared):
    config, root, name, rows = configured(prepared, 1, 0)
    job = load_job(root, name)
    row = next(r for r in rows if r['id'] == job['rows'][0]['id'])
    rows = [row, *[r for r in rows if r is not row]]
    Producer(config, 'cpu').generate(root, job, sources=read_json(
        root/'jobs'/name/'sources.json'), host='local', minimum_gib=0)
    from explore_a.frontend import shared_root
    shared = shared_root(root)
    Producer(config, 'cpu').generate(shared, load_job(shared, name), sources=read_json(
        shared/'jobs'/name/'sources.json'), host='local', minimum_gib=0)
    before = {p:p.read_bytes() for p in root.rglob('*.npz')}
    runtime = make_runtime(config, ['farneback'], device='cpu')
    result = runtime.compute(read_pair(rows[0], config['data']), 'farneback')
    assert runtime.statistics()['cache_consumed'] == 3
    assert not runtime.local.frozen.models
    assert before == {p:p.read_bytes() for p in root.rglob('*.npz')}
    runtime.close()
    # A missing component is assigned locally before submission, not after I/O failure.
    path = artifact(root, 'frames', rows[0])
    path.unlink()
    runtime = make_runtime(config, ['farneback'], device='cpu')
    runtime.compute(read_pair(rows[0], config['data']), 'farneback')
    assert runtime.statistics()['cache_unavailable'] == 1
    assert runtime.statistics()['local_consumed'] == 1
    assert not path.exists()
    runtime.close()
    path.write_bytes(b'corrupt')
    runtime = make_runtime(config, ['farneback'], device='cpu')
    with pytest.raises(Exception):
        runtime.compute(read_pair(rows[0], config['data']), 'farneback')
    assert not runtime.local.frozen.models
    assert path.read_bytes() == b'corrupt'
    runtime.close()


def test_prefetch_bounds_nested_phase_and_main_thread_cuda_owner(prepared, monkeypatch):
    config, _, _, rows = configured(prepared, 0, .5)
    monkeypatch.setattr('explore_a.remote_online.client.Client', MemoryClient)
    MemoryClient.gate = None
    runtime = make_runtime(config, ['farneback'], device='cpu')
    main, execute = get_ident(), runtime.local.execute
    def tracked(*args, **kwargs):
        assert get_ident() == main
        return execute(*args, **kwargs)
    runtime.local.execute = tracked
    train = OnlinePairs(config, rows, runtime, 'farneback', phase='train')
    iterator = train.iter_batches()
    try:
        next(iterator)
        dev = OnlinePairs(config, rows, runtime, 'farneback', phase='dev')
        list(dev.iter_batches())
        list(iterator)
        stats = runtime.statistics()
        assert stats.get('remote_inflight_peak', 0) <= 12
        assert stats.get('cache_inflight_peak', 0) <= 4
        assert len(runtime.jobs) <= runtime.limit
    finally:
        iterator.close()
        runtime.close()


def test_native_producer_and_transform_identity(prepared):
    from explore_a.cache.native import produce
    from explore_a.online.cache_components import native_binding
    config, _, _, rows = configured(prepared, 1, 0)
    report = produce(config, 'manifest', 'train', device='cpu')
    assert report['written'] == 3
    raw = read_pair(rows[0], config['data'])
    path, identity = native_binding(config, raw)
    assert path.is_file()
    runtime = make_runtime(config, ['farneback'], device='cpu')
    raw['frozen_ticket'] = runtime.prepare(rows[0], 0, {}, native=True)
    runtime.depth_pair(raw)
    assert runtime.statistics()['cache_consumed'] == 2
    assert not runtime.local.frozen.models
    altered = {**raw, 'row':copy.deepcopy(raw['row'])}
    altered['row']['input_transform'] = {'crop':[0,10,0,10]}
    other, _ = native_binding(config, altered)
    assert other != path
    runtime.close()
    assert produce(config, 'manifest', 'train', device='cpu')['written'] == 0


def test_real_component_memory_service_and_fail_stop(tmp_path, monkeypatch):
    from explore_a.remote_online.client import Client
    from test_explore_a_pipeline import online_experiment
    config = online_experiment(tmp_path)
    monkeypatch.setattr('depth_project.paths.PROJECT', tmp_path)
    client = Client(config, ['farneback'], False, DEFAULTS,
                    [sys.executable,'-u','-m','explore_a.remote_online.service','--fake'])
    payload = {'operation':'prediction', 'images':[np.ones((20,30,3),np.uint8)],
               'canvas':[16,24], 'flow':None, 'native':True, 'key':'native-identity'}
    try:
        values, stats = client.infer(payload)
        assert values['prediction'].shape == (20,30)
        assert values['prediction'].dtype == np.float32
        assert stats['remote_rpc_s'] >= 0
        process = client.process
        process.kill()
        process.wait()
        with pytest.raises((OSError, EOFError, BrokenPipeError)):
            client.infer(payload)
    finally:
        client.close()


def test_native_production_interruption_is_resumable(prepared, monkeypatch):
    from explore_a.cache import native
    config, root, _, _ = configured(prepared, 0, 0)
    writer = native.write_npz
    def interrupted(*args, **kwargs):
        raise OSError('interrupted native commit')
    monkeypatch.setattr(native, 'write_npz', interrupted)
    with pytest.raises(OSError, match='interrupted native'):
        native.produce(config, 'manifest', 'train', device='cpu')
    directory = root/'native'
    assert list(directory.glob('*.json')) and not list(directory.glob('*.npz'))
    monkeypatch.setattr(native, 'write_npz', writer)
    assert native.produce(config, 'manifest', 'train', device='cpu')['written'] == 3


@pytest.mark.parametrize('frontend', ['da2'])
def test_labeled_nyu_uses_pipeline_and_exact_no_history(tmp_path, monkeypatch, fake_backbones,
                                                       frontend):
    import cv2
    from test_explore_a_pipeline import online_experiment
    from explore_a.training.prediction import make_fusion
    from explore_a.evaluation.nyu_best import run
    config = online_experiment(tmp_path)
    if frontend == 'da2':
        config['frontend'] = {'name':'da2'}
        config['frontend'] = {'name':'da2'}
        monkeypatch.setattr('explore_a.frontend.make', lambda config:fake_backbones[0]())
    root = tmp_path/'nyu'
    for folder in ('rgb','gt'):
        (root/'test'/folder).mkdir(parents=True)
    cv2.imwrite(str(root/'test/rgb/0001.png'), np.full((480,640,3),100,np.uint8))
    np.save(root/'test/gt/0001.npy', np.full((480,640),3,np.float32))
    write_json(root/'test/manifest.json', {'test_ids':[1]})
    checkpoint = tmp_path/'best.pt'
    identity = {'experiment':{'flow':'farneback','occlusion':'rule','fusion':'unet_guided'},
                'training':config['training']}
    if frontend == 'da2':
        identity['frontend'] = {'name':'da2','kind':'relative_inverse'}
    torch.save({'model':make_fusion('unet_guided',config).state_dict(), 'identity':identity,
                'step':1, 'best':.1}, checkpoint)
    result = run(config, root, checkpoint, tmp_path/'evaluated-nyu', 'cpu')
    assert result['fallback_exact_images'] == 1
    branch = 'da2_canvas_control_ssi' if frontend == 'da2' else 'da2_canvas_control_ssi'
    assert result['mean']['best'] == result['mean'][branch]
    assert not list(tmp_path.rglob('*.npz'))


def test_route_weights_resume_and_independent_rng():
    import random
    from collections import Counter
    from explore_a.remote_online.policy import route
    states = random.getstate(), np.random.get_state(), torch.get_rng_state()
    values = [route(2026, i, 8, .2, .28) for i in range(20000)]
    counts = Counter(values)
    for source, p in [('cache',.2),('remote',.28),('local',.52)]:
        assert abs(counts[source]-20000*p) < 4*(20000*p*(1-p))**.5
    assert values[4000:] == [route(2026,i,8,.2,.28) for i in range(4000,20000)]
    assert random.getstate() == states[0]
    assert np.array_equal(np.random.get_state()[1], states[1][1])
    assert torch.equal(torch.get_rng_state(), states[2])


def test_cache_backpressure_retains_all_prefetched_results(prepared):
    config, root, _, rows = configured(prepared, 1, 0)
    producer = Producer(config, 'cpu')
    for folder in (root/'jobs').iterdir():
        job = read_json(folder/'job.json')
        producer.generate(root, job, sources=read_json(folder/'sources.json'),
                          host='local', minimum_gib=0)
    from explore_a.frontend import shared_root
    shared = shared_root(root)
    for folder in (shared/'jobs').iterdir():
        producer.generate(shared, read_json(folder/'job.json'), sources=read_json(
            folder/'sources.json'), host='local', minimum_gib=0)
    runtime = make_runtime(config, ['farneback'], device='cpu')
    try:
        dataset = OnlinePairs(config, rows, runtime, 'farneback', phase='dev')
        output = list(dataset.iter_batches())
        assert sum(len(indices) for indices, _ in output) == len(rows)
        assert runtime.statistics()['cache_inflight_peak'] <= 4
        assert not runtime.local.frozen.models
        assert runtime.statistics()['cache_consumed'] > 4
    finally:
        runtime.close()


def test_io_queue_prioritizes_newly_ready_earlier_dependency():
    from explore_a.online.pool import Workers
    pool, entered, release = Workers(1, 'priority-test'), Event(), Event()
    order = []
    def first():
        entered.set()
        release.wait()
        order.append('first')
    try:
        active = pool.submit(first)
        assert entered.wait(2)
        later = pool.submit(lambda:order.append('later'), priority=20)
        earlier = pool.submit(lambda:order.append('earlier-dependency'), priority=1)
        release.set()
        active.result(2)
        earlier.result(2)
        later.result(2)
        assert order == ['first','earlier-dependency','later']
    finally:
        release.set()
        pool.close()


def test_phase_drain_does_not_count_unconsumed_training_results(prepared):
    config, _, _, rows = configured(prepared, 0, 0)
    runtime = make_runtime(config, ['farneback'], device='cpu')
    raw = read_pair(rows[0], config['data'])
    raw['frozen_ticket'] = runtime.prepare(rows[0], 0, phase='train')
    runtime.submit_decoded(raw)
    try:
        with runtime.input_phase('train'):
            with runtime.input_phase('dev'):
                assert runtime.statistics().get('local_consumed', 0) == 0
                assert not any(j.leased for j in runtime.jobs.values())
            runtime.compute(raw, 'farneback')
            assert runtime.statistics()['local_consumed'] == 3
    finally:
        runtime.close()
