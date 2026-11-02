"""One blocking, bounded hybrid scheduler for training and every fusion evaluation phase."""
from collections import Counter, OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Condition, get_ident
import random
import time

import numpy as np
import torch

from explore_a.cache.read_policy import settings as cache_settings
from explore_a.data.reading import fit_image
from explore_a.frontend import kind
from explore_a.online.cache_components import CacheComponents, digest, image_identity
from explore_a.online.components import LocalExecutor, validate
from explore_a.online.pool import Workers
from explore_a.remote_online.policy import settings as remote_settings, route


@dataclass(eq=False)
class Component:
    key: str
    operation: str
    images: list
    canvas: tuple
    flow: str | None
    native: bool
    origin: str
    phase: str
    order: int
    binding: object = None
    dependency: object = None
    future: object = None
    values: dict | None = None
    representation: str = ""
    buffered: bool = False
    consumed: bool = False
    leased: bool = False
    submitted: float = field(default_factory=time.monotonic)


class FrozenPipeline:
    def __init__(self, config, flows, teacher=False, device='cuda'):
        self.config, self.flows, self.teacher, self.device = config, flows, teacher, device
        self.backend = config.get('execution', {}).get('backend', 'online')
        self.policy = cache_settings(config) or {'probability':0., 'max_inflight':4}
        self.remote_policy = remote_settings(config)
        self.pc = self.policy['probability'] if self.backend == 'hybrid' else 0.
        if self.backend == 'cache':
            self.pc = 1.
        self.pr = self.remote_policy['probability'] if self.remote_policy['enabled'] else 0.
        self.seed, self.batch = config['training']['seed'], config['training']['effective_batch']
        self.local = LocalExecutor(config, device)
        self.cache = CacheComponents(config) if self.backend != 'online' else None
        self.condition, self.jobs = Condition(), OrderedDict()
        self.counts, self.phase, self.closed = Counter(), 'evaluation', False
        self.main_thread, self.client = get_ident(), None
        self.cache_pool, self.remote_pool = None, None
        self.limits = {'cache':self.policy['max_inflight'],
                       'remote':self.remote_policy['max_inflight']}
        self.limit = config['online']['prefetch'] * (3+2*max(1, len(flows)))

    def statistics(self):
        with self.condition:
            return dict(self.counts)

    def prepare(self, row, occurrence, flows=None, teacher=None, phase=None, native=False):
        teacher = self.teacher if teacher is None else teacher
        return {'occurrence':occurrence, 'phase':phase or self.phase, 'native':native,
                'flows':{self.flows[0]:teacher} if flows is None else flows}

    def _origin(self, raw, operation, old, flow, native, occurrence):
        chosen = route(self.seed, occurrence, self.batch, self.pc, self.pr)
        self.counts['route_'+chosen] += 1
        binding = None
        if chosen == 'cache':
            binding = self.cache.discover(raw, operation, old, flow, native)
            self.counts['cache_available' if binding else 'cache_unavailable'] += 1
            if binding:
                return 'cache', binding, binding[2] if operation == 'prediction' else native
            if self.backend == 'cache':
                raise FileNotFoundError('Assigned cache component unavailable: '+raw['row']['id'])
            denominator = 1-self.pc
            probability = self.pr/denominator if denominator > 0 else 0.
            # Availability is resolved before submission, not by retrying a failed executor.
            value = random.Random(f'missing-component:{self.seed}:{occurrence}').random()
            chosen = 'remote' if value < probability else 'local'
        return chosen, binding, native

    def submit_component(self, raw, operation, old=False, flow=None, native=False, dependency=None):
        ticket = raw['frozen_ticket']
        images = ([raw['images'][old]] if operation == 'prediction' else
                  [raw['fitted'][True], raw['fitted'][False]])
        identities = [image_identity(raw, old)] if operation == 'prediction' else [
            image_identity(raw, True), image_identity(raw, False)]
        key = digest({'images':identities, 'operation':operation, 'flow':flow,
                      'canvas':list(raw['canvas']), 'native':native, 'kind':kind(self.config)})
        with self.condition:
            if self.closed:
                raise RuntimeError('Frozen pipeline closed')
            job = self.jobs.get(key)
            if job is not None:
                job.order = min(job.order, ticket['occurrence'])
                self.jobs.move_to_end(key)
                self.counts['component_reuse'] += 1
                return job
            origin, binding, native = self._origin(raw, operation, old, flow, native,
                                                   ticket['occurrence'])
            job = Component(key, operation, images, tuple(raw['canvas']), flow, native, origin,
                            ticket['phase'], ticket['occurrence'], binding, dependency,
                            representation=kind(self.config))
            self.jobs[key] = job
            self.counts[origin+'_assigned'] += 1
            # Evict only consumed results. Submitted and prefetched work stays owned and bounded
            # by the loader window, not by an unbounded executor queue.
            for old_key in list(self.jobs):
                if len(self.jobs) <= self.limit:
                    break
                if self.jobs[old_key].consumed or self.jobs[old_key].buffered:
                    del self.jobs[old_key]
            return job

    def submit_decoded(self, raw):
        if 'frozen_jobs' in raw:
            return
        ticket = raw.setdefault('frozen_ticket', self.prepare(raw['row'], 0))
        jobs = {}
        for old, prefix in ((True, 'previous'), (False, 'current')):
            jobs[prefix+'_depth'] = self.submit_component(
                raw, 'prediction', old, native=ticket['native'])
        for flow, teacher in ticket['flows'].items():
            job = self.submit_component(raw, 'flow', flow=flow)
            jobs['flow_'+flow] = job
            if teacher:
                jobs['teacher_'+flow] = self.submit_component(
                    raw, 'teacher', flow=flow, dependency=job)
        raw['frozen_jobs'] = jobs

    def _notify(self, future):
        with self.condition:
            self.condition.notify_all()

    def _execute(self, job):
        images = list(job.images)
        if job.dependency is not None:
            images.append(job.dependency.values['backward'])
        if job.origin == 'cache':
            values = self.cache.read(job.binding)
        elif job.origin == 'remote':
            values, stats = self.client.infer({'operation':job.operation, 'images':images,
                'canvas':job.canvas, 'flow':job.flow, 'native':job.native, 'key':job.key})
            with self.condition:
                self.counts.update(stats)
        else:
            values = self.local.execute(job.operation, images, job.canvas,
                                        flow=job.flow, native=job.native)
        return validate(job.operation, values, job.canvas, images[0].shape, self.config, job.native)

    def _activate(self, phase):
        for job in sorted(self.jobs.values(), key=lambda item:item.order):
            if job.phase != phase or job.values is not None or job.future is not None:
                continue
            if job.dependency is not None and job.dependency.values is None:
                continue
            if job.origin == 'local':
                continue
            leased = sum(j.leased and j.origin == job.origin for j in self.jobs.values())
            if leased >= self.limits[job.origin]:
                continue
            if job.origin == 'cache':
                if self.cache_pool is None:
                    self.cache_pool = Workers(self.limits['cache'], 'frozen-cache')
                pool = self.cache_pool
            else:
                if self.client is None:
                    from explore_a.remote_online.client import Client
                    self.client = Client(self.config, self.flows, self.teacher, self.remote_policy)
                    self.remote_pool = Workers(self.remote_policy['send_ahead'], 'frozen-rpc')
                pool = self.remote_pool
            job.leased = True
            job.future = pool.submit(lambda job=job:self._execute(job), priority=job.order)
            job.future.add_done_callback(self._notify)
            active = sum(j.leased and j.origin == job.origin for j in self.jobs.values())
            self.counts[job.origin+'_inflight_peak'] = max(
                self.counts[job.origin+'_inflight_peak'], active)

    def _harvest(self):
        for job in self.jobs.values():
            if job.future is not None and job.future.done() and job.values is None:
                # Exceptions, including a corrupted NPZ, propagate without changing the route.
                job.values = job.future.result()
                self.counts[job.origin+'_completed'] += 1

    def _pump(self, phase):
        self._harvest()
        self._activate(phase)
        local = next((j for j in sorted(self.jobs.values(), key=lambda j:j.order)
                      if j.phase == phase and j.origin == 'local' and j.values is None
                      and (j.dependency is None or j.dependency.values is not None)), None)
        if local is not None:
            # No decoding or I/O worker may initialize/use CUDA.
            self.condition.release()
            try:
                began = time.monotonic()
                local.values = self._execute(local)
            finally:
                self.condition.acquire()
            self.counts['local_compute_s'] += time.monotonic()-began
            self.counts['local_completed'] += 1
            return True
        return False

    def resolve(self, job, consume=True):
        if get_ident() != self.main_thread:
            raise RuntimeError('Frozen result consumption must run on the main GPU thread')
        began = time.monotonic()
        with self.condition:
            while job.values is None:
                if self.closed:
                    raise RuntimeError('Frozen pipeline closed')
                ran_local = self._pump(job.phase)
                self._harvest()
                if job.values is None and not ran_local:
                    self.condition.wait()
            if not consume:
                job.buffered, job.leased = True, False
            elif not job.consumed:
                job.consumed, job.leased = True, False
                self.counts[job.origin+'_consumed'] += 1
                self.counts['component_wait_s'] += time.monotonic()-began
            return job.values

    @contextmanager
    def input_phase(self, phase):
        previous, self.phase = self.phase, phase
        if previous != phase:
            # Drain submitted work into the bounded CPU result window before nested dev/test.
            with self.condition:
                active = [j for j in self.jobs.values() if j.phase == previous]
            began = time.monotonic()
            for job in active:
                if not job.consumed:
                    self.resolve(job, consume=False)
            while True:
                with self.condition:
                    outstanding = [j for j in self.jobs.values()
                                   if j.phase == previous and j.leased]
                if not outstanding:
                    break
                for job in outstanding:
                    self.resolve(job, consume=False)
            self.counts['phase_drain_s'] += time.monotonic()-began
        try:
            yield
        finally:
            self.phase = previous

    def _ensure(self, raw):
        self.submit_decoded(raw)

    def _tensor(self, value):
        with torch.inference_mode(False):
            return torch.from_numpy(np.ascontiguousarray(value)).to(self.device)

    def depth_pair(self, raw, ordered=False):
        self._ensure(raw)
        data = {k:v.to(self.device) for k,v in raw['data'].items()}
        for prefix in ('previous', 'current'):
            job = raw['frozen_jobs'][prefix+'_depth']
            value = self.resolve(job)['prediction']
            if job.native:
                raw.setdefault('native_predictions', {})[prefix] = value
                value = fit_image(value, raw['canvas'])[0][None]
            data[prefix+'_depth'] = self._tensor(value)
        return data

    def add_flow(self, raw, base, flow, teacher=False):
        self._ensure(raw)
        data = dict(base)
        for key, value in self.resolve(raw['frozen_jobs']['flow_'+flow]).items():
            data[key] = self._tensor(value)
        if teacher:
            values = self.resolve(raw['frozen_jobs']['teacher_'+flow])
            data['teacher_visibility'] = self._tensor(values['visibility'])
        if not raw.get("frozen_pair_consumed"):
            origins = {job.origin for job in raw["frozen_jobs"].values()}
            with self.condition:
                for origin in origins:
                    self.counts[origin+"_pairs"] += 1
                self.counts["pairs_consumed"] += 1
            raw["frozen_pair_consumed"] = True
        return data

    def compute(self, raw, flow, teacher=False, ordered=False):
        return self.add_flow(raw, self.depth_pair(raw, ordered), flow, teacher)

    def timed_depth(self, raw):
        began = time.monotonic()
        result = self.depth_pair(raw)
        return result, time.monotonic()-began

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        if self.client is not None:
            self.client.close()
        for pool in (self.cache_pool, self.remote_pool):
            if pool is not None:
                pool.close()
        self.jobs.clear()
