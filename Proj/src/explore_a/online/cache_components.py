"""Read-only component discovery; native-cache writes are exclusive to explicit producers."""
import hashlib
import json
from pathlib import Path
from threading import RLock

import numpy as np

from depth_project.io import read_json
from explore_a.cache.paths import artifact
from explore_a.cache.reader import configured_root, cache_spec, check_component
from explore_a.cache.source_identity import CacheSources
from explore_a.cache.storage import read_npz
from explore_a.frontend import field, metadata, name, shared_root


def image_identity(raw, previous=False):
    prefix = 'previous' if previous else 'current'
    row = raw['row']
    path = Path(row[prefix+'_rgb']).resolve()
    stat = path.stat()
    return {'path':str(path), 'bytes':stat.st_size, 'mtime_ns':stat.st_mtime_ns,
            'transform':row.get('input_transform', 'training-original-rgb'),
            'shape':list(raw['images'][previous].shape)}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def native_binding(config, raw, previous=False):
    weight = Path(config['weights'][name(config)])
    identity = {'schema':1, 'frontend':metadata(config),
                'weight':{'name':weight.name, 'bytes':weight.stat().st_size,
                          'mtime_ns':weight.stat().st_mtime_ns},
                'image':image_identity(raw, previous)}
    root = shared_root(configured_root(config)) / 'frontends' / name(config) / 'native'
    return root / (digest(identity)+'.npz'), identity


class CacheComponents:
    def __init__(self, config):
        self.config, self.root = config, configured_root(config)
        self.sources, self.shared = CacheSources(self.root), CacheSources(shared_root(self.root))
        self.lock, self.spec, self.loaded = RLock(), None, False

    def discover(self, raw, operation, previous=False, flow=None, native=False):
        with self.lock:
            return self._discover(raw, operation, previous, flow, native)

    def _discover(self, raw, operation, previous, flow, native):
        if operation == 'prediction':
            path, identity = native_binding(self.config, raw, previous)
            sidecar = path.with_suffix('.json')
            if path.exists() and sidecar.exists():
                if read_json(sidecar) != identity:
                    raise ValueError('Native cache input identity differs: '+str(path))
                return path, field(self.config), True
            if native:
                return None
        # Benchmark cropping and RGB preprocessing have their own native identity.
        if raw['row'].get('input_transform') is not None:
            return None
        if not self.loaded:
            if (self.root/'recipe.json').exists() or (
                    shared_root(self.root)/'recipe.json').exists():
                self.spec = cache_spec(self.config)
            self.loaded = True
        if self.spec is None:
            return None
        row = raw['row']
        if operation == 'prediction':
            registered = self.sources.validate(row)
            path, key = artifact(self.root, 'frames', row, previous), field(self.config)
        else:
            category = ('flow_' if operation == 'flow' else 'teacher_')+flow
            check_component(self.spec, category)
            registered = self.shared.validate(row)
            kind = ('flow/' if operation == 'flow' else 'teacher/')+flow
            path, key = artifact(self.root, kind, row), None
        return (path, key, False) if registered and path.is_file() else None

    @staticmethod
    def read(binding):
        path, key, native = binding
        values = read_npz(path)
        if key is None:
            return values
        if set(values) != {key}:
            raise ValueError('Unexpected prediction cache fields: '+str(path))
        value = values[key]
        if native:
            value = np.asarray(value).reshape(value.shape[-2:])
        return {'prediction':value}
