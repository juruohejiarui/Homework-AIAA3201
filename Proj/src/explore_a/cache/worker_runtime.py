"""Lazily load one frozen frontend set per recipe, on the GPU worker's main thread."""
from depth_project.io import read_json
import yaml


class WorkerRuntime:
    def __init__(self):
        self.producer, self.spec = None, None

    def get(self, root):
        if (root / ".migration.json").exists():
            raise ValueError("Cache migration is incomplete")
        spec = read_json(root / "recipe.json")
        if spec != self.spec:
            if self.producer is not None:
                self.producer = None
                import gc
                import torch
                gc.collect()
                torch.cuda.empty_cache()
            from explore_a.cache.plan import recipe_spec
            from explore_a.cache.producer import Producer
            # Producer jobs contain frozen settings, not training scheduler policies.
            config = yaml.safe_load((root / "producer.yaml").read_text())
            from explore_a.cache.identity import assert_compatible
            assert_compatible(spec, recipe_spec(config, spec["id"]))
            self.producer = Producer(config)
            self.spec = spec
        return self.producer
