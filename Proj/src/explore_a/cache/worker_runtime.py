"""Lazily load one frozen frontend set per recipe, on the GPU worker's main thread."""
from depth_project.io import read_json
from explore_a.config import load_config


class WorkerRuntime:
    def __init__(self):
        self.producer, self.spec = None, None

    def get(self, root):
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
            config = load_config(root / "producer.yaml")
            if recipe_spec(config, spec["id"]) != spec:
                raise ValueError("Producer configuration/weights differ from recipe")
            self.producer = Producer(config)
            self.spec = spec
        return self.producer
