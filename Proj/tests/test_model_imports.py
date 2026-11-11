"""Fresh interpreters exercise metadata and parallel model imports before any warmup."""
from pathlib import Path
import os
import subprocess
import sys
import textwrap


def run_fresh(code):
    environment = os.environ.copy()
    source = str(Path(__file__).parents[1] / "src")
    environment["PYTHONPATH"] = os.pathsep.join((source, environment.get("PYTHONPATH", "")))
    result = subprocess.run([sys.executable, "-c", textwrap.dedent(code)],
                            env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_provenance_import_does_not_load_adapters():
    run_fresh("""
        import importlib.abc
        import sys

        class RejectAdapters(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in {"depth_project.models.da2", "depth_project.models.vda",
                                "depth_project.models.unidepth"}:
                    raise AssertionError("Metadata import loaded adapter: " + fullname)

        sys.meta_path.insert(0, RejectAdapters())
        from depth_project.models.provenance import source_signature
        from explore_a.remote_online.sync import ensure_synchronized
        assert callable(source_signature) and callable(ensure_synchronized)
    """)


def test_parallel_metadata_and_adapter_imports_keep_public_api():
    run_fresh("""
        from concurrent.futures import ThreadPoolExecutor
        from importlib import import_module
        from threading import Barrier

        names = ("depth_project.models.da2", "depth_project.models.provenance",
                 "depth_project.models.vda", "explore_a.remote_online.sync")
        barrier = Barrier(len(names))
        def load(name):
            barrier.wait(timeout=10)
            return import_module(name)
        with ThreadPoolExecutor(max_workers=len(names)) as executor:
            modules = list(executor.map(load, names))

        from depth_project.models import DepthAnythingAdapter, VideoDepthAdapter, adapter_type
        assert DepthAnythingAdapter is modules[0].DepthAnythingAdapter is adapter_type("da2")
        assert VideoDepthAdapter is modules[2].VideoDepthAdapter is adapter_type("vda")
    """)
