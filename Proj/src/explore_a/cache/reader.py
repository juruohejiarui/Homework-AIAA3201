"""Cache roots and numeric component identities; execution lives in the shared pipeline."""
from depth_project.io import read_json
from explore_a.cache.settings import LOCAL_ROOT, root_path


def configured_root(config):
    settings = config.get("cache", {})
    from explore_a.frontend import name
    name(config)
    if settings.get("recipe", "glob-da2") != "glob-da2":
        raise ValueError("DA predictions require cache.recipe: glob-da2")
    return root_path(settings.get("root", LOCAL_ROOT), settings.get("recipe", "glob-da2"))


def check_component(spec, category):
    if category in spec.get("shared_cache_conflicts", ()):
        raise ValueError("Shared cache component identity differs: " + category)


def cache_spec(config):
    from explore_a.frontend import shared_root
    from explore_a.cache.plan import recipe_spec
    root = configured_root(config)
    if (root / ".migration.json").exists():
        raise ValueError("DA cache migration is incomplete")
    if (shared_root(root) / ".migration.json").exists():
        raise ValueError("Shared cache migration is incomplete")
    requested = recipe_spec(config, "glob-da2")
    shared = shared_root(root) / "recipe.json"
    conflicts = []
    if shared.exists():
        stored = read_json(shared)
        for key in ("geometry", "data"):
            expected = config["geometry"] if key == "geometry" else requested[key]
            if stored[key] != expected:
                raise ValueError("Shared flow preprocessing differs from DA request")
        from pathlib import Path
        for category, names in (("flow_farneback", ()), ("flow_raft", ("raft",)),
                                ("teacher_farneback", ("mftiq", "resnet", "dinov2")),
                                ("teacher_raft", ("mftiq", "resnet", "dinov2", "raft"))):
            for key in names:
                path = Path(config["weights"][key])
                value = {"filename": path.name, "bytes": path.stat().st_size}
                if stored["weights"].get(key) != value:
                    conflicts.append(category)
                    break
    if (root / "recipe.json").exists():
        from explore_a.cache.identity import assert_compatible
        assert_compatible(read_json(root / "recipe.json"), requested)
    return {**requested, "shared_cache_compatible": shared.exists(),
            "shared_cache_conflicts": conflicts}
