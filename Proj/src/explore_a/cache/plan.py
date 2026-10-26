"""Export the fixed local split into RGB-only, per-sequence portable producer jobs."""
from collections import defaultdict
from pathlib import Path

from depth_project.io import read_json, write_json
from explore_a.cache.paths import component, expected, job_id
from explore_a.cache.settings import LOCAL_ROOT, RECIPE, root_path
from explore_a.config import bind_identity
from explore_a.data.prepare import load_manifest


def recipe_spec(config, recipe=RECIPE):
    return {"schema": 3, "id": component(recipe), "data": config["data"],
            "geometry": config["geometry"], "flows": ["raft", "farneback"], "teacher": True,
            "storage": "fp32-npz-deflate-9", "attention": "flash_prefer",
            "precision": {"depthpro": "fp16", "flow": "fp32", "mftiq": "fp32"},
            "weights": {name: {"filename": Path(path).name, "bytes": Path(path).stat().st_size}
                        for name, path in config["weights"].items()}}


def prepare_plan(config, recipe=RECIPE, base=LOCAL_ROOT):
    root = root_path(base, recipe)
    origin = root / "dispatch.origin.json"
    if not origin.exists():
        write_json(origin, {"legacy": (root / "plan.json").exists()})
    manifest = load_manifest(config)
    if set(config["data"]["datasets"]) - {"kitti", "tum"}:
        raise ValueError("Use the KITTI/TUM profile for the first distributed recipe")
    spec = recipe_spec(config, recipe)
    bind_identity(root / "recipe.json", spec)
    import copy
    import yaml
    producer = copy.deepcopy(config)
    producer.pop("config_path", None)
    producer["execution"] = {"backend": "online"}
    write_yaml(root / "producer.yaml", yaml.safe_dump(producer, sort_keys=False))
    consumer = copy.deepcopy(config)
    consumer.pop("config_path", None)
    consumer["execution"] = {"backend": "cache"}
    consumer["cache"] = {"root": str(Path(base).expanduser()), "recipe": recipe}
    consumer["paths"]["runs"] = str(Path(config["paths"]["runs"]).with_name("runs_cached_60k"))
    write_yaml(root / "consumer.yaml", yaml.safe_dump(consumer, sort_keys=False))
    write_hybrid_config(root, config)
    groups = defaultdict(list)
    for row in manifest["samples"]:
        groups[job_id(row)].append(row)
    jobs = []
    for name, rows in sorted(groups.items()):
        files, exported = {}, []
        for row in rows:
            selected = {key: row[key] for key in
                        ("id", "dataset", "sequence", "camera", "previous_id", "current_id",
                         "split", "group")}
            for flag in ("previous", "current"):
                path = Path(row[flag + "_rgb"])
                filename = component(path.name)
                rel = str(Path(row["dataset"]) / row["sequence"] / row["camera"] / filename)
                if rel in files and files[rel]["source"] != str(path):
                    raise ValueError(f"Ambiguous duplicate RGB ID: {rel}")
                if rel not in files:
                    files[rel] = {"source": str(path), "bytes": path.stat().st_size}
                selected[flag + "_rgb"] = rel
            exported.append(selected)
        job = {"recipe": recipe, "job": name, "rows": exported,
               "inputs": {rel: value["bytes"] for rel, value in sorted(files.items())},
               "outputs": expected(rows), "history_drives": sorted(manifest["history"]),
               "test_label": manifest["test_label"]}
        bind_identity(root / "jobs" / name / "job.json", job)
        bind_identity(root / "jobs" / name / "sources.json", files)
        jobs.append({"job": name, "pairs": len(rows), "input_files": len(files),
                     "input_bytes": sum(value["bytes"] for value in files.values())})
    summary = {"recipe": recipe, "counts": manifest["counts"], "jobs": jobs,
               "pairs": len(manifest["samples"]), "history": manifest["history"]}
    bind_identity(root / "plan.json", summary)
    from explore_a.cache.policy import refresh_policy
    from explore_a.cache.catalog import refresh_catalog
    policy = refresh_policy(root, config)
    refresh_catalog(root)
    skipped = sum(not video["eligible"] for video in policy["videos"].values())
    print(f"Plan {root}: {len(jobs)} sequences / {summary['pairs']} pairs", flush=True)
    print(f"Cache policy: {skipped} videos skipped; all samples remain in training/evaluation.")
    return root


def load_job(root, name):
    return read_json(Path(root) / "jobs" / component(name) / "job.json")


def write_yaml(path, source):
    """Workers may already be waiting while a coordinator reopens an existing plan."""
    if path.exists() and path.read_text() == source:
        return
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(source)
    temporary.replace(path)


def write_hybrid_config(root, config):
    import copy
    import yaml
    hybrid = copy.deepcopy(config)
    hybrid.pop("config_path", None)
    hybrid["execution"] = {"backend": "hybrid"}
    hybrid["cache"] = {"root": str(root.parents[1]), "recipe": root.name}
    hybrid["paths"]["runs"] = str(Path(config["paths"]["runs"]).with_name("runs_hybrid_60k"))
    write_yaml(root / "consumer_hybrid.yaml", yaml.safe_dump(hybrid, sort_keys=False))
