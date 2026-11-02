"""Export the fixed local split into RGB-only, per-sequence portable producer jobs."""
from collections import Counter, defaultdict
from pathlib import Path

from depth_project.io import read_json, write_json
from explore_a.cache.paths import component, expected, job_id
from explore_a.cache.settings import LOCAL_ROOT, RECIPE, root_path
from explore_a.config import bind_identity
from explore_a.data.prepare import load_manifest


def recipe_spec(config, recipe=RECIPE):
    from explore_a.frontend import metadata
    prediction = recipe == "glob-da2"
    names = ("da2",) if prediction else ("raft", "mftiq", "resnet", "dinov2")
    result = {"schema": 4, "id": component(recipe),
              "data": {key: config["data"][key] for key in ("canvas_kitti", "canvas_indoor")},
              "geometry": {} if prediction else config["geometry"],
              "flows": [] if prediction else ["raft", "farneback"],
              "teacher": not prediction, "storage": "fp32-npz-deflate-9",
              "attention": "sdpa" if prediction else "flash_prefer",
              "precision": {"da2": "fp16"} if prediction else {"flow":"fp32", "mftiq":"fp32"},
              "weights": {name: {"filename": Path(config["weights"][name]).name,
                                 "bytes": Path(config["weights"][name]).stat().st_size}
                          for name in names}}
    if prediction:
        result["frontend"] = metadata(config)
    return result


def prepare_plan(config, recipe=RECIPE, base=LOCAL_ROOT):
    from explore_a.frontend import relative
    if recipe not in ("glob", "glob-da2"):
        raise ValueError("Use glob-da2 for DA predictions or glob for shared flow/visibility")
    root = root_path(base, recipe)
    if (root / ".migration.json").exists():
        raise ValueError("Complete the pending migration before preparing jobs")
    origin = root / "dispatch.origin.json"
    if not origin.exists():
        write_json(origin, {"legacy": (root / "plan.json").exists()})
    manifest = load_manifest(config)
    if set(config["data"]["datasets"]) - {"kitti", "tum", "nyuv2"}:
        raise ValueError("Distributed recipes support KITTI, TUM and NYUv2 H5")
    spec = recipe_spec(config, recipe)
    from explore_a.cache.identity import assert_compatible
    if (root / "recipe.json").exists():
        assert_compatible(read_json(root / "recipe.json"), spec)
    groups = defaultdict(list)
    for row in manifest["samples"]:
        groups[job_id(row)].append(row)
    jobs, bindings = [], []
    for name, rows in sorted(groups.items()):
        files, exported = {}, []
        for row in rows:
            selected = {key: row[key] for key in
                        ("id", "dataset", "sequence", "camera", "previous_id", "current_id",
                         "split", "group")}
            for flag in ("previous", "current"):
                path = Path(row[flag + "_rgb"])
                filename = component(path.stem + ".png" if row["camera"] == "h5_rgb"
                                     else path.name)
                rel = str(Path(row["dataset"]) / row["sequence"] / row["camera"] / filename)
                if rel in files and files[rel]["source"] != str(path):
                    raise ValueError(f"Ambiguous duplicate RGB ID: {rel}")
                if rel not in files:
                    files[rel] = {"source": str(path), "bytes": path.stat().st_size}
                    if row["camera"] == "h5_rgb":
                        files[rel]["encoding"] = "nyu_h5_rgb"
                selected[flag + "_rgb"] = rel
            exported.append(selected)
        job = {"recipe": recipe, "job": name, "rows": exported,
               "inputs": {rel: value["bytes"] for rel, value in sorted(files.items())},
               "outputs": expected(rows,
                                   flows=() if recipe == "glob-da2" else ("raft", "farneback"),
                                   teacher=recipe != "glob-da2", frames=recipe == "glob-da2"),
               "history_drives": sorted(manifest["history"]),
               "test_label": manifest["test_label"]}
        job_path = root / "jobs" / name / "job.json"
        if job_path.exists():
            # A consumer may select fewer pairs or different splits from a production job.
            # Production membership stays fixed once dispatched; no receipt is invalidated.
            stored = read_json(job_path)
            known = {r["id"]: r for r in stored["rows"]}
            original_sources = read_json(job_path.with_name("sources.json"))
            for row in exported:
                previous = known.get(row["id"])
                if previous is None or any(previous[k] != v for k, v in row.items()
                                           if k not in ("split", "group")):
                    raise ValueError("Settings changed; conflicting shared cache pair: "
                                     + row["id"])
            if any(original_sources.get(key) != value for key, value in files.items()):
                raise ValueError(f"Settings changed; conflicting shared cache sources: {name}")
            job, files = stored, original_sources
        for path, value in ((root / "jobs" / name / "job.json", job),
                            (root / "jobs" / name / "sources.json", files)):
            if path.exists() and read_json(path) != value:
                raise ValueError(f"Settings changed; conflicting shared cache job: {path}")
            bindings.append((path, value))
        jobs.append({"job": name, "pairs": len(job["rows"]), "input_files": len(files),
                     "input_bytes": sum(value["bytes"] for value in files.values())})
    # Preflight every binding before publishing any new job or consumer configuration.
    for path, value in bindings:
        bind_identity(path, value)
    write_json(root / "recipe.json", spec)
    import copy
    import yaml
    producer = {key: copy.deepcopy(config[key]) for key in
                ("paths", "weights", "data", "geometry", "online", "observability")}
    if "frontend" in config:
        producer["frontend"] = copy.deepcopy(config["frontend"])
    write_yaml(root / "producer.yaml", yaml.safe_dump(producer, sort_keys=False))
    consumer = copy.deepcopy(config)
    consumer.pop("config_path", None)
    consumer["execution"] = {"backend": "cache"}
    consumer.pop("remote_online", None)
    consumer["cache"] = {"root": str(Path(base).expanduser()), "recipe": "glob-da2"}
    if not relative(config):
        consumer["paths"]["runs"] = str(Path(config["paths"]["runs"]).with_name("runs_cached_60k"))
    write_yaml(root / "consumer.yaml", yaml.safe_dump(consumer, sort_keys=False))
    profile = root / "profiles" / component(Path(config["paths"]["manifest"]).parent.name)
    profile.mkdir(parents=True, exist_ok=True)
    write_yaml(profile / "consumer.yaml", yaml.safe_dump(consumer, sort_keys=False))
    write_hybrid_config(root, config)
    # A training profile can add jobs but cannot remove the shared catalog's other jobs.
    old = read_json(root / "plan.json") if (root / "plan.json").exists() else None
    combined = {item["job"]: item for item in old["jobs"]} if old else {}
    combined.update({item["job"]: item for item in jobs})
    counts = Counter()
    for name in combined:
        for row in load_job(root, name)["rows"]:
            counts[f"{row['dataset']}/{row['split']}"] += 1
    summary = {"recipe": recipe, "counts": dict(counts),
               "jobs": sorted(combined.values(), key=lambda item: item["job"]),
               "pairs": sum(counts.values()), "history": manifest["history"]}
    write_json(root / "plan.json", summary)
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
    path.parent.mkdir(parents=True, exist_ok=True)
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
    hybrid["paths"]["runs"] = config["paths"]["runs"]
    from explore_a.frontend import shared_root
    hybrid["cache"] = {"root": str(shared_root(root).parents[1]), "recipe": "glob-da2"}
    write_yaml(root / "consumer_hybrid.yaml", yaml.safe_dump(hybrid, sort_keys=False))
    profile = root / "profiles" / component(Path(config["paths"]["manifest"]).parent.name)
    write_yaml(profile / "consumer_hybrid.yaml", yaml.safe_dump(hybrid, sort_keys=False))
