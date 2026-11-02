"""Run settings are provenance; only the selected model task constrains resume."""
from datetime import datetime, timezone
import json


def assert_same_task(previous, requested):
    if representation(previous) != representation(requested):
        raise ValueError("Checkpoint frontend/representation differs")
    if previous.get("experiment") != requested.get("experiment"):
        raise ValueError("Checkpoint model/task differs (flow, occlusion or fusion)")


def frontend(identity):
    return identity.get("frontend", identity.get("input", {}).get("frontend", {})).get(
        "name", "unknown")


def representation(identity):
    description = identity.get("frontend", identity.get("input", {}).get("frontend", {}))
    selected = frontend(identity)
    default = "relative_inverse" if selected == "da2" else "unknown"
    return selected, description.get("kind", default)


def check_frontend(identity, config):
    from explore_a.frontend import name, kind
    if representation(identity) != (name(config), kind(config)):
        raise ValueError("Checkpoint frontend differs from evaluation config")


def record_change(path, previous, requested):
    record = {"time": datetime.now(timezone.utc).isoformat(), "identity_file": path.name,
              "previous": previous, "current": requested}
    with path.with_name("identity_history.jsonl").open("a") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
