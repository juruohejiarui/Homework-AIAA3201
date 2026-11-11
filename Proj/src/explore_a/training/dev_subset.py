"""Deterministic, source-balanced continuous development clips shared by all consumers."""
from collections import defaultdict
from hashlib import sha256
import json
import math

PROTOCOL = "continuous-sqrt-332-v1"
SOURCE_WEIGHTS = {"kitti": 3, "nyuv2": 3, "tum": 2}


def quotas(capacities, target, weights, minimum=0):
    """Capped largest-remainder allocation, with stable ties and optional coverage."""
    target = min(target, sum(capacities.values()))
    floor = minimum if target >= len(capacities)*minimum else 0
    total = sum(weights[k] for k in sorted(weights))
    ideal = {k: target*weights[k]/total for k in capacities}
    result = {k: min(capacities[k], max(floor, int(ideal[k]))) for k in capacities}
    while sum(result.values()) < target:
        key = max((k for k in capacities if result[k] < capacities[k]),
                  key=lambda k: (ideal[k]-result[k], k))
        result[key] += 1
    while sum(result.values()) > target:
        key = max((k for k in capacities if result[k] > floor),
                  key=lambda k: (result[k]-ideal[k], k))
        result[key] -= 1
    return result


def rank(seed, value):
    return sha256(f"{PROTOCOL}:{seed}:{value}".encode()).hexdigest()


def chains(rows):
    """Only exact existing pair endpoints connect; absent metadata means isolated pairs."""
    groups = defaultdict(list)
    for row in rows:
        groups[row.get("camera", "")].append(row)
    result = []
    for camera, group in sorted(groups.items()):
        def endpoint(row, previous):
            prefix = "previous" if previous else "current"
            return str(row.get(prefix+"_rgb", row.get(prefix+"_id", prefix+row["id"])))
        starts = {endpoint(r, True): r for r in group}
        if len(starts) != len(group):
            raise ValueError("Duplicate development pair start within a camera")
        ends = {endpoint(r, False) for r in group}
        roots = sorted((r for r in group if endpoint(r, True) not in ends), key=lambda r: r["id"])
        seen = set()
        for root in roots:
            chain, row = [], root
            while row is not None and row["id"] not in seen:
                chain.append(row)
                seen.add(row["id"])
                row = starts.get(endpoint(row, False))
            result.append((camera, chain))
        if len(seen) != len(group):
            raise ValueError("Cyclic development pair endpoints")
    return result


def clips(rows, target, seed):
    by_camera = defaultdict(list)
    for camera, chain in chains(rows):
        by_camera[camera].append(chain)
    capacities = {k: sum(map(len, v)) for k, v in by_camera.items()}
    allocation = quotas(capacities, target, capacities, minimum=1)
    selected = []
    for camera, segments in sorted(by_camera.items()):
        remaining = allocation[camera]
        segments.sort(key=lambda s: (-len(s), rank(seed, s[0]["id"])))
        # Use one/two time strata on a long chain; more chains only when gaps require it.
        split = remaining >= 64 and bool(segments) and len(segments[0]) >= remaining
        if split:
            chain = segments.pop(0)
            middle = len(chain)//2
            segments = [chain[:middle], chain[middle:]]
        if not split and sum(len(s) for s in segments[:1]) >= remaining:
            segments = segments[:1]
        elif not split:
            used, capacity = [], 0
            for segment in segments:
                used.append(segment)
                capacity += len(segment)
                if capacity >= remaining:
                    break
            segments = used
        for index, chain in enumerate(segments):
            if not remaining:
                break
            later = sum(len(s) for s in segments[index+1:])
            wanted = max(remaining-later, math.ceil(remaining/(len(segments)-index)))
            # A single sufficient chain should not become many tiny fragments.
            if target < 64 and len(chain) >= remaining:
                wanted = remaining
            take = min(len(chain), remaining, wanted)
            if not take:
                continue
            offset = int(rank(seed, chain[0]["id"]), 16) % (len(chain)-take+1)
            part = chain[offset:offset+take]
            clip_id = f"{PROTOCOL}:{part[0]['id']}:{part[-1]['id']}"
            selected.extend({**r, "evaluation_clip": clip_id} for r in part)
            remaining -= take
        if remaining:
            raise AssertionError("Continuous clip allocation lost pairs")
    return selected


def select_dev(rows, fraction=1., seed=2026):
    if not 0 < fraction <= 1:
        raise ValueError("training.dev_sample_fraction must be in (0, 1]")
    if fraction == 1 or not rows:
        return rows
    sources = defaultdict(lambda: defaultdict(list))
    for row in rows:
        sources[row["dataset"]][row["sequence"]].append(row)
    capacities = {s: sum(map(len, groups.values())) for s, groups in sources.items()}
    target = max(1, round(len(rows)*fraction))
    weights = {s: SOURCE_WEIGHTS.get(s, 1) for s in sources}
    budgets = quotas(capacities, target, weights)
    result = []
    for source, groups in sorted(sources.items()):
        sizes = {name: len(group) for name, group in groups.items()}
        allocation = quotas(sizes, budgets[source], {k: math.sqrt(v) for k, v in sizes.items()}, 1)
        for name, group in sorted(groups.items()):
            result.extend(clips(group, allocation[name], seed))
    return result


def selection_identity(rows):
    content = [(r["id"], r.get("evaluation_clip")) for r in rows]
    protocol = PROTOCOL if any(r.get("evaluation_clip") for r in rows) else "full-cohort-v1"
    return {"protocol": protocol, "pairs": len(rows),
            "sha256": sha256(json.dumps(content).encode()).hexdigest()}


def selection_record(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row.get("evaluation_clip", row["sequence"])].append(row)
    return {**selection_identity(rows), "sample_ids": [r["id"] for r in rows],
            "clips": [{"id": key, "dataset": group[0]["dataset"],
                       "sequence": group[0]["sequence"], "camera": group[0].get("camera"),
                       "pairs": len(group), "first": group[0]["id"], "last": group[-1]["id"]}
                      for key, group in groups.items()]}
