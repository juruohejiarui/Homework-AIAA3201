"""Evaluation-only equal-weight ensembles of registered prediction fields."""
from pathlib import Path

import cv2
import numpy as np


def plan(entries, selected=None):
    """Resolve dependencies once, allowing nested ensembles but never cycles or weights."""
    registry = {entry["name"]: entry for entry in entries}
    requested = list(registry) if selected is None else list(selected)
    leaves, ensembles, visiting, done = [], [], set(), set()

    def visit(name):
        if name not in registry:
            raise ValueError(f"Unknown ensemble/model member: {name}")
        if name in visiting:
            raise ValueError(f"Cyclic ensemble membership: {name}")
        if name in done:
            return
        entry = registry[name]
        if entry.get("kind", "fusion") == "ensemble":
            members = entry.get("members")
            if (not isinstance(members, list) or not members
                    or any(not isinstance(m, str) for m in members)
                    or len(set(members)) != len(members)):
                raise ValueError(f"Ensemble {name} needs a nonempty list of unique member names")
            visiting.add(name)
            for member in members:
                visit(member)
            visiting.remove(name)
            ensembles.append(entry)
        else:
            leaves.append(entry)
        done.add(name)

    for name in requested:
        visit(name)
    return leaves, ensembles, requested


def mean_fields(fields):
    """Average raw fields, before GT fitting, clipping or metric conversion."""
    if not fields:
        raise ValueError("Cannot average an empty ensemble")
    shape = fields[0].shape
    if len(shape) != 2 or any(field.shape != shape for field in fields):
        raise ValueError("Ensemble members must return matching HW fields")
    total = np.zeros(shape, np.float64)
    for field in fields:
        if not np.isfinite(field).all():
            raise ValueError("Nonfinite ensemble member prediction")
        total += field
    return (total / len(fields)).astype(np.float32)


class EnsembleFields:
    """Stream frame ensembles; spool only when a member needs a complete sequence."""
    def __init__(self, root, definitions, deferred=False):
        self.root, self.deferred = Path(root) / "ensemble", deferred
        self.definitions = {e["name"]: e for e in definitions}
        self.leaf_names = {m for e in definitions for m in e["members"]} - self.definitions.keys()
        self.representations, self.current = {}, {}

    def store(self, name, index, field, representation, shape):
        if name not in self.leaf_names:
            return
        field = np.asarray(field, np.float32)
        if field.ndim != 2:
            raise ValueError("Ensemble members must return HW fields")
        if field.shape != shape:
            field = cv2.resize(field, (shape[1], shape[0]))
        prior = self.representations.setdefault(name, representation)
        if prior != representation:
            raise ValueError(f"Ensemble member changed representation: {name}")
        if self.deferred:
            path = self.root / name / f"{index}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, field, allow_pickle=False)
        else:
            self.current[name] = field

    def predictions(self, index, names):
        values = {}

        def get(name):
            if name not in values:
                if name in self.definitions:
                    fields = [get(member) for member in self.definitions[name]["members"]]
                    if len({representation for _, representation in fields}) != 1:
                        raise ValueError(f"Ensemble {name} mixes prediction representations")
                    values[name] = mean_fields([field for field, _ in fields]), fields[0][1]
                else:
                    field = (np.load(self.root/name/f"{index}.npy", allow_pickle=False)
                             if self.deferred else self.current[name])
                    values[name] = field, self.representations[name]
            return values[name]

        result = {name: get(name) for name in names}
        self.current.clear()
        return result
