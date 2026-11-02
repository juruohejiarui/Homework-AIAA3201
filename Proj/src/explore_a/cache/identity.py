"""Numerical cache identity excludes experiment membership, splits and sampling weights."""


def numeric_spec(spec):
    """Normalize legacy schema 3 and shared schema 4 without reinterpreting predictions."""
    if spec.get("schema") not in (3, 4):
        raise ValueError("Unsupported cache identity schema")
    data = spec["data"]
    result = {key: spec[key] for key in
            ("geometry", "flows", "teacher", "storage", "attention", "precision", "weights")} | {
                "data": {key: data[key] for key in ("canvas_kitti", "canvas_indoor")}}
    if "frontend" in spec:
        result["frontend"] = spec["frontend"]
    return result


def assert_compatible(stored, requested):
    if numeric_spec(stored) != numeric_spec(requested):
        raise ValueError("Cache models/preprocessing differ from the requested numeric recipe")
