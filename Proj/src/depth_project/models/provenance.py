"""Content identities for the same inference code on host and DGX."""
import hashlib
from depth_project.paths import PROJECT


def source_signature():
    digest = hashlib.sha256()
    for package in ("depth_project", "explore_a"):
        for path in sorted((PROJECT/"src"/package).rglob("*.py")):
            digest.update(str(path.relative_to(PROJECT/"src")).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def shared_signature():
    """Adding a new adapter does not invalidate completed common baseline results."""
    files = ["depth_project/models/api.py", "depth_project/models/__init__.py", "depth_project/models/da2.py",
             "depth_project/models/attention.py", "depth_project/models/provenance.py",
             "depth_project/models/serialization.py", "depth_project/io.py",
             "explore_a/frontend.py", "explore_a/data/reading.py", "explore_a/training/prediction.py"]
    roots = [PROJECT/"src/explore_a"/name for name in ("geometry", "online", "remote_online")]
    paths = [PROJECT/"src"/name for name in files]
    paths += [p for root in roots for p in root.rglob("*.py")]
    paths += [PROJECT/"src/explore_a/evaluation"/f"{name}.py" for name in
              ("session", "transport", "alignment", "sintel_ac", "accumulator", "state")]
    paths += list((PROJECT/"third_party/Depth-Anything-V2").rglob("*.py"))
    digest=hashlib.sha256()
    for path in sorted(set(paths)):
        digest.update(str(path.relative_to(PROJECT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
