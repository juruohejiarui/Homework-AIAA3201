"""Full manifests are used only for explicit copy-before-delete migrations."""
import os
from pathlib import Path

from depth_project.io import file_hash


def inventory(root):
    root = Path(root)
    entries = {}

    def visit(directory):
        for entry in sorted(os.scandir(directory), key=lambda item: item.name):
            path = Path(entry.path)
            relative = str(path.relative_to(root))
            if entry.is_symlink():
                entries[relative] = {"kind": "symlink", "target": os.readlink(path)}
            elif entry.is_dir(follow_symlinks=False):
                entries[relative] = {"kind": "directory"}
                visit(path)
            elif entry.is_file(follow_symlinks=False):
                entries[relative] = {"kind": "file", "bytes": entry.stat().st_size, "sha256": file_hash(path)}
            else:
                raise ValueError(f"Unsupported special file: {path}")

    visit(root)
    return entries
