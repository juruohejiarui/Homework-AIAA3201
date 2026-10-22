"""Small shared file operations, without dataset or model policy."""
import hashlib
import json
from pathlib import Path

import cv2


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def read_rgb(path):
    image = cv2.imread(str(path))
    if image is None:
        raise OSError(f"Cannot decode image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def file_hash(path, algorithm="sha256"):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()
