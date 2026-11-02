"""Download only the pinned official UniDepthV2 ViT-S inference files; no inference."""
import os

from depth_project.io import write_json
from depth_project.models.unidepth import DIRECTORY, REPOSITORY, REVISION
from depth_project.paths import CHECKPOINTS, PROJECT


def main():
    # All download metadata stays in the workspace rather than the user's home cache.
    os.environ.setdefault("HF_HOME", str(PROJECT / ".runtime/huggingface"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import snapshot_download

    folder = CHECKPOINTS / DIRECTORY
    snapshot_download(REPOSITORY, revision=REVISION, local_dir=folder,
                      allow_patterns=["config.json", "model.safetensors", "README.md"])
    write_json(folder / "source.json", {"repository": REPOSITORY, "revision": REVISION,
                                      "variant": "UniDepthV2 ViT-S/14"})
    print(folder)


if __name__ == "__main__":
    main()
