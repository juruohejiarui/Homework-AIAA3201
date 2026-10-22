"""Prepare Bonn sequence metadata without copying raw RGB or depth."""
from depth_project.data.association import associate_tum


def prepare_bonn(source):
    scenes = sorted(path.parent for path in source.rglob("rgb.txt")
                    if (path.parent / "depth.txt").exists())
    sequences = []
    for scene in scenes:
        frames = []
        for image, depth, association in associate_tum(scene)[:510]:
            frames.append({"image": str(image.resolve()), "gt": str(depth.resolve()), **association})
        sequences.append({"name": scene.name, "frames": frames})
    return sequences
