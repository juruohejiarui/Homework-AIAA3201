"""Read calibrated ScanNet/NYU exports and matching KITTI raw/depth frames."""
import numpy as np

from depth_project.data.association import natural_order, pair_frame_ids
from depth_project.data.protocol import NYU_SCENES


def prepare_exported(source, dataset):
    names = NYU_SCENES if dataset == "nyuv2" else sorted(path.name for path in source.iterdir() if path.is_dir())[:100]
    sequences = []
    for name in names:
        scene = source / name
        image_root = scene / ("color" if dataset == "scannet" else "rgb")
        images = list(image_root.glob("*.jpg")) + list(image_root.glob("*.png"))
        pairs = pair_frame_ids(images, (scene / "depth").glob("*.png"))[:510]
        frames = []
        for index, (image, depth) in enumerate(pairs):
            row = {"image": str(image.resolve()), "gt": str(depth.resolve()), "source_index": index}
            if dataset == "scannet":
                row["K"] = np.loadtxt(scene / "intrinsic/intrinsic_depth.txt")[:3, :3].tolist()
                row["pose"] = np.loadtxt(scene / "pose" / (image.stem + ".txt")).tolist()
                row["pose_convention"] = "camera_to_world"
            frames.append(row)
        if not frames:
            raise FileNotFoundError(f"No aligned RGB/depth pairs in {scene}")
        sequences.append({"name": name, "frames": frames})
    return sequences


def prepare_kitti(source, depth_root):
    sequences = []
    for scene in sorted(path for path in depth_root.iterdir() if path.is_dir()):
        depths = natural_order((scene / "proj_depth/groundtruth/image_02").glob("*.png"))[:510]
        image_root = source / scene.name[:10] / scene.name / "image_02/data"
        frames = [{"image": str((image_root / depth.name).resolve()), "gt": str(depth.resolve()),
                   "source_index": index} for index, depth in enumerate(depths)]
        if frames:
            sequences.append({"name": scene.name, "frames": frames})
    return sequences
