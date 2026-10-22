"""Frozen VDA code-selection rules, separate from dataset decoding."""
CONFIG = {
    "sintel": {"folder": "MPI-Sintel", "crop": [0, 436, 0, 1024], "max_depth": 70, "factor": 1},
    "bonn": {"folder": "Bonn-RGBD", "crop": [0, 480, 0, 640], "max_depth": 10, "factor": 5000},
    "nyuv2": {"folder": "nyu_depth_v2", "crop": [45, 471, 41, 601], "max_depth": 10, "factor": 6000},
    "scannet": {"folder": "ScanNet", "crop": [8, -8, 11, -11], "max_depth": 10, "factor": 1000},
    "kitti": {"folder": "KITTI", "crop": [0, 374, 0, 1242], "max_depth": 80, "factor": 256},
}
NYU_SCENES = ["basement_0001a", "bookstore_0001a", "cafe_0001a", "classroom_0001a",
              "kitchen_0003", "office_0004", "playroom_0002", "study_0002"]
SETTINGS = ["long", "short", "tae", "length-110", "length-192", "length-300", "length-400", "length-500"]


def select_sequences(manifest, setting):
    dataset = manifest["dataset"]
    sequences = manifest["sequences"]
    length = 500
    if setting.startswith("length-"):
        length = int(setting.split("-")[1])
        if dataset not in {"bonn", "scannet", "nyuv2"}:
            raise ValueError("Length curves require Bonn, ScanNet, or NYUv2")
        if dataset in {"bonn", "scannet"}:
            sequences = [sequence for sequence in sequences if len(sequence["frames"]) >= 500][:10]
    if setting == "tae":
        if dataset != "scannet":
            raise ValueError("Original TAE requires ScanNet camera metadata")
        sequences = sequences[:20]
    selected = []
    for sequence in sequences:
        frames = sequence["frames"]
        if setting == "tae":
            frames = frames[10:180]
        elif setting == "short" and dataset == "bonn":
            frames = frames[30:140]
        elif setting == "short" and dataset == "scannet":
            frames = frames[:270:3]
        elif setting == "short":
            frames = frames[:100 if dataset == "sintel" else 110]
        else:
            frames = frames[:100 if dataset == "sintel" else length]
        required = length if setting.startswith("length-") else 170 if setting == "tae" else 2
        if len(frames) < required:
            raise ValueError(f"Insufficient frames: {sequence['name']}")
        selected.append({"name": sequence["name"], "frames": frames})
    if not selected:
        raise ValueError("No sequences satisfy the selected protocol")
    return selected
