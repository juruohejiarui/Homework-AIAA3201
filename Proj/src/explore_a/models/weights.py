"""Report model paths and official URLs; checking never downloads or constructs a model."""
from pathlib import Path

URLS = {
    "depthpro": "https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt",
    "raft": "https://download.pytorch.org/models/raft_large_C_T_SKHT_V2-ff5fadd5.pth",
    "mftiq": "https://cmp.felk.cvut.cz/~serycjon/MFTIQ/materials/UOM_bs4_200k.pth",
    "resnet": "https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
    "dinov2": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/"
              "dinov2_vits14_reg4_pretrain.pth",
}
EXPECTED_BYTES = {"raft": 21106607, "mftiq": 108034251, "resnet": 102540417,
                  "dinov2": 88291785}


def check_weights(config, required=None):
    missing = []
    for name in required or URLS:
        path = Path(config["weights"][name])
        available = path.is_file() and path.stat().st_size > 0
        if available and name in EXPECTED_BYTES:
            available = path.stat().st_size == EXPECTED_BYTES[name]
        print(f"{'FOUND' if available else 'MISSING'} {name}: {path}")
        if not available:
            print("  Download:", URLS[name])
            missing.append(name)
    if required is None or "mftiq" in required:
        path = Path(config["paths"]["dinov2_source"])
        if not (path / "hubconf.py").is_file():
            print("MISSING DINOv2 source:", path,
                  "\n  Source: https://github.com/facebookresearch/dinov2")
            missing.append("dinov2_source")
    if missing:
        raise FileNotFoundError("Download the missing assets manually: " + ", ".join(missing))
