"""Project locations; input datasets are shared across projects."""
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
DATASETS = Path.home() / "Documents/ResourcesLow/Datasets"
CHECKPOINTS = PROJECT / "checkpoints"
OUTPUTS = PROJECT / "outputs"
THIRD_PARTY = PROJECT / "third_party"

SMALL = {
    "encoder": "vits",
    "features": 64,
    "out_channels": [48, 96, 192, 384],
}
