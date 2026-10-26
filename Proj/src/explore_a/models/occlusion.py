"""Independent visibility student distilled from MFTIQ; not an occlusion GT classifier."""
from torch import nn

from explore_a.geometry.features import occlusion_features
from explore_a.models.blocks import SmallUNet


class OcclusionStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = SmallUNet(13, 1)

    def forward(self, data, geometry):
        return self.network(occlusion_features(data, geometry))
