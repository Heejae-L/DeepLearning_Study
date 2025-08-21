import torch

from block.nf_wsconv import ScaledWSConv2d
from block.nf_block import NFBlock

class NFNet(torch.nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = torch.nn.Sequential(
            ScaledWSConv2d(3, 16, kernel_size=3),
            ScaledWSConv2d(16, 32, kernel_size=3),
            ScaledWSConv2d(32, 64, kernel_size=3),
            ScaledWSConv2d(64, 128, kernel_size=3)
        )

        self.stage1 = NFBlock(128, 128, 256)

        self.stage2 = torch.nn.Sequential(
            NFBlock(256, 256, 512),
            NFBlock(512, 256, 512)
        )

        self.stage3 = torch.nn.Sequential(
            NFBlock(512, 768, 1536),
            NFBlock(1536, 768, 1536),
            NFBlock(1536, 768, 1536),
            NFBlock(1536, 768, 1536),
            NFBlock(1536, 768, 1536),
            NFBlock(1536, 768, 1536),
        )

        self.stage4 = torch.nn.Sequential(
            NFBlock(1536, 768, 1536),
            NFBlock(1536, 768, 1536),
            NFBlock(1536, 768, 1536),
        )