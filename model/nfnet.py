import torch

from block.nf_wsconv import ScaledWSConv2d
from block.nf_block import NFBlock

class NFNet(torch.nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = torch.nn.Sequential(
            ScaledWSConv2d(3, 16, kernel_size=3, stride=2),
            ScaledWSConv2d(16, 32, kernel_size=3),
            ScaledWSConv2d(32, 64, kernel_size=3),
            ScaledWSConv2d(64, 128, kernel_size=3, stride=2)
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

        self.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1,1)),
            torch.nn.Conv2d(1536, num_classes, kernel_size=1, bias=True)
        )

    def forward(self, x):
        out = self.stem(x)

        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)

        out = self.classifier(out)
        out = torch.flatten(out, 1)

        return out