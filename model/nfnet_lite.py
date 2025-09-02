import torch

from block.nf_wsconv import ScaledWSConv2d
from block.nf_block import NFBlock

class NFNetLite(torch.nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.stem = torch.nn.Sequential(
            ScaledWSConv2d(3, 16, kernel_size=3),
            ScaledWSConv2d(16, 32, kernel_size=3),
            ScaledWSConv2d(32, 64, kernel_size=3),
            ScaledWSConv2d(64, 128, kernel_size=3),
        )


        # Stage 1: 128 -> 128 (mid 축소)
        self.stage1 = NFBlock(128, 64, 128)  # was (128,128,256) in big model

        # Stage 2: 128 -> 256
        self.stage2 = torch.nn.Sequential(
            NFBlock(128, 128, 256),   # down/up as your NFBlock defines
            NFBlock(256, 128, 256),
        )

        # Stage 3: 256 -> 768
        self.stage3 = torch.nn.Sequential(
            NFBlock(256, 384, 768),
            NFBlock(768, 384, 768),
            NFBlock(768, 384, 768),
        )

        # Stage 4: 768 -> 768
        self.stage4 = torch.nn.Sequential(
            NFBlock(768, 384, 768),
            NFBlock(768, 384, 768),
        )

        # Classifier head: 입력 채널 768로 맞춤
        self.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Conv2d(768, num_classes, kernel_size=1, bias=True),
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
