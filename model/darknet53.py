import torch

from block.darknet53_block import DarknetBlock
from block.darknet53_downsample_block import DownSampleBlock

class Darknet53(torch.nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(0.1, inplace=True),

            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(0.1, inplace=True)
        )

        self.block1 = DarknetBlock(64)
        self.down1 = DownSampleBlock(64, 128)
        self.block2 = torch.nn.Sequential(*[DarknetBlock(128) for _ in range(2)])
        self.down2 = DownSampleBlock(128, 256)
        self.block3 = torch.nn.Sequential(*[DarknetBlock(256) for _ in range(8)])
        self.down3 = DownSampleBlock(256, 512)
        self.block4 = torch.nn.Sequential(*[DarknetBlock(512) for _ in range(8)])
        self.down4 = DownSampleBlock(512, 1024)
        self.block5 = torch.nn.Sequential(*[DarknetBlock(1024) for _ in range(4)])

        self.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1,1)),
            torch.nn.Conv2d(1024, num_classes, kernel_size=1, bias=True)
        )
    
    def forward(self, x):
        out = self.stem(x)

        out = self.block1(out)
        out = self.down1(out)

        out = self.block2(out)
        out = self.down2(out)

        out = self.block3(out)
        out = self.down3(out)

        out = self.block4(out)
        out = self.down4(out)

        out = self.block5(out)

        out = self.classifier(out)
        out = out.view(out.size(0),-1)

        return out