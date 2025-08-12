import torch

from block.mobilenetv2_block import InvertedResidualBlock

class MobilenetV2(torch.nn.Module):
    def __init__(self, num_classes=1000, expansion_rate = 6):
        super().__init__()

        self.expansion_rate = expansion_rate

        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU6(inplace=True)
        )

        self.bottleneck1 = InvertedResidualBlock(32, 16, 1, 1)
        self.bottleneck2 = self.make_conv(16, 24, 2, 2)
        self.bottleneck3 = self.make_conv(24, 32, 3, 2)
        self.bottleneck4 = self.make_conv(32, 64, 4, 2)
        self.bottleneck5 = self.make_conv(64, 96, 3, 1)
        self.bottleneck6 = self.make_conv(96, 160, 3, 2)
        self.bottleneck7 = self.make_conv(160, 320, 1, 1)

        self.pconv = torch.nn.Sequential(
            torch.nn.Conv2d(320, 1280, kernel_size=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(1280),
            torch.nn.ReLU6(inplace=True)
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1,1)),
            torch.nn.Conv2d(1280, num_classes, kernel_size=1, bias=False)
        )
    
    def make_conv(self, in_channels, out_channels, num_blocks, stride):
        layers = []

        layers.append(
            InvertedResidualBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    expansion_rate=self.expansion_rate,
                    stride=stride
                )
        )

        for i in range(num_blocks-1):
            layers.append(
                InvertedResidualBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    expansion_rate=self.expansion_rate,
                    stride=1
                )
            )
        return torch.nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.stem(x)

        out = self.bottleneck1(out)
        out = self.bottleneck2(out)
        out = self.bottleneck3(out)
        out = self.bottleneck4(out)
        out = self.bottleneck5(out)
        out = self.bottleneck6(out)
        out = self.bottleneck7(out)

        out = self.pconv(out)

        out = self.classifier(out)
        out = out.view(out.size(0), -1)

        return out