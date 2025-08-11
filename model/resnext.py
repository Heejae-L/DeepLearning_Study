import torch

from block.resnext_block import ResNeXtBlock

class ResNeXt(torch.nn.Module):
    def __init__(self, num_classes=1000, config=(3, 4, 6, 3), cardinality=32, bottleneck_width=4):
        super().__init__()

        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width

        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),  # padding=3 권장
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
        )
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.in_channels = 64
        self.layer2 = self.make_conv(out_channels=256,  num_blocks=config[0], stride=1)
        self.layer3 = self.make_conv(out_channels=512,  num_blocks=config[1], stride=2)
        self.layer4 = self.make_conv(out_channels=1024, num_blocks=config[2], stride=2)
        self.layer5 = self.make_conv(out_channels=2048, num_blocks=config[3], stride=2)

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(2048, num_classes)

    def make_conv(self, out_channels, num_blocks, stride):

        layers = []

        layers.append(
            ResNeXtBlock(
                in_channels=self.in_channels,
                out_channels=out_channels,
                cardinality=self.cardinality,
                bottleneck_width=self.bottleneck_width,
                stride=stride
            )
        )
        self.in_channels = out_channels

        for i in range(num_blocks-1):
            layers.append(
                ResNeXtBlock(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    cardinality=self.cardinality,
                    bottleneck_width=self.bottleneck_width,
                    stride=1
                )
            )
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        out = self.stem(x)
        out = self.maxpool(out)

        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def resnext50_32x4d(num_classes=1000):
    return ResNeXt(num_classes=num_classes, config=(3,4,6,3), cardinality=32, bottleneck_width=4)

def resnext101_32x4d(num_classes=1000):
    return ResNeXt(num_classes=num_classes, config=(3,4,23,3), cardinality=32, bottleneck_width=4)
