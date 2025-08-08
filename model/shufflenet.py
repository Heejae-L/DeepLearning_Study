import torch

from block.shufflenet_block import ShuffleNetUnit

class ShuffleNet(torch.nn.Module):
    def __init__(self, groups, config = [], num_classes=1000):
        super().__init__()
        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=24, kernel_size=3, stride=2, padding=1),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        num_channel1, num_channel2, num_channel3 = config

        self.stage2 = torch.nn.Sequential(
            ShuffleNetUnit(24, num_channel1, stride=2, groups=groups),

            ShuffleNetUnit(num_channel1, num_channel1, stride=1, groups=groups),
            ShuffleNetUnit(num_channel1, num_channel1, stride=1, groups=groups),
            ShuffleNetUnit(num_channel1, num_channel1, stride=1, groups=groups)
        )
        self.stage3 = torch.nn.Sequential(
            ShuffleNetUnit(num_channel1, num_channel2, stride=2, groups=groups),

            ShuffleNetUnit(num_channel2, num_channel2, stride=1, groups=groups),
            ShuffleNetUnit(num_channel2, num_channel2, stride=1, groups=groups),
            ShuffleNetUnit(num_channel2, num_channel2, stride=1, groups=groups),
            ShuffleNetUnit(num_channel2, num_channel2, stride=1, groups=groups),
            ShuffleNetUnit(num_channel2, num_channel2, stride=1, groups=groups),
            ShuffleNetUnit(num_channel2, num_channel2, stride=1, groups=groups),
            ShuffleNetUnit(num_channel2, num_channel2, stride=1, groups=groups)
        )
        self.stage4 = torch.nn.Sequential(
            ShuffleNetUnit(num_channel2, num_channel3, stride=2, groups=groups),

            ShuffleNetUnit(num_channel3, num_channel3, stride=1, groups=groups),
            ShuffleNetUnit(num_channel3, num_channel3, stride=1, groups=groups),
            ShuffleNetUnit(num_channel3, num_channel3, stride=1, groups=groups)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1,1)),
            torch.nn.Conv2d(num_channel3, num_classes, kernel_size=1, bias=True)
        )
    def forward(self, x):

        out = self.stem(x)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.classifier(out)
        out = torch.flatten(out, 1)


        return out