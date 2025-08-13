import torch

from block.ghost_bottleneck import Gbneck

class GhostNet(torch.nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(inplace=True)
        )

        self.stage1 = torch.nn.Sequential(
            Gbneck(in_channels=16, out_channels=16, expansion_size=16),
            Gbneck(in_channels=16, out_channels=24, expansion_size=48, stride=2)
        )

        self.stage2 = torch.nn.Sequential(
            Gbneck(in_channels=24, out_channels=24, expansion_size=72),
            Gbneck(in_channels=24, out_channels=40, expansion_size=72, stride=2)
        )

        self.stage3 = torch.nn.Sequential(
            Gbneck(in_channels=40, out_channels=40, expansion_size=120, se=True),
            Gbneck(in_channels=40, out_channels=80, expansion_size=240, stride=2)
        )

        self.stage4 = torch.nn.Sequential(
            Gbneck(in_channels=80, out_channels=80, expansion_size=200),
            Gbneck(in_channels=80, out_channels=80, expansion_size=184),
            Gbneck(in_channels=80, out_channels=80, expansion_size=184),
            Gbneck(in_channels=80, out_channels=112, expansion_size=480, se=True),
            Gbneck(in_channels=112, out_channels=112, expansion_size=672, se=True),
            Gbneck(in_channels=112, out_channels=160, expansion_size=672, se=True, stride=2)
        )

        self.stage5 = torch.nn.Sequential(
            Gbneck(in_channels=160, out_channels=160, expansion_size=960),
            Gbneck(in_channels=160, out_channels=160, expansion_size=960, se=True),
            Gbneck(in_channels=160, out_channels=160, expansion_size=960),
            Gbneck(in_channels=160, out_channels=160, expansion_size=960, se=True),
            torch.nn.Conv2d(160, 960, kernel_size=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(960),
            torch.nn.ReLU(inplace=True)
        )

        
        self.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(960, 1280, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(1280),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.2),
            torch.nn.Flatten(),                 # (B, 1280, 1, 1) -> (B, 1280)
            torch.nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        out = self.stem(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.stage5(out)
        out = self.classifier(out)
        return out
