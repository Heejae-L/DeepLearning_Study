import torch

from block.mbconv_block import MBConvBlock
class EfficientNet(torch.nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.SiLU(inplace=True)
        )

        self.stage2 = MBConvBlock(32,16,stride=2,kernel_size=3,expand_ratio=1)
        self.stage3 = torch.nn.Sequential(
            MBConvBlock(16,24,stride=1,kernel_size=3),
            MBConvBlock(24,24,kernel_size=3, stride=1)
        )
        self.stage4 = torch.nn.Sequential(
            MBConvBlock(24,40,kernel_size=5,stride=2),
            MBConvBlock(40,40,kernel_size=5,stride=1),
        )
        self.stage5 = torch.nn.Sequential(
            MBConvBlock(40,80, kernel_size=3, stride=2),
            MBConvBlock(80,80,kernel_size=3, stride=1),
            MBConvBlock(80,80,kernel_size=3, stride=1)
        )
        self.stage6 = torch.nn.Sequential(
            MBConvBlock(80,112, kernel_size=5, stride=2),
            MBConvBlock(112,112,kernel_size=5, stride=1),
            MBConvBlock(112,112,kernel_size=5, stride=1)
        )
        self.stage7 = torch.nn.Sequential(
            MBConvBlock(112,192, kernel_size=5, stride=1),
            MBConvBlock(192,192,kernel_size=5, stride=1),
            MBConvBlock(192,192,kernel_size=5, stride=1),
            MBConvBlock(192,192,kernel_size=5, stride=1)
        )
        self.stage8 = torch.nn.Sequential(
            MBConvBlock(192, 320, kernel_size=3, stride=2)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Conv2d(320, 1280, kernel_size=1, stride=1),
            torch.nn.AdaptiveAvgPool2d((1,1)),
            torch.nn.Flatten(),  
            torch.nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        out = self.stem(x)

        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.stage5(out)
        out = self.stage6(out)
        out = self.stage7(out)
        out = self.stage8(out)

        out = self.classifier(out)

        return out