import torch

from block.mobilev3_block import MobileV3Block

class MobileNetV3(torch.nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(16),
            torch.nn.Hardswish(inplace=True),
            torch.nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(16),
            torch.nn.Hardswish(inplace=True)
        )

        self.stage1 = torch.nn.Sequential(
            MobileV3Block(16,16,kernel_size=3, expansion_size=16, stride=2, se=True, non_linear=torch.nn.ReLU),
            MobileV3Block(16,16,kernel_size=3, expansion_size=16, stride=1, se=True, non_linear=torch.nn.ReLU)
        )

        self.stage2 = torch.nn.Sequential(
            MobileV3Block(16,24,kernel_size=3, expansion_size=72, stride=2, se=False, non_linear=torch.nn.ReLU),
            MobileV3Block(24,24,kernel_size=3, expansion_size=72, stride=1, se=False, non_linear=torch.nn.ReLU)
        )

        self.stage3 = torch.nn.Sequential(
            MobileV3Block(24,24,kernel_size=3, expansion_size=88, stride=1, se=False, non_linear=torch.nn.ReLU)
        )

        self.stage4 = torch.nn.Sequential(
            MobileV3Block(24,40,kernel_size=5, expansion_size=96, stride=2, se=True, non_linear=torch.nn.Hardswish),
            MobileV3Block(40,40,kernel_size=5, expansion_size=96, stride=1, se=True, non_linear=torch.nn.Hardswish)
        )

        self.stage5 = MobileV3Block(40,40,kernel_size=5, expansion_size=240, stride=1, se=True, non_linear=torch.nn.Hardswish)
        self.stage6 = MobileV3Block(40,40,kernel_size=5, expansion_size=240, stride=1, se=True, non_linear=torch.nn.Hardswish)
        self.stage7 = MobileV3Block(40,48,kernel_size=5, expansion_size=120, stride=1, se=True, non_linear=torch.nn.Hardswish)
        self.stage8 = MobileV3Block(48,48,kernel_size=5, expansion_size=144, stride=1, se=True, non_linear=torch.nn.Hardswish)

        self.stage9 = torch.nn.Sequential(
            MobileV3Block(48, 96, kernel_size=5, expansion_size=288, stride=2, se=True, non_linear=torch.nn.Hardswish),
            MobileV3Block(96, 96, kernel_size=5, expansion_size=288, stride=1, se=True, non_linear=torch.nn.Hardswish)
        )

        self.stage10 = MobileV3Block(96, 96, kernel_size=5, expansion_size=576, stride=1, se=True, non_linear=torch.nn.Hardswish)
        self.stage11 = MobileV3Block(96, 96, kernel_size=5, expansion_size=576, stride=1, se=True, non_linear=torch.nn.Hardswish)

        self.classifier = torch.nn.Sequential(
            torch.nn.Conv2d(96, 576, kernel_size=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(576),
            torch.nn.Hardswish(inplace=True),
            torch.nn.AdaptiveAvgPool2d((1,1)),
            torch.nn.Conv2d(576, 1024, kernel_size=1, stride=1, bias=True),
            torch.nn.Conv2d(1024, num_classes, kernel_size=1, stride=1, bias=True),
        )
    def forward(self, x):
        out = self.stem(x)

        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.stage5(out)
        out = self.stage6(out)
        out = self.stage7(out)
        out = self.stage8(out)
        out = self.stage9(out)
        out = self.stage10(out)
        out = self.stage11(out)

        out = self.classifier(out)
        out = out.view(out.size(0), -1) 

        return out