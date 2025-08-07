import torch

from block.separable_conv import SeparableConv2D

class ExitFlow(torch.nn.Module):
    def __init__(self, num_classes = 1000):
        super().__init__()

        self.relu1 = torch.nn.ReLU(inplace=False)
        self.conv1 = SeparableConv2D(in_channels=728, out_channels=728)

        self.relu2 = torch.nn.ReLU(inplace=False)
        self.conv2 = SeparableConv2D(in_channels=728, out_channels=1024)

        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.shortcut = torch.nn.Sequential(
            torch.nn.Conv2d(728, 1024, kernel_size=1, stride=2, bias=False),
            torch.nn.BatchNorm2d(1024)
        )

        self.conv3 = SeparableConv2D(in_channels=1024, out_channels=1536)
        self.relu3 = torch.nn.ReLU(inplace=False)

        self.conv4 = SeparableConv2D(in_channels=1536, out_channels=2048)
        self.relu4 = torch.nn.ReLU(inplace=False)

        self.gap = torch.nn.AdaptiveAvgPool2d((1,1))
        self.fc = torch.nn.Linear(2048, num_classes)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu1(x)
        out = self.conv1(out)
        out = self.relu2(out)
        out = self.conv2(out)
        out = self.maxpool(out)

        out += identity

        out = self.relu3(self.conv3(out))
        out = self.relu4(self.conv4(out))

        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out