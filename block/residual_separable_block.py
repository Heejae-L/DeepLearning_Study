import torch
from block.separable_conv import SeparableConv2D

class ResidualSeparableBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, enter_relu=True):
        super().__init__()
        
        self.enter_relu = enter_relu
        self.conv1 = SeparableConv2D(in_channels=in_channels, out_channels=out_channels)
        self.relu = torch.nn.ReLU(inplace=False)
        self.conv2 = SeparableConv2D(in_channels=out_channels, out_channels=out_channels)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.shortcut = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2, bias=False),
            torch.nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = self.shortcut(x)
        if self.enter_relu:
            x = self.relu(x)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.maxpool(out)

        return out + identity

