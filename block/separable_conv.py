import torch

class SeparableConv2D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.depthwise = torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, groups=in_channels, bias=False)
        
        self.pointwise = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.bn = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn(out)

        return out