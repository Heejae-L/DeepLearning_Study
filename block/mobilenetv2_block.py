import torch

class InvertedResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, expansion_rate=6, stride=1):
        super().__init__()
        
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.intermediate_channels = (int)(in_channels * expansion_rate)

        self.expention = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=self.intermediate_channels, kernel_size=1, stride = 1, bias=False),
            torch.nn.BatchNorm2d(self.intermediate_channels),
            torch.nn.ReLU6(inplace=True)
        )

        self.dconv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.intermediate_channels, out_channels=self.intermediate_channels, kernel_size=3, groups=self.intermediate_channels, stride=stride, padding=1, bias=False),
            torch.nn.BatchNorm2d(self.intermediate_channels),
            torch.nn.ReLU6(inplace=True)
        )

        self.projection = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.intermediate_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.expention(x)
        out = self.dconv(out)
        out = self.projection(out)

        if self.stride == 1 and self.in_channels == self.out_channels:
            out = x + out
        
        return out


