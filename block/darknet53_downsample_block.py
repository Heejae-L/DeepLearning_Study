import torch

class DownSampleBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        out = self.down(x)
        return out