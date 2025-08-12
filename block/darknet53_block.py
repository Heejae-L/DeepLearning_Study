import torch

from block.darknet_residual_block import DarknetResidualBlock

class DarknetBlock(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        intermediate_channels = (int)(in_channels//2)
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=intermediate_channels, kernel_size=1, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(intermediate_channels),
            torch.nn.LeakyReLU(0.1, inplace=True),

            torch.nn.Conv2d(in_channels=intermediate_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.LeakyReLU(0.1, inplace=True),
        )

        self.residual = DarknetResidualBlock(in_channels=in_channels, out_channels=in_channels)
    
    def forward(self, x):
        out = self.features(x)
        out = out + x
        return out