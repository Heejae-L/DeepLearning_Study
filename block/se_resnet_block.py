import torch

from block.se_block import SEBlock

class SERenetBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, reduction_ratio=16):
        super().__init__()
        self.residual = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels)
        )

        self.SE = SEBlock(in_channels=out_channels, reduction_ratio=reduction_ratio)

        if stride > 1 or in_channels!=out_channels :
            self.shorcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), 
                torch.nn.BatchNorm2d(out_channels)
                )
        else:
            self.shorcut = torch.nn.Identity()
        
        self.relu = torch.nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = self.shorcut(x)
        out = self.residual(x)
        out = out * self.SE(out)
        out = out + identity
        out = self.relu(out)

        return out
