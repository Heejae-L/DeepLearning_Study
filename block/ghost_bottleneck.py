import torch

from block.ghost_module import GhostModule

class SqueezeExcite(torch.nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.se = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(channels, hidden, 1, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(hidden, channels, 1, bias=True),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class Gbneck(torch.nn.Module):
    def __init__(self, in_channels, out_channels, expansion_size, se = False, stride=1):
        super().__init__()

        # Ghost module2
        self.module1 = torch.nn.Sequential(
            GhostModule(in_channels,expansion_size),
            torch.nn.ReLU(inplace=True)
        )

        # DWConv stride=2
        if stride>1 :
            self.dwconv = torch.nn.Sequential(
                torch.nn.Conv2d(expansion_size, expansion_size, kernel_size=3, 
                                groups=expansion_size, stride=stride, padding=1),
                torch.nn.BatchNorm2d(expansion_size)
            )
        else:
            self.dwconv=torch.nn.Identity()

        # Squeeze and Excitation = True
        self.se = SqueezeExcite(expansion_size) if se else torch.nn.Identity()

        # Ghost module2
        self.module2 = torch.nn.Sequential(
            GhostModule(expansion_size, out_channels),
        )

        # Shortcut
        if stride == 1 and in_channels == out_channels:
            self.shortcut = torch.nn.Identity()
        else:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False),  # depthwise
                torch.nn.BatchNorm2d(in_channels),
                torch.nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),  # pointwise
                torch.nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.module1(x)
        out = self.dwconv(out)
        out = self.se(out) 
        out = self.module2(out)
        out = out + identity

        return out
        