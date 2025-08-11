import torch

class ResNeXtBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, cardinality, bottleneck_width, stride = 1):
        super().__init__()

        D = bottleneck_width
        C = cardinality
        bottlenect_channel = D*C

        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=bottlenect_channel, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(bottlenect_channel),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(in_channels=bottlenect_channel, out_channels=bottlenect_channel, kernel_size=3, stride=stride, padding=1, groups=C, bias=False),
            torch.nn.BatchNorm2d(bottlenect_channel),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(in_channels=bottlenect_channel, out_channels=out_channels, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )
        

        if stride>1 or in_channels != out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = None

    def forward(self, x):
        identity = x if self.shortcut is None else self.shortcut(x)

        out = self.block(x)
        return out + identity