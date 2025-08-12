import torch

class DarknetResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(0.1, inplace=True),
        )
        self.layer2 = torch.nn.Sequential(

            torch.nn.Conv2d(in_channels=out_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.LeakyReLU(0.1, inplace=True),
        )