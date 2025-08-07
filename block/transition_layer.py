import torch

class TransitionLayer(torch.nn.Module):
    def __init__(self, in_channels, theta=0.5):
        super().__init__()
        out_channels=int(in_channels*theta)
        self.transition = torch.nn.Sequential(
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size=1, bias=False),
            torch.nn.AvgPool2d(kernel_size=2, stride=2)
        )
    def forward(self, x):
        return self.transition(x)