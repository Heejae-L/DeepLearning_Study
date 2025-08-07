import torch

from block.separable_conv import SeparableConv2D

class MiddleFlow(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = SeparableConv2D(in_channels=728, out_channels=728)
        self.relu = torch.nn.ReLU(inplace=False)
        self.conv2 = SeparableConv2D(in_channels=728, out_channels=728)
        self.conv3 = SeparableConv2D(in_channels=728, out_channels=728)
    
    def forward(self, x):
        identity = x

        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)

        out = out + identity

        return out