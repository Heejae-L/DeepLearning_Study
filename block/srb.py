import torch

class SRB(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        padding = int((kernel_size -1)/2)
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self,x):
        identity = x
        out = self.conv(x)
        out = out + identity
        out = self.relu(out)
        return out