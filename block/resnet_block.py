

import torch

class ResnetBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)

        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)

        if stride > 1 :
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), 
                torch.nn.BatchNorm2d(out_channels)
                )
        else:
            self.downsample = torch.nn.Identity()

        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity # 입력 다시 더함
        out = self.relu(out)
        return out