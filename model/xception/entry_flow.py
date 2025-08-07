import torch
from block.residual_separable_block import ResidualSeparableBlock


class EntryFlow(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=False)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=False)
        )

        self.block1 = ResidualSeparableBlock(64,128,False)
        self.block2 = ResidualSeparableBlock(128,256)
        self.block3 = ResidualSeparableBlock(256,728)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)

        return out

        
