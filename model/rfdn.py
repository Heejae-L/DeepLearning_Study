import torch

from block.rfdb import RFDB
from block.rfdb import pixelshuffle_block


class RFDN(torch.nn.Module):
    def __init__(self, hidden=50, num_module=4, upscale=4):
        super().__init__()
        
        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=hidden, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True)
        )

        self.rfdb1 = RFDB(hidden)
        self.rfdb2 = RFDB(hidden)
        self.rfdb3 = RFDB(hidden)
        self.rfdb4 = RFDB(hidden)

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(hidden*num_module, hidden, kernel_size=1, stride=1, padding=0),
            torch.nn.ReLU(inplace=True)
        )

        self.conv2 = torch.nn.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding=1)

        self.upsample = pixelshuffle_block(hidden, 3, upscale_factor=upscale)

    def forward(self, x):
        init = self.stem(x)
        b1 = self.rfdb1(init)
        b2 = self.rfdb2(b1)
        b3 = self.rfdb3(b2)
        b4 = self.rfdb4(b3)

        out = torch.cat([b1,b2,b3,b4], dim=1)  
        out = self.conv1(out)
        out = self.conv2(out)
        out = out + init

        out = self.upsample(out)

        return out