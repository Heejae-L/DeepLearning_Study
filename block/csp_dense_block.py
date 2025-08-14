import torch

from block.densenet_block import DensenetBlock

class CSPDenseBlock(torch.nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers, part_ratio=0.5):
        super().__init__()
        self.part1_channels = (int)(in_channels*part_ratio)
        self.part2_channels = in_channels - self.part1_channels
        self.denseblock = DensenetBlock(num_layers, self.part2_channels, growth_rate=growth_rate)

    def forward(self, x):
        part1 = x[:, :self.part1_channels,:,:]
        part2 = x[:, self.part1_channels:,:,:]

        part2 = self.denseblock(part2)
        out = torch.cat((part1, part2), dim=1)

        return out