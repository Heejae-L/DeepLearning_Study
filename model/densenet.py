import torch

from block.densenet_block import DensenetBlock
from layer.transition_layer import TransitionLayer

class Densenet(torch.nn.Module):
    def __init__(self, growth_rate, theta, num_classes=1000, config=()):
        super().__init__()
        out_channels = 2 * growth_rate

        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(3, out_channels, kernel_size=7, stride=2, padding=3, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        block_list = []
        for num_layers in config:
            block_list.append(DensenetBlock(num_layers = num_layers,
                                             int_channels = out_channels,
                                             growth_rate = growth_rate))
            out_channels += growth_rate * num_layers

            if num_layers != config[-1]:
                block_list.append(TransitionLayer(in_channels=out_channels, theta=theta))
                out_channels = int(out_channels * theta)


        self.features = torch.nn.Sequential(*block_list)

        self.classification = torch.nn.Sequential(
            
        )

