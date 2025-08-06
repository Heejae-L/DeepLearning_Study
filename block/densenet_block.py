import torch

from layer.dense_layer import DenseLayer

class DensenetBlock(torch.nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super().__init__()

        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + (i * growth_rate), growth_rate))

        self.block = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
