import torch

class SEBlock(torch.nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.squeeze = torch.nn.AdaptiveAvgPool2d((1,1))
        self.excitation = torch.nn.Sequential(
            torch.nn.Linear(in_channels, out_features=in_channels//reduction_ratio),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_channels//reduction_ratio, in_channels),
            torch.nn.Sigmoid()
        )
    def forward(self, x):
        out = self.squeeze(x)
        out = out.view(out.size(0), -1)
        out = self.excitation(out)
        out = out.view(out.size(0),out.size(1), 1, 1)
        return out