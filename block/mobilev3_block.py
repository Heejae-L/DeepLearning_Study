import torch

class SEBlock(torch.nn.Module):
    def __init__(self, in_channels, se_ratio = 0.25):
        super().__init__()
        hidden = max(1, int(in_channels * se_ratio))
        self.squeeze = torch.nn.AdaptiveAvgPool2d(1)
        self.excitation = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, hidden, kernel_size=1, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(hidden, in_channels, kernel_size=1, bias=True),
            torch.nn.Hardsigmoid()
        )

    def forward(self, x):
        s = self.squeeze(x)        
        s = self.excitation(s)     
        return x * s    
    
class MobileV3Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, expansion_size, stride = 1, se=False, se_ratio = 0.25, non_linear=torch.nn.ReLU):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = expansion_size
        self.has_se = se and (se_ratio is not None) and (se_ratio > 0)
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        layers = []
        layers += [
                torch.nn.Conv2d(in_channels, self.mid_channels, kernel_size=1, bias=False),
                torch.nn.BatchNorm2d(self.mid_channels),
                non_linear(inplace=True)
            ]
        padding = kernel_size//2
        layers+=[
            torch.nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=self.mid_channels, bias=False),
            torch.nn.BatchNorm2d(self.mid_channels),
            non_linear(inplace=True)
        ]

        self.block = torch.nn.Sequential(*layers)

        self.has_se = (se_ratio is not None) and (se_ratio > 0.0)
        if self.has_se:
            self.se = SEBlock(
                self.mid_channels,
                se_ratio=se_ratio,
            )
        self.projection = torch.nn.Sequential(
            torch.nn.Conv2d(self.mid_channels, out_channels, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(out_channels)
        )
    def forward(self, x):
        out = self.block(x)
        if self.has_se:
            out = self.se(out)
        out = self.projection(out)
        if self.use_residual:
            out = out + x
        
        return out
        