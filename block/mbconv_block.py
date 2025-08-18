import torch

class SEBlock(torch.nn.Module):
    def __init__(self, in_channels, se_ratio = 0.25):
        super().__init__()
        hidden = max(1, int(in_channels * se_ratio))
        self.squeeze = torch.nn.AdaptiveAvgPool2d(1)
        self.excitation = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, hidden, kernel_size=1, bias=True),
            torch.nn.SiLU(inplace=True),
            torch.nn.Conv2d(hidden, in_channels, kernel_size=1, bias=True),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        s = self.squeeze(x)        # (N,C,1,1)
        s = self.excitation(s)     # (N,C,1,1)
        return x * s               

    
class drop(torch.nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.p = float(p)

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        # per-sample mask, broadcast to [N,1,1,1]
        mask = torch.empty(x.shape[0], 1, 1, 1, device=x.device).bernoulli_(keep)
        return x / keep * mask

class MBConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=6, kernel_size = 3, se_ratio = 0.25, drop_ratio = 0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)
        mid_channels = in_channels * expand_ratio

        layers = []

        if expand_ratio != 1:
            layers += [
                torch.nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
                torch.nn.BatchNorm2d(mid_channels),
                torch.nn.SiLU(inplace=True)
            ]
        
        padding = kernel_size//2
        layers+=[
            torch.nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=mid_channels, bias=False),
            torch.nn.BatchNorm2d(mid_channels),
            torch.nn.SiLU(inplace=True)
        ]

        self.block = torch.nn.Sequential(*layers)

        self.has_se = (se_ratio is not None) and (se_ratio > 0.0)
        if self.has_se:
            self.se = SEBlock(
                mid_channels,
                se_ratio=se_ratio,
            )

        self.projection = torch.nn.Sequential(
            torch.nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(out_channels)
        )

        self.drop = drop(drop_ratio) if self.use_residual and drop_ratio > 0 else torch.nn.Identity()

    def forward(self, x):
        out = self.block(x)
        if self.has_se:
            out = self.se(out)
        out = self.projection(out)
        if self.use_residual:
            out = self.drop(out)
            out = out + x
        
        return out


        
        
        