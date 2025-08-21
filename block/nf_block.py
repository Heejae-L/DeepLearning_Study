import torch
import torch.nn.functional as F
from block.nf_wsconv import ScaledWSConv2d  # 당신이 만든 WSConv

class SEBlock(torch.nn.Module):
    def __init__(self, in_channels, se_ratio=0.5):
        super().__init__()
        hidden = max(1, int(in_channels * se_ratio))
        self.squeeze = torch.nn.AdaptiveAvgPool2d(1)
        self.excitation = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, hidden, kernel_size=1, bias=True),
            torch.nn.SiLU(inplace=True),
            torch.nn.Conv2d(hidden, in_channels, kernel_size=1, bias=True),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        s = self.squeeze(x)
        s = self.excitation(s)
        return x * s


class DropPath(torch.nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if (not self.training) or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep).div_(keep)
        return x * mask


class NFBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        mid_channels,
        out_channels,
        stride=1,
        alpha=0.2,
        drop=0.0,
        use_se=True,
        se_ratio=0.5,
        eps=1e-5,
    ):
        super().__init__()
        mid_channels = mid_channels
        groups = max(1, mid_channels // 128)

        self.alpha = alpha
        self.eps = eps
        self.act = torch.nn.SiLU(inplace=True)
        self.drop_path = DropPath(drop)

        self.conv1 = ScaledWSConv2d(in_channels, mid_channels, kernel_size=1, bias=True)
        self.conv2 = ScaledWSConv2d(
            mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, groups=groups, bias=True
        )
        self.conv3 = ScaledWSConv2d(
            mid_channels, mid_channels, kernel_size=3, padding=1, groups=groups, bias=True
        )
        self.conv4 = ScaledWSConv2d(mid_channels, out_channels, kernel_size=1, bias=True)

        self.se = SEBlock(out_channels, se_ratio) if use_se else torch.nn.Identity()

        if stride != 1 or in_channels != out_channels:
            self.skip = torch.nn.Sequential(
                torch.nn.AvgPool2d(kernel_size=stride, stride=stride),
                ScaledWSConv2d(in_channels, out_channels, kernel_size=1, bias=True),
            )
        else:
            self.skip = torch.nn.Identity()

    def scaled_activation(self, x, eps=1e-5):

        var = x.var(dim=(0, 1, 2, 3), unbiased=False, keepdim=True)
        beta = torch.sqrt(var + eps)
        return x / beta

    def forward(self, x):
        s = self.scaled_activation(x, self.eps)
        out = self.conv1(self.act(s))

        out = self.conv2(self.act(self.scaled_activation(out, self.eps)))
        out = self.conv3(self.act(self.scaled_activation(out, self.eps)))
        out = self.conv4(self.act(self.scaled_activation(out, self.eps)))

        out = self.se(out)
        out = self.drop_path(out)
        out = self.alpha * out

        return self.skip(x) + out
