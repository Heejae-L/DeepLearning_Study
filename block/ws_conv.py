import torch
import torch.nn.functional as F

class WSConv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, eps=1e-5):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.eps = eps

    def forward(self, x):
        # weight: [C_out, C_in/groups, kH, kW]
        w = self.weight

        mean = w.mean(dim=(1,2,3), keepdim=True)
        w = w - mean

        std = (w * w).mean(dim=(1,2,3), keepdim=True)
        w = w / (std + self.eps)

        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
