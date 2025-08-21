import torch
import torch.nn.functional as F

class ScaledWSConv2d(torch.nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, eps=1e-5, gain = True, gamma=1.0):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.eps = eps
        self.gain = torch.nn.Parameter(torch.ones(self.out_channels, 1, 1, 1)) if gain else None
        self.scale = gamma * self.weight[0].numel() ** -0.5

    def forward(self, x):

        w = self.weight

        mean = w.mean(dim=(1,2,3), keepdim=True)
        w = w - mean

        std = (w * w).mean(dim=(1,2,3), keepdim=True)
        w = self.scale * w / (std + self.eps)

        if self.gain is not None:
            w = w * self.gain
        
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

        
        
    
