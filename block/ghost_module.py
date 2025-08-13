import math
import torch

class GhostModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels, ghost_ratio = 2, kernel_size=1, stride=1, dw_kernel_size=3):
        super().__init__()

        self.intrinsic_channels = (int)(out_channels//ghost_ratio)
        self.new_channels = out_channels - self.intrinsic_channels
        self.out_channels = out_channels

        padding = kernel_size//2 if kernel_size>1 else 0
        dw_padding = dw_kernel_size//2 if dw_kernel_size>1 else 0
        
        self.intrinsic = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=self.intrinsic_channels, 
                                         kernel_size=kernel_size, stride=stride, padding=padding,
                                         bias=False),
            torch.nn.BatchNorm2d(self.intrinsic_channels),
            torch.nn.ReLU(inplace=True)
        )
        
        self.cheap_operation = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.intrinsic_channels, out_channels=self.intrinsic_channels,
                            kernel_size=dw_kernel_size, stride=1, padding = dw_padding,
                            groups=self.intrinsic_channels),
            torch.nn.BatchNorm2d(self.intrinsic_channels)
        )

    def forward(self, x):
        instrinsic = self.intrinsic(x)

        ghost = self.cheap_operation(instrinsic)

        if self.new_channels > 0:
            n = ghost.shape[1]
            reps = int(math.ceil(self.new_channels / float(n)))  # 몇 번 쌓아야 충분한가
            ghosts_list = [ghost] * reps
            ghosts = torch.cat(ghosts_list, dim=1)[:, :self.new_channels, :, :]
        else:
            ghosts = ghost[:, :0, :, :] # 비어있는 텐서
        
        out = torch.cat([instrinsic, ghosts], dim=1)
        out = out[:, :self.out_channels, :, :] # 넘칠 경우 out_channels에 맞춰서 자르기

        return out
