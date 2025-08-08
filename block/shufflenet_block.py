import torch

class ShuffleNetUnit(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride, groups):
        super().__init__()


        conv_out_channels = out_channels-in_channels if stride>1 else out_channels
        bottle_neck = conv_out_channels // 4
        bottle_neck = (bottle_neck // groups) * groups
        
        self.is_downscaling = True if stride > 1 else False
        self.groups = groups

        self.gconv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=bottle_neck, kernel_size=1, stride=1, groups=groups, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(bottle_neck)
        self.relu = torch.nn.ReLU(inplace=False)

        self.dwconv = torch.nn.Conv2d(in_channels=bottle_neck, out_channels=bottle_neck,kernel_size=3, stride=stride, padding=1, groups = bottle_neck, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(bottle_neck)

        self.gconv2 = torch.nn.Conv2d(in_channels=bottle_neck, out_channels=conv_out_channels, kernel_size=1,stride=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(conv_out_channels)

        self.shortcut = torch.nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
            
    
    def forward(self, x):
        identity = self.shortcut(x) if self.is_downscaling else x

        out = self.gconv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = ChannelShuffle(out, self.groups)

        out = self.dwconv(out)
        out = self.bn2(out)

        out = self.gconv2(out)
        out = self.bn3(out)

        if self.is_downscaling:
            out = torch.concat([out,identity], dim=1)
        else:
            out = out + identity
        
        return out


def ChannelShuffle(x, groups):
    batchsize, num_channels, height, width = x.size()
    if num_channels % groups > 0 :
        return x
    
    n = num_channels // groups
    x = x.view(batchsize, groups, n, height, width)
    x = torch.transpose(x, 1, 2)
    x = x.contiguous().view(batchsize, num_channels, height, width) # 메모리를 재배치 + flatten

    return x