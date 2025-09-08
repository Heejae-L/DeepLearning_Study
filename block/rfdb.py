import torch

from block.srb import SRB

class RFDB(torch.nn.Module):
    def __init__(self, in_channels, distillation_rate = 0.25):
        super().__init__()
        self.distilled_channels = in_channels//2
        self.remaining_channels = in_channels

        self.c1_d = torch.nn.Conv2d(in_channels, self.distilled_channels, kernel_size=1, stride=1, padding=0)
        self.c1_r = SRB(in_channels, self.remaining_channels, kernel_size=3, stride=1, padding=1)

        self.c2_d = torch.nn.Conv2d(self.remaining_channels, self.distilled_channels, kernel_size=1, stride=1, padding=0)
        self.c2_r = SRB(self.remaining_channels, self.remaining_channels, kernel_size=3, stride=1, padding=1)

        self.c3_d = torch.nn.Conv2d(self.remaining_channels, self.distilled_channels, kernel_size=1, stride=1, padding=0)
        self.c3_r = SRB(self.remaining_channels, self.remaining_channels, kernel_size=3, stride=1, padding=1)

        self.c4 = torch.nn.Conv2d(self.remaining_channels, self.distilled_channels, kernel_size=3, stride=1, padding=1)

        self.c5 = torch.nn.Conv2d(self.distilled_channels*4, in_channels, kernel_size=1, stride=1, padding=0)
        self.esa = ESA(in_channels, torch.nn.Conv2d)

    def forward(self, x):
        identity = x
        c1_d = self.c1_d(x)
        c1_r = self.c1_r(x)

        c2_d = self.c2_d(c1_r)
        c2_r = self.c2_r(c1_r)

        c3_d = self.c3_d(c2_r)
        c3_r = self.c3_r(c2_r)

        c4_d = self.c4(c3_r)

        out = torch.cat([c1_d, c2_d, c3_d, c4_d], dim = 1)

        out = self.c5(out)
        out = self.esa(out)

        out = out+identity

        return out

class ESA(torch.nn.Module):
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = torch.nn.functional.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = torch.nn.functional.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False) 
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3+cf)
        m = self.sigmoid(c4)
        
        return x * m

def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = torch.nn.Conv2d(in_channels, out_channels*(upscale_factor ** 2), kernel_size=kernel_size, stride=stride)
    pixel_shuffle = torch.nn.PixelShuffle(upscale_factor)
    return torch.nn.Sequential(conv, pixel_shuffle)