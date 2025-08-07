import torch

from block.mobilenet_depthwise_separable_conv import DepthwiseSeparableConv2D

class MobileNet(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        config = [(32, 64, 1),
                  (64, 128, 2),
                  (128, 128, 1),
                  (128, 256, 2),
                  (256, 256, 1),
                  (256, 512, 2),
                  (512, 512, 1),
                  (512, 512, 1),
                  (512, 512, 1),
                  (512, 512, 1),
                  (512, 512, 1),
                  (512, 1024, 2),
                  (1024, 1024, 2)]
        
        self.stem = torch.nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)

        features_list = []
        for (in_channels, out_channels, stride) in config:
            features_list.append(DepthwiseSeparableConv2D(in_channels=in_channels,
                                             out_channels=out_channels,
                                             stride=stride))

        self.features = torch.nn.Sequential(*features_list)

        self.classifer = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Conv2d(in_channels=out_channels, 
                            out_channels=num_classes, 
                            kernel_size=1, bias=True)
        )

    def forward(self, x):
        out = self.stem(x)
        out = self.features(out)
        out = self.classifer(out)
        out = torch.flatten(out, 1)

        return out
