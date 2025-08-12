import torch
from block.se_resnet_block import SERenetBlock

class SEResnet(torch.nn.Module):
    def __init__(self, num_classes=1000, config=[]):
        super().__init__()
        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        features_list = []
        for (in_channels, out_channels, stride) in config:
            features_list.append(SERenetBlock(in_channels=in_channels,
                                             out_channels=out_channels,
                                             stride=stride))

        self.features = torch.nn.Sequential(*features_list)


        _, final_channels, _ = config[-1]


        self.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Conv2d(in_channels=final_channels, 
                            out_channels=num_classes, 
                            kernel_size=1, bias=True)
        )

    def forward(self, x):
        out = self.stem(x)
        out = self.features(out)
        out = self.classifier(out)
        out = out.view(out.size(0), -1)
        return out
    
def SeResnet18(num_classes=1000):

    config = [
        (64, 64, 1),
        (64, 64, 1),
        (64, 128, 2),
        (128, 128, 1),
        (128, 256, 2),
        (256, 256, 1),
        (256, 512, 2),
        (512, 512, 1),
    ]
    return SEResnet(num_classes=num_classes, config=config)