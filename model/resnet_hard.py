import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride > 1 :
            self.downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(out_channels))
        else:
            self.downsample = None

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity # 입력 다시 더함
        out = self.relu(out)
        return out
    
class ResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        
        # 블록 A
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 블록 B
        self.block1 = BasicBlock(64, 64)
        self.block2 = BasicBlock(64, 64)

        # 블록 C
        self.block3 = BasicBlock(64, 128, stride=2)
        self.block4 = BasicBlock(128, 128)

        # 블록 D
        self.block5 = BasicBlock(128, 256, stride=2)
        self.block6 = BasicBlock(256, 256)

        # 블록 E
        self.block7 = BasicBlock(256, 512, stride=2)
        self.block8 = BasicBlock(512, 512)

        # classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)

        #self.conv_final = nn.Conv2d(1,num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)

        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)

        return x