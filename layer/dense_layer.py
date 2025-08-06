import torch

class DenseLayer(torch.nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()

        out_channels = 4*growth_rate

        self.bn1 = torch.nn.BatchNorm2d(in_channels)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                     kernel_size=1, bias=False)
        
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(in_channels=out_channels, out_channels=growth_rate, 
                                     kernel_size=3, padding=1, bias=False)
        
    def forward(self, x):
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        return torch.cat([x, out], dim=1) # shape : [batch_size, channels, height, width]
