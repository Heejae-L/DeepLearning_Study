import torch

from model.xception.entry_flow import EntryFlow
from model.xception.middle_flow import MiddleFlow
from model.xception.exit_flow import ExitFlow

class Xception(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.entry = EntryFlow()
        self.middle = MiddleFlow()
        self.exit = ExitFlow(num_classes=num_classes)

    def forward(self, x):
        out = self.entry(x)
        out = self.middle(out)
        out = self.exit(out)

        return out