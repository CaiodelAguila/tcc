import torch as t
import torch.nn as nn

class downblock(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=4,stride=2,padding=1):
        super().__init__()
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(out_channels)
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)

        res = self.shortcut(x)
        out += res

        out = self.relu(out)
        return out