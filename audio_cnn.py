import torch as t
import torch.nn as nn

class downblock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride > 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        res = self.shortcut(x)
        out = self.bn(self.conv(x))

        out+=res
        out = self.relu(out)
        return out



class AudioNet1D(nn.Module):
    def __init__(self,num_classes=35):
        super().__init__()
        self.down1 = downblock(1, 32)
        self.down2 = downblock(32, 64)
        self.down3 = downblock(64, 128)
        self.down4 = downblock(128, 256)
        
        self.pool = nn.AdaptiveAvgPool1d(32)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256*32, num_classes)



    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)

        x = self.pool(x)
        x = self.flatten(x)
        logits = self.fc(x)

        return logits
    
class AudioNet1DV2(nn.Module):
    def __init__(self,num_classes=35):
        super().__init__()
        self.down1 = downblock(64, 64)
        self.down2 = downblock(64, 128)
        self.down3 = downblock(128, 256)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256*32, num_classes)



    def forward(self, x):

        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)

        x = self.flatten(x)
        logits = self.fc(x)

        return logits