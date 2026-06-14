import torch as t
import torch.nn as nn
from downblock import down_block


class AudioNet1D(nn.Module):
    def __init__(self,num_classes=10):
        super().__init__()
        self.down1 = down_block(1, 32)
        self.down2 = down_block(32, 64)
        self.down3 = down_block(64, 128)
        self.down4 = down_block(128, 256)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256*32, num_classes)



    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)

        x = self.flatten(x)
        logits = self.fc(x)

        return logits