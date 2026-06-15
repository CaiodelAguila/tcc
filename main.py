import numpy as np
import torch as t
import torch.nn as nn
import audio_cnn as ac


def main():
    num_classes = 10
    model = ac.AudioNet1D(num_classes=num_classes)


    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = t.optim.Adam(model.parameters(), lr=0.001)

    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {parameters}")

    