import torch as t
import torch.nn as nn

def down_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
    layers = [nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)]
    layers.append(nn.BatchNorm1d(out_channels))
    layers.append(nn.ReLU())
    return nn.Sequential(*layers)