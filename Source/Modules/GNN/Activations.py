import torch
from torch import nn as nn


class Sigmoid(nn.Module):
    """
    Sigmoid activation function
    """
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        return torch.sigmoid(x)

class Tanh(nn.Module):
    """
    Tanh activation function
    """
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        return torch.tanh(x)


class Swish(nn.Module):
    """
    Swish activation function
    """
    def __init__(self, beta=1):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta*x)
