import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LinearClassifier(nn.Module):
    def __init__(self, output_class):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(2048, output_class)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)

        return x
