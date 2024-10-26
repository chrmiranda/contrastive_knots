import torch
import numpy as np
from dataset import KnotCNData
from torch.nn import Linear, ReLU, Softmax


class MLP(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.fc1 = Linear(input_size, 64)
        self.fc2 = Linear(64, 64)
        self.fc3 = Linear(64, 32)
        self.fc4 = Linear(32, 14)
        self.relu = ReLU()
        self.softmax = Softmax()

    def forward(self, braid):
        x = self.fc1(braid)
        x = self.fc2(self.relu(x))
        x = self.fc3(self.relu(x))
        x = self.fc4(self.relu(x))
        return self.softmax(x)