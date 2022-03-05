import torch
import torch.nn as nn

class GeneratorBase(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def train_step(self):
        raise NotImplementedError()

    def forward(self):
        raise NotImplementedError()

    def save(self, path):
        torch.save(self.state_dict(), path)