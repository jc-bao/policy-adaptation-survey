import torch

class Random:
    def __init__(self, action_dim):
        self.action_dim = action_dim
    def __call__(self, state):
        return torch.rand((1, self.action_dim))*2-1