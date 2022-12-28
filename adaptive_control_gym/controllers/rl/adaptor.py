import torch
from torch import nn

class AdaptorTConv(nn.Module):
    def __init__(self, state_dim, action_dim, horizon, output_dim):
        super().__init__()
        self.state_dim = state_dim
        self.horizon = horizon
        self.embedding_size = 32
        self.output_size = output_dim
        self.input_size = state_dim + action_dim
        self.channel_transform = nn.Sequential(
            nn.Linear(self.input_size, self.embedding_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.ReLU(inplace=True),
        )
        self.temporal_aggregation = nn.Sequential(
            nn.Conv1d(self.embedding_size, self.embedding_size, (4,), stride=(2,)),
            nn.ReLU(inplace=True),
            # nn.Conv1d(self.embedding_size, self.embedding_size, (3,), stride=(1,)),
            # nn.ReLU(inplace=True),
            nn.Conv1d(self.embedding_size, self.embedding_size, (2,), stride=(1,)),
            nn.ReLU(inplace=True),
        )
        self.low_dim_proj = nn.Linear(self.embedding_size * 3, self.output_size)

    def forward(self, x):
        x = x.reshape(-1, self.horizon, self.input_size)  # (N, 10, 20)
        x = self.channel_transform(x)  # (N, 10, self.embedding_size)
        x = x.permute((0, 2, 1))  # (N, self.embedding_size, 10)
        x = self.temporal_aggregation(x)  # (N, self.embedding_size, 3)
        x = self.low_dim_proj(x.flatten(1))
        return x

class AdaptorMLP(nn.Module):
    def __init__(self, state_dim, action_dim, horizon, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear((state_dim+action_dim)*horizon, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return torch.tanh(self.mlp(x))

class AdaptorOracle(nn.Module):
    def __init__(self, state_dim, action_dim, horizon, output_dim):
        super().__init__()

    def forward(self, info):
        total_force = info['acc'] * 0.01
        disturb = total_force - info['action_force']
        disturb[:,1] -= 0.01*9.8
        mass_normed = torch.zeros_like(info['mass'], device=info['mass'].device)
        disturb_normed = disturb / 0.1
        delay_normed = torch.zeros_like(info['delay'], device=info['delay'].device)
        decay_normed = torch.zeros_like(info['decay'], device=info['decay'].device)
        return torch.concat([mass_normed, disturb_normed, delay_normed, decay_normed], dim=-1)