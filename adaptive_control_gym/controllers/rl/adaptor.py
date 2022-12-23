import torch
from torch import nn

class AdaptorTConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_size = 4
        self.output_size = 2
        self.input_size = 16 + 16
        self.channel_transform = nn.Sequential(
            nn.Linear(self.input_size, self.embedding_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.ReLU(inplace=True),
        )
        self.temporal_aggregation = nn.Sequential(
            nn.Conv1d(self.embedding_size, self.embedding_size, (5,), stride=(2,)),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.embedding_size, self.embedding_size, (4,), stride=(1,)),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.embedding_size, self.embedding_size, (3,), stride=(1,)),
            nn.ReLU(inplace=True),
        )
        self.low_dim_proj = nn.Linear(self.embedding_size * 3, self.output_size)

    def forward(self, x):
        x = self.channel_transform(x)  # (N, 50, self.embedding_size)
        x = x.permute((0, 2, 1))  # (N, self.embedding_size, 50)
        x = self.temporal_aggregation(x)  # (N, self.embedding_size, 3)
        x = self.low_dim_proj(x.flatten(1))
        return x

class AdaptorMLP(nn.Module):
    def __init__(self, state_dim, horizon, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(state_dim*horizon, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.mlp(x)