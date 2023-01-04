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
    def __init__(self, adapt_dim, horizon, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(adapt_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, output_dim),
        )

    def forward(self, adapt_obs):
        return torch.tanh(self.mlp(adapt_obs))*2

class Adaptor3D(nn.Module):
    def __init__(self, adapt_dim, horizon, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(adapt_dim//3, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, output_dim//3),
        )

    def forward(self, adapt_obs):
        x = torch.tanh(self.mlp(adapt_obs))*2
        return x.reshape(x.shape[0], -1)

class AdaptorOracle(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, info):
        if hasattr(self, 'device') is False:
            self.device = info['acc_his'].device
        da2, da3 = info['d_acc_his'][-2], info['d_acc_his'][-1]
        du2, du3 = info['d_u_force_his'][-2], info['d_u_force_his'][-1]
        a1, a2, a3 = info['acc_his'][-3], info['acc_his'][-2], info['acc_his'][-1]
        u2, u3 = info['u_force_his'][-2], info['u_force_his'][-1]
        v2, v3 = info['v_his'][-2], info['v_his'][-1]

        k = (da2*du3 - da3*du2) / (a2*da2 - a1*da3) * 30
        k = torch.where(torch.isnan(k)|torch.isinf(k), torch.ones_like(k, device=self.device)*0.15, k)
        m = (du3-k*a2*1/30) / da3
        m = torch.where(torch.isnan(m)|torch.isinf(m), torch.ones_like(m, device=self.device)*0.03, m)
        F = m * a3 - u3 + k * v2 + m * torch.tensor([0,9.8,0], device=self.device)
        F = torch.where(torch.isnan(F)|torch.isinf(F), torch.ones_like(F, device=self.device)*0.0, F)

        mass_normed = (m - 0.03)/0.02
        disturb_normed = F / 0.3
        delay_normed = torch.zeros_like(info['delay'], device=self.device)
        decay_normed = (k-0.15)/0.15
        return torch.concat([mass_normed, disturb_normed, delay_normed, decay_normed], dim=-1)