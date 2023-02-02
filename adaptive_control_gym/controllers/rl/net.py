import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

class ActorBase(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = None  # build_mlp(dims=[state_dim, *dims, action_dim])
        self.explore_noise_std = None  # standard deviation of exploration action noise
        self.ActionDist = torch.distributions.normal.Normal

        self.state_avg = nn.Parameter(
            torch.zeros((state_dim,)), requires_grad=False)
        self.state_std = nn.Parameter(
            torch.ones((state_dim,)), requires_grad=False)

    def state_norm(self, state: Tensor) -> Tensor:
        return (state - self.state_avg) / self.state_std


class ActorPPO(ActorBase):
    def __init__(self, state_dim: int, expert_dim: int, action_dim: int, expert_mode: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.expert_dim = expert_dim
        self.expert_mode = expert_mode
        embedding_dim = 256
        n_layers = 2
        if expert_mode == 0:
            fc_list = [nn.Linear(state_dim, embedding_dim)]
            for _ in range(n_layers):
                fc_list.append(nn.Linear(embedding_dim, embedding_dim))
            fc_list.append(nn.Linear(embedding_dim, action_dim))
            self.fc = nn.Sequential(*fc_list)
        elif expert_mode == 1:
            fc_list = [nn.Linear(state_dim + expert_dim, embedding_dim)]
            for _ in range(n_layers):
                fc_list.append(nn.Linear(embedding_dim, embedding_dim))
            fc_list.append(nn.Linear(embedding_dim, action_dim))
            self.fc = nn.Sequential(*fc_list)
        elif expert_mode == 2:
            fc_list = [nn.Linear(state_dim, embedding_dim)]
            for _ in range(n_layers):
                fc_list.append(nn.Linear(embedding_dim, embedding_dim))
            fc_list.append(nn.Linear(embedding_dim+expert_dim, action_dim))
            self.fc = nn.Sequential(*fc_list)
        elif expert_mode == 3:
            fc_list = [nn.Linear(state_dim+expert_dim, embedding_dim)]
            for _ in range(n_layers):
                fc_list.append(nn.Linear(embedding_dim+expert_dim, embedding_dim))
            fc_list.append(nn.Linear(embedding_dim+expert_dim, action_dim))
            self.fc = nn.Sequential(*fc_list)
        else:
            raise NotImplementedError
        layer_init_with_orthogonal(self.fc[-1], std=0.1)

        self.action_std_log = nn.Parameter(torch.zeros(
            (1, action_dim)), requires_grad=True)  # trainable parameter

    def forward(self, state: Tensor, e: Tensor) -> Tensor:
        return self.convert_action_for_env(self.get_action_avg(state, e))
        
    def get_action_avg(self, state: Tensor, e: Tensor) -> Tensor:
        if self.expert_mode == 0:
            for fc in self.fc[:-1]:
                state = F.relu(fc(state))
        elif self.expert_mode == 1:
            state = torch.cat([state, e], dim=-1)
            for fc in self.fc[:-1]:
                state = F.relu(fc(state))
        elif self.expert_mode == 2:
            # pass state into last layers of the net
            for fc in self.fc[:-1]:
                state = F.relu(fc(state))
            state = torch.cat([state, e], dim=-1)
        elif self.expert_mode == 3:
            # pass state into each layers of the net
            for fc in self.fc[:-1]:
                state = torch.cat([state, e], dim=-1)
                state = F.relu(fc(state))
            state = torch.cat([state, e], dim=-1)
        return self.fc[-1](state)

    def get_action(self, state: Tensor, e:Tensor) -> (Tensor, Tensor):  # for exploration
        action_avg = self.get_action_avg(state, e)
        action_std = self.action_std_log.exp()

        dist = self.ActionDist(action_avg, action_std)
        action = dist.sample()
        logprob = dist.log_prob(action).sum(1)
        return action, logprob

    def get_logprob_entropy(self, state: Tensor, action: Tensor, e:Tensor) -> (Tensor, Tensor):
        action_avg = self.get_action_avg(state, e)
        action_std = self.action_std_log.exp()

        dist = self.ActionDist(action_avg, action_std)
        logprob = dist.log_prob(action).sum(1)
        entropy = dist.entropy().sum(1)
        return logprob, entropy

    @staticmethod
    def convert_action_for_env(action: Tensor) -> Tensor:
        return action.tanh()


class CriticBase(nn.Module):  # todo state_norm, value_norm
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = None  # build_mlp(dims=[state_dim + action_dim, *dims, 1])

        self.state_avg = nn.Parameter(
            torch.zeros((state_dim,)), requires_grad=False)
        self.state_std = nn.Parameter(
            torch.ones((state_dim,)), requires_grad=False)
        self.value_avg = nn.Parameter(torch.zeros((1,)), requires_grad=False)
        self.value_std = nn.Parameter(torch.ones((1,)), requires_grad=False)

    def state_norm(self, state: Tensor) -> Tensor:
        return (state - self.state_avg) / self.state_std  # todo state_norm

    def value_re_norm(self, value: Tensor) -> Tensor:
        return value * self.value_std + self.value_avg  # todo value_norm


class CriticPPO(CriticBase):
    def __init__(self, state_dim: int, expert_dim: int, action_dim: int, expert_mode: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.expert_dim, self.expert_mode = expert_dim, expert_mode
        embedding_dim = 256
        n_layers = 2
        if expert_mode == 0:
            fc_list = [nn.Linear(state_dim, embedding_dim)]
            for _ in range(n_layers):
                fc_list.append(nn.Linear(embedding_dim, embedding_dim))
            fc_list.append(nn.Linear(embedding_dim, 1))
            self.fc = nn.Sequential(*fc_list)
        elif expert_mode == 1:
            fc_list = [nn.Linear(state_dim + expert_dim, embedding_dim)]
            for _ in range(n_layers):
                fc_list.append(nn.Linear(embedding_dim, embedding_dim))
            fc_list.append(nn.Linear(embedding_dim, 1))
            self.fc = nn.Sequential(*fc_list)
        elif expert_mode == 2:
            fc_list = [nn.Linear(state_dim, embedding_dim)]
            for _ in range(n_layers):
                fc_list.append(nn.Linear(embedding_dim, embedding_dim))
            fc_list.append(nn.Linear(embedding_dim+expert_dim, 1))
            self.fc = nn.Sequential(*fc_list)
        elif expert_mode == 3:
            fc_list = [nn.Linear(state_dim+expert_dim, embedding_dim)]
            for _ in range(n_layers):
                fc_list.append(nn.Linear(embedding_dim+expert_dim, embedding_dim))
            fc_list.append(nn.Linear(embedding_dim+expert_dim, 1))
            self.fc = nn.Sequential(*fc_list)
        elif expert_mode == 4:
            fc_list = [nn.Linear(state_dim, embedding_dim)]
            for _ in range(n_layers):
                fc_list.append(nn.Linear(embedding_dim, embedding_dim))
            fc_list.append(nn.Linear(embedding_dim, 8))
            self.fc = nn.Sequential(*fc_list)
            fc_list = [nn.Linear(expert_dim, embedding_dim)]
            for _ in range(n_layers):
                fc_list.append(nn.Linear(embedding_dim, embedding_dim))
            fc_list.append(nn.Linear(embedding_dim, 8))
            self.fc_expert = nn.Sequential(*fc_list)
            layer_init_with_orthogonal(self.fc_expert[-1], std=0.5)
        else:
            raise NotImplementedError
        layer_init_with_orthogonal(self.fc[-1], std=0.5)

    def forward(self, state: Tensor, e: Tensor) -> Tensor:
        if self.expert_mode == 0:
            for fc in self.fc[:-1]:
                state = F.relu(fc(state))
            return self.fc[-1](state)
        elif self.expert_mode == 1:
            state = torch.cat([state, e], dim=-1)
            for fc in self.fc[:-1]:
                state = F.relu(fc(state))
            return self.fc[-1](state)
        elif self.expert_mode == 2:
            # pass state into last layers of the net
            for fc in self.fc[:-1]:
                state = F.relu(fc(state))
            state = torch.cat([state, e], dim=-1)
            return self.fc[-1](state)
        elif self.expert_mode == 3:
            # pass state into each layers of the net
            for fc in self.fc[:-1]:
                state = torch.cat([state, e], dim=-1)
                state = F.relu(fc(state))
            state = torch.cat([state, e], dim=-1)
            return self.fc[-1](state)
        elif self.expert_mode == 4:
            for fc in self.fc[:-1]:
                state = F.relu(fc(state))
            state = self.fc[-1](state)
            for fc in self.fc_expert[:-1]:
                e = F.relu(fc(e))
            e = self.fc_expert[-1](e)
            return torch.sum(state * e, dim=-1, keepdim=True)


class Compressor(nn.Module):
    def __init__(self, expert_dim, embedding_dim) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        if embedding_dim > 0:
            self.mlp = build_mlp_net([expert_dim, 128, embedding_dim], activation=nn.ReLU, if_raw_out=True)
            self.std_log = nn.Parameter(torch.ones((1, embedding_dim))*(-10.0), requires_grad=True)  # trainable parameter
        else:
            self.std_log = nn.Parameter(torch.ones((1, expert_dim))*(-10.0), requires_grad=True)  # trainable parameter
        self.dist = torch.distributions.normal.Normal
    
    def forward(self, x):
        return self.get_compress_mean(x)

    def get_compress(self, x):
        mean = self.get_compress_mean(x)
        std = self.std_log.exp()

        dist = self.dist(mean, std)
        z = dist.sample()
        logprob = dist.log_prob(z).sum(1)
        return z, logprob

    def get_compress_mean(self, x):
        if self.embedding_dim > 0:
            # use tanh as activation
            return torch.tanh(self.mlp(x))
        else:
            return x

    def get_logprob_entropy(self, x, h):
        avg = self.get_compress(x)
        std = self.std_log.exp()

        dist = self.dist(avg, std)
        logprob = dist.log_prob(h).sum(1)
        entropy = dist.entropy().sum(1)
        return logprob, entropy

def build_mlp_net(dims: [int], activation: nn = None, if_raw_out: bool = True) -> nn.Sequential:
    """
    build MLP (MultiLayer Perceptron)

    dims: the middle dimension, `dims[-1]` is the output dimension of this network
    activation: the activation function
    if_remove_out_layer: if remove the activation function of the output layer.
    """
    if activation is None:
        activation = nn.ReLU
    net_list = []
    for i in range(len(dims) - 1):
        net_list.extend([nn.Linear(dims[i], dims[i+1]), activation()])
    if if_raw_out:
        # delete the activation function of the output layer to keep raw output
        del net_list[-1]
    return nn.Sequential(*net_list)


def layer_init_with_orthogonal(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
