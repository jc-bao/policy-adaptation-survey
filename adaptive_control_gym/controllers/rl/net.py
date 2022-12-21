import torch
import torch.nn as nn
from torch import Tensor

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
    def __init__(self, dims: [int], state_dim: int, expert_dim: int, action_dim: int, expert_mode: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.expert_dim = expert_dim
        self.expert_mode = expert_mode
        dims = [state_dim, *dims, action_dim]
        if expert_mode == 1:
            dims[0] += expert_dim
        elif expert_mode == 2:
            dims[-2] += expert_dim
        elif expert_mode == 3:
            dims[:-1] += expert_dim
        elif expert_mode != 0:
            raise NotImplementedError
        self.net = build_mlp_net(dims=dims)
        layer_init_with_orthogonal(self.net[-1], std=0.1)

        self.action_std_log = nn.Parameter(torch.zeros(
            (1, action_dim)), requires_grad=True)  # trainable parameter

    def forward(self, state: Tensor, e: Tensor) -> Tensor:
        return self.convert_action_for_env(self.get_action_avg(state, e))
        
    def get_action_avg(self, state: Tensor, e: Tensor) -> Tensor:
        if self.expert_mode == 0:
            return self.net(state)
        elif self.expert_mode == 1:
            state = torch.cat([state, e], dim=-1)
            return self.net(state)
        elif self.expert_mode == 2:
            # pass state into last layers of the net
            for i in range(len(self.net) - 1):
                state = self.net[i](state)
            state = torch.cat([state, e], dim=-1)
            return self.net[-1](state)
        elif self.expert_mode == 3:
            # pass state into each layers of the net
            for i in range(len(self.net) - 1):
                state = torch.cat([state, e], dim=-1)
                state = self.net[i](state)
            state = torch.cat([state, e], dim=-1)
            return self.net[-1](state)
            

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
    def __init__(self, dims: [int], state_dim: int, expert_dim: int, action_dim: int, expert_mode: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.expert_dim, self.expert_mode = expert_dim, expert_mode
        dims = [state_dim, *dims, 1]
        if expert_mode == 1:
            dims[0] += expert_dim
        elif expert_mode == 2:
            dims[-2] += expert_dim
        elif expert_mode == 3:
            dims[:-1] += expert_dim
        elif expert_mode == 4:
            dims = [int(d/1.4) for d in dims]
            dims[0], dims[-1] = expert_dim, 8
            self.net_expert = build_mlp_net(dims=dims)
            layer_init_with_orthogonal(self.net_expert[-1], std=0.5)
            dims[0] = state_dim
        elif expert_mode != 0:
            raise NotImplementedError
        self.net = build_mlp_net(dims=dims)
        layer_init_with_orthogonal(self.net[-1], std=0.5)

    def forward(self, state: Tensor, e: Tensor) -> Tensor:
        if self.expert_mode == 0:
            return self.net(state)
        elif self.expert_mode == 1:
            state = torch.cat([state, e], dim=-1)
            return self.net(state)
        elif self.expert_mode == 2:
            # pass state into last layers of the net
            for i in range(len(self.net) - 1):
                state = self.net[i](state)
            state = torch.cat([state, e], dim=-1)
            return self.net[-1](state)
        elif self.expert_mode == 3:
            # pass state into each layers of the net
            for i in range(len(self.net) - 1):
                state = torch.cat([state, e], dim=-1)
                state = self.net[i](state)
            state = torch.cat([state, e], dim=-1)
            return self.net[-1](state)
        elif self.expert_mode == 4:
            embed_x = self.net(state)
            embed_e = self.net_expert(e)
            # return the dot product of two embeddings
            return torch.sum(embed_x * embed_e, dim=-1, keepdim=True)


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
        net_list.extend([nn.Linear(dims[i], dims[i + 1]), activation()])
    if if_raw_out:
        # delete the activation function of the output layer to keep raw output
        del net_list[-1]
    return nn.Sequential(*net_list)


def layer_init_with_orthogonal(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
