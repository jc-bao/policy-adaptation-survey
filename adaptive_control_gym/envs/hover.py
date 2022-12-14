import torch
from torch import nn
import gym
import numpy as np
import torch
import control as ct
from icecream import ic
import os
from matplotlib import pyplot as plt

import adaptive_control_gym
from adaptive_control_gym import controllers as ctrl


class HoverEnv(gym.Env):
    def __init__(self, 
        dim: int = 1, env_num: int = 1, gpu_id: int = 0, seed:int = 0, 
        expert_mode:bool = False, ood_mode:bool = False, 
        mass_uncertainty_rate:float=0.0, 
        disturb_uncertainty_rate:float=0.0, disturb_period: int = 15,
        delay_mean:float = 0.0, delay_std:float = 0.0,
        decay_mean:float = 0.2, decay_std:float = 0.0, 
        res_dyn_scale: float = 0.6, res_dyn_param_std:float = 1.0,
        ):
        torch.manual_seed(seed)
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id>=0)) else "cpu")
        # parameters
        self.mass_mean, self.mass_std = 0.1, 0.1 * mass_uncertainty_rate
        self.mass_min, self.mass_max = 0.04, 3.0
        self.delay_mean, self.delay_std = delay_mean, delay_std
        self.delay_min, self.delay_max = 0, 10
        self.decay_mean, self.decay_std = decay_mean, decay_std
        self.decay_min, self.decay_max = 0.0, 0.5
        self.disturb_mean, self.disturb_std = 0.0, 1.0 * disturb_uncertainty_rate
        self.disturb_period = disturb_period

        self.res_dyn_scale = res_dyn_scale
        self.res_dyn_mlp = ResDynMLP(input_dim=dim*2, output_dim=dim).to(self.device)
        self.res_dyn_param_mean, self.res_dyn_param_std = 0.0, res_dyn_param_std
        self.res_dyn_param_min, self.res_dyn_param_max = -1.0, 1.0

        self.traj_T, self.traj_A = 60, 1
        # generate a sin trajectory with torch
        self.traj_t = torch.arange(0, self.traj_T, 1).float().to(self.device)
        self.traj_x = self.traj_A * torch.sin(2 * np.pi * self.traj_t / self.traj_T)

        self.init_x_mean, self.init_x_std = 0.0, 1.0
        self.init_v_mean, self.init_v_std = 0.0, 1.0
        self.tau = 1.0/30.0  # seconds between state updates
        self.force_scale = 1.0
        self.max_force = 1.0

        # state
        self.dim = dim
        self.env_num = env_num
        self.ood_mode = ood_mode
        self.expert_mode = expert_mode
        if expert_mode:
            self.state_dim = 2*dim+1+dim
        else:
            self.state_dim = 2*dim
        self.action_dim = dim
        self.max_steps = 60

        self.reset()

        # dynamic
        A = np.array([[0, 1], [0, 0]])
        self.A = np.stack([A] * self.env_num, axis=0)
        B = (1 / self.mass).detach().cpu().numpy()
        self.B = np.expand_dims(np.stack([np.zeros_like(B), B], axis=1), axis=2)
        # self.dynamic_info = self._get_dynamic_info()
        # ic(self.dynamic_info)

        self.action_space = gym.spaces.Box(
            low=-self.max_force, high=self.max_force, shape=(self.dim,), dtype=float)

    def _get_dynamic_info(self):
        eig_A = np.linalg.eig(self.A[0])
        ctrb_AB = ct.ctrb(self.A[0], self.B[0])
        ctrb_rank = np.linalg.matrix_rank(ctrb_AB)
        return {
            'eig_A': eig_A,
            'ctrb_AB': ctrb_AB,
            'ctrb_rank': ctrb_rank,
        }
    
    def _get_initial_state(self, size = None):
        if size is None:
            size = self.env_num
        if self.ood_mode:
            x = torch.abs(torch.randn((size, self.dim), device=self.device)) * self.init_x_std + self.init_x_mean
            v = torch.abs(torch.randn((size, self.dim), device=self.device)) * self.init_v_std + self.init_v_mean
            mass = torch.abs(torch.randn((size, self.dim), device=self.device)) * self.mass_std + self.mass_mean
            delay = torch.abs(torch.randn((size, 1), device=self.device)) * self.delay_std + self.delay_mean
            decay = torch.abs(torch.randn((size, self.dim), device=self.device)) * self.decay_std + self.decay_mean
            res_dyn_param = torch.abs(torch.randn(((size, self.dim), self.res_dyn_mlp.param_dim), device=self.device)) * self.res_dyn_param_std + self.res_dyn_param_mean
        else:
            x = torch.randn((size, self.dim), device=self.device) * self.init_x_std + self.init_x_mean
            v = torch.randn((size, self.dim), device=self.device) * self.init_v_std + self.init_v_mean
            mass = torch.randn((size, self.dim), device=self.device) * self.mass_std + self.mass_mean
            delay = torch.randn((size, 1), device=self.device) * self.delay_std + self.delay_mean
            decay = torch.randn((size, self.dim), device=self.device) * self.decay_std + self.decay_mean
            res_dyn_param = torch.randn((size, self.dim), device=self.device) * self.res_dyn_param_std + self.res_dyn_param_mean
        mass = torch.clip(mass, self.mass_min, self.mass_max)
        delay = torch.clip(delay, self.delay_min, self.delay_max).type(torch.int)
        decay = torch.clip(decay, self.decay_min, self.decay_max)
        res_dyn_param = torch.clip(res_dyn_param, self.res_dyn_param_min, self.res_dyn_param_max)
        return x, v, mass, delay, decay, res_dyn_param
    
    def step(self, action):
        current_force = torch.clip(action*self.force_scale, -self.max_force, self.max_force) 
        if (self.delay == 0).all():
            self.force = current_force
        else:
            self.force_history.append(current_force)
            self.force_history.pop(0)
            self.force = torch.zeros((self.env_num, self.dim), device=self.device)
            for i in range(self.delay_max):
                env_mask = (self.delay == i)
                self.force[env_mask] = self.force_history[-i-1][env_mask]
        
        self.force += self.disturb
        self.decay_force = self.decay * self.v
        self.force -= self.decay_force
        self.res_dyn_force = self.res_dyn_mlp(torch.cat([self.v, self.res_dyn_param], dim=-1)) * self.res_dyn_scale
        self.force += self.res_dyn_force
        
        self.x += self.v * self.tau
        self.v += self.force / self.mass * self.tau
        self.step_cnt += 1
        single_done = self.step_cnt >= self.max_steps
        done = torch.ones(self.env_num, device=self.device)*single_done
        reward = 1.0 - torch.norm(self.x-self.traj_x[self.step_cnt],dim=1) - torch.norm(self.v,dim=1)*0.0
        # update disturb
        if self.step_cnt % self.disturb_period == 0:
            self._set_disturb()
        if single_done:
            self.reset()
        info = {
            'mass': self.mass, 
            'disturb': self.disturb,
        }
        return self._get_obs(), reward, done, info

    def reset(self):
        self.step_cnt = 0
        self._set_disturb()
        self.x, self.v, self.mass, self.delay, self.decay, self.res_dyn_param = self._get_initial_state()
        self.force_history = [torch.zeros((self.env_num, self.dim), device=self.device)] * self.delay_max
        return self._get_obs()

    def _get_obs(self):
        if self.expert_mode:
            return torch.concat([self.x,self.v,self.disturb, self.mass], dim=-1)
        else:
            return torch.concat([self.x,self.v], dim=-1)

    def _set_disturb(self):
        self.disturb = torch.randn((self.env_num,self.dim), device=self.device) * self.disturb_std + self.disturb_mean

class ResDynMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        embedding_size = 64
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, embedding_size),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_size, output_dim),
            nn.Tanh()
        )
        # freeze the network
        for p in self.mlp.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.mlp(x)*3 # empirical value for output range

def get_hover_policy(env, policy_name = "ppo"):
    if policy_name == "lqr":
        Q = np.array([[50, 0],[0, 1]])
        R = 1
        policy = ctrl.LRQ(env.A, env.B, Q, R, gpu_id = -1)
    elif policy_name == "random":
        policy = ctrl.Random(env.action_space)
    elif policy_name == "ppo":
        policy = torch.load('../../results/rl/actor_ppo.pt').to('cpu')
    else: 
        raise NotImplementedError
    return policy


def test_hover(env, policy):
    state = env.reset()
    x_list, v_list, a_list, force_list, disturb_list, decay_list, res_dyn_list, r_list, done_list = [], [], [], [], [], [], [], [], []

    for t in range(180):
        act = policy(state)
        state, rew, done, info = env.step(act)  # take a random action
        x_list.append(state[0,0].item())
        v_list.append(state[0,1].item()*0.3)
        a_list.append(act[0,0].item())
        r_list.append(rew[0].item())
        force_list.append(env.force[0].item())
        disturb_list.append(env.disturb[0].item())
        decay_list.append(env.decay_force[0].item())
        res_dyn_list.append(env.res_dyn_force[0].item())
        if done:
            done_list.append(t)
    # set matplotlib style
    plt.style.use('seaborn')
    # plot x_list, v_list, action_list in three subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    axs[0].plot(x_list, label="x")
    axs[0].plot(v_list, label="v*0.3", alpha=0.3)
    axs[1].plot(res_dyn_list, label="res_dyn")
    axs[1].plot(a_list, label="action", linestyle='--', alpha=0.5)
    axs[1].plot(force_list, label="force", alpha=0.5)
    axs[1].plot(disturb_list, label="disturb", alpha=0.2)
    axs[1].plot(decay_list, label="decay", alpha=0.3)
    axs[2].plot(r_list, 'y', label="reward")
    # draw vertical lines for done
    for t in done_list:
        axs[0].axvline(t, color="red", linestyle="--", label='reset')
    # plot horizontal line for x=0
    axs[0].plot(env.traj_x.numpy(), color="black", linestyle="--", label='ref traj')
    for i in range(3):
        axs[i].legend()
    # save the plot as image
    package_path = os.path.dirname(adaptive_control_gym.__file__)
    plt.savefig(f"{package_path}/../results/hover.png")
    env.close()

if __name__ == "__main__":
    env = HoverEnv(env_num=1, gpu_id = -1, seed=0)
    policy = get_hover_policy(env, policy_name = "ppo")
    test_hover(env, policy)