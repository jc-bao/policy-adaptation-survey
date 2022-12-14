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
        mass_mean:float = 0.05, mass_std:float=0.02, 
        disturb_uncertainty_rate:float=0.0, disturb_period: int = 15,
        delay_mean:float = 5.0, delay_std:float = 2.0,
        decay_mean:float = 0.2, decay_std:float = 0.1, 
        res_dyn_scale: float = 1.0, res_dyn_param_std:float = 1.0,
        ):
        torch.manual_seed(seed)
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id>=0)) else "cpu")
        # parameters
        self.mass_mean, self.mass_std = mass_mean, mass_std
        self.mass_min, self.mass_max = 0.01, 1.0
        self.delay_mean, self.delay_std = delay_mean, delay_std
        self.delay_min, self.delay_max = 0, 10
        self.decay_mean, self.decay_std = decay_mean, decay_std
        self.decay_min, self.decay_max = 0.0, 0.5
        self.disturb_mean, self.disturb_std = 0.0, 1.0 * disturb_uncertainty_rate
        self.disturb_period = disturb_period

        self.res_dyn_scale = res_dyn_scale
        self.res_dyn_mlp = ResDynMLP(input_dim=dim*3, output_dim=dim).to(self.device)
        self.res_dyn_param_mean, self.res_dyn_param_std = 0.0, res_dyn_param_std
        self.res_dyn_param_min, self.res_dyn_param_max = -1.0, 1.0

        self.tau = 1.0/30.0  # seconds between state updates
        self.force_scale = 1.0
        self.max_force = 1.0

        # generate a sin trajectory with torch
        self.max_steps = 120
        self.obs_traj_len = 10
        # self.traj_T_cnt, self.traj_A = 60, 1
        # self.traj_T = self.traj_T_cnt * self.tau
        # self.traj_t = torch.arange(0, self.traj_T_cnt*10, 1).float().to(self.device)
        # self.traj_x = self.traj_A * torch.cos(2 * np.pi * self.traj_t / self.traj_T_cnt)
        # self.traj_v = -self.traj_A * 2 * np.pi * torch.sin(2 * np.pi * self.traj_t / self.traj_T_cnt) / self.traj_T

        self.init_x_mean, self.init_x_std = 0.0, 1.0 # self.traj_A, 0.0
        self.init_v_mean, self.init_v_std = 0.0, 1.0 # 0.0, 0.0

        # state
        self.dim = dim
        self.env_num = env_num
        self.ood_mode = ood_mode
        self.expert_mode = expert_mode
        self.state_dim = 4*dim+2*dim*self.obs_traj_len
        if expert_mode:
            self.state_dim += (1+dim*3)
        self.action_dim = dim

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

    def _generate_traj(self):
        base_w = 2 * np.pi / self.max_steps
        t = torch.arange(0, self.max_steps+self.obs_traj_len, 1, device=self.device)
        t = torch.tile(t, (self.env_num, self.dim))
        x = torch.zeros((self.env_num, self.max_steps+self.obs_traj_len), device=self.device)
        v = torch.zeros((self.env_num, self.max_steps+self.obs_traj_len), device=self.device)
        for i in range(4):
            A = torch.rand((self.env_num,self.dim), device=self.device)*(2**(-i))
            w = base_w*(2**i)
            phase = torch.rand((self.env_num,self.dim), device=self.device)*(2*np.pi)
            x += A*torch.cos(t*w+phase)
            v -= w*A*torch.sin(t*w+phase)/self.tau
        return x, v

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
        self.res_dyn_force = self.res_dyn_mlp(torch.cat([self.v, action, self.res_dyn_param], dim=-1)) * self.res_dyn_scale
        self.force += self.res_dyn_force
        
        self.x += self.v * self.tau
        self.v += self.force / self.mass * self.tau
        self.step_cnt += 1
        single_done = self.step_cnt >= self.max_steps
        done = torch.ones(self.env_num, device=self.device)*single_done
        reward = 1.0 - torch.norm(self.x-self.traj_x[:,self.step_cnt-1],dim=1) - torch.norm(self.v-self.traj_v[:,self.step_cnt-1],dim=1)*0.2
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
        self.traj_x, self.traj_v = self._generate_traj()
        return self._get_obs()

    def _get_obs(self):
        future_traj_x = self.traj_x[:, self.step_cnt:self.step_cnt+self.obs_traj_len]
        future_traj_v = self.traj_v[:, self.step_cnt:self.step_cnt+self.obs_traj_len]
        err_x, err_v = future_traj_x[:,0] - self.x, future_traj_v[:,0] - self.v
        obs = torch.concat([self.x, self.v, err_x, err_v, torch.tile(future_traj_x, dims=(self.env_num, 1)), torch.tile(future_traj_v, dims=(self.env_num, 1))], dim=-1)
        if self.expert_mode:
            obs = torch.concat([obs, self.mass, self.delay, self.decay, self.res_dyn_param], dim=-1)
        return obs

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
        policy = ctrl.Random(env.action_dim)
    elif policy_name == "ppo":
        policy = torch.load('../../results/rl/actor_ppo_OODFalse_EXPTrue_S0.pt').to('cpu')
    else: 
        raise NotImplementedError
    return policy


def test_hover(env, policy, save_path = None):
    state = env.reset()
    x_list, v_list, a_list, force_list, disturb_list, decay_list, res_dyn_list, mass_list, delay_list, res_dyn_param_list, traj_x_list, traj_v_list, r_list, done_list = [], [], [], [], [], [], [], [], [], [], [], [], [], []

    time_limit = 120*5
    for t in range(time_limit):
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
        mass_list.append(env.mass[0,0].item()*10)
        delay_list.append(env.delay[0,0].item()*0.2)
        traj_x_list.append(env.traj_x[0,t%env.max_steps].item())
        traj_v_list.append(env.traj_v[0,t%env.max_steps].item()*0.3)
        res_dyn_param_list.append(env.res_dyn_param[0,0].item())
        if done:
            done_list.append(t)
    # set matplotlib style
    plt.style.use('seaborn')
    # plot x_list, v_list, action_list in three subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    axs[0].plot(x_list, label="x")
    axs[0].plot(traj_x_list, label="traj_x")
    axs[0].plot(v_list, label="v*0.3", alpha=0.3)
    axs[0].plot(traj_v_list, label="traj_v*0.3", alpha=0.3)
    axs[1].plot(res_dyn_list, label="res_dyn")
    axs[1].plot(a_list, label="action", linestyle='--', alpha=0.5)
    axs[1].plot(force_list, label="force", alpha=0.5)
    axs[1].plot(disturb_list, label="disturb", alpha=0.2)
    axs[1].plot(decay_list, label="decay", alpha=0.3)
    axs[2].plot(mass_list, label="mass*10", alpha=0.5)
    axs[2].plot(delay_list, label="delay*0.2", alpha=0.5)
    axs[2].plot(res_dyn_param_list, label="res_dyn_param", alpha=0.5)
    axs[2].plot(r_list, 'y', label="reward")
    # add mean reward to axs 2 as text
    axs[2].text(0.5, 0.5, f"mean reward: {np.mean(r_list):.3f}")
    # draw vertical lines for done
    for t in done_list:
        axs[0].axvline(t, color="red", linestyle="--", label='reset')
    for i in range(3):
        axs[i].legend()
    # save the plot as image
    if save_path == None:
        package_path = os.path.dirname(adaptive_control_gym.__file__)
        save_path = f"{package_path}/../results/hover.png"
    plt.savefig(save_path)
    env.close()

if __name__ == "__main__":
    env = HoverEnv(env_num=1, gpu_id = -1, seed=0, expert_mode=True)
    policy = get_hover_policy(env, policy_name = "random")
    test_hover(env, policy)