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


class DodgerEnv(gym.Env):
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
        self.max_steps = 60
        self.obs_traj_len = 1

        self.init_x_mean, self.init_x_std = -1.0, 0.3 # self.traj_A, 0.0
        self.init_v_mean, self.init_v_std = 0.0, 1.0 # 0.0, 0.0
        self.obstacle_radius = 0.6
        self.obstacle_pos = torch.zeros(dim, device=self.device)

        # state
        self.dim = dim
        self.env_num = env_num
        self.ood_mode = ood_mode
        self.expert_mode = expert_mode
        self.state_dim = 3*dim
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
            goal = torch.abs(torch.randn((size, self.dim), device=self.device)) * self.init_x_std - self.init_x_mean
            mass = torch.abs(torch.randn((size, self.dim), device=self.device)) * self.mass_std + self.mass_mean
            delay = torch.abs(torch.randn((size, 1), device=self.device)) * self.delay_std + self.delay_mean
            decay = torch.abs(torch.randn((size, self.dim), device=self.device)) * self.decay_std + self.decay_mean
            res_dyn_param = torch.abs(torch.randn(((size, self.dim), self.res_dyn_mlp.param_dim), device=self.device)) * self.res_dyn_param_std + self.res_dyn_param_mean
        else:
            x = torch.randn((size, self.dim), device=self.device) * self.init_x_std + self.init_x_mean
            v = torch.randn((size, self.dim), device=self.device) * self.init_v_std + self.init_v_mean
            goal = torch.randn((size, self.dim), device=self.device) * self.init_x_std - self.init_x_mean
            mass = torch.randn((size, self.dim), device=self.device) * self.mass_std + self.mass_mean
            delay = torch.randn((size, 1), device=self.device) * self.delay_std + self.delay_mean
            decay = torch.randn((size, self.dim), device=self.device) * self.decay_std + self.decay_mean
            res_dyn_param = torch.randn((size, self.dim), device=self.device) * self.res_dyn_param_std + self.res_dyn_param_mean
        mass = torch.clip(mass, self.mass_min, self.mass_max)
        delay = torch.clip(delay, self.delay_min, self.delay_max).type(torch.int)
        decay = torch.clip(decay, self.decay_min, self.decay_max)
        res_dyn_param = torch.clip(res_dyn_param, self.res_dyn_param_min, self.res_dyn_param_max)
        return x, v, mass, delay, decay, res_dyn_param, goal
    
    def step(self, action):
        current_force = torch.clip(action*self.force_scale, -self.max_force, self.max_force) 
        if (self.delay == 0).all():
            self.force = current_force
        else:
            self.force_history.append(current_force)
            self.force_history.pop(0)
            self.force = torch.zeros((self.env_num, self.dim), device=self.device)
            for i in range(self.delay_max):
                env_mask = (self.delay == i)[:,0]
                self.force[env_mask] = self.force_history[-i-1][env_mask]
        
        self.force += self.disturb
        self.decay_force = self.decay * self.v
        self.force -= self.decay_force
        self.res_dyn_force = self.res_dyn_mlp(torch.cat([self.v, action, self.res_dyn_param], dim=-1)) * self.res_dyn_scale
        self.force += self.res_dyn_force
        
        self.x += self.v * self.tau
        self.v += self.force / self.mass * self.tau
        reward = 1.0 - torch.norm(self.x-self.goal,dim=1) - torch.norm(self.v,dim=1)*0.1
        dist2obstacle = torch.norm(self.x-self.obstacle_pos,dim=1) - self.obstacle_radius
        reward[dist2obstacle < 0] += (dist2obstacle[dist2obstacle < 0]-1.0)

        self.step_cnt += 1

        single_done = self.step_cnt >= self.max_steps
        done = torch.ones(self.env_num, device=self.device)*single_done
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
        self.x, self.v, self.mass, self.delay, self.decay, self.res_dyn_param, self.goal = self._get_initial_state()
        self.force_history = [torch.zeros((self.env_num, self.dim), device=self.device)] * self.delay_max
        return self._get_obs()

    def _get_obs(self):
        obs = torch.concat([self.x, self.v, self.goal], dim=-1)
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

def get_dodger_policy(env, policy_name = "ppo"):
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


def test_dodger(env, policy, save_path = None):
    state = env.reset()
    x_list, v_list, a_list, force_list, disturb_list, decay_list, res_dyn_list, mass_list, delay_list, res_dyn_param_list, traj_x_list, traj_v_list, r_list, done_list = [], [], [], [], [], [], [], [], [], [], [], [], [], []

    time_limit = 60*5
    for t in range(time_limit):
        act = policy(state)
        state, rew, done, info = env.step(act)  # take a random action
        x_list.append(state[0,:env.dim].numpy())
        v_list.append(state[0,env.dim:env.dim*2].numpy()*0.3)
        traj_x_list.append(env.goal[0].numpy())
        traj_v_list.append(env.obstacle_pos.numpy()*0.3)
        a_list.append(act[0].numpy())
        r_list.append(rew[0].item())
        force_list.append(env.force[0].numpy())
        disturb_list.append(env.disturb[0].numpy())
        decay_list.append(env.decay_force[0].numpy())
        res_dyn_list.append(env.res_dyn_force[0].numpy())
        mass_list.append(env.mass[0,:].numpy()*10)
        delay_list.append(env.delay[0,0].item()*0.2)
        res_dyn_param_list.append(env.res_dyn_param[0,:].numpy())
        if done:
            done_list.append(t)
    # set matplotlib style
    plt.style.use('seaborn')
    # plot x_list, v_list, action_list in three subplots
    plot_num = 2*env.dim+1
    fig, axs = plt.subplots(plot_num, 1, figsize=(10, 3*plot_num))
    x_numpy, v_array = np.array(x_list), np.array(v_list)
    traj_x_array, traj_v_array = np.array(traj_x_list), np.array(traj_v_list)
    for i in range(env.dim):
        axs[i].set_title(f"kinematics measurement dim={i}")
        axs[i].plot(x_numpy[:,i], label="x")
        axs[i].plot(v_array[:,i], label="v*0.3", alpha=0.3)
        axs[i].plot(traj_x_array[:,i], label="goal")
        axs[i].plot(traj_v_array[:,i], label="obstacle", alpha=0.3)
    res_dyn_numpy, a_numpy, force_array = np.array(res_dyn_list), np.array(a_list), np.array(force_list)
    disturb_array, decay_array = np.array(disturb_list), np.array(decay_list)
    for i in range(env.dim):
        axs[env.dim+i].set_title(f"force measurement dim={i}")
        axs[env.dim+i].plot(res_dyn_numpy[:,i], label="res_dyn")
        axs[env.dim+i].plot(a_numpy[:,i], label="action", linestyle='--', alpha=0.5)
        axs[env.dim+i].plot(force_array[:,i], label="force", alpha=0.5)
        axs[env.dim+i].plot(disturb_array[:,i], label="disturb", alpha=0.2)
        axs[env.dim+i].plot(decay_array[:,i], label="decay", alpha=0.3)
    mass_array = np.array(mass_list)
    res_dyn_param_numpy = np.array(res_dyn_param_list)
    axs[env.dim*2].set_title(f"system parameters and reward")
    for i in range(env.dim):
        axs[env.dim*2].plot(mass_array[:,i], label=f"mass-{i}*10", alpha=0.5)
        axs[env.dim*2].plot(res_dyn_param_numpy[:,i], label=f"res_dyn_param-{i}", alpha=0.5)
    axs[env.dim*2].plot(delay_list, label="delay*0.2", alpha=0.5)
    axs[env.dim*2].plot(r_list, 'y', label="reward")
    # add mean reward to axs 2 as text
    axs[env.dim*2].text(0.5, 0.5, f"mean reward: {np.mean(r_list):.3f}")
    # draw vertical lines for done
    for t in done_list:
        axs[0].axvline(t, color="red", linestyle="--", label='reset')
    for i in range(plot_num):
        axs[i].legend()
    # save the plot as image
    if save_path == None:
        package_path = os.path.dirname(adaptive_control_gym.__file__)
        save_path = f"{package_path}/../results/dodger.png"
    plt.savefig(save_path)
    env.close()

if __name__ == "__main__":
    env = DodgerEnv(env_num=1, gpu_id = -1, seed=0, expert_mode=True, dim=2)
    policy = get_dodger_policy(env, policy_name = "random")
    test_dodger(env, policy)