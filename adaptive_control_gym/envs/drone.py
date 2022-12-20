import os

import control as ct
import gym
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from icecream import ic
from matplotlib import pyplot as plt
from torch import nn
from tqdm import trange

import adaptive_control_gym
from adaptive_control_gym import controllers as ctrl


class DroneEnv(gym.Env):
    def __init__(self, 
        env_num: int = 1, gpu_id: int = 0, seed:int = 0, 
        expert_mode:bool = False, ood_mode:bool = False,
        ):
        torch.random.set_rng_state(torch.manual_seed(1024).get_state())
        torch.manual_seed(seed)
        
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id>=0)) else "cpu")
        # parameters
        self.dim=dim=3
        self.disturb_period = 120
        self.model_delay_alpha = 0.9
        self.res_dyn_scale = 0.0  / (2**dim)
        self.res_dyn_param_dim = 0
        self.curri_param = 1.0

        self.mass_min, self.mass_max = 0.01, 0.04
        self.delay_min, self.delay_max = 0, 0
        self.decay_min, self.decay_max = 0.0, 0.2 
        self.res_dyn_param_min, self.res_dyn_param_max = -1.0, 1.0
        self.disturb_min, self.disturb_max = -0.8, 0.8
        self.action_noise_std, self.obs_noise_std = 0.00, 0.00

        # generated parameters
        self.rotate_mass_scale = 1
        self.mass_mean, self.mass_std = (self.mass_min+self.mass_max)/2, (self.mass_max-self.mass_min)/2
        if self.mass_std == 0:
            self.mass_std = 1.0
        self.delay_mean, self.delay_std = 0, 1
        self.decay_mean, self.decay_std = (self.decay_max+self.decay_min)/2, (self.decay_max-self.decay_min)/2
        if self.decay_std == 0:
            self.decay_std = 1.0
        self.disturb_mean, self.disturb_std = 0, self.mass_mean*10
        if self.disturb_std == 0:
            self.disturb_std = 1.0

        self.res_dyn_mlp = ResDynMLP(input_dim=dim*3+self.res_dyn_param_dim, output_dim=dim).to(self.device)
        self.res_dyn_param_mean, self.res_dyn_param_std = (self.res_dyn_param_min+self.res_dyn_param_max)/2, (self.res_dyn_param_max-self.res_dyn_param_min)/2
        if self.res_dyn_param_std == 0:
            self.res_dyn_param_std = 1.0
        

        self.tau = 1.0/30.0  # seconds between state updates
        self.force_scale = 1.0
        self.max_force = 1.0
        self.gravity = 9.8

        # generate a sin trajectory with torch
        self.obs_traj_len = 5
        self.traj_scale = 0.0
        self.traj_T = 360
        if self.traj_scale == 0:
            self.max_steps = 120
        else:
            self.max_steps = 360

        self.init_x_mean, self.init_x_std = 0.0, 0.8 # self.traj_A, 0.0
        self.init_v_mean, self.init_v_std = 0.0, 1.0 # 0.0, 0.0
        self.x_min, self.x_max = -2.0, 2.0

        # state
        self.dim = dim
        self.env_num = env_num
        self.ood_mode = ood_mode
        self.expert_mode = expert_mode
        self.state_dim = 4*dim+2*dim*self.obs_traj_len
        self.expert_dim = (self.res_dyn_param_dim+dim*3+1)
        if expert_mode:
            self.state_dim += self.expert_dim
        self.action_dim = dim - 1

        self.reset()

        # dynamic
        A = np.array([[0, 1], [0, 0]])
        self.A = np.stack([A] * self.env_num, axis=0)
        B = (1 / self.mass).cpu().numpy()
        self.B = np.expand_dims(np.stack([np.zeros_like(B), B], axis=1), axis=2)
        # self.dynamic_info = self._get_dynamic_info()
        # ic(self.dynamic_info)

        self.action_space = gym.spaces.Box(
            low=-self.max_force, high=self.max_force, shape=(self.dim,), dtype=float)

    def _generate_traj(self):
        base_w = 2 * np.pi / self.traj_T
        t = torch.arange(0, self.max_steps+self.obs_traj_len, 1, device=self.device)
        t = torch.tile(t, (self.env_num, self.dim, 1))
        x = torch.zeros((self.env_num, self.dim, self.max_steps+self.obs_traj_len), device=self.device)
        v = torch.zeros((self.env_num, self.dim, self.max_steps+self.obs_traj_len), device=self.device)
        for i in np.arange(0,1,1):
            A = (torch.rand((self.env_num,self.dim, 1), device=self.device)*0.5+0.5)*(2.0**(-i))
            w = base_w*(2**i)
            phase = torch.rand((self.env_num,self.dim, 1), device=self.device)*(2*np.pi)
            x += A*torch.cos(t*w+phase)
            v -= w*A*torch.sin(t*w+phase)/self.tau
        x *= self.traj_scale
        v *= self.traj_scale
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
            mass = -(torch.rand((size, self.dim), device=self.device))* (self.mass_max-self.mass_min) * 0.5 * self.curri_param + (self.mass_max+self.mass_min) * 0.5
            delay = -(torch.rand((size, 1), device=self.device)) * (self.delay_max-self.delay_min) * 0.5 * self.curri_param + (self.delay_max+self.delay_min) * 0.5
            decay = -(torch.rand((size, self.dim), device=self.device))* (self.decay_max-self.decay_min) * 0.5 * self.curri_param + (self.decay_max+self.decay_min) * 0.5
            res_dyn_param = -(torch.rand((size, self.res_dyn_param_dim), device=self.device)) * (self.res_dyn_param_max-self.res_dyn_param_min) * 0.5 * self.curri_param + (self.res_dyn_param_max+self.res_dyn_param_min) * 0.5
        else:
            mass = (torch.rand((size, self.dim), device=self.device)*2-1)* (self.mass_max-self.mass_min) *self.curri_param + (self.mass_min+self.mass_max)*0.5
            delay = (torch.rand((size, 1), device=self.device)*2-1) * (self.delay_max-self.delay_min) *self.curri_param + (self.delay_min+self.delay_max)*0.5
            decay = (torch.rand((size, self.dim), device=self.device)*2-1)* (self.decay_max-self.decay_min) *self.curri_param + (self.decay_min+self.decay_max)*0.5
            res_dyn_param = (torch.rand((size, self.res_dyn_param_dim), device=self.device)*2-1) * (self.res_dyn_param_max-self.res_dyn_param_min) *self.curri_param + (self.res_dyn_param_min+self.res_dyn_param_max)*0.5
        mass = torch.clip(mass, self.mass_min, self.mass_max)
        mass[:, 1] = mass[:, 0]
        mass[:, 2] *= self.rotate_mass_scale
        delay = torch.clip(delay, self.delay_min, self.delay_max).type(torch.int)
        decay = torch.clip(decay, self.decay_min, self.decay_max)
        res_dyn_param = torch.clip(res_dyn_param, self.res_dyn_param_min, self.res_dyn_param_max)

        x = (torch.rand((size, self.dim), device=self.device)*2-1)* self.init_x_std + self.init_x_mean
        x[:, 2] = torch.rand((size), device=self.device) * 2 * np.pi - np.pi
        v = (torch.rand((size, self.dim), device=self.device)*2-1)* self.init_v_std + self.init_v_mean
        x = torch.clip(x, self.x_min, self.x_max)
        return x, v, mass, delay, decay, res_dyn_param
    
    def step(self, action):
        # add noise to action
        action += torch.randn_like(action, device=self.device) * self.action_noise_std
        current_force = torch.clip(action*self.force_scale, -self.max_force, self.max_force) 
        if (self.delay == 0).all():
            force = current_force
        else:
            self.force_history.append(current_force)
            self.force_history.pop(0)
            force = torch.zeros((self.env_num, self.dim), device=self.device)
            for i in range(self.delay_max):
                env_mask = (self.delay == i)[:,0]
                self.force[env_mask] = self.force_history[-i-1][env_mask]
        theta = self.x[:,[2]]
        F, M = force[:,[0]], force[:,[1]]
        force = torch.cat([F*torch.sin(theta), F*torch.cos(theta), M], dim=-1)
        self.force = force*self.model_delay_alpha + self.force*(1-self.model_delay_alpha)
        
        self.force += self.disturb
        self.decay_force = self.decay * self.v
        self.force -= self.decay_force
        if self.res_dyn_scale > 0:
            self.res_dyn_force = self.res_dyn_mlp(torch.cat([self.x, self.v*0.3, self.force, self.res_dyn_param], dim=-1)) * self.res_dyn_scale
            self.force += self.res_dyn_force
        else:
            self.res_dyn_force = torch.zeros_like(self.force)
        
        self.x += self.v * self.tau
        self.v += self.force / self.mass * self.tau
        self.v[:,1] -= self.gravity * self.tau
        self.x[:,:2] = torch.clip(self.x[:,:2], self.x_min, self.x_max)
        err_x = torch.norm((self.x-self.traj_x[...,self.step_cnt])[:,:2],dim=1)
        err_v = torch.norm((self.v-self.traj_v[...,self.step_cnt])[:,:2],dim=1)
        reward = 1.0 - err_x - err_v*0.1

        # for hover task, add penalty for angular velocity
        if self.traj_scale == 0:
            reward -= torch.abs(self.x[:,2])*0.00
            reward -= torch.abs(self.v[:,2])*0.02

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
            'err_x': err_x, 
            'err_v': err_v
        }
        next_obs = self._get_obs()
        # add gaussian noise to next_obs
        next_obs += torch.randn_like(next_obs, device=self.device) * self.obs_noise_std
        return next_obs, reward, done, info

    def reset(self):
        self.step_cnt = 0
        self.x, self.v, self.mass, self.delay, self.decay, self.res_dyn_param = self._get_initial_state()
        self.force_history = [torch.zeros((self.env_num, self.dim), device=self.device)] * self.delay_max
        self.force = torch.zeros((self.env_num, self.dim), device=self.device)
        self.traj_x, self.traj_v = self._generate_traj()
        self._set_disturb()
        return self._get_obs()

    def _get_obs(self):
        future_traj_x = self.traj_x[..., self.step_cnt:self.step_cnt+self.obs_traj_len]
        future_traj_v = self.traj_v[..., self.step_cnt:self.step_cnt+self.obs_traj_len]
        err_x, err_v = future_traj_x[..., 0] - self.x, future_traj_v[..., 0] - self.v
        obs = torch.concat([self.x, self.v*0.3, err_x, err_v*0.3, future_traj_x.reshape(self.env_num,-1), future_traj_v.reshape(self.env_num, -1)], dim=-1)
        if self.expert_mode:
            obs = torch.concat([obs, (self.mass-self.mass_mean)/self.mass_std, (self.disturb-self.disturb_mean)/self.disturb_std, self.delay/10.0, (self.decay-self.decay_mean)/self.decay_std, self.res_dyn_param], dim=-1)
        return obs

    def _set_disturb(self):
        if self.ood_mode:
            self.disturb = -(torch.rand((self.env_num,self.dim), device=self.device)) * (self.disturb_max-self.disturb_min) * 0.5 * self.curri_param + (self.disturb_min+self.disturb_max)*0.5
        else:
            self.disturb = (torch.rand((self.env_num,self.dim), device=self.device)*2-1) * (self.disturb_max-self.disturb_min) * 0.5 * self.curri_param + (self.disturb_min+self.disturb_max)*0.5
        self.disturb *= (self.mass*self.gravity)
        # self.disturb = torch.ones_like(self.mass, device=self.device) * 0.25

class ResDynMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        embedding_size = 128
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, embedding_size),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_size, output_dim),
            nn.Tanh()
        )
        # initialize weights with uniform and bias with uniform
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1)
                nn.init.uniform_(m.bias, -0.2, 0.21)
        # freeze the network
        for p in self.mlp.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.mlp(x) # empirical value for output range

def get_drone_policy(env, policy_name = "ppo"):
    if policy_name == "lqr":
        Q = np.array([[50, 0],[0, 1]])
        R = 1
        policy = ctrl.LRQ(env.A, env.B, Q, R, gpu_id = -1)
    elif policy_name == "random":
        policy = ctrl.Random(env.action_dim)
    elif policy_name == "ppo":
        policy = torch.load('../../results/rl/actor_ppo_EXPTrue_OODFalse_S0.pt').to('cpu')
        # freeze the policy
        # for p in policy.parameters():
        #     p.requires_grad = False
    else: 
        raise NotImplementedError
    return policy

def eval_drone(policy, env_args, gpu_id):
    # freeze policy
    for p in policy.parameters():
        p.requires_grad = False
    device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")
    n_mass = 6 #4
    n_decay = 6 #4
    n_param = 1 # 4
    env_num = (n_mass ** 3) * (n_decay**3) * (n_param**4)
    eval_times = 100
    ic(env_num)
    env = DroneEnv(env_num = env_num, gpu_id=gpu_id, **env_args)
    # mass_x, mass_y, mass_z, decay_x, decay_y, decay_z, param_0, param_1, param_2, param_3
    mm_x = torch.linspace(env.mass_min, env.mass_max, n_mass, device=device)
    mm_y = torch.linspace(env.mass_min, env.mass_max, n_mass, device=device)
    mm_z = torch.linspace(env.mass_min, env.mass_max, n_mass, device=device)
    dd_x = torch.linspace(env.decay_min, env.decay_max, n_decay, device=device)
    dd_y = torch.linspace(env.decay_min, env.decay_max, n_decay, device=device)
    dd_z = torch.linspace(env.decay_min, env.decay_max, n_decay, device=device)
    pp_0 = torch.linspace(env.res_dyn_param_min, env.res_dyn_param_max, n_param, device=device)
    pp_1 = torch.linspace(env.res_dyn_param_min, env.res_dyn_param_max, n_param, device=device)
    pp_2 = torch.linspace(env.res_dyn_param_min, env.res_dyn_param_max, n_param, device=device)
    pp_3 = torch.linspace(env.res_dyn_param_min, env.res_dyn_param_max, n_param, device=device)
    # create meshgrid
    mm_x, mm_y, mm_z, dd_x, dd_y, dd_z, pp_0, pp_1, pp_2, pp_3 = torch.meshgrid(mm_x, mm_y, mm_z, dd_x, dd_y, dd_z, pp_0, pp_1, pp_2, pp_3)
    # flatten
    mm_x, mm_y, mm_z, dd_x, dd_y, dd_z, pp_0, pp_1, pp_2, pp_3 = mm_x.reshape(-1), mm_y.reshape(-1), mm_z.reshape(-1), dd_x.reshape(-1), dd_y.reshape(-1), dd_z.reshape(-1), pp_0.reshape(-1), pp_1.reshape(-1), pp_2.reshape(-1), pp_3.reshape(-1)
    # concat
    mass = torch.stack([mm_x, mm_y, mm_z], dim=-1)
    decay = torch.stack([dd_x, dd_y, dd_z], dim=-1)
    res_dyn_param = torch.stack([pp_0, pp_1, pp_2, pp_3], dim=-1)
    all_params = torch.cat([mass, decay, res_dyn_param], dim=-1)
    # evaluate env
    rews = torch.zeros(env_num, device=device)
    for i in trange(eval_times):
        state = env.reset()
        # env.mass, env.decay, env.res_dyn_param = mass, decay, res_dyn_param
        env.mass, env.decay = mass, decay
        for t in range(env.max_steps-1):
            act = policy(state)
            state, reward, done, _ = env.step(act)
            rews += reward
    rews /= (eval_times*env.max_steps)
    # concat all params and rews
    all_params = torch.cat([all_params, rews.unsqueeze(-1)], dim=-1)
    header_list = ['mass_x', 'mass_y', 'mass_z', 'decay_x', 'decay_y', 'decay_z', 'param_0', 'param_1', 'param_2', 'param_3', 'rew']
    df = pd.DataFrame(all_params.cpu().numpy(), columns=header_list)
    df.to_csv('../../results/rl/eval.csv', index=False)

def plot_drone():
    # read the file
    df = pd.read_csv('../../results/rl/eval.csv')
    # use seaborn to plot
    sns.set()
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
    # plot
    fig, axs = plt.subplots(3, 4, figsize=(40, 20))
    sns.lineplot(x='mass_x', y='rew', data=df, ax=axs[0,0])
    sns.lineplot(x='mass_y', y='rew', data=df, ax=axs[0,1])
    sns.lineplot(x='mass_z', y='rew', data=df, ax=axs[0,2])
    sns.lineplot(x='decay_x', y='rew', data=df, ax=axs[1,0])
    sns.lineplot(x='decay_y', y='rew', data=df, ax=axs[1,1])
    sns.lineplot(x='decay_z', y='rew', data=df, ax=axs[1,2])
    sns.lineplot(x='param_0', y='rew', data=df, ax=axs[2,0])
    sns.lineplot(x='param_1', y='rew', data=df, ax=axs[2,1])
    sns.lineplot(x='param_2', y='rew', data=df, ax=axs[2,2])
    sns.lineplot(x='param_3', y='rew', data=df, ax=axs[2,3])
    # save the plot
    plt.savefig('../../results/rl/sensitivity.png')


def test_drone(env:DroneEnv, policy, save_path = None):
    state = env.reset()
    x_list, v_list, a_list, force_list, disturb_list, decay_list, decay_param_list, res_dyn_list, mass_list, delay_list, res_dyn_param_list, traj_x_list, traj_v_list, r_list, done_list = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    js = []
    # check if the policy is torch neural network
    if_policy_grad = (isinstance(policy, nn.Module) and env.expert_mode) and False

    n_ep = 5
    time_limit = env.max_steps*n_ep
    for t in range(time_limit):
        if if_policy_grad:
            # set state as required grad
            state.requires_grad=True
            policy.zero_grad()
        act = policy(state)
        if if_policy_grad:
            # calculate jacobian respect to state
            ic(act.requires_grad, state.requires_grad)
            jacobian = torch.autograd.grad(act, state, grad_outputs=torch.ones_like(act), create_graph=True)[0][0, -env.expert_dim:]
        with torch.no_grad():
            state, rew, done, info = env.step(act)  # take a random action
        x_list.append(state[0,:env.dim].numpy())
        v_list.append(state[0,env.dim:env.dim*2].numpy())
        traj_x_list.append(env.traj_x[0,:,t%env.max_steps].numpy())
        traj_v_list.append(env.traj_v[0,:,t%env.max_steps].numpy()*0.3)
        a_list.append(act[0].detach().numpy())
        r_list.append(rew[0].item())
        force_list.append(env.force[0].numpy())
        disturb_list.append(env.disturb[0].numpy())
        decay_list.append(env.decay_force[0].numpy())
        decay_param_list.append(env.decay[0,:].numpy())
        res_dyn_list.append(env.res_dyn_force[0].numpy())
        mass_list.append(env.mass[0,:].numpy()*10)
        delay_list.append(env.delay[0,0].item()*0.2)
        if if_policy_grad:
            js.append(jacobian.detach().numpy())
        res_dyn_param_list.append(env.res_dyn_param[0,:].numpy())
        if done:
            done_list.append(t)
    # set matplotlib style
    plt.style.use('seaborn')
    # plot x_list, v_list, action_list in three subplots
    plot_num = 2*env.dim+1+if_policy_grad
    fig, axs = plt.subplots(plot_num, 1, figsize=(10, 3*plot_num))
    x_numpy, v_array = np.array(x_list), np.array(v_list)
    traj_x_array, traj_v_array = np.array(traj_x_list), np.array(traj_v_list)
    for i in range(env.dim):
        axs[i].set_title(f"kinematics measurement dim={i}")
        axs[i].plot(x_numpy[:,i], label="x")
        axs[i].plot(v_array[:,i], label="v*0.3", alpha=0.3)
        if i < 2:
            axs[i].plot(traj_x_array[:,i], label="traj_x")
            axs[i].plot(traj_v_array[:,i], label="traj_v*0.3", alpha=0.3)
        for t in done_list:
            axs[i].axvline(t, color="red", linestyle="--", label='reset')
        if i == 2:
            # plot horizontal line for the ground
            axs[i].axhline(0, color="black", linestyle="--", label='ground')
    res_dyn_numpy, a_numpy, force_array = np.array(res_dyn_list), np.array(a_list), np.array(force_list)
    disturb_array, decay_array, decay_param_array = np.array(disturb_list), np.array(decay_list), np.array(decay_param_list)
    for i in range(env.dim):
        axs[env.dim+i].set_title(f"force measurement dim={i}")
        axs[env.dim+i].plot(res_dyn_numpy[:,i], label="res_dyn")
        if i < 2:
            axs[env.dim+i].plot(a_numpy[:,0], label="action", linestyle='--', alpha=0.5)
        else:
            axs[env.dim+i].plot(a_numpy[:,1], label="action", linestyle='--', alpha=0.5)
        axs[env.dim+i].plot(force_array[:,i], label="force", alpha=0.5)
        axs[env.dim+i].plot(disturb_array[:,i], label="disturb", alpha=0.2)
        axs[env.dim+i].plot(decay_array[:,i], label="decay", alpha=0.3)
    mass_array = np.array(mass_list)
    res_dyn_param_numpy = np.array(res_dyn_param_list)
    axs[env.dim*2].set_title(f"system parameters and reward")
    for i in range(env.dim):
        axs[env.dim*2].plot(mass_array[:,i], label=f"mass-{i}*10", alpha=0.5)
    for i in range(env.res_dyn_param_dim):
        axs[env.dim*2].plot(res_dyn_param_numpy[:,i], label=f"res_dyn_param-{i}", alpha=0.5)
    axs[env.dim*2].plot(delay_list, label="delay*0.2", alpha=0.5)
    axs[env.dim*2].plot(r_list, 'y', label="reward")
    # add mean reward to axs 2 as text
    axs[env.dim*2].text(0.5, 0.5, f"mean reward: {np.mean(r_list):.3f}")
    # plot jacobian respect to different parameters
    if if_policy_grad:
        js_array = np.array(js)
        axs[env.dim*2+1].set_title(f"jacobian respect to extra parameters")
        for j in range(env.expert_dim):
            axs[env.dim*2+1].plot(js_array[j], label=f"jacobian-{j}")
    for i in range(plot_num):
        axs[i].legend()
    # save the plot as image
    if save_path == None:
        package_path = os.path.dirname(adaptive_control_gym.__file__)
        save_path = f"{package_path}/../results"
    plt.savefig(save_path+'/plot.png')

    # plot the movement of the drone over different timesteps
    fig, axs = plt.subplots(n_ep, 1, figsize=(5, 5*n_ep))
    for i in range(n_ep):
        axs[i].set_title(f"drone movement experiment {i+1} of {n_ep}")
        axs[i].set_xlabel("x")
        axs[i].set_ylabel("y")
        axs[i].set_xlim(-1, 1)
        axs[i].set_ylim(-1, 1)
        # get drone position and direction
        pos_info = x_numpy[i*env.max_steps:(i+1)*env.max_steps-1]
        x, y, theta = pos_info[:,0], pos_info[:,1], pos_info[:,2]
        # draw arrow for the drone
        for t in range(0, len(x), 5):
            # set color for each arrow according to t
            axs[i].arrow(x[t], y[t], np.sin(theta[t])*0.1, np.cos(theta[t])*0.1, head_width=0.05, head_length=0.1, fc='k', ec='k', alpha=t/len(x))
            # plot reference trajectory as dot
            axs[i].scatter(traj_x_array[i*env.max_steps+t,0], traj_x_array[i*env.max_steps+t,1], alpha=t/len(x), color='red', marker='*', s=10)
        # add related parameters as text
        # mass
        axs[i].text(0.3, 0.5, f"mass*10: {mass_array[i*env.max_steps,0]:.3f}, {mass_array[i*env.max_steps,1]:.3f}, {mass_array[i*env.max_steps,2]:.3f}")
        # decay
        axs[i].text(0.3, 0.4, f"decay: {decay_param_array[i*env.max_steps,0]:.3f}, {decay_array[i*env.max_steps,1]:.3f}, {decay_param_array[i*env.max_steps,2]:.3f}")
        # disturb
        axs[i].text(0.3, 0.3, f"disturb: {disturb_array[i*env.max_steps,0]:.3f}, {disturb_array[i*env.max_steps,1]:.3f}, {disturb_array[i*env.max_steps,2]:.3f}")
    plt.savefig(save_path+'/vis.png')
    
    env.close()

if __name__ == "__main__":
    expert_mode = True
    env = DroneEnv(env_num=1, gpu_id = -1, seed=0, expert_mode=expert_mode)
    policy = get_drone_policy(env, policy_name = "ppo")
    test_drone(env, policy)
    # eval_drone(policy.to("cuda:0"), {'expert_mode':expert_mode, 'seed': 0}, gpu_id = 0)
    # plot_drone()