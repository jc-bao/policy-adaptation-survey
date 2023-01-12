import os

import control as ct
import gym
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from icecream import install
from matplotlib import pyplot as plt
from torch import nn
from tqdm import trange

import adaptive_control_gym
from adaptive_control_gym import controllers as ctrl
from adaptive_control_gym.utils import sample_inv_norm

install()

class DroneEnv(gym.Env):
    def __init__(self, 
        env_num: int = 1, gpu_id: int = 0, seed:int = 1, 
        res_dyn_param_dim: int = 0
        ):
        torch.random.set_rng_state(torch.manual_seed(1024).get_state())
        torch.manual_seed(seed)

        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id>=0)) else "cpu")
        # parameters
        self.mode = 0 # evaluation mode
        self.dim=dim=3
        self.disturb_period = 30
        self.model_delay_alpha = 0.9
        self.res_dyn_scale = 1.0
        self.res_dyn_param_dim = res_dyn_param_dim 
        self.curri_param = 0.0
        self.adapt_horizon = 3

        self.mass_min, self.mass_max = 0.006, 0.03
        self.delay_min, self.delay_max = 0, 0
        self.decay_min, self.decay_max = 0.0, 0.1
        self.res_dyn_param_min, self.res_dyn_param_max = -1.0, 1.0
        self.disturb_min, self.disturb_max = -0.8, 0.8
        self.action_noise_std, self.obs_noise_std = 0.00, 0.00

        # generated parameters
        self.rotate_mass_scale = 1
        # self.mass_mean, self.mass_std = (self.mass_min+self.mass_max)/2, (self.mass_max-self.mass_min)/2
        self.mass_mean, self.mass_std = 0.018, 0.012
        self.delay_mean, self.delay_std = 0, 1
        # self.decay_mean, self.decay_std = (self.decay_max+self.decay_min)/2, (self.decay_max-self.decay_min)/2
        self.decay_mean, self.decay_std = 0.15, 0.3
        self.disturb_mean, self.disturb_std = 0, 0.3
        self.v_mean, self.v_std = 0, 1.0 / 0.3
        self.acc_mean, self.acc_std = 0, 1.0 / 0.03
        self.d_acc_mean, self.d_acc_std = 0, 2.0 / 0.03

        self.res_dyn = ResDynMLP(input_dim=dim+2+self.res_dyn_param_dim, output_dim=dim).to(self.device)
        self.res_dyn_param_mean, self.res_dyn_param_std = 0, 1.0


        self.tau = 1.0/30.0  # seconds between state updates
        self.force_scale = 1.0
        self.max_force = 1.0
        self.gravity = torch.zeros((env_num, 3), device=self.device)
        self.gravity[:, 1] = -9.8

        # generate a sin trajectory with torch
        self.obs_traj_len = 1
        self.traj_scale = 0.0
        self.traj_T = 360
        self.max_steps = 120 if self.traj_scale == 0 else 360
        self.init_x_mean, self.init_x_std = 0.0, 0.8 # self.traj_A, 0.0
        self.init_v_mean, self.init_v_std = 0.0, 1.0 # 0.0, 0.0
        self.x_min, self.x_max = -2.0, 2.0

        # state
        self.dim = dim
        self.env_num = env_num
        self.state_dim = 4*dim+2*dim*self.obs_traj_len
        self.expert_dim = self.res_dyn_param_dim
        if self.mass_min != self.mass_max:
            self.expert_dim += self.dim
        if self.decay_min != self.decay_max:
            self.expert_dim += self.dim
        if self.disturb_min != self.disturb_max:
            self.expert_dim += self.dim
        if self.delay_min != self.delay_max:
            self.expert_dim += 1
        self.action_dim = dim - 1
        self.adapt_dim = (3+3+3+3+3)*self.adapt_horizon
        # [info['u_force_his'], info['d_u_force_his'], info['v_his'], info['acc_his'], info['d_acc_his']]

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

    def _update_history(self):
        pass

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
        std = 0.4 - 0.2 * self.curri_param
        
        mass = sample_inv_norm(std, [size, self.dim], device=self.device) * (self.mass_max-self.mass_min)*0.5 + (self.mass_min+self.mass_max)*0.5
        delay = sample_inv_norm(std, [size, 1], device=self.device) * (self.delay_max-self.delay_min) * 0.5 + (self.delay_min+self.delay_max)*0.5
        decay = sample_inv_norm(std, [size, self.dim], device=self.device)* (self.decay_max-self.decay_min) * 0.5 + (self.decay_min+self.decay_max)*0.5
        # decay[:, 1] = decay[:, 0]
        # decay[:, 2] = decay[:, 0]
        res_dyn_param = sample_inv_norm(std, [size, self.res_dyn_param_dim], device=self.device) * (self.res_dyn_param_max-self.res_dyn_param_min) * 0.5 + (self.res_dyn_param_min+self.res_dyn_param_max)*0.5

        mass = torch.clip(mass, self.mass_min, self.mass_max)
        mass[:, 1] = mass[:, 0]
        # mass[:, 2] = mass[:, 0]
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
        # delay action
        if not (self.delay == 0).all():
            self.action_history.append(action)
            self.action_history.pop(0)
            action_delay = torch.zeros((self.env_num, self.dim), device=self.device)
            for i in range(self.delay_max):
                env_mask = (self.delay == i)[:,0]
                action_delay[env_mask] = self.action_history[-i-1][env_mask]
        else:
            action_delay = action

        # calculate u
        u = action_delay
        u[:,0] = (u[:,0]+1)/2 # make sure thrust 0 is always positive
        theta = self.x[:,[2]]
        F, M = u[:,[0]], u[:,[1]]
        u_force = torch.cat([F*torch.sin(theta), F*torch.cos(theta), M], dim=-1)
        self.u_his.append(u)
        self.u_his.pop(0)
        self.u_force_his.append(u_force)
        self.u_force_his.pop(0)
        if self.adapt_horizon > 1:
            self.d_u_his.append((self.u_his[-1]-self.u_his[-2]))
            self.d_u_his.pop(0)
            self.d_u_force_his.append((self.u_force_his[-1]-self.u_force_his[-2]))
            self.d_u_force_his.pop(0)

        # calculate u_delay
        if self.adapt_horizon > 1:
            u_delay = u*self.model_delay_alpha + self.u_his[-2]*(1-self.model_delay_alpha)
        else:
            u_delay = u

        # add noise
        u_delay_noise = torch.randn_like(action, device=self.device) * self.action_noise_std + u_delay
        u_delay_noise = torch.clip(u_delay_noise*self.force_scale, -self.max_force, self.max_force) 

        # calculate force
        F, M = u_delay_noise[:,[0]], u_delay_noise[:,[1]]
        self.action_force = torch.cat([F*torch.sin(theta), F*torch.cos(theta), M], dim=-1)
        
        self.decay_force = self.decay * self.v
        self.force = self.action_force + self.disturb - self.decay_force + self.gravity * self.mass
        if self.res_dyn_scale > 0:
            self.res_dyn_force = self.res_dyn(torch.cat([self.v*0.3, action, self.res_dyn_param], dim=-1)) * self.res_dyn_scale
            # self.res_dyn_force = torch.clip(self.res_dyn_force, -self.max_force/2, self.max_force/2)
        else:
            self.res_dyn_force = torch.zeros_like(self.force)
        self.force += self.res_dyn_force
        
        # system dynamics
        self.x += self.v * self.tau
        self.acc = self.force / self.mass
        self.v += self.acc * self.tau
        self.x[:,:2] = torch.clip(self.x[:,:2], self.x_min, self.x_max)
        self.v_his.append(self.v.clone())
        self.v_his.pop(0)
        self.acc_his.append(self.acc)
        self.acc_his.pop(0)
        if self.adapt_horizon > 1:
            self.d_acc_his.append((self.acc_his[-1]-self.acc_his[-2]))
            self.d_acc_his.pop(0)

        # calculate reward
        err_x = torch.norm((self.x-self.traj_x[...,self.step_cnt])[:,:2],dim=1)
        err_v = torch.norm((self.v-self.traj_v[...,self.step_cnt])[:,:2],dim=1)
        reward = 1.0 - torch.clip(err_x, 0, 2)*0.5 - torch.clip(err_v, 0, 1)*0.1
        reward -= torch.clip(torch.log(err_x+1)*5, 0, 1)*0.1 # for 0.2
        reward -= torch.clip(torch.log(err_x+1)*10, 0, 1)*0.1 # for 0.1
        reward -= torch.clip(torch.log(err_x+1)*20, 0, 1)*0.1 # for 0.05
        reward -= torch.clip(torch.log(err_x+1)*50, 0, 1)*0.1 # for 0.02
        # for hover task, add penalty for angular velocity
        if self.traj_scale == 0:
            reward -= torch.abs(self.x[:,2])*0.00
            reward -= torch.tanh(torch.abs(self.v[:,2]))*0.05

        self.step_cnt += 1

        # auto reset
        single_done = self.step_cnt >= self.max_steps
        done = torch.ones(self.env_num, device=self.device)*single_done
        # update disturb
        if self.step_cnt % self.disturb_period == 0:
            self._set_disturb()
        if single_done:
            self.reset()
        
        next_obs = self._get_obs()
        # update observation history
        self.obs_his.append(torch.cat((next_obs, action), dim=-1))
        self.obs_his.pop(0)
        next_info = self._get_info()
        # add gaussian noise to next_obs
        next_obs += torch.randn_like(next_obs, device=self.device) * self.obs_noise_std
        return next_obs, reward, done, next_info

    def reset(self, mode = None):
        if mode is None: 
            mode = self.mode
        self.step_cnt = 0
        self.x, self.v, self.mass, self.delay, self.decay, self.res_dyn_param = self._get_initial_state()
        self.force = torch.zeros((self.env_num, self.dim), device=self.device)
        self.action_force = torch.zeros((self.env_num, self.dim), device=self.device)
        self.action = torch.zeros((self.env_num, self.dim-1), device=self.device)
        self.acc = torch.zeros((self.env_num, self.dim), device=self.device)
        self.traj_x, self.traj_v = self._generate_traj()
        self._set_disturb()
        obs = self._get_obs()
        self._set_history(obs)
        info = self._get_info()
        return obs, info

    def _set_history(self, obs):
        # for delay controller
        self.action_history = [torch.zeros((self.env_num, 2), device=self.device)] * self.delay_max
        # for adaptive controller
        self.u_his = [torch.zeros((self.env_num, 2), device=self.device)] * self.adapt_horizon
        self.d_u_his = [torch.zeros((self.env_num, 2), device=self.device)] * self.adapt_horizon
        self.u_force_his = [torch.zeros((self.env_num, 3), device=self.device)] * self.adapt_horizon
        self.d_u_force_his = [torch.zeros((self.env_num, 3), device=self.device)] * self.adapt_horizon
        self.v_his = [torch.zeros((self.env_num, 3), device=self.device)] * self.adapt_horizon
        self.acc_his = [torch.zeros((self.env_num, 3), device=self.device)] * self.adapt_horizon
        self.d_acc_his = [torch.zeros((self.env_num, 3), device=self.device)] * self.adapt_horizon
        obs_his_shape = list(obs.shape)
        obs_his_shape[-1] += 2
        self.obs_his = [torch.zeros(obs_his_shape, device=self.device)] * self.adapt_horizon

        self.v_his.append(self.v.clone())
        self.v_his.pop(0)

    def _get_obs(self):
        future_traj_x = self.traj_x[..., self.step_cnt:self.step_cnt+self.obs_traj_len]
        future_traj_v = self.traj_v[..., self.step_cnt:self.step_cnt+self.obs_traj_len]
        err_x, err_v = future_traj_x[..., 0] - self.x, future_traj_v[..., 0] - self.v
        obs = torch.concat([self.x, self.v*0.3, err_x, err_v*0.3, future_traj_x.reshape(self.env_num,-1), future_traj_v.reshape(self.env_num, -1)], dim=-1)
        return obs

    def _get_info(self):
        err_x = torch.norm((self.x-self.traj_x[...,self.step_cnt])[:,:2],dim=1)
        err_v = torch.norm((self.v-self.traj_v[...,self.step_cnt])[:,:2],dim=1)
        info = {
            'pos': self.x,
            'vel': self.v,
            'acc': self.acc, 
            'action': self.action, 
            'action_force': self.action_force, 
            'mass': self.mass, 
            'disturb': self.disturb,
            'decay': self.decay,
            'err_x': err_x, 
            'err_v': err_v, 
            'e': self._get_e(),
            'obs_his': torch.stack(self.obs_his, dim=0),
            'u_his': torch.stack(self.u_his, dim=0),
            'd_u_his': torch.stack(self.d_u_his, dim=0),
            # 'u_force_his': torch.ones_like(torch.stack(self.u_force_his, dim=0), device=self.device)*1000,
            'u_force_his': torch.stack(self.u_force_his, dim=0), 
            'd_u_force_his': torch.stack(self.d_u_force_his, dim=0),
            'v_his': torch.stack(self.v_his, dim=0),
            'acc_his': torch.stack(self.acc_his, dim=0),
            'd_acc_his': torch.stack(self.d_acc_his, dim=0),
            'delay': self.delay,
        }

        info['adapt_obs'] = torch.concat(
            [info['u_force_his'], info['d_u_force_his'], (info['v_his']-self.v_mean)/self.v_std, 
            (info['acc_his']-self.acc_mean)/self.acc_std, (info['d_acc_his']-self.d_acc_mean)/self.d_acc_std], 
            dim=-1
        ) # Tensor(adapt_horizon, env_num, param)]
        info['adapt_obs'] = info['adapt_obs'].permute(1,0,2)
        info['adapt_obs'] = info['adapt_obs'].reshape(self.env_num, -1)
        # info['adapt_obs'] = torch.concat([(info['acc_his'][-1]-self.acc_mean)/self.acc_std, info['u_force_his'][-1], self.gravity/9.8], dim=-1) # Tensor(env_num, 3[dim], 3[param])

        if self.delay_max > 0:
            info['action_history'] = torch.stack(self.action_history, dim=1)
        return info

    def _get_e(self):
        es = []
        if self.mass_min != self.mass_max:
            mass_normed = (self.mass-self.mass_mean)/self.mass_std
            es.append(mass_normed)
        if self.disturb_min != self.disturb_max:
            disturb_normed = (self.disturb-self.disturb_mean)/self.disturb_std # * 0.0
            es.append(disturb_normed)
        if self.delay_min != self.delay_max:
            delay_normed = self.delay/10.0 # * 0.0
            es.append(delay_normed)
        if self.decay_min != self.decay_max:
            decay_normed = (self.decay-self.decay_mean)/self.decay_std # * 0.0
            es.append(decay_normed)
        if self.res_dyn_param_dim > 0:
            res_dyn_param_normed = (self.res_dyn_param-self.res_dyn_param_mean)/self.res_dyn_param_std
            es.append(res_dyn_param_normed)
        return torch.concat(es, dim=-1)

    def _set_disturb(self):
        std = 0.4 - 0.2 * self.curri_param
        self.disturb = sample_inv_norm(std, [self.env_num, self.dim], device=self.device) * (self.disturb_max-self.disturb_min)*0.5 + (self.disturb_max+self.disturb_min)*0.5
        self.disturb *= (self.mass*self.gravity[0,1])

class ResDynPolynomial:
    def __init__(self, input_dim:int, output_dim:int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.matrix = torch.rand((output_dim, input_dim, input_dim))*2-1
        self.vector = torch.rand((output_dim))*2-1

    def to(self, device:torch.device):
        self.matrix = self.matrix.to(device)
        self.vector = self.vector.to(device)
        return self

    def __call__(self, x:torch.Tensor)-> torch.Tensor: 
        y = self.vector + torch.einsum('bi,oij,bj->bo', x, self.matrix, x)
        return y/self.input_dim

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
        raw = self.mlp(x)
        ''''
        # for f(v, w_1)
        raw[..., 1] += 0.2 
        return raw * 5.0
        '''
        '''
        # for f(v, u, w_0)
        raw[..., 0] -= 0.1
        raw[..., 1] -= 0.3
        raw[..., 2] += 0.05
        return raw*3.0
        '''
        # for f(v, u, w_1)
        raw[..., 0] += 0.1
        raw[..., 1] += 0.25
        raw[..., 2] -= 0.1
        return raw*3.0

def get_drone_policy(env, policy_name = "ppo"):
    if policy_name == "lqr":
        Q = np.array([[50, 0],[0, 1]])
        R = 1
        policy = ctrl.LRQ(env.A, env.B, Q, R, gpu_id = -1)
    if policy_name == "pid":
        policy = ctrl.PID()
    elif policy_name == "random":
        policy = ctrl.Random(env.action_dim)
    elif policy_name == "ppo":
        loaded_agent = torch.load('/home/pcy/rl/policy-adaptation-survey/results/rl/ppo_ActEx0_CriEx0_S1.pt', map_location='cpu')
        policy = loaded_agent['actor']
        # freeze the policy
        # for p in policy.parameters():
        #     p.requires_grad = False
    else: 
        raise NotImplementedError
    return policy


def test_drone(env:DroneEnv, policy, adaptor, save_path = None):
    state, info = env.reset()
    obs_his = info['obs_his']
    x_list, v_list, a_list, force_list, disturb_list, decay_list, decay_param_list, res_dyn_list, mass_list, delay_list, res_dyn_param_list, traj_x_list, traj_v_list, r_list, done_list = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    js = []
    # check if the policy is torch neural network
    if_policy_grad =  False

    n_ep = 5
    time_limit = env.max_steps*n_ep
    for t in range(time_limit):
        if if_policy_grad:
            # set state as required grad
            state.requires_grad=True
            policy.zero_grad()
        act = policy(state, adaptor(info['adapt_obs']))
        # act = policy(state, info)
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
        save_path = f"{package_path}/../results/test"
    plt.savefig(f'{save_path}_plot.png')

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
        axs[i].text(0.3, 0.4, f"decay: {decay_param_array[i*env.max_steps,0]:.3f}, {decay_param_array[i*env.max_steps,1]:.3f}, {decay_param_array[i*env.max_steps,2]:.3f}")
        # disturb
        axs[i].text(0.3, 0.3, f"disturb: {disturb_array[i*env.max_steps,0]:.3f}, {disturb_array[i*env.max_steps,1]:.3f}, {disturb_array[i*env.max_steps,2]:.3f}")
    plt.savefig(f'{save_path}_vis.png')
    
    env.close()

if __name__ == "__main__":
    env = DroneEnv(env_num=1, gpu_id = -1, res_dyn_param_dim=1, seed=1)
    # policy = get_drone_policy(env, policy_name = "ppo")
    loaded_agent = torch.load('/home/pcy/rl/policy-adaptation-survey/results/rl/ppo_mlp_expert_Dw1_C4.pt', map_location='cpu')
    policy, adaptor = loaded_agent['actor'], loaded_agent['adaptor']
    test_drone(env, policy, adaptor)
    # eval_drone(policy.to("cuda:0"), {'seed': 0}, gpu_id = 0)
    # plot_drone()