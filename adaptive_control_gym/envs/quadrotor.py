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
from typing import List

import adaptive_control_gym
from adaptive_control_gym import controllers as ctrl
from adaptive_control_gym.utils import sample_inv_norm, rpy2quat, quat2rotmat, quat2rpy, quat_mul

install()

class QuadEnv(gym.Env):
    def __init__(self, 
        env_num: int = 1, gpu_id: int = 0, seed:int = 1, 
        res_dyn_param_dim: int = 3
        ):
        torch.random.set_rng_state(torch.manual_seed(1024).get_state())
        torch.manual_seed(seed)

        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id>=0)) else "cpu")
        self.gpu_id = gpu_id
        self.seed = seed
        # parameters
        self.mode = 0 # evaluation mode
        self.disturb_period = 120 #30
        self.model_delay_alpha = 0.9
        self.res_dyn_scale = 1.0
        self.res_dyn_param_dim = res_dyn_param_dim 
        self.curri_param = 0.0
        self.adapt_horizon = 3

        self.mass_min, self.mass_max = 0.018, 0.018 #0.006, 0.03 
        self.delay_min, self.delay_max = 0, 0
        self.decay_min, self.decay_max =  0.05, 0.05 #0.0, 0.1 #0.0, 0.0
        self.res_dyn_param_min, self.res_dyn_param_max = -0.25, 0.25
        self.disturb_min, self.disturb_max = 0.0, 0.0 # -0.8, 0.8
        self.force_scale_min, self.force_scale_max = 1.0, 1.0 # 0.75, 1.25
        self.action_noise_std, self.obs_noise_std = 0.00, 0.00

        # generated parameters
        self.rotate_mass_scale = 1
        self.mass_mean, self.mass_std = 0.018, 0.012
        self.delay_mean, self.delay_std = 0, 1
        self.decay_mean, self.decay_std = 0.05, 0.05
        self.disturb_mean, self.disturb_std = 0, 0.8
        self.v_mean, self.v_std = 0, 1.0 / 0.3
        self.acc_mean, self.acc_std = 0, 1.0 / 0.03
        self.d_acc_mean, self.d_acc_std = 0, 2.0 / 0.03
        self.force_scale_mean, self.force_scale_std = 1.0, 0.25

        self.res_dyn_origin = ResDynNeuralFly(input_dim=6+4+self.res_dyn_param_dim, output_dim=6, res_dyn_param_dim = res_dyn_param_dim, path='/home/pcy/rl/policy-adaptation-survey/adaptive_control_gym/envs/results/v-q.pth').to(self.device)
        self.res_dyn_fit = ResDynNeuralFly(input_dim=6+4+self.res_dyn_param_dim, output_dim=6, dropout_rate=0.1, res_dyn_param_dim=res_dyn_param_dim, path='/home/pcy/rl/policy-adaptation-survey/adaptive_control_gym/envs/results/v.pth').to(self.device)
        # self.res_dyn_fit.load_state_dict(self.res_dyn_origin.state_dict())
        # self.res_dyn_fit = lambda x: torch.zeros((self.env_num, 3), device=self.device)
        self.res_dyn = self.res_dyn_origin
        self.res_dyn_param_mean, self.res_dyn_param_std = 0, 0.25


        self.tau = 1.0/30.0  # seconds between state updates
        self.force_scale = torch.ones([env_num, 2], device=self.device)
        self.max_force = 1.0
        self.gravity = torch.zeros((env_num, 6), device=self.device)
        self.gravity[:, 2] = -9.8

        # generate a sin trajectory with torch
        self.obs_traj_len = 1
        self.traj_scale = 0.0
        self.traj_T = 360
        self.max_steps = 120 if self.traj_scale == 0 else 360
        self.init_x_mean, self.init_x_std = 0.0, 0.8 # self.traj_A, 0.0
        self.init_rpy_mean, self.init_rpy_std = 0.0, np.pi
        self.init_v_mean, self.init_v_std = 0.0, 1.0 # 0.0, 0.0
        self.x_min, self.x_max = -2.0, 2.0

        # state
        self.env_num = env_num
        self.state_dim = 7+6+3+3+2*3*self.obs_traj_len
        self.expert_dim = self.res_dyn_param_dim
        if self.mass_min != self.mass_max:
            self.expert_dim += 6
        if self.decay_min != self.decay_max:
            self.expert_dim += 6
        if self.disturb_min != self.disturb_max:
            self.expert_dim += 6
        if self.delay_min != self.delay_max:
            self.expert_dim += 1
        if self.force_scale_min != self.force_scale_max:
            self.expert_dim += 4
        self.action_dim = 4 # force and angle rate
        self.adapt_dim = (6+6+6+6+6)*self.adapt_horizon
        # [info['u_force_his'], info['d_u_force_his'], info['v_his'], info['acc_his'], info['d_acc_his']]

        self.reset()

        # dynamic
        A = np.array([[0, 1], [0, 0]])
        self.A = np.stack([A] * self.env_num, axis=0)
        B = (1 / self.mass).cpu().numpy()
        self.B = np.expand_dims(np.stack([np.zeros_like(B), B], axis=1), axis=2)
        # self.dynamic_info = self._get_dynamic_info()

        self.action_space = gym.spaces.Box(
            low=-self.max_force, high=self.max_force, shape=(3,), dtype=float)

    def _update_history(self):
        pass

    def _generate_traj(self):
        base_w = 2 * np.pi / self.traj_T
        t = torch.arange(0, self.max_steps+self.obs_traj_len, 1, device=self.device)
        t = torch.tile(t, (self.env_num, 3, 1))
        x = torch.zeros((self.env_num, 3, self.max_steps+self.obs_traj_len), device=self.device)
        v = torch.zeros((self.env_num, 3, self.max_steps+self.obs_traj_len), device=self.device)
        for i in np.arange(0,1,1):
            A = (torch.rand((self.env_num, 3, 1), device=self.device)*0.5+0.5)*(2.0**(-i))
            w = base_w*(2**i)
            phase = torch.rand((self.env_num, 3, 1), device=self.device)*(2*np.pi)
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
        
        mass = sample_inv_norm(std, [size, 3+3], device=self.device) * (self.mass_max-self.mass_min)*0.5 + (self.mass_min+self.mass_max)*0.5
        delay = sample_inv_norm(std, [size, 1], device=self.device) * (self.delay_max-self.delay_min) * 0.5 + (self.delay_min+self.delay_max)*0.5
        decay = sample_inv_norm(std, [size, 3+3], device=self.device)* (self.decay_max-self.decay_min) * 0.5 + (self.decay_min+self.decay_max)*0.5
        res_dyn_param = sample_inv_norm(std, [size, self.res_dyn_param_dim], device=self.device) * (self.res_dyn_param_max-self.res_dyn_param_min) * 0.5 + (self.res_dyn_param_min+self.res_dyn_param_max)*0.5

        force_scale = sample_inv_norm(std, [size, 1+3], device=self.device) * (self.force_scale_max-self.force_scale_min) * 0.5 + (self.force_scale_min+self.force_scale_max)*0.5

        mass = torch.clip(mass, self.mass_min, self.mass_max)
        mass[..., 0:3] = mass[..., :1]
        mass[:, 3:6] *= self.rotate_mass_scale
        delay = torch.clip(delay, self.delay_min, self.delay_max).type(torch.int)
        decay = torch.clip(decay, self.decay_min, self.decay_max)
        res_dyn_param = torch.clip(res_dyn_param, self.res_dyn_param_min, self.res_dyn_param_max)
        force_scale = torch.clip(force_scale, self.force_scale_min, self.force_scale_max)

        x = (torch.rand((size, 3), device=self.device)*2-1)* self.init_x_std + self.init_x_mean
        x = torch.clip(x, self.x_min, self.x_max)
        # uniformly generate xyz euler angles uniformly
        rpy = (torch.rand((size, 3), device=self.device)*2-1) * self.init_rpy_std + self.init_rpy_mean
        # convert to quaternion
        quat = rpy2quat(rpy)
        # concat quat to x
        x = torch.cat([x, quat], dim=1)
        v = (torch.rand((size, 3+3), device=self.device)*2-1)* self.init_v_std + self.init_v_mean
        return x, v, mass, delay, decay, res_dyn_param, force_scale
    
    def step(self, action):
        # delay action
        if not (self.delay == 0).all():
            self.action_history.append(action)
            self.action_history.pop(0)
            action_delay = torch.zeros((self.env_num, self.action_dim), device=self.device)
            for i in range(self.delay_max):
                env_mask = (self.delay == i)[:,0]
                action_delay[env_mask] = self.action_history[-i-1][env_mask]
        else:
            action_delay = action

        # get information from state
        quat = self.x[..., 3:7]
        rot_mat = quat2rotmat(quat)
        omega = self.v[..., 3:6]
        # quadrotor dynamics
        u = action_delay
        # make sure u[0] is always positive
        u[..., 0] = (u[..., 0] + 1.0) / 2.0
        u[..., 3] *= 0.1 # weaken the rotation along z-axis

        self.u_his.append(u)
        self.u_his.pop(0)
        if self.adapt_horizon > 1:
            self.d_u_his.append((self.u_his[-1]-self.u_his[-2]))
            self.d_u_his.pop(0)

        # calculate u_delay
        if self.adapt_horizon > 1:
            u_delay = u*self.model_delay_alpha + self.u_his[-2]*(1-self.model_delay_alpha)
        else:
            u_delay = u

        # add noise
        u_delay_noise = torch.randn_like(action, device=self.device) * self.action_noise_std + u_delay
        u_delay_noise = torch.clip(u_delay_noise*self.force_scale, -self.max_force, self.max_force) 

        # calculate force
        f_u_local = torch.zeros((self.env_num, 3), device=self.device)
        f_u_local[..., 2] = u_delay_noise[..., 0]
        f_u = torch.matmul(rot_mat, f_u_local.unsqueeze(-1)).squeeze(-1)
        tau_u_local = u_delay_noise[..., 1:4]
        tau_u = torch.matmul(rot_mat, tau_u_local.unsqueeze(-1)).squeeze(-1)
        u_force = torch.cat([f_u, tau_u], dim=1)
        self.action_force = u_force
        # record parameters
        self.u_force_his.append(u_force)
        self.u_force_his.pop(0)
        if self.adapt_horizon > 1:
            self.d_u_force_his.append((self.u_force_his[-1]-self.u_force_his[-2]))
            self.d_u_force_his.pop(0)
        
        self.decay_force = self.decay * self.v
        self.force = self.action_force + self.disturb - self.decay_force + self.gravity * self.mass
        if self.res_dyn_scale > 0:
            self.res_dyn_force = self.res_dyn(torch.cat([self.v*0.3, action, self.res_dyn_param], dim=-1)) * self.res_dyn_scale
            # self.res_dyn_force = torch.clip(self.res_dyn_force, -self.max_force/2, self.max_force/2)
            self.res_dyn_fit_force = self.res_dyn_fit(torch.cat([self.v*0.3, action, self.res_dyn_param], dim=-1)) * self.res_dyn_scale
        else:
            self.res_dyn_force = torch.zeros_like(self.force)
            self.res_dyn_fit_force = torch.zeros_like(self.force)
        self.force += self.res_dyn_force
        
        # system dynamics
        self.x[..., :3] += self.v[..., :3] * self.tau
        d_rpy = omega * self.tau
        d_quat = rpy2quat(d_rpy)
        new_quat = quat_mul(d_quat, quat)
        self.x[..., 3:7] = new_quat

        self.acc = self.force / self.mass
        self.v += self.acc * self.tau
        self.x[...,:3] = torch.clip(self.x[...,:3], self.x_min, self.x_max)
        self.v_his.append(self.v.clone())
        self.v_his.pop(0)
        self.acc_his.append(self.acc)
        self.acc_his.pop(0)
        if self.adapt_horizon > 1:
            self.d_acc_his.append((self.acc_his[-1]-self.acc_his[-2]))
            self.d_acc_his.pop(0)

        # calculate reward
        err_x = torch.norm((self.x[..., :3]-self.traj_x[...,self.step_cnt])[...,:3],dim=1)
        err_v = torch.norm((self.v[..., :3]-self.traj_v[...,self.step_cnt])[...,:3],dim=1)
        reward = 1.0 - torch.clip(err_x, 0, 2)*0.5 - torch.clip(err_v, 0, 1)*0.1
        reward -= torch.clip(torch.log(err_x+1)*5, 0, 1)*0.1 # for 0.2
        reward -= torch.clip(torch.log(err_x+1)*10, 0, 1)*0.1 # for 0.1
        reward -= torch.clip(torch.log(err_x+1)*20, 0, 1)*0.1 # for 0.05
        reward -= torch.clip(torch.log(err_x+1)*50, 0, 1)*0.1 # for 0.02
        # for hover task, add penalty for angular velocity
        if self.traj_scale == 0:
            reward -= torch.abs(self.x[...,2])*0.00
            reward -= torch.tanh(torch.abs(self.v[...,2]))*0.05

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

    def reset(self, mode = None, env_params = None):
        if mode is None: 
            mode = self.mode
        self.step_cnt = 0
        self.x, self.v, self.mass, self.delay, self.decay, self.res_dyn_param, self.force_scale = self._get_initial_state()
        if env_params is not None:
            self.set_env_params(*env_params)
        # record parameters for visualization
        self.force = torch.zeros((self.env_num, 3+3), device=self.device)
        self.action_force = torch.zeros((self.env_num, 3+3), device=self.device)
        self.action = torch.zeros((self.env_num, 4), device=self.device)
        self.acc = torch.zeros((self.env_num, 3+3), device=self.device)
        self.traj_x, self.traj_v = self._generate_traj()
        self._set_disturb()
        obs = self._get_obs()
        self._set_history(obs)
        info = self._get_info()
        return obs, info

    def _set_history(self, obs):
        # for delay controller
        self.action_history = [torch.zeros((self.env_num, self.action_dim), device=self.device)] * self.delay_max
        # for adaptive controller
        self.u_his = [torch.zeros((self.env_num, self.action_dim), device=self.device)] * self.adapt_horizon
        self.d_u_his = [torch.zeros((self.env_num, self.action_dim), device=self.device)] * self.adapt_horizon
        self.u_force_his = [torch.zeros((self.env_num, 6), device=self.device)] * self.adapt_horizon
        self.d_u_force_his = [torch.zeros((self.env_num, 6), device=self.device)] * self.adapt_horizon
        self.v_his = [torch.zeros((self.env_num, 6), device=self.device)] * self.adapt_horizon
        self.acc_his = [torch.zeros((self.env_num, 6), device=self.device)] * self.adapt_horizon
        self.d_acc_his = [torch.zeros((self.env_num, 6), device=self.device)] * self.adapt_horizon
        obs_his_shape = list(obs.shape)
        obs_his_shape[-1] += self.action_dim
        self.obs_his = [torch.zeros(obs_his_shape, device=self.device)] * self.adapt_horizon

        self.v_his.append(self.v.clone())
        self.v_his.pop(0)

    def _get_obs(self):
        future_traj_x = self.traj_x[..., self.step_cnt:self.step_cnt+self.obs_traj_len]
        future_traj_v = self.traj_v[..., self.step_cnt:self.step_cnt+self.obs_traj_len]
        err_x, err_v = future_traj_x[..., 0] - self.x[..., :3], future_traj_v[..., 0] - self.v[..., :3]
        return torch.concat(
            [
                self.x,
                self.v * 0.3,
                err_x,
                err_v * 0.3,
                future_traj_x.reshape(self.env_num, -1),
                future_traj_v.reshape(self.env_num, -1),
            ],
            dim=-1,
        )

    def _get_info(self):
        err_x = torch.norm((self.x[..., :3]-self.traj_x[...,self.step_cnt])[:,:2],dim=1)
        err_v = torch.norm((self.v[..., :3]-self.traj_v[...,self.step_cnt])[:,:2],dim=1)
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
        if self.expert_dim == 0: 
            return torch.zeros([self.env_num, 0], device=self.device)
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
        if self.force_scale_min != self.force_scale_max:
            force_scale_normed = (self.force_scale-self.force_scale_mean)/self.force_scale_std
            es.append(force_scale_normed)
        return torch.concat(es, dim=-1)

    def _set_disturb(self):
        std = 0.4 - 0.2 * self.curri_param
        self.disturb = sample_inv_norm(std, [self.env_num, 3+3], device=self.device) * (self.disturb_max-self.disturb_min)*0.5 + (self.disturb_max+self.disturb_min)*0.5
        self.disturb *= (self.mass*self.gravity[0,2])
    
    def get_env_params(self):
        return (self.mass, self.delay, self.decay, self.res_dyn_param, self.force_scale)

    def set_env_params(self, mass, delay, decay, res_dyn_param, force_scale):
        self.mass[:] = mass
        self.delay[:] = delay
        self.decay[:] = decay
        self.res_dyn_param[:] = res_dyn_param
        self.force_scale[:] = force_scale

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

class Phi_Net(nn.Module):
    def __init__(self, input_dim:int):
        super(Phi_Net, self).__init__()

        self.fc1 = nn.Linear(input_dim, 50)
        self.fc2 = nn.Linear(50, 60)
        self.fc3 = nn.Linear(60, 50)
        self.fc4 = nn.Linear(50, 2)
        
    def forward(self, x:torch.Tensor):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        if len(x.shape) == 1:
            # single input
            return torch.cat([x, torch.ones(1, device=x.device)])
        else:
            # batch input for training
            return torch.cat([x, torch.ones([x.shape[0], 1], device=x.device)], dim=-1)

class ResDynNeuralFly(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate = 0.0, res_dyn_param_dim = 0, path=None):
        if path is None: 
            path = '/home/pcy/rl/policy-adaptation-survey/adaptive_control_gym/envs/results/v-q.pth'
        if 'v-q' in path:
            self.mode='v-q'
            input_dim=7
        elif 'v' in path:
            self.mode='v'
            input_dim=3
        else:
            raise ValueError('path should contain v-q or v')
        super().__init__()
        self.phi_net = Phi_Net(input_dim=input_dim)
        model = torch.load(path)
        self.phi_net.load_state_dict(model['phi_net_state_dict'])
        self.res_dyn_param_dim = res_dyn_param_dim
        if res_dyn_param_dim == 0:
            self.A = nn.Parameter(torch.tensor([
                [0.1, 0.05, 0.05], 
                [0.05, 0.1, 0.05], 
                [0.05, 0.05, 0.1]
            ]), requires_grad=False)
        else:
            assert res_dyn_param_dim in [3,9], 'res_dyn_param_dim should be 0, 3 or 9'
        self.phi_net.eval()
        for p in self.phi_net.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = x[..., 7:10]
        if self.mode == 'v-q':
            q = x[..., 3:7]
            phi = self.phi_net(torch.cat([v, q], dim=-1))
        else:
            phi = self.phi_net(v)
        if self.res_dyn_param_dim == 0:
            A = self.A
        elif self.res_dyn_param_dim ==3: 
            # create diagonal matrix from x[..., -3:]
            A = torch.diag_embed(x[..., -3:])
        elif self.res_dyn_param_dim == 9:
            A = x[..., -9:].reshape([*x.shape[:-1], 3, 3])
        f = torch.einsum('bij,bj->bi', A, phi)
        return torch.cat([f, torch.zeros([*f.shape[:-1], 3], device=f.device)], dim=-1)

class ResDynMLP(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate = 0.0, res_dyn_param_dim: int = 1):
        super().__init__()
        embedding_size = 128
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, embedding_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate, inplace=True),
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate, inplace=True),
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
        if res_dyn_param_dim == 0:
            # for f(v, u, w_0)
            self.offset = nn.Parameter(torch.tensor([-0.25, -0.25, 0.0, 0.0, 0.0, 0.0]), requires_grad=False)
            self.scale = nn.Parameter(torch.tensor([2.0, 2.0, 2.0, 0.2, 0.2, 0.1]), requires_grad=False)
        else:
            self.offset = nn.Parameter(torch.tensor([0.0, 0.0, 0.0]), requires_grad=False)
            self.scale = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]), requires_grad=False)


        # for f(v, u, w_0)
        # self.offset = nn.Parameter(torch.tensor([-0.1, -0.3, 0.05]), requires_grad=False)
        # self.scale = nn.Parameter(torch.tensor([3.0, 3.0, 3.0]), requires_grad=False)
        # for f(v, w_1)
        # self.offset = nn.Parameter(torch.tensor([0.0, 0.2, 0.0]), requires_grad=False)
        # self.scale = nn.Parameter(torch.tensor([5.0, 5.0, 5.0]), requires_grad=False)

    def forward(self, x):
        raw = self.mlp(x)
        return (raw+self.offset)*self.scale

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


def test_quad(env:QuadEnv, policy, adaptor, compressor=lambda x: x, save_path = None):
    state, info = env.reset()
    obs_his = info['obs_his']
    x_list, v_list, a_list, force_list, disturb_list, decay_list, decay_param_list, res_dyn_list, res_dyn_fit_list, mass_list, delay_list, res_dyn_param_list, traj_x_list, traj_v_list, r_list, done_list = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    e_diff_list, action_force_list = [], []
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
        e_pred = adaptor(info['adapt_obs'])
        e_diff_list.append((e_pred - compressor(info['e'])).detach().numpy()[0])
        act = policy(state, e_pred)
        # act = policy(state, info)
        if if_policy_grad:
            # calculate jacobian respect to state
            jacobian = torch.autograd.grad(act, state, grad_outputs=torch.ones_like(act), create_graph=True)[0][0, -env.expert_dim:]
        with torch.no_grad():
            state, rew, done, info = env.step(act)  # take a random action
        x_list.append(state[0,:7].numpy())
        v_list.append(state[0,7:7+6].numpy())
        traj_x_list.append(env.traj_x[0,:,t%env.max_steps].numpy())
        traj_v_list.append(env.traj_v[0,:,t%env.max_steps].numpy()*0.3)
        a_list.append(act[0].detach().numpy())
        r_list.append(rew[0].item())
        force_list.append(env.force[0].numpy())
        disturb_list.append(env.disturb[0].numpy())
        decay_list.append(env.decay_force[0].numpy())
        decay_param_list.append(env.decay[0,:].numpy())
        res_dyn_list.append(env.res_dyn_force[0].numpy())
        res_dyn_fit_list.append(env.res_dyn_fit_force[0].numpy())
        mass_list.append(env.mass[0,:].numpy()*10)
        delay_list.append(env.delay[0,0].item()*0.2)
        action_force_list.append(env.action_force[0].numpy())
        if if_policy_grad:
            js.append(jacobian.detach().numpy())
        res_dyn_param_list.append(env.res_dyn_param[0,:].numpy())
        if done:
            done_list.append(t)
    # set matplotlib style
    plt.style.use('seaborn')
    # plot x_list, v_list, action_list in three subplots
    plot_num = 16
    fig, axs = plt.subplots(plot_num, 1, figsize=(10, 3*plot_num))
    x_array, v_array = np.array(x_list), np.array(v_list)
    quat_array = x_array[:,3:7]
    # convert quaternion to rpy angle
    rpy_array = quat2rpy(torch.tensor(quat_array)).numpy()
    traj_x_array, traj_v_array = np.array(traj_x_list), np.array(traj_v_list)
    for i in range(3):
        axs[i].set_title(f"position dim={i}")
        axs[i].plot(x_array[:,i], label="x")
        axs[i].plot(v_array[:,i], label="v*0.3", alpha=0.3)
        axs[i].plot(traj_x_array[:,i], label="traj_x")
        axs[i].plot(traj_v_array[:,i], label="traj_v*0.3", alpha=0.3)
        for t in done_list:
            axs[i].axvline(t, color="red", linestyle="--", label='reset')
        if i == 2:
            # plot horizontal line for the ground
            axs[i].axhline(0, color="black", linestyle="--", label='ground')
    for j in range(3):
        axs[3+j].set_title(f"rpy angle dim={j}")
        axs[3+j].plot(rpy_array[:,j], label="rpy")
        for t in done_list:
            axs[3+j].axvline(t, color="red", linestyle="--", label='reset')
    res_dyn_numpy, a_numpy, force_array = np.array(res_dyn_list), np.array(a_list), np.array(force_list)
    res_dyn_fit_numpy = np.array(res_dyn_fit_list)
    disturb_array, decay_array, decay_param_array = np.array(disturb_list), np.array(decay_list), np.array(decay_param_list)
    action_force_array = np.array(action_force_list)
    for i in range(6):
        axs[6+i].set_title(f"force measurement dim={i}")
        axs[6+i].plot(res_dyn_numpy[:,i], label="res_dyn")
        axs[6+i].plot(res_dyn_fit_numpy[:,i], label="res_dyn_fit")
        axs[6+i].plot(force_array[:,i], label="force", alpha=0.5)
        axs[6+i].plot(disturb_array[:,i], label="disturb", alpha=0.2)
        axs[6+i].plot(decay_array[:,i], label="decay", alpha=0.3)
        axs[6+i].plot(action_force_array[:,i], label="action_force", alpha=1.0)
    mass_array = np.array(mass_list)
    res_dyn_param_numpy = np.array(res_dyn_param_list)
    axs[12].set_title(f"system parameters and reward")
    for i in range(6):
        axs[12].plot(mass_array[:,i], label=f"mass-{i}*10", alpha=0.5)
    for i in range(env.res_dyn_param_dim):
        axs[12].plot(res_dyn_param_numpy[:,i], label=f"res_dyn_param-{i}", alpha=0.5)
    axs[12].plot(delay_list, label="delay*0.2", alpha=0.5)
    axs[12].plot(r_list, 'y', label="reward")
    # add mean reward to axs 2 as text
    axs[12].text(0.5, 0.5, f"mean reward: {np.mean(r_list):.3f}")
    # plot e_diff_list respect to different parameters
    e_diff_array = np.array(e_diff_list)
    for i in range(e_diff_array.shape[-1]):
        axs[13].plot(e_diff_array[:,i], label=f"e_diff-{i}")
    # plot jacobian respect to different parameters
    if if_policy_grad:
        js_array = np.array(js)
        axs[14].set_title(f"jacobian respect to extra parameters")
        for j in range(env.expert_dim):
            axs[14].plot(js_array[j], label=f"jacobian-{j}")
    # plot action
    axs[15].set_title(f"action")
    for i in range(env.action_dim):
        axs[15].plot(a_numpy[:,i], label=f"action-{i}")
    for i in range(plot_num):
        axs[i].legend()
    # save the plot as image
    if save_path == None:
        package_path = os.path.dirname(adaptive_control_gym.__file__)
        save_path = f"{package_path}/../results/test"
    plt.savefig(f'{save_path}_plot.png')

    # plot 3D the movement of the drone over different timesteps
    fig = plt.figure(figsize=(10, 3*n_ep))
    for i in range(n_ep):
        # 3d projection
        ax = fig.add_subplot(n_ep, 1, i+1, projection='3d')
        ax.set_title(f"drone movement experiment {i+1} of {n_ep}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(0, 2)
        # get drone position and direction
        pos_info = x_array[i*env.max_steps:(i+1)*env.max_steps-1]
        x, y, z, quat = pos_info[:,0], pos_info[:,1], pos_info[:,2], pos_info[:,3:]
        # draw arrow for the drone
        for t in range(0, len(x), 5):
            # set color for each arrow according to t
            color = (t/len(x), 0, 1-t/len(x))
            # get the direction of the arrow
            direction = np.array([2*(quat[t,1]*quat[t,3]-quat[t,0]*quat[t,2]), 2*(quat[t,0]*quat[t,1]+quat[t,2]*quat[t,3]), 1-2*(quat[t,1]**2+quat[t,2]**2)])
            # draw the arrow
            ax.quiver(x[t], y[t], z[t], direction[0], direction[1], direction[2], color=color, length=0.1, normalize=True)
            # plot reference trajectory traj_x_array as dot
            ax.scatter(traj_x_array[i*env.max_steps+t,0], traj_x_array[i*env.max_steps+t,1], traj_x_array[i*env.max_steps+t,2], color=color, marker='.')
        # add related parameters as text
        # mass
        ax.text(0.3, 0.5, -1.0, f"mass*10: {mass_array[i*env.max_steps,0]:.3f}, {mass_array[i*env.max_steps,1]:.3f}, {mass_array[i*env.max_steps,2]:.3f}")
        # rotation mass
        ax.text(0.3, 0.6, -1.0, f"rotation mass: {mass_array[i*env.max_steps,3]:.3f}, {mass_array[i*env.max_steps,4]:.3f}, {mass_array[i*env.max_steps,5]:.3f}")
        # decay
        ax.text(0.3, 0.7, -1.0, f"decay: {decay_param_array[i*env.max_steps,0]:.3f}, {decay_param_array[i*env.max_steps,1]:.3f}, {decay_param_array[i*env.max_steps,2]:.3f}, {decay_param_array[i*env.max_steps,3]:.3f}, {decay_param_array[i*env.max_steps,4]:.3f}, {decay_param_array[i*env.max_steps,5]:.3f}")
        # disturb
        ax.text(0.3, 0.8, -1.0, f"disturb: {disturb_array[i*env.max_steps,0]:.3f}, {disturb_array[i*env.max_steps,1]:.3f}, {disturb_array[i*env.max_steps,2]:.3f}, {disturb_array[i*env.max_steps,3]:.3f}, {disturb_array[i*env.max_steps,4]:.3f}, {disturb_array[i*env.max_steps,5]:.3f}")
    plt.savefig(f'{save_path}_vis.png')
    
    env.close()

    np.save(f'{save_path}_x.npy', x_array)
    np.save(f'{save_path}_res_dyn.npy', res_dyn_numpy)
    np.save(f'{save_path}_disturb.npy', disturb_array)

def vis_data(path = None):
    # load data
    x = np.load(f'{path}_x.npy')
    res_dyn = np.load(f'{path}_res_dyn.npy')
    disturb = np.load(f'{path}_disturb.npy')
    
    # use meshcat to visualize the data
    import meshcat
    import meshcat.geometry as g
    import meshcat.transformations as tf
    import time

    # create a visualizer
    vis = meshcat.Visualizer()
    vis.open()
    # set camera position
    vis["/Cameras/default"].set_transform(
		tf.translation_matrix([0,0,0]).dot(
		tf.euler_matrix(0,np.radians(-30),-np.pi/2)))
    vis["/Cameras/default/rotated/<object>"].set_transform(
        tf.translation_matrix([1, 0, 0]))
    # set quadrotor position
    vis["Quadrotor"].set_object(g.StlMeshGeometry.from_file('../assets/crazyflie2.stl'))

    # update quadrotor position with x
    while True:
        for state, res_force, dist in zip(x, res_dyn, disturb):
            pos = state[:3]
            quat = state[3:]
            # quat from [x,y,z,w] to [w,x,y,z]
            quat = np.array([quat[3], quat[0], quat[1], quat[2]])
            vis["Quadrotor"].set_transform(
                tf.translation_matrix(pos).dot(
                    tf.quaternion_matrix(quat)))
            vis["ResForce"].set_object(g.LineSegments(
                g.PointsGeometry(position=np.array([
                    pos, pos+res_force[:3]]).astype(np.float32).T,
                    color=np.array([
                    [1, 1, 0], [1, 1, 0]]).astype(np.float32).T
                ),
                g.LineBasicMaterial(vertexColors=True)))
            vis['Disturb'].set_object(g.LineSegments(
                g.PointsGeometry(position=np.array([
                    pos, pos+dist[:3]]).astype(np.float32).T,
                    color=np.array([
                    [0, 1, 1], [0, 1, 1]]).astype(np.float32).T
                ),
                g.LineBasicMaterial(vertexColors=True)))
            time.sleep(1/30)

class PID():	
	def __init__(self, env:QuadEnv):
		self.name = 'PID'
		self.mass = env.mass[..., :3]
		self.J = env.mass[..., 3:6]

		# Note: we assume here that our control is forces
		arm_length = 0.046 # m
		arm = 0.707106781 * arm_length
		self.arm = arm
		t2t = 0.006 # thrust-to-torque ratio
		self.t2t = t2t
		self.g = 9.81 # not signed
		self.a_min = np.array([0, 0, 0, 0])
		self.a_max = np.array([12, 12, 12, 12]) / 1000 * 9.81 # g->N

		# PID parameters
		self.K_i = 0.5 # eigs[0] * eigs[1] * eigs[2]
		self.K_p = 3.0 # eigs[0] * eigs[1] + eigs[1] * eigs[2] + eigs[2] * eigs[0]
		self.K_d = 6.0 # sum(eigs)
		self.Att_p = 400
		self.Att_d = 150
		self.time = 0.0 
		self.int_p_e = np.zeros(3)

	def policy(self, state, time=None):
		p_e = -state[:3]
		v_e = -state[3:6]
		int_p_e = self.integrate_error(p_e, time)
		F_d = (self.K_i * int_p_e + self.K_p * p_e + self.K_d * v_e) * self.mass # TODO: add integral term
		F_d[2] += self.g * self.mass
		T_d = np.linalg.norm(F_d)
		yaw_d = 0
		roll_d = np.arcsin((F_d[0]*np.sin(yaw_d)-F_d[1]*np.cos(yaw_d)) / T_d)
		pitch_d = np.arctan((F_d[0]*np.cos(yaw_d)+F_d[1]*np.sin(yaw_d)) / F_d[2])
		euler_d = np.array([roll_d, pitch_d, yaw_d])
		euler = rowan.to_euler(rowan.normalize(state[6:10]), 'xyz')
		att_e = -(euler - euler_d)
		att_v_e = -state[10:]
		torque = self.Att_p * att_e + self.Att_d * att_v_e
		torque[0] *= self.J[0]
		torque[1] *= self.J[1]
		torque[2] *= self.J[2]
		Jomega = np.array([self.J[0]*state[10], self.J[1]*state[11], self.J[2]*state[12]])
		torque -= np.cross(Jomega, state[10:])

		yawpart = -0.25 * torque[2] / self.t2t
		rollpart = 0.25 / self.arm * torque[0]
		pitchpart = 0.25 / self.arm * torque[1]
		thrustpart = 0.25 * T_d

		motorForce = np.array([
			thrustpart - rollpart - pitchpart + yawpart,
			thrustpart - rollpart + pitchpart - yawpart,
			thrustpart + rollpart + pitchpart + yawpart,
			thrustpart + rollpart - pitchpart - yawpart
		])
		motorForce = np.clip(motorForce, self.a_min, self.a_max)

		return motorForce

	def integrate_error(self, p_e, time):
		if not self.time:
			dt = 0.0
			self.time = time 
			return np.zeros(3)
		else:
			dt = time - self.time
			self.time = time 
			self.int_p_e += dt * p_e
			return self.int_p_e


if __name__ == "__main__":
    env_num = 1
    env = QuadEnv(env_num=env_num, gpu_id = -1, res_dyn_param_dim=9, seed=1)
    # env.init_x_mean = env.init_x_std = env.init_v_mean = env.init_v_std = env.init_rpy_mean = env.init_rpy_std = 0.0
    # env.disturb_max, env.disturb_min = 1e-5, 0.0
    # env.res_dyn_scale = 0.0
    # policy = lambda x,y: torch.tensor([[9.8*0.018, 0, 0, 0]])
    # adaptor = lambda x: torch.zeros([env_num, env.expert_dim])

    agent = torch.load('/home/pcy/rl/policy-adaptation-survey/results/rl/ppo_3D.pt', map_location='cpu')
    test_quad(env, agent['actor'], agent['adaptor'], agent['compressor'])

    # vis_data(path='/home/pcy/rl/policy-adaptation-survey/results/rl/ppo_3Dneural')