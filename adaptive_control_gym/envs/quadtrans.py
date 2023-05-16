import gym
import os
import torch
import numpy as np
import adaptive_control_gym
from adaptive_control_gym.utils import rpy2rotmat
from icecream import ic
from matplotlib import pyplot as plt

class QuadTransEnv(gym.Env):
    '''
    This is a class for quadrotor transportation task.
    The control input is the thrust and attitude, i.e. 4 control inputs.
    The state is the position, velocity, attitude, and angular velocity.
    The reward is the object distance to the target position.
    TODO 
    * modelling related: delay of controller, pid control error (especially for attitude control), environment damping ratio
    '''
    def __init__(self, env_num=1024, gpu_id=0, seed=0, **kwargs) -> None:
        super().__init__()

        # environment related parameters
        self.seed = seed
        torch.manual_seed(self.seed)
        # set torch default float precision
        torch.set_default_dtype(torch.float32)
        self.env_num = env_num
        self.dt = 1/100 # s
        self.g = 9.81 # m/s^2
        self.communication_delay = 0.02 # s
        self.substep_num = 10
        self.max_steps = 80
        self.gpu_id = gpu_id
        self.device = torch.device(
            f"cuda:{self.gpu_id}" if (torch.cuda.is_available() & (gpu_id>=0)) else "cpu"
        )

        # object related parameters
        self.mass_obj_min, self.mass_obj_max = 0.01, 0.02 # kg
        self.mass_obj_min, self.mass_obj_max = torch.tensor([self.mass_obj_min]).to(self.device), torch.tensor([self.mass_obj_max]).to(self.device)
        self.mass_obj_mean, self.mass_obj_std = (self.mass_obj_min + self.mass_obj_max)/2, (self.mass_obj_max - self.mass_obj_min)/2
        self.mass_obj_std[self.mass_obj_std == 0.0] = 1.0

        self.length_rope_min, self.length_rope_max = 0.1, 0.3
        self.length_rope_min, self.length_rope_max = torch.tensor([self.length_rope_min]).to(self.device), torch.tensor([self.length_rope_max]).to(self.device)
        self.length_rope_mean, self.length_rope_std = (self.length_rope_min + self.length_rope_max)/2, (self.length_rope_max - self.length_rope_min)/2
        self.length_rope_std[self.length_rope_std == 0.0] = 1.0

        self.tp_obj_min, self.tp_obj_max = np.array([-np.pi, -np.pi/2]), np.array([np.pi, np.pi/2])
        self.tp_obj_min, self.tp_obj_max = torch.tensor([self.tp_obj_min], dtype=torch.float32).to(self.device), torch.tensor([self.tp_obj_max], dtype=torch.float32).to(self.device)
        self.tp_obj_mean, self.tp_obj_std = (self.tp_obj_min + self.tp_obj_max)/2, (self.tp_obj_max - self.tp_obj_min)/2
        
        self.vtp_obj_min, self.vtp_obj_max = np.array([-10, -10]), np.array([10, 10])
        self.vtp_obj_min, self.vtp_obj_max = torch.tensor([self.vtp_obj_min], dtype=torch.float32).to(self.device), torch.tensor([self.vtp_obj_max], dtype=torch.float32).to(self.device)
        self.vtp_obj_mean, self.vtp_obj_std = (self.vtp_obj_min + self.vtp_obj_max)/2, (self.vtp_obj_max - self.vtp_obj_min)/2

        # drone related parameters
        self.mass_drone_min, self.mass_drone_max = 0.025, 0.03
        self.mass_drone_min, self.mass_drone_max = torch.tensor([self.mass_drone_min]).to(self.device), torch.tensor([self.mass_drone_max]).to(self.device)
        self.mass_drone_mean, self.mass_drone_std = (self.mass_drone_min + self.mass_drone_max)/2, (self.mass_drone_max - self.mass_drone_min)/2
        self.mass_drone_std[self.mass_drone_std == 0.0] = 1.0

        self.xyz_drone_min, self.xyz_drone_max = np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0])
        self.xyz_drone_min, self.xyz_drone_max = torch.tensor(self.xyz_drone_min, dtype=torch.float32).to(self.device), torch.tensor(self.xyz_drone_max, dtype=torch.float32).to(self.device)
        self.xyz_drone_mean, self.xyz_drone_std = (self.xyz_drone_min + self.xyz_drone_max)/2, (self.xyz_drone_max - self.xyz_drone_min)/2

        self.xyz_target_min, self.xyz_target_max = np.array([-1.0, -1.0, -1.0])*0.7, np.array([1.0, 1.0, 1.0])*0.7
        self.xyz_target_min, self.xyz_target_max = torch.tensor(self.xyz_target_min, dtype=torch.float32).to(self.device), torch.tensor(self.xyz_target_max, dtype=torch.float32).to(self.device)
        self.xyz_target_mean, self.xyz_target_std = (self.xyz_target_min + self.xyz_target_max)/2, (self.xyz_target_max - self.xyz_target_min)/2
        self.xyz_target_std[self.xyz_target_std==0] = 1.0

        self.vxyz_drone_min, self.vxyz_drone_max = np.array([-1.0, -1.0, -1.0])*2.0, np.array([1.0, 1.0, 1.0])*2.0
        self.vxyz_drone_min, self.vxyz_drone_max = torch.tensor(self.vxyz_drone_min, dtype=torch.float32).to(self.device), torch.tensor(self.vxyz_drone_max, dtype=torch.float32).to(self.device)
        self.vxyz_drone_mean, self.vxyz_drone_std = (self.vxyz_drone_min + self.vxyz_drone_max)/2, (self.vxyz_drone_max - self.vxyz_drone_min)/2

        self.rpy_drone_min, self.rpy_drone_max = np.array([-np.pi, -np.pi, 0.0])/6, np.array([np.pi, np.pi, 0.0])/6
        self.rpy_drone_min, self.rpy_drone_max = torch.tensor(self.rpy_drone_min, dtype=torch.float32).to(self.device), torch.tensor(self.rpy_drone_max, dtype=torch.float32).to(self.device)
        self.rpy_drone_mean, self.rpy_drone_std = (self.rpy_drone_min + self.rpy_drone_max)/2, (self.rpy_drone_max - self.rpy_drone_min)/2
        self.rpy_drone_std[2] = 1.0

        self.vrpy_drone_min, self.vrpy_drone_max = np.array([-30, -30, 0.0]), np.array([30, 30, 0.0])
        self.vrpy_drone_min, self.vrpy_drone_max = torch.tensor(self.vrpy_drone_min, dtype=torch.float32).to(self.device), torch.tensor(self.vrpy_drone_max, dtype=torch.float32).to(self.device)
        self.vrpy_drone_mean, self.vrpy_drone_std = (self.vrpy_drone_min + self.vrpy_drone_max)/2, (self.vrpy_drone_max - self.vrpy_drone_min)/2
        self.vrpy_drone_std[2] = 1.0

        # control related parameters
        self.action_dim = 3
        self.action_min, self.action_max = -1.0, 1.0
        self.action_mean, self.action_std = (self.action_min + self.action_max)/2, (self.action_max - self.action_min)/2
        self.thrust_min, self.thrust_max = 0.0, 0.6 # N
        self.thrust_mean, self.thrust_std = (self.thrust_min + self.thrust_max)/2, (self.thrust_max - self.thrust_min)/2
        self.ctl_roll_min, self.ctl_roll_max = -np.pi/6, np.pi/6 # rad
        self.ctl_roll_mean, self.ctl_roll_std = (self.ctl_roll_min + self.ctl_roll_max)/2, (self.ctl_roll_max - self.ctl_roll_min)/2
        self.ctl_pitch_min, self.ctl_pitch_max = -np.pi/6, np.pi/6 # rad
        self.ctl_pitch_mean, self.ctl_pitch_std = (self.ctl_pitch_min + self.ctl_pitch_max)/2, (self.ctl_pitch_max - self.ctl_pitch_min)/2
        self.ctl_roll_rate_min, self.ctl_roll_rate_max = -20, 20 # rad/s
        self.ctl_roll_rate_mean, self.ctl_roll_rate_std = (self.ctl_roll_rate_min + self.ctl_roll_rate_max)/2, (self.ctl_roll_rate_max - self.ctl_roll_rate_min)/2
        self.ctl_pitch_rate_min, self.ctl_pitch_rate_max = -20, 20 # rad/s
        self.ctl_pitch_rate_mean, self.ctl_pitch_rate_std = (self.ctl_pitch_rate_min + self.ctl_pitch_rate_max)/2, (self.ctl_pitch_rate_max - self.ctl_pitch_rate_min)/2
        # TBD
        self.damping_rate_drone_min, self.damping_rate_drone_max = 0.0, 0.05
        self.damping_rate_drone_min, self.damping_rate_drone_max = torch.tensor([self.damping_rate_drone_min]).to(self.device), torch.tensor([self.damping_rate_drone_max]).to(self.device)
        self.damping_rate_drone_mean, self.damping_rate_drone_std = (self.damping_rate_drone_min + self.damping_rate_drone_max)/2, (self.damping_rate_drone_max - self.damping_rate_drone_min)/2
        # TBD
        self.damping_rate_obj_min, self.damping_rate_obj_max = 0.0, 0.05
        self.damping_rate_obj_min, self.damping_rate_obj_max = torch.tensor(self.damping_rate_obj_min).to(self.device), torch.tensor([self.damping_rate_obj_max]).to(self.device)
        self.damping_rate_obj_mean, self.damping_rate_obj_std = (self.damping_rate_obj_min + self.damping_rate_obj_max)/2, (self.damping_rate_obj_max - self.damping_rate_obj_min)/2
        # TBD
        self.attitude_pid_p_min, self.attitude_pid_p_max = 0.4, 0.5
        self.attitude_pid_p_min, self.attitude_pid_p_max = torch.tensor([self.attitude_pid_p_min]).to(self.device), torch.tensor([self.attitude_pid_p_max]).to(self.device)
        self.attitude_pid_p_mean, self.attitude_pid_p_std = (self.attitude_pid_p_min + self.attitude_pid_p_max)/2, (self.attitude_pid_p_max - self.attitude_pid_p_min)/2
        self.attitude_pid_p_std[self.attitude_pid_p_std == 0.0] = 1.0

        self.thrust_pid_p_min, self.thrust_pid_p_max = 0.4, 0.6
        self.thrust_pid_p_min, self.thrust_pid_p_max = torch.tensor([self.thrust_pid_p_min]).to(self.device), torch.tensor([self.thrust_pid_p_max]).to(self.device)
        self.thrust_pid_p_mean, self.thrust_pid_p_std = (self.thrust_pid_p_min + self.thrust_pid_p_max)/2, (self.thrust_pid_p_max - self.thrust_pid_p_min)/2
        self.thrust_pid_p_std[self.thrust_pid_p_std == 0.0] = 1.0

        # RMA related parameters
        self.rma_params = ['mass_obj', 'length_rope', 'mass_drone', 'damping_rate_drone', 'damping_rate_obj', 'attitude_pid_p', 'thrust_pid_p']
        self.expert_dim = 1 + 1 + 1 + 1 + 1 + 1 + 1
        self.obs_traj_len = 5
        self.adapt_horizon = 3
        self.adapt_store_len = self.adapt_horizon + 5
        self.adapt_dim = self.adapt_horizon * 30

        # RL parameters
        self.state_dim = 3 + 3 + 3 + 3 + 3 + 3 + 3 + 3 + 5 * 6

        # render related parameters
        self.state_params = ['mass_obj', 'length_rope', 'mass_drone', 'xyz_drone', 'xyz_target', 'vxyz_drone', 'rpy_drone', 'vrpy_drone', 'tp_obj', 'vtp_obj', 'damping_rate_drone', 'damping_rate_obj', 'attitude_pid_p', 'thrust_pid_p']
        self.setup_params()
        self.plot_params = ['xyz_drone', 'vxyz_drone', 'rpy_drone', 'xyz_obj', 'xyz_target', 'tp_obj', 'vtp_obj'] + self.rma_params

        self.reset()

    def setup_params(self):
        '''
        environment parameters setup
        '''
        self.mass_obj = torch.zeros((self.env_num, 1), device=self.device, dtype=torch.float32)
        self.length_rope = torch.zeros((self.env_num, 1), device=self.device, dtype=torch.float32)
        self.mass_drone = torch.zeros((self.env_num, 1), device=self.device, dtype=torch.float32)
        self.xyz_drone = torch.zeros((self.env_num, 3), device=self.device, dtype=torch.float32)
        self.xyz_target = torch.zeros((self.env_num, 3), device=self.device, dtype=torch.float32)
        self.vxyz_drone = torch.zeros((self.env_num, 3), device=self.device, dtype=torch.float32)
        self.rpy_drone = torch.zeros((self.env_num, 3), device=self.device, dtype=torch.float32)
        self.vrpy_drone = torch.zeros((self.env_num, 3), device=self.device, dtype=torch.float32)
        self.tp_obj = torch.zeros((self.env_num, 2), device=self.device, dtype=torch.float32)
        self.vtp_obj = torch.zeros((self.env_num, 2), device=self.device, dtype=torch.float32)
        self.damping_rate_drone = torch.zeros((self.env_num, 1), device=self.device, dtype=torch.float32)
        self.damping_rate_obj = torch.zeros((self.env_num, 1), device=self.device, dtype=torch.float32)
        self.attitude_pid_p = torch.zeros((self.env_num, 1), device=self.device, dtype=torch.float32)
        self.thrust_pid_p = torch.zeros((self.env_num, 1), device=self.device, dtype=torch.float32)
        self.ctl_thrust = torch.zeros((self.env_num, 1), device=self.device, dtype=torch.float32)
        self.ctl_roll = torch.zeros((self.env_num, 1), device=self.device, dtype=torch.float32)
        self.ctl_pitch = torch.zeros((self.env_num, 1), device=self.device, dtype=torch.float32)

        # TODO 1d adapt vs robust, PID
        self.ctl_thrust_his = torch.zeros((self.adapt_store_len, self.env_num, 1), device=self.device, dtype=torch.float32)
        self.ctl_roll_his = torch.zeros((self.adapt_store_len, self.env_num, 1), device=self.device, dtype=torch.float32)
        self.ctl_pitch_his = torch.zeros((self.adapt_store_len, self.env_num, 1), device=self.device, dtype=torch.float32)
        self.xyz_drone_his = torch.zeros((self.adapt_store_len, self.env_num, 3), device=self.device, dtype=torch.float32)
        self.xyz_obj_his = torch.zeros((self.adapt_store_len, self.env_num, 3), device=self.device, dtype=torch.float32)
    
    def sample_params(self, key):
        param_max = self.__dict__[key + '_max']
        param_min = self.__dict__[key + '_min']
        param_dim = param_max.shape[0]
        return torch.rand((self.env_num, param_dim), device=self.device) * (param_max - param_min) + param_min

    def reset(self):
        # sample parameters uniformly using pytorch
        for key in self.state_params:
            self.__dict__[key] = self.sample_params(key)
        # DEBUG
        self.xyz_drone *= 1.0
        self.vxyz_drone *= 1.0
        self.rpy_drone *= 0.2
        self.vrpy_drone *= 0.2
        self.tp_obj *= 0.2
        self.vtp_obj *= 0.2

        self.traj_x, self.traj_v = self._generate_traj()

        # other parameters
        self.step_cnt = 0
        self.xyz_target = self.traj_x[self.step_cnt]
        self.vxyz_target = self.traj_v[self.step_cnt]

        # phisical parameters
        self.thrust = torch.ones((self.env_num, 1), device=self.device, dtype=torch.float32) * 0.027 * 9.81
        self.vxyz_obj = self.vxyz_drone + self.get_obj_rel_vel(self.tp_obj, self.vtp_obj, self.length_rope)
        self.xyz_obj = self.xyz_drone + self.get_obj_disp(self.tp_obj, self.length_rope)
        self.obj2goal = self.xyz_obj - self.xyz_target
        self.drone2goal = self.xyz_drone - self.xyz_target
        self.drone2goal[..., [-1]] -= self.length_rope

        # PID expert information, mainly for reference
        self.thrust_pid = torch.zeros((self.env_num, 1), device=self.device, dtype=torch.float32)
        self.ctl_roll_pid = torch.zeros((self.env_num, 1), device=self.device, dtype=torch.float32)
        self.ctl_pitch_pid = torch.zeros((self.env_num, 1), device=self.device, dtype=torch.float32)
        self.ctl_roll_rate_pid = torch.zeros((self.env_num, 1), device=self.device, dtype=torch.float32)
        self.ctl_pitch_rate_pid = torch.zeros((self.env_num, 1), device=self.device, dtype=torch.float32)

        # for delay
        self.ctl_thrust = torch.zeros((self.env_num, 1), device=self.device, dtype=torch.float32)
        self.ctl_roll = torch.zeros((self.env_num, 1), device=self.device, dtype=torch.float32)
        self.ctl_pitch = torch.zeros((self.env_num, 1), device=self.device, dtype=torch.float32)
        self.ctl_thrust_old = torch.zeros((self.env_num, 1), device=self.device, dtype=torch.float32)
        self.ctl_roll_old = torch.zeros((self.env_num, 1), device=self.device, dtype=torch.float32)
        self.ctl_pitch_old = torch.zeros((self.env_num, 1), device=self.device, dtype=torch.float32)

        # for reward
        self.delta_roll = torch.zeros((self.env_num, 1), device=self.device, dtype=torch.float32)
        self.delta_pitch = torch.zeros((self.env_num, 1), device=self.device, dtype=torch.float32)

        # for adapt
        self.ctl_thrust_his = torch.zeros((self.adapt_store_len, self.env_num, 1), device=self.device, dtype=torch.float32)
        self.ctl_roll_his = torch.zeros((self.adapt_store_len, self.env_num, 1), device=self.device, dtype=torch.float32)
        self.ctl_pitch_his = torch.zeros((self.adapt_store_len, self.env_num, 1), device=self.device, dtype=torch.float32)
        self.xyz_drone_his = torch.zeros((self.adapt_store_len, self.env_num, 3), device=self.device, dtype=torch.float32)
        self.xyz_obj_his = torch.zeros((self.adapt_store_len, self.env_num, 3), device=self.device, dtype=torch.float32)

        return self._get_obs(), self._get_info()

    def step(self, action):
        '''
        action: (env_num, action_dim)
            action=[thrust, ctl_roll, ctl_pitch]
        '''
        # convert action to thrust, ctl_roll, ctl_pitch
        self.ctl_thrust = (action[:, [0]]-self.action_mean)/self.action_std * self.thrust_std + self.thrust_mean
        self.ctl_roll = (action[:, [1]]-self.action_mean)/self.action_std * self.ctl_roll_std + self.ctl_roll_rate_mean
        self.ctl_pitch = (action[:, [2]]-self.action_mean)/self.action_std * self.ctl_pitch_std + self.ctl_pitch_rate_mean

        # calculate delta angle
        max_angle_change = np.pi/2
        self.delta_roll = torch.clip(self.ctl_roll - self.rpy_drone[..., [0]], -max_angle_change, max_angle_change)
        self.delta_pitch = torch.clip(self.ctl_pitch - self.rpy_drone[..., [1]], -max_angle_change, max_angle_change)
        self.ctl_roll = self.rpy_drone[..., [0]] + self.delta_roll
        self.ctl_pitch = self.rpy_drone[..., [1]] + self.delta_pitch
        # ===================== velocity control =====================
        # self.delta_roll = ctl_roll * self.dt * self.substep_num
        # self.delta_pitch = ctl_pitch * self.dt * self.substep_num

        for i in range(self.substep_num):
            if (i * self.dt) < self.communication_delay:
                debug_info = self.substep(self.ctl_thrust_old, self.ctl_roll_old, self.ctl_pitch_old)
            else:
                debug_info = self.substep(self.ctl_thrust, self.ctl_roll, self.ctl_pitch)
        
        # save parameters
        self.ctl_roll_old = self.ctl_roll
        self.ctl_pitch_old = self.ctl_pitch
        self.ctl_thrust_old = self.ctl_thrust
        self.ctl_thrust_his[self.step_cnt%self.adapt_store_len] = self.ctl_thrust
        self.ctl_roll_his[self.step_cnt%self.adapt_store_len] = self.ctl_roll
        self.ctl_pitch_his[self.step_cnt%self.adapt_store_len] = self.ctl_pitch
        self.xyz_drone_his[self.step_cnt%self.adapt_store_len] = self.xyz_drone
        self.xyz_obj_his[self.step_cnt%self.adapt_store_len] = self.xyz_obj
        
        reward = self._get_reward()

        # calculate done
        self.step_cnt += 1
        self.xyz_target = self.traj_x[self.step_cnt]
        self.vxyz_target = self.traj_v[self.step_cnt]
        single_done = (self.step_cnt >= self.max_steps)
        done = torch.ones((self.env_num), device=self.device) * single_done
        if single_done:
            self.reset()
        next_obs = self._get_obs()
        next_info = self._get_info()
        next_info['debug']= debug_info

        return next_obs, reward, done, next_info

    def _get_reward(self):
        # calculate reward
        err_x = torch.norm(self.obj2goal,dim=1)
        err_v = torch.norm((self.vxyz_obj - self.vxyz_target),dim=1)
        reward = 1.0 - torch.clip(err_x, 0, 2)*0.5 - torch.clip(err_v, 0, 2)*0.5
        reward -= torch.clip(torch.log(err_x+1)*5, 0, 1)*0.1 # for 0.2
        reward -= torch.clip(torch.log(err_x+1)*10, 0, 1)*0.1 # for 0.1
        # panelty for large delta control
        reward -= torch.clip(torch.abs(self.delta_roll[:,0]), 0.0, np.pi/2)*0.1
        reward -= torch.clip(torch.abs(self.delta_pitch[:,0]), 0.0, np.pi/2)*0.1
        return reward

    def substep(self, thrust, ctl_roll, ctl_pitch):
        # analysis two point mass system dynamics which connected with a rope
        # thrust force
        self.thrust = thrust * self.thrust_pid_p + self.thrust * (1-self.thrust_pid_p)
        force_thrust_local = torch.zeros((self.env_num, 3), device=self.device)
        force_thrust_local[:, [2]] = self.thrust
        rotmat_drone = rpy2rotmat(self.rpy_drone)
        force_thrust = torch.bmm(rotmat_drone, force_thrust_local.unsqueeze(-1)).squeeze(-1)
        # gravity force
        force_gravity_drone = torch.zeros((self.env_num, 3), device=self.device)
        force_gravity_drone[:, [2]] = -self.mass_drone * self.g
        force_gravity_obj = torch.zeros((self.env_num, 3), device=self.device)
        force_gravity_obj[:, [2]] = -self.mass_obj * self.g
        # drag
        vxyz_obj_rel = self.get_obj_rel_vel(self.tp_obj, self.vtp_obj, self.length_rope)
        self.vxyz_obj = self.vxyz_drone + vxyz_obj_rel
        force_drag_drone = -self.damping_rate_drone * self.vxyz_drone
        force_drag_obj = -self.damping_rate_obj * self.vxyz_obj
        # set force_drag_obj to zero when the object mass is zero
        force_drag_obj = force_drag_obj * (self.mass_obj > 0).float()
        # analysis the center of mass of the two point mass system
        acc_com = (force_thrust + force_gravity_drone + force_gravity_obj + force_drag_drone + force_drag_obj) / (self.mass_drone + self.mass_obj)
        # calculate the acceleration of the object with respect to center of mass
        # object distance to center of mass
        dist2com = self.length_rope * self.mass_drone / (self.mass_drone + self.mass_obj)
        vel2com = vxyz_obj_rel * self.mass_drone / (self.mass_drone + self.mass_obj)
        acc_rad_obj = vel2com.square().sum(dim=-1, keepdim=True) / dist2com
        # calculate the unit vector of object local frame
        z_hat_obj = torch.stack(
            [torch.cos(self.tp_obj[:, 0]) * torch.sin(self.tp_obj[:, 1]),
            torch.sin(self.tp_obj[:, 0]) * torch.sin(self.tp_obj[:, 1]),
            -torch.cos(self.tp_obj[:, 1])], dim=-1)
        y_hat_obj = torch.stack(
            [-torch.sin(self.tp_obj[:, 0]),
            torch.cos(self.tp_obj[:, 0]),
            torch.zeros(self.env_num, device=self.device)], dim=-1)
        x_hat_obj = torch.cross(y_hat_obj, z_hat_obj, dim=-1)
        # calculate radial force
        obj_external_force = force_gravity_obj + force_drag_obj - self.mass_obj * acc_com
        force_rope = torch.einsum('bi,bi->b', z_hat_obj, obj_external_force).unsqueeze(-1) + self.mass_obj * acc_rad_obj
        force_obj_local_x = torch.einsum('bi,bi->b', x_hat_obj, obj_external_force).unsqueeze(-1)
        force_obj_local_y = torch.einsum('bi,bi->b', y_hat_obj, obj_external_force).unsqueeze(-1)
        # calculate object angular acceleration
        thetadotdot = force_obj_local_y / self.mass_obj
        phidotdot = -force_obj_local_x / self.mass_obj
        # set thetadotdot where mass_obj==0 to 0
        thetadotdot[self.mass_obj==0] = 0.0
        phidotdot[self.mass_obj==0] = 0.0
        # calculate drone acceleration
        acc_drone = (force_thrust + force_drag_drone + force_gravity_drone + force_rope*z_hat_obj) / self.mass_drone
        # update state
        self.xyz_drone = torch.clip(self.xyz_drone + self.vxyz_drone * self.dt, -5, 5)
        self.vxyz_drone = torch.clip(self.vxyz_drone + acc_drone * self.dt, -30, 30)
        reach_x_bound = torch.abs(self.xyz_drone[:, 0]) >= 5
        self.vxyz_drone[reach_x_bound, 0] = 0.0
        reach_y_bound = torch.abs(self.xyz_drone[:, 1]) >= 5
        self.vxyz_drone[reach_y_bound, 1] = 0.0
        reach_z_bound = torch.abs(self.xyz_drone[:, 2]) >= 5
        self.vxyz_drone[reach_z_bound, 2] = 0.0

        # =================== angular rate control ===================
        # ctl_roll_acc = torch.clip((ctl_roll - self.rpy_drone[:, [0]])*60 - self.vrpy_drone[:, [0]] * 7, -100.0, 100.0)
        # ctl_pitch_acc = torch.clip((ctl_pitch - self.rpy_drone[:, [1]])*60 - self.vrpy_drone[:, [1]] * 7, -100.0, 100.0)
        # self.vrpy_drone[:, [0]] = ctl_roll_acc * self.dt + self.vrpy_drone[:, [0]]
        # self.vrpy_drone[:, [1]] = ctl_pitch_acc * self.dt + self.vrpy_drone[:, [1]]
        # self.rpy_drone[:, [0]] = torch.clip(self.vrpy_drone[:, [0]] * self.dt + self.rpy_drone[:, [0]], -np.pi/4, np.pi/4)
        # self.rpy_drone[:, [1]] = torch.clip(self.vrpy_drone[:, [1]] * self.dt + self.rpy_drone[:, [1]], -np.pi/4, np.pi/4)
        # roll_hit_bound = torch.abs(self.rpy_drone[:, 0]) >= np.pi/4
        # self.vrpy_drone[roll_hit_bound, 0] = 0.0
        # pitch_hit_bound = torch.abs(self.rpy_drone[:, 1]) >= np.pi/4
        # self.vrpy_drone[pitch_hit_bound, 1] = 0.0
        # =================== attitude control ===================
        self.rpy_drone[:, [0]] = ctl_roll * self.attitude_pid_p + self.rpy_drone[:, [0]] * (1-self.attitude_pid_p)
        self.rpy_drone[:, [1]] = ctl_pitch * self.attitude_pid_p + self.rpy_drone[:, [1]] * (1-self.attitude_pid_p)

        self.tp_obj = self.tp_obj + self.vtp_obj * self.dt
        self.tp_obj[:, 0] = (self.tp_obj[:, 0] + np.pi) % (2*np.pi) - np.pi
        self.tp_obj[:, 1] = torch.clip(self.tp_obj[:, 1], -np.pi/2, np.pi/2)
        self.vtp_obj[:, [0]] = torch.clip(self.vtp_obj[:, [0]] + thetadotdot * self.dt, -20, 20)
        self.vtp_obj[:, [1]] = torch.clip(self.vtp_obj[:, [1]] + phidotdot * self.dt, -20, 20)
        reach_p_bound = torch.abs(self.tp_obj[:, 1]) >= np.pi/2
        self.vtp_obj[reach_p_bound, 1] = - self.vtp_obj[reach_p_bound, 1]
        # set tp_obj, vtp_obj where mass_obj==0 to 0
        self.tp_obj[self.mass_obj[:,0]==0,:] = 0.0
        self.vtp_obj[self.mass_obj[:,0]==0,:] = 0.0
        xyz_obj2drone = self.get_obj_disp(self.tp_obj, self.length_rope)
        self.xyz_obj = self.xyz_drone + xyz_obj2drone
        self.obj2goal = self.xyz_obj - self.xyz_target
        self.drone2goal = self.xyz_drone - self.xyz_target
        self.drone2goal[..., [-1]] -= self.length_rope

        # PID control expert information
        total_force_obj = - force_gravity_obj - force_drag_obj - self.obj2goal * 1.0 - (self.vxyz_obj - self.vxyz_target) * 0.3
        total_force_obj[self.mass_obj[:,0]==0,:] = 0.0
        total_force_obj_projected = torch.einsum('bi,bi->b', z_hat_obj, total_force_obj).unsqueeze(-1) * z_hat_obj
        z_hat_obj_target = - total_force_obj / torch.norm(total_force_obj, dim=-1, keepdim=True)
        z_hat_obj_target[self.mass_obj[:,0]==0] = torch.tensor([0.0, 0.0, -1.0], device=self.device)
        # z_hat_obj_target = torch.where(z_hat_obj_target[:, [2]] > 0, -z_hat_obj_target, z_hat_obj_target)
        xyz_drone_target = self.xyz_obj - z_hat_obj_target * self.length_rope
        xyz_drone_target = torch.where(self.mass_obj[:,[0]]==0, self.xyz_target - z_hat_obj_target * self.length_rope, xyz_drone_target)
        total_force_drone = total_force_obj_projected - force_gravity_drone - force_drag_drone - (self.xyz_drone - xyz_drone_target) * 0.5 - (self.vxyz_drone - self.vxyz_target) * 0.3
        total_force_drone_projected = torch.einsum('bij,bj->bi', torch.inverse(rotmat_drone), total_force_drone)[..., [2]]
        self.thrust_pid = torch.clip(total_force_drone_projected, self.thrust_min, self.thrust_max)
        self.ctl_roll_pid = torch.atan2(-total_force_drone[..., [1]], torch.sqrt(total_force_drone[..., [0]]**2 + total_force_drone[..., [2]]**2))
        self.ctl_pitch_pid = torch.atan2(total_force_drone[..., [0]], total_force_drone[..., [2]])
        self.ctl_roll_rate_pid = (self.ctl_roll_pid - self.rpy_drone[:, [0]])/(self.dt * self.substep_num)
        self.ctl_pitch_rate_pid = self.ctl_pitch_pid - self.rpy_drone[:, [1]]/(self.dt * self.substep_num) 

        return {
            # position information
            'xyz_drone': self.xyz_drone,
            'xyz_target': self.xyz_target,
            'vxyz_drone': self.vxyz_drone,
            'rpy_drone': self.rpy_drone,
            'rotmat_drone': rotmat_drone,
            'tp_obj': self.tp_obj,
            'vtp_obj': self.vtp_obj,
            'xyz_obj': self.xyz_drone + xyz_obj2drone,
            'vxyz_obj_rel': vxyz_obj_rel,
            'z_hat_obj': z_hat_obj,
            'y_hat_obj': y_hat_obj,
            'x_hat_obj': x_hat_obj,
            'xyz_obj2drone': xyz_obj2drone,
            # force information
            'force_thrust_drone': force_thrust,
            'force_gravity_drone': force_gravity_drone,
            'force_drag_drone': force_drag_drone,
            'force_rope_drone': force_rope * z_hat_obj,
            'force_rope_obj': -force_rope * z_hat_obj,
            'force_gravity_obj': force_gravity_obj,
            'force_drag_obj': force_drag_obj,
            'acc_com': acc_com,
            'acc_rad_obj': acc_rad_obj,
            'obj_external_force': obj_external_force,
            # PID control expert information
            'total_force_obj': total_force_obj,
            'total_force_obj_projected': total_force_obj_projected,
            'z_hat_obj_target': z_hat_obj_target,
            'xyz_drone_target': xyz_drone_target,
            'total_force_drone': total_force_drone,
            'total_force_drone_projected': total_force_drone_projected,
        }


    def _get_obs(self):
        xyz_drone_normed = (self.xyz_drone - self.xyz_drone_mean) / self.xyz_drone_std
        xyz_obj_normed = (self.xyz_obj - self.xyz_drone_mean) / self.xyz_drone_std
        xyz_target_normed = (self.xyz_target - self.xyz_target_mean) / self.xyz_target_std
        vxyz_drone_normed = (self.vxyz_drone - self.vxyz_drone_mean) / self.vxyz_drone_std
        vxyz_obj_normed = (self.vxyz_obj - self.vxyz_drone_mean) / self.vxyz_drone_std
        rpy_drone_normed = (self.rpy_drone - self.rpy_drone_mean) / self.rpy_drone_std
        future_traj_x = self.traj_x[self.step_cnt:self.step_cnt+self.obs_traj_len].reshape(self.env_num, -1)
        future_traj_v = self.traj_v[self.step_cnt:self.step_cnt+self.obs_traj_len].reshape(self.env_num, -1)
        # tp_obj_normed = (self.tp_obj - self.tp_obj_mean) / self.tp_obj_std
        # vtp_obj_normed = (self.vtp_obj - self.vtp_obj_mean) / self.vtp_obj_std
        # ic(xyz_drone_normed, xyz_obj_normed, xyz_target_normed, vxyz_drone_normed, vxyz_obj_normed, rpy_drone_normed, self.obj2goal, self.vxyz_obj - self.vxyz_target, future_traj_x, future_traj_v)
        obs = torch.cat([xyz_drone_normed, xyz_obj_normed, xyz_target_normed, vxyz_drone_normed, vxyz_obj_normed, rpy_drone_normed, self.obj2goal, self.vxyz_obj - self.vxyz_target, future_traj_x, future_traj_v], dim=1)
        return obs

    def _get_info(self):
        # expert info
        e_list = []
        for key in self.rma_params:
            param = getattr(self, key)
            if len(param.shape) == 1:
                param = param.unsqueeze(-1)
            param_mean, param_std = getattr(self, f'{key}_mean'), getattr(self, f'{key}_std')
            e_list.append((param - param_mean) / param_std)
        e = torch.cat(e_list, dim=-1)

        # adapt info
        adapt_obs_idx = torch.arange(self.step_cnt - self.adapt_horizon, self.step_cnt, device=self.device) % self.adapt_horizon
        thrust = self.ctl_thrust_his[adapt_obs_idx]
        roll = self.ctl_roll_his[adapt_obs_idx]
        pitch = self.ctl_pitch_his[adapt_obs_idx]
        d_thrust = self.ctl_thrust_his[adapt_obs_idx] - self.ctl_thrust_his[adapt_obs_idx-1]
        d_roll = self.ctl_roll_his[adapt_obs_idx] - self.ctl_roll_his[adapt_obs_idx-1]
        d_pitch = self.ctl_pitch_his[adapt_obs_idx] - self.ctl_pitch_his[adapt_obs_idx-1]
        xyz_drone = self.xyz_drone_his[adapt_obs_idx]
        xyz_obj = self.xyz_obj_his[adapt_obs_idx]
        d_xyz_drone = self.xyz_drone_his[adapt_obs_idx] - self.xyz_drone_his[adapt_obs_idx-1]
        d_xyz_obj = self.xyz_obj_his[adapt_obs_idx] - self.xyz_obj_his[adapt_obs_idx-1]
        dd_xyz_drone = self.xyz_drone_his[adapt_obs_idx] - 2 * self.xyz_drone_his[adapt_obs_idx-1] + self.xyz_drone_his[adapt_obs_idx-2]
        dd_xyz_obj = self.xyz_obj_his[adapt_obs_idx] - 2 * self.xyz_obj_his[adapt_obs_idx-1] + self.xyz_obj_his[adapt_obs_idx-2]
        ddd_xyz_drone = self.xyz_drone_his[adapt_obs_idx] - 3 * self.xyz_drone_his[adapt_obs_idx-1] + 3 * self.xyz_drone_his[adapt_obs_idx-2] - self.xyz_drone_his[adapt_obs_idx-3]
        ddd_xyz_obj = self.xyz_obj_his[adapt_obs_idx] - 3 * self.xyz_obj_his[adapt_obs_idx-1] + 3 * self.xyz_obj_his[adapt_obs_idx-2] - self.xyz_obj_his[adapt_obs_idx-3]
        adapt_obs = torch.concat(
            [
                thrust, roll, pitch, d_thrust, d_roll, d_pitch, # dim=6
                xyz_drone, xyz_obj, d_xyz_drone, d_xyz_obj, dd_xyz_drone, dd_xyz_obj, ddd_xyz_drone, ddd_xyz_obj # dim=6*4=24
            ],
            dim=-1
        ).permute(1,0,2).reshape(self.env_num, -1)

        return {
            'e': e,
            'adapt_obs': adapt_obs,
            'err_x': torch.norm(self.obj2goal, dim=-1),
            'err_v': torch.norm(self.vxyz_obj-self.vxyz_target, dim=-1),
        }
        

    def get_obj_disp(self, obj_tp, rope_length):
        '''
        drone_pos: (env_num, 3)
        obj_tp: (env_num, 3)
        rope_length: (env_num, 1)
        '''
        disp = torch.zeros((self.env_num, 3), device=self.device)
        disp[:, [0]] = rope_length * torch.cos(obj_tp[:, [0]]) * torch.sin(obj_tp[:, [1]])
        disp[:, [1]] = rope_length * torch.sin(obj_tp[:, [0]]) * torch.sin(obj_tp[:, [1]])
        disp[:, [2]] = -rope_length * torch.cos(obj_tp[:, [1]])
        return disp

    def get_obj_rel_vel(self, obj_tp, obj_vtp, rope_length):
        '''
        drone_vxyz: (env_num, 3)
        obj_vtp: (env_num, 3)
        rope_length: (env_num, 1)
        '''
        theta = obj_tp[:, [0]]
        phi = obj_tp[:, [1]]
        dtheta = obj_vtp[:, [0]]
        dphi = obj_vtp[:, [1]]

        vt = rope_length * torch.sin(phi) * dtheta
        vp = rope_length * dphi
        vtx = - vt * torch.sin(theta)
        vty = vt * torch.cos(theta)
        vpz = vp * torch.sin(phi)
        vpxy = vp * torch.cos(phi)
        vpx = vpxy * torch.cos(theta)
        vpy = vpxy * torch.sin(theta)
        
        vx = vtx + vpx
        vy = vty + vpy
        vz = vpz

        return torch.concat([vx, vy, vz], dim=-1)

    def _generate_traj(self):
        traj_len = self.max_steps + self.obs_traj_len
        delta_t = self.dt * self.substep_num
        base_w = 2 * np.pi / (40.0 * delta_t)
        t = torch.arange(0, traj_len, 1, device=self.device) * delta_t
        t = torch.tile(t.unsqueeze(-1).unsqueeze(-1), (1, self.env_num, 3))
        x = torch.zeros((traj_len, self.env_num, 3), device=self.device)
        v = torch.zeros((traj_len, self.env_num, 3), device=self.device)
        for i in np.arange(0,2,1):
            # DEBUG
            A = 0.5*torch.tile(((torch.rand((1, self.env_num, 3), device=self.device)*0.3+0.7)*(2.0**(-i))), (traj_len,1,1))

            w = base_w*(2**i)

            phase = torch.tile(torch.rand((1, self.env_num, 3), device=self.device)*(2*np.pi), (traj_len,1,1))

            x += A*torch.cos(t*w+phase)
            v -= w*A*torch.sin(t*w+phase)
        return x, v
    
    def get_env_params(self):
        return None

def test_quadtrans(env:QuadTransEnv, policy, adaptor=None, compressor=None, save_path=None):
    '''
    test the environment
    '''
    state, info = env.reset()
    total_steps = env.max_steps*5
    record_params = {key: [] for key in env.plot_params}
    for _ in range(total_steps):
        with torch.no_grad():
            e_pred = compressor(info['e'])
            act = policy(state, e_pred)
            state, rew, done, info = env.step(act)  # take a random action
        # record parameters
        for key in env.plot_params:
            record_params[key].append(getattr(env, key).numpy().copy())
    # concate all parameters
    for key in env.plot_params:
        record_params[key] = np.concatenate(record_params[key], axis=0)
    # plot all parameters
    plot_num = len(env.plot_params)
    fig, axs = plt.subplots(plot_num, 1, figsize=(10, plot_num*5))
    for i, key in enumerate(env.plot_params):
        # set title
        axs[i].set_title(key)
        param = record_params[key]
        for j in range(param.shape[1]):
            axs[i].plot(param[:, j], label=f'{key}-{j}')
        axs[i].legend()
    env.close()
    if save_path == None:
        package_path = os.path.dirname(adaptive_control_gym.__file__)
        save_path = f"{package_path}/../results/test"
    plt.savefig(f'{save_path}.png')

def playground():
    '''
    use meshcat to visualize the environment
    '''
    import meshcat
    import meshcat.geometry as g
    import meshcat.transformations as tf
    import time

    # set torch print precision
    torch.set_printoptions(precision=2)

    env = QuadTransEnv(env_num=1, gpu_id=-1)
    env.mass_drone_min[:], env.mass_drone_max[:] = 0.027, 0.027
    env.mass_obj_min[:], env.mass_obj_max[:] = 0.00, 0.00
    env.damping_rate_drone_min[:], env.damping_rate_drone_max[:] = 0.00, 0.00
    # env.max_steps = 40
    env.reset()
    vis_env_id = 0

    '''
    create visualizer
    '''
    vis = meshcat.Visualizer()
    # set camera position
    vis["/Cameras/default"].set_transform(
		tf.translation_matrix([0,0,0]).dot(
		tf.euler_matrix(0,np.radians(-45),-np.pi/2)))
    vis["/Cameras/default/rotated/<object>"].set_transform(
        tf.translation_matrix([1.5, 0, 0]))
    # set quadrotor model
    vis["drone"].set_object(g.StlMeshGeometry.from_file('../assets/crazyflie2.stl'))
    # set object model as a sphere
    vis["obj"].set_object(g.Sphere(0.01))
    vis["obj_axes"].set_object(g.StlMeshGeometry.from_file('../assets/axes.stl'), material=g.MeshLambertMaterial(color=0x00FF00))
    # set target model as a sphere
    vis["target"].set_object(g.Sphere(0.01), material=g.MeshLambertMaterial(color=0xff0000))
    # create arrow for visualize the trajectory
    for i in range(env.traj_x.shape[0]):
        vis[f"traj_x{i}"].set_object(g.StlMeshGeometry.from_file('../assets/arrow.stl'), material=g.MeshLambertMaterial(color=0xf000ff))
    # set force as a cylinder and a cone
    vis_force_list = ["force_thrust_drone", "force_drag_drone", "force_gravity_drone", "force_rope_drone", "force_rope_obj", "force_drag_obj", "force_gravity_obj", "total_force_drone"]
    for key in vis_force_list:
        vis[key].set_object(g.StlMeshGeometry.from_file('../assets/arrow.stl'), material=g.MeshLambertMaterial(color=np.random.randint(0xffffff)))
    def vis_vector(obj, origin, vec, scale = 2.0):
        # visualize the force with arrow    
        vec_norm = np.linalg.norm(vec)
        if vec_norm == 0:
            return
        vec = vec / vec_norm
        # gernerate two unit vectors perpendicular to the force vector
        if vec[0] == 0 and vec[1] == 0:
            vec_1 = np.array([1, 0, 0])
            vec_2 = np.array([0, 1, 0])
        else:
            vec_1 = np.array([vec[1], -vec[0], 0])
            vec_1 /= np.linalg.norm(vec_1)
            vec_2 = np.cross(vec, vec_1)
        rot_mat = np.eye(4)
        rot_mat[:3, 2] = vec
        rot_mat[:3, 0] = vec_1
        rot_mat[:3, 1] = vec_2
        rot_mat[:3, :3] *= vec_norm*scale
        obj.set_transform(tf.translation_matrix(origin) @ rot_mat)
    def vis_traj(vis_input, traj_x, traj_v):
        for i in range(traj_x.shape[0]):
            vis_vector(vis_input[f'traj_x{i}'], traj_x[i], traj_v[i], scale=0.5)
    # for PID
    vis["force_pid"].set_object(g.StlMeshGeometry.from_file('../assets/arrow.stl'), material=g.MeshLambertMaterial(color=0x000fff))
    # for neural
    # loaded_agent = torch.load('/home/pcy/rl/policy-adaptation-survey/results/rl/ppo_trans_robust.pt', map_location='cpu')
    # policy = loaded_agent['actor']
    # compressor = loaded_agent['compressor']

    '''
    running experinment
    '''
    rotmat_drone = torch.eye(3, device=env.device)
    rope_force = torch.zeros(3, device=env.device)
    vis_traj(vis, env.traj_x[:, vis_env_id], env.traj_v[:, vis_env_id])
    xyz_drone_his = np.zeros((env.max_steps, 3))
    rpy_drone_his = np.zeros((env.max_steps, 3))
    action_his = np.zeros((env.max_steps, 3))
    for step in range(env.max_steps-1):
        for t in range(env.substep_num):
            # ============= z-axis control =============
            # rope_force_projected = (torch.inverse(rotmat_drone) @ rope_force)[2].item()
            # thrust = env.mass_drone*env.g / torch.cos(torch.norm(env.rpy_drone[0], dim=-1)) - rope_force_projected
            # angle = 0.0
            # if env.step_cnt%20 < 10:
            #     ctl_roll = angle
            #     ctl_pitch = 0.0
            # else:
            #     ctl_roll = -angle
            #     ctl_pitch = 0.0
            # ============= PID control =============
            if t % env.substep_num == 0:
                thrust = env.thrust_pid[vis_env_id]
                ctl_roll = env.ctl_roll_pid[vis_env_id]
                ctl_pitch = env.ctl_pitch_pid[vis_env_id]
                total_force = -rope_force + env.mass_drone[vis_env_id]*env.g*torch.tensor([0, 0, 1], device=env.device) - env.drone2goal[vis_env_id] * 1.0 - env.vxyz_drone[vis_env_id] * 0.2
                total_force_projected = (torch.inverse(rotmat_drone) @ total_force)[2].item()
                thrust = total_force_projected
                total_force_normed = total_force / torch.norm(total_force)
                ctl_roll = torch.atan2(-total_force_normed[1], torch.sqrt(total_force_normed[vis_env_id]**2 + total_force_normed[2]**2))
                ctl_pitch = torch.atan2(total_force_normed[vis_env_id], total_force_normed[2])
                vis_vector(vis["force_pid"], env.xyz_drone[vis_env_id].numpy(), total_force.numpy())
            # ============= random control =============
            # if t % 10 == 0:
            #     thrust = np.random.uniform(0.0, 0.8)
            #     ctl_roll = np.random.uniform(-np.pi/3, np.pi/3)
            #     ctl_pitch = np.random.uniform(-np.pi/3, np.pi/3)
            # ============= neural control =============
            # if t % env.substep_num == 0:
            #     obs = env._get_obs()
            #     action = policy(obs, compressor(env._get_info()['e']))
            #     thrust = action[vis_env_id,0].item() * env.thrust_std + env.thrust_mean
            #     ctl_roll = action[vis_env_id, 1].item() * env.ctl_roll_std + env.ctl_roll_mean
            #     ctl_pitch = action[vis_env_id, 2].item() * env.ctl_pitch_std + env.ctl_pitch_mean
            # ============= debug control =============
            # if t % env.substep_num == 0:
            #     roll = env.rpy_drone[vis_env_id, 0].item()
            #     pitch = env.rpy_drone[vis_env_id, 1].item()
            #     thrust = (0.027+0.01)*9.81 / np.cos(np.sqrt(pitch**2+roll**2))
            #     if env.step_cnt%40 < 10:
            #         ctl_pitch = np.pi/24
            #     elif env.step_cnt%40 < 30:
            #         ctl_pitch = -np.pi/24
            #     else:
            #         ctl_pitch = np.pi/24
            #     ctl_roll = ctl_pitch
            thrust = torch.tensor([thrust], dtype=torch.float32)
            ctl_roll = torch.tensor([ctl_roll], dtype=torch.float32)
            ctl_pitch = torch.tensor([ctl_pitch], dtype=torch.float32)
            debug_info = env.substep(thrust, ctl_roll, ctl_pitch)
            rope_force = debug_info['force_rope_drone'][vis_env_id]
            rotmat_drone = debug_info['rotmat_drone'][vis_env_id]
            # visualize the drone
            xyz_drone = env.xyz_drone[vis_env_id].numpy()
            rpy_drone = env.rpy_drone[vis_env_id].numpy()
            vis["drone"].set_transform(tf.translation_matrix(xyz_drone) @ tf.euler_matrix(*rpy_drone))
            # visualize the target
            xyz_target = env.xyz_target[vis_env_id].numpy()
            vis["target"].set_transform(tf.translation_matrix(xyz_target))
            # visualize the object
            xyz_obj = env.xyz_drone[vis_env_id].numpy() + env.get_obj_disp(env.tp_obj, env.length_rope)[vis_env_id].numpy()
            vis["obj"].set_transform(tf.translation_matrix(xyz_obj))
            obj_rot_mat = np.eye(4)
            obj_rot_mat[:3, 0] = debug_info['x_hat_obj'][vis_env_id].numpy()
            obj_rot_mat[:3, 1] = debug_info['y_hat_obj'][vis_env_id].numpy()
            obj_rot_mat[:3, 2] = debug_info['z_hat_obj'][vis_env_id].numpy()
            vis["obj_axes"].set_transform(tf.translation_matrix(xyz_obj)@obj_rot_mat)
            # visualize the force
            for key in vis_force_list:
                if 'drone' in key:  
                    origin = xyz_drone
                elif 'obj' in key:
                    origin = xyz_obj
                else:
                    raise "unknown key: {}".format(key)
                vis_vector(obj=vis[key], origin=origin, vec=debug_info[key][vis_env_id].numpy(), scale=2.0)
            # record drone xyz
            xyz_drone_his[env.step_cnt] = env.xyz_drone[vis_env_id].numpy()
            # record drone rpy
            rpy_drone_his[env.step_cnt] = env.rpy_drone[vis_env_id].numpy()
            action_his[env.step_cnt] = np.array([thrust.item(), ctl_roll.item(), ctl_pitch.item()])
            # DEBUG
            time.sleep(env.dt)
        env.step_cnt += 1
        env.xyz_target = env.traj_x[env.step_cnt]
        env.vxyz_target = env.traj_v[env.step_cnt]
        if env.step_cnt == env.max_steps:
            env.reset()
            vis_traj(vis, env.traj_x[:, vis_env_id], env.traj_v[:, vis_env_id])
    # plot the trajectory xyz in three subplot
    fig, axs = plt.subplots(6, 1)
    axs[0].plot(xyz_drone_his[:, 0], label='x')
    axs[0].plot(env.traj_x[:,vis_env_id,0], label='x_ref', linestyle='--')
    axs[1].plot(xyz_drone_his[:, 1], label='y')
    axs[1].plot(env.traj_x[:,vis_env_id,1], label='y_ref', linestyle='--')
    axs[2].plot(xyz_drone_his[:, 2], label='z')
    axs[2].plot(env.traj_x[:,vis_env_id,2]+0.2, label='z_ref', linestyle='--')
    axs[3].plot(rpy_drone_his[:, 0], label='roll')
    axs[3].plot(action_his[:, 1], label='roll_ctl', linestyle='--')
    axs[4].plot(rpy_drone_his[:, 1], label='pitch')
    axs[4].plot(action_his[:, 2], label='pitch_ctl', linestyle='--')
    axs[5].plot(rpy_drone_his[:, 2], label='yaw')
    for i in range(6):
        axs[i].grid()
        axs[i].legend()
    # save the plot
    plt.savefig('results/traj.png')
        
if __name__ == '__main__':
    playground()