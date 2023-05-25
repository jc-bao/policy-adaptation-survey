import gym
import torch
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic
import meshcat
from meshcat.animation import Animation, convert_frames_to_video
import meshcat.geometry as g
import meshcat.transformations as tf
import time
import pandas as pd
import os
import tyro
from dataclasses import dataclass

import adaptive_control_gym
from adaptive_control_gym.utils import geom


class QuadTransEnv(gym.Env):
    def __init__(self,
                 env_num: int = 1024,
                 drone_num: int = 1,
                 gpu_id: int = 0,
                 seed: int = 0,
                 enable_log: bool = False,
                 enable_vis: bool = False,
                 task: str = 'track',  # 'hover', 'track', 'avoid'
                 disable_obj: bool = False, 
                 disable_hook: bool = False,
                 disable_loose_rope: bool = False,
                 **kwargs) -> None:

        super().__init__()

        # setup torch
        torch.manual_seed(seed)
        torch.set_default_dtype(torch.float32)

        # set task and related parameters
        self.task = task
        assert self.task in {
            'hover',
            'track',
            'avoid',
        }, 'task must be one of hover, track, avoid'

        # set simulator parameters
        self.disable_obj = disable_obj
        self.disable_hook = disable_hook
        self.disable_loose_rope = disable_loose_rope
        self.seed = seed
        self.env_num = env_num
        self.sim_dt = 4e-4
        self.ctl_substeps = 5
        self.drone_num = drone_num
        self.ctl_dt = self.sim_dt * self.ctl_substeps
        self.step_substeps = 50
        self.max_steps = 50
        self.step_dt = self.ctl_dt * self.step_substeps
        self.gpu_id = gpu_id
        self.device = torch.device(
            f"cuda:{self.gpu_id}" if (
                torch.cuda.is_available() & (gpu_id >= 0)) else "cpu"
        )

        # set RL parameters
        self.state_dim = 3 + 3 + (3 + 3 + 4 + 3) * \
            self.drone_num + 3 + 3
        self.expert_dim = 12 * drone_num + 3
        self.adapt_horizon = 1
        self.record_len = 2
        self.adapt_dim = (29*drone_num+12) * self.adapt_horizon
        self.action_dim = 3 * drone_num
        self.curri_param = 0.0

        # set record history
        self.set_history()

        # set state
        self.set_state()

        # controller
        self.set_control_params()

        # state variables, physical parameters
        self.reset()

        # set logger and visualizer
        self.logger = Logger(drone_num=drone_num, enable=enable_log)
        self.visualizer = MeshVisulizer(env=self, enable=enable_vis)

    def set_state(self):
        # enviroment variables
        self.step_cnt = torch.zeros(
            (self.env_num), device=self.device, dtype=torch.long)
        self.reach_goal_step = torch.zeros(
            (self.env_num), device=self.device, dtype=torch.long)
        self.xyz_traj = torch.zeros(
            (self.max_steps*2, self.env_num, 3), device=self.device)
        self.vxyz_traj = torch.zeros(
            (self.max_steps*2, self.env_num, 3), device=self.device)
        self.xyz_obj_target = self.xyz_traj[0]
        self.vxyz_obj_target = self.vxyz_traj[0]
        self.xyz_drones = torch.zeros(
            (self.env_num, self.drone_num, 3), device=self.device)
        self.vxyz_drones = torch.zeros(
            (self.env_num, self.drone_num, 3), device=self.device)
        self.quat_drones = torch.zeros(
            (self.env_num, self.drone_num, 4), device=self.device)
        self.vrpy_drones = torch.zeros(
            (self.env_num, self.drone_num, 3), device=self.device)
        self.vrpy_target = torch.zeros(
            (self.env_num, self.drone_num, 3), device=self.device)
        self.thrust = torch.zeros(
            (self.env_num, self.drone_num), device=self.device)
        self.xyz_obj = torch.zeros((self.env_num, 3), device=self.device)
        self.vxyz_obj = torch.zeros((self.env_num, 3), device=self.device)
        self.reward = torch.zeros((self.env_num), device=self.device)

    def set_history(self):
        # record parameters
        # control related parameters
        self.thrust_his = torch.zeros(
            (self.record_len, self.env_num, self.drone_num), device=self.device)
        self.vrpy_target_his = torch.zeros(
            (self.record_len, self.env_num, self.drone_num, 3), device=self.device)
        # drone state
        zeros3 = torch.zeros(
            (self.record_len, self.env_num, self.drone_num, 3), device=self.device)
        self.xyz_drones_his = zeros3.clone()
        self.vxyz_drones_his = zeros3.clone()
        self.axyz_drones_his = zeros3.clone()
        self.jxyz_drones_his = zeros3.clone()
        self.quat_drones_his = torch.zeros(
            (self.record_len, self.env_num, self.drone_num, 4), device=self.device)
        self.vrpy_drones_his = zeros3.clone()
        self.arpy_drones_his = zeros3.clone()
        self.jrpy_drones_his = zeros3.clone()
        # object state
        zeros3 = torch.zeros(
            (self.record_len, self.env_num, 3), device=self.device)
        self.xyz_obj_his = zeros3.clone()
        self.vxyz_obj_his = zeros3.clone()
        self.axyz_obj_his = zeros3.clone()
        self.jxyz_obj_his = zeros3.clone()

    def reset_history(self, done):
        # reset records
        self.xyz_drones_his[:, done] *= 0
        self.vxyz_drones_his[:, done] *= 0
        self.axyz_drones_his[:, done] *= 0
        self.jxyz_drones_his[:, done] *= 0
        self.quat_drones_his[:, done] *= 0
        self.vrpy_drones_his[:, done] *= 0
        self.arpy_drones_his[:, done] *= 0
        self.jrpy_drones_his[:, done] *= 0
        self.xyz_obj_his[:, done] *= 0
        self.vxyz_obj_his[:, done] *= 0
        self.axyz_obj_his[:, done] *= 0
        self.jxyz_obj_his[:, done] *= 0
        self.save_state()

    def step(self, action):
        # preprocess action
        action = action.reshape(self.env_num, self.drone_num, 3)
        thrust = (action[..., 0] + 1.0) / 2.0 * self.max_thrust
        vrp_target = action[..., 1:] * self.max_vrp

        # state placeholder
        reward_tensor = torch.zeros(
            (self.step_substeps, self.env_num), device=self.device, dtype=torch.float32)
        hitwall = torch.zeros(
            (self.env_num), device=self.device, dtype=torch.bool)
        rope_force_done = torch.zeros(
            (self.env_num), device=self.device, dtype=torch.bool)

        # main step: ctl_step to run controller
        for i in range(self.step_substeps):
            state = self.ctlstep(vrp_target, thrust)

            # record state
            reward_tensor[i] = state['reward_groundtruth']
            hitwall = hitwall | state['hitwall']
            rope_force_done = rope_force_done | state['rope_force_done']

        # update goals and steps
        self.step_cnt += 1
        rows = torch.arange(self.env_num, device=self.device, dtype=torch.long)
        self.xyz_obj_target = self.xyz_traj[self.step_cnt, rows]
        self.vxyz_obj_target = self.vxyz_traj[self.step_cnt, rows]

        # record history for adaptation
        self.save_state()
        reward = reward_tensor.mean(dim=0)
        self.reward = reward  # for logging purpose

        # calculate auto done
        # reach goal done
        reached_goal = (torch.norm(
            self.xyz_obj - self.xyz_obj_target, dim=-1) < self.reach_threshold).long()
        self.reach_goal_step = (
            reached_goal + self.reach_goal_step) * reached_goal
        reach_done = self.reach_goal_step >= self.max_reach_step
        # reach time limit done
        time_done = (self.step_cnt >= self.max_steps)
        # reach angle limit done
        cos_err_div2 = self.quat_drones[..., 3]
        quat_done = (cos_err_div2 < np.cos(self.max_angle/2)).any(dim=-1)
        # reach position limit done
        xyz_done = (torch.abs(self.xyz_drones) >=
                    self.max_xyz).any(dim=1).any(dim=1)
        # unomoved for a long time done
        past_idx = (torch.arange(self.unmoved_time_limit, device=self.device,
                    dtype=torch.long).unsqueeze(1) - self.unmoved_time_limit + 1 + self.step_cnt.unsqueeze(0)) % self.record_len
        col = torch.arange(self.env_num, device=self.device, dtype=torch.long)
        past_traj = self.xyz_obj_his[past_idx, col]
        unmoved_done = (torch.norm(
            past_traj - self.xyz_obj, dim=-1).max(dim=0)[0] < self.reach_threshold) & (self.step_cnt >= 5)
        # hit the wall done
        if self.task == 'hitwall':
            hitwall_done = hitwall
        else:
            hitwall_done = torch.zeros(
                (self.env_num), device=self.device, dtype=torch.bool)

        done = reach_done | time_done | quat_done | xyz_done | unmoved_done | rope_force_done | hitwall_done
        self.reset(done)
        next_obs = self.get_obs()
        next_info = self.get_info()

        return next_obs, reward, done, next_info

    def ctlstep(self, vrp_target: torch.Tensor, thrust: torch.Tensor):
        # yaw rate controller to make sure yaw angle close to zero
        rpy_drones = geom.quat2rpy(self.quat_drones)
        yaw_drones = rpy_drones[..., [2]]
        vy_target = - yaw_drones * 15.0

        # set target value to attitude rate controller
        vrpy_target = torch.cat([vrp_target, vy_target], dim=-1)
        self.vrpy_target = vrpy_target
        self.thrust = thrust

        # run lower level attitude rate L1 adaptive controller
        self.vrpy_err = vrpy_target - self.vrpy_drones
        torque = (self.J_drones @ self.attirate_controller.update(self.vrpy_drones,
                  self.vrpy_target, self.u_attirate).unsqueeze(-1)).squeeze(-1)
        thrust = torch.clip(thrust, 0.0, self.max_thrust)
        torque = torch.clip(torque, -self.max_torque, self.max_torque)
        self.u_attirate = (torch.inverse(self.J_drones) @
                           torque.unsqueeze(-1)).squeeze(-1)

        # substep state variables to assist further information calculation
        reward_tensor = torch.zeros(
            (self.ctl_substeps, self.env_num), device=self.device)
        hitwall = torch.zeros(
            (self.env_num), device=self.device, dtype=torch.bool)
        rope_force_done = torch.zeros(
            (self.env_num), device=self.device, dtype=torch.bool)

        # run simulation substeps
        for i in range(self.ctl_substeps):
            state = self.simstep(torque, thrust)

            # record required states.
            reward_tensor[i] = state['reward_groundtruth'].squeeze(-1)
            hitwall = hitwall | state['hitwall']
            rope_force_done = state['rope_force_done']

        state['reward_groundtruth'] = reward_tensor.mean(dim=0)
        state['rope_force_done'] = rope_force_done

        return state

    def simstep(self, torque: torch.Tensor, thrust: torch.Tensor):
        # state variables
        rotmat_drone = geom.quat2rotmat(self.quat_drones)
        xyz_hook = self.xyz_drones + \
            (rotmat_drone @ self.hook_disp.unsqueeze(-1)).squeeze(-1)

        # force of drones
        # gravity
        gravity_drones = self.g * self.mass_drones
        # thrust
        thrust_drones = torch.zeros(
            (self.env_num, self.drone_num, 3), device=self.device)
        thrust_drones[:, :, 2] = thrust
        thrust_drones = (
            rotmat_drone @ thrust_drones.unsqueeze(-1)).squeeze(-1)
        # rope force
        xyz_obj2hook = self.xyz_obj.unsqueeze(1) - xyz_hook
        xyz_obj2hook_normed = xyz_obj2hook / \
            torch.norm(xyz_obj2hook, dim=-1, keepdim=True)
        rope_origin = self.rope_length * xyz_obj2hook_normed
        rope_disp = xyz_obj2hook - rope_origin
        if self.disable_loose_rope:
            loose_rope = torch.zeros(
                (self.env_num, self.drone_num), device=self.device, dtype=torch.bool)
        else:
            loose_rope = torch.norm(
                xyz_obj2hook, dim=-1) < self.rope_length.squeeze(-1)
        vxyz_hook = self.vxyz_drones + \
            torch.cross(self.vrpy_drones, self.hook_disp, dim=-1)
        vxyz_obj2hook = self.vxyz_obj.unsqueeze(1) - vxyz_hook
        rope_vel = torch.sum(vxyz_obj2hook * xyz_obj2hook_normed,
                             dim=-1, keepdim=True) * xyz_obj2hook_normed
        mass_joint = self.mass_drones * \
            self.mass_obj.unsqueeze(
                1) / (self.mass_drones + self.mass_obj.unsqueeze(1))
        rope_force_drones = mass_joint * \
            ((self.rope_wn ** 2) * rope_disp + 2 *
             self.rope_zeta * self.rope_wn * rope_vel)
        # clip rope force to avoid numerical instability
        rope_force_drones = torch.clip(
            rope_force_drones, -self.max_rope_force, self.max_rope_force)
        # corner case: disable object by setting its weight to zero
        zero_mass_obj = (self.mass_obj < 1e-6).all(dim=1).unsqueeze(1)
        rope_force_drones[loose_rope | zero_mass_obj, :] = 0.0
        # total force
        force_drones = gravity_drones + thrust_drones + rope_force_drones
        # total moment
        dist2center = torch.norm(self.xyz_drones, dim=-1, keepdim=True)
        rope_torque = torch.cross(self.hook_disp, rope_force_drones, dim=-1)
        moment_drones = torque + rope_torque + \
            (self.J_drones @ torch.cross(self.vrpy_drones,
             self.vrpy_drones, axis=-1).unsqueeze(-1)).squeeze(-1)

        # force of the object
        # gravity
        gravity_obj = self.g * self.mass_obj
        # rope force
        rope_force_obj = -torch.sum(rope_force_drones, dim=1)
        # total force
        force_obj = gravity_obj + rope_force_obj - \
            self.vxyz_obj * 0.01  # TODO set as parameter
        force_obj[zero_mass_obj.squeeze(-1), :] = 0.0

        # update the state variables
        # drone
        self.vxyz_drones = self.vxyz_drones + \
            self.sim_dt * force_drones / self.mass_drones
        self.xyz_drones = self.xyz_drones + self.sim_dt * self.vxyz_drones
        self.vrpy_drones = self.vrpy_drones + self.sim_dt * \
            (torch.inverse(self.J_drones) @ moment_drones.unsqueeze(-1)).squeeze(-1)
        # integrate the quaternion
        self.quat_drones = geom.integrate_quat(
            self.quat_drones, self.vrpy_drones, self.sim_dt)
        # object
        axyz_obj = force_obj / self.mass_obj
        axyz_obj[zero_mass_obj.squeeze(-1), :] = 0.0
        self.vxyz_obj = self.vxyz_obj + self.sim_dt * axyz_obj
        self.vxyz_obj[zero_mass_obj.squeeze(-1)
                      ] = self.vxyz_drones[zero_mass_obj.squeeze(-1), 0]
        self.xyz_obj = self.xyz_obj + self.sim_dt * self.vxyz_obj
        self.xyz_obj[zero_mass_obj.squeeze(-1)
                     ] = self.xyz_drones[zero_mass_obj.squeeze(-1), 0]

        # log and visualize for debug purpose
        state = {
            'reward_groundtruth': self.get_reward().unsqueeze(-1),
            'hitwall': (self.get_hit_penalty(self.xyz_drones) < 0.0) | (self.get_hit_penalty(self.xyz_obj) < 0.0),
            'rope_force_done': (torch.norm(rope_force_drones, dim=-1) > 500.0).any(dim=-1),
        }
        if self.logger.enable or self.visualizer.enable:
            state = {
                'step': self.step_cnt,
                'xyz_traj': self.xyz_traj,
                'vxyz_traj': self.vxyz_traj,
                'xyz_drones': self.xyz_drones,
                'vxyz_drones': self.vxyz_drones,
                'quat_drones': self.quat_drones,
                'rpy_drones': geom.quat2rpy(self.quat_drones),
                'vrpy_drones': self.vrpy_drones,
                'xyz_obj': self.xyz_obj,
                'xyz_obj_err': self.xyz_obj - self.xyz_obj_target,
                'vrpy_target': self.vrpy_target,
                'vrpy_err': self.vrpy_err,
                'vxyz_obj': self.vxyz_obj,
                'vxyz_obj_err': self.vxyz_obj - self.vxyz_obj_target,
                'force_drones': force_drones,
                'moment_drones': moment_drones,
                'force_obj': force_obj,
                'rope_force_drones': rope_force_drones,
                'rope_force_norm': rope_force_drones.norm(dim=-1),
                'rope_force_obj': rope_force_obj,
                'gravity_drones': gravity_drones,
                'thrust_drones': thrust_drones,
                'rope_disp': rope_disp,
                'rope_vel': rope_vel,
                'torque': torque,
                'xyz_obj_target': self.xyz_obj_target,
                'vxyz_obj_target': self.vxyz_obj_target,
                'reward': self.reward.unsqueeze(-1),
                **state
            }
        self.logger.log(state)
        self.visualizer.update(state)

        return state

    def get_env_params(self):
        # TODO required by throughout parametric search
        return None

    def save_state(self):
        col = torch.arange(self.env_num, device=self.device, dtype=torch.long)
        step_idx = self.step_cnt % self.record_len
        mask = (self.step_cnt > 0).unsqueeze(-1).unsqueeze(-1)
        axyz_drones = (
            self.vxyz_drones_his[step_idx, col] - self.vxyz_drones_his[step_idx-1, col]) / self.step_dt * mask
        axyz_obj = (self.vxyz_obj_his[step_idx, col] -
                    self.vxyz_obj_his[step_idx-1, col]) / self.ctl_dt * mask.squeeze(-1)
        arpy_drones = (
            self.vrpy_drones_his[step_idx, col] - self.vrpy_drones_his[step_idx-1, col]) / self.step_dt * mask
        mask = (self.step_cnt > 1).unsqueeze(-1).unsqueeze(-1)
        jxyz_drones = (
            self.axyz_drones_his[step_idx, col] - self.axyz_drones_his[step_idx-1, col]) / self.step_dt * mask
        jxyz_obj = (self.axyz_obj_his[step_idx, col] -
                    self.axyz_obj_his[step_idx-1, col]) / self.ctl_dt * mask.squeeze(-1)
        jrpy_drones = (
            self.arpy_drones_his[step_idx, col] - self.arpy_drones_his[step_idx-1, col]) / self.step_dt * mask

        self.thrust_his[step_idx, col] = self.thrust
        self.vrpy_target_his[step_idx, col] = self.vrpy_target

        self.xyz_drones_his[step_idx, col] = self.xyz_drones
        self.vxyz_drones_his[step_idx, col] = self.vxyz_drones
        self.axyz_drones_his[step_idx, col] = axyz_drones
        self.jxyz_drones_his[step_idx, col] = jxyz_drones

        self.quat_drones_his[step_idx, col] = self.quat_drones
        self.vrpy_drones_his[step_idx, col] = self.vrpy_drones
        self.arpy_drones_his[step_idx, col] = arpy_drones
        self.jrpy_drones_his[step_idx, col] = jrpy_drones

        self.xyz_obj_his[step_idx, col] = self.xyz_obj
        self.vxyz_obj_his[step_idx, col] = self.vxyz_obj
        self.axyz_obj_his[step_idx, col] = axyz_obj
        self.jxyz_obj_his[step_idx, col] = jxyz_obj

    def get_hit_penalty(self, xyz):
        within_obs_x_range = torch.abs(xyz[..., 0]) < self.wall_half_width
        outof_obs_z_range = torch.abs(xyz[..., 2]) > self.wall_half_height

        hit_x_bound = within_obs_x_range & outof_obs_z_range

        hit_panelty = - torch.clip(hit_x_bound.float() * torch.min(self.wall_half_width-torch.abs(
            xyz[..., 0]), torch.abs(xyz[..., 2])-self.wall_half_height) * 500.0, 0, 1)

        if len(hit_panelty.shape) == 2:
            hit_panelty = hit_panelty.sum(dim=-1)

        return hit_panelty

    def get_avoid_reward(self):

        err_x = torch.norm(self.xyz_obj - self.xyz_obj_target, dim=1)
        err_v = torch.norm(self.vxyz_obj - self.vxyz_obj_target, dim=1)

        close2target = (err_x < 0.2)
        reward = 1.5 + \
            (1-torch.clip(err_v, 0, 2)*0.5) * close2target.float() - \
            torch.clip(err_x, 0, 2)*0.5 - \
            torch.clip(torch.log(err_x+1)*5, 0, 1)*0.1 - \
            torch.clip(torch.log(err_v+1)*5, 0, 1)*0.1

        drone_panelty = self.get_hit_penalty(self.xyz_drones) * 10.0
        obj_panelty = self.get_hit_penalty(self.xyz_obj) * 10.0
        reward *= ((drone_panelty >= 0) | (obj_panelty >= 0)).float()
        reward += drone_panelty
        reward += obj_panelty

        return reward

    def get_track_reward(self):
        err_x = torch.norm(self.xyz_obj - self.xyz_obj_target, dim=1)
        err_v = torch.norm(self.vxyz_obj - self.vxyz_obj_target, dim=1)

        reward = 1.5 - torch.clip(err_x, 0, 3)*0.5 - \
            torch.clip(err_v, 0, 2)*0.5 - \
            torch.clip(torch.log(err_x+1)*5, 0, 1)*0.1 - \
            torch.clip(torch.log(err_x+1)*10, 0, 1)*0.1

        return reward

    def get_reward(self):
        if self.task == 'avoid':
            return self.get_avoid_reward()
        elif self.task in ['track', 'hover']:
            return self.get_track_reward()
        else:
            raise NotImplementedError

    def get_obs(self):
        # calculate observation
        # make sure all element is not NaN
        assert not torch.isnan(self.xyz_obj).any()
        assert not torch.isnan(self.vxyz_obj).any()
        assert not torch.isnan(self.xyz_drones).any()
        assert not torch.isnan(self.vxyz_drones).any()
        assert not torch.isnan(self.quat_drones).any()
        assert not torch.isnan(self.vrpy_drones).any()

        return torch.cat(
            [
                self.xyz_obj,
                self.vxyz_obj,
                self.xyz_drones.reshape(self.env_num, -1),
                self.vxyz_drones.reshape(self.env_num, -1) / 2.0,
                self.quat_drones.reshape(self.env_num, -1),
                self.vrpy_drones.reshape(self.env_num, -1) / 15.0,
                self.xyz_obj - self.xyz_obj_target,
                self.vxyz_obj - self.vxyz_obj_target,
            ],
            dim=-1,
        )

    def get_info(self):
        drone_info = torch.cat([
            self.mass_drones/0.027,
            self.J_drones[:, :, 0, [0]]/1.7e-5,
            self.J_drones[:, :, 1, [1]]/1.7e-5,
            self.J_drones[:, :, 2, [2]]/2.98e-5,
            self.hook_disp / 0.015,
            self.rope_length/0.2,
            self.rope_zeta/0.75,
            self.rope_wn/1000.0,
        ], dim=-1).reshape(self.env_num, -1)  # 10*drone_num
        expert_info = torch.cat([
            drone_info,
            self.mass_obj/0.01,
        ], dim=-1)
        adapt_idx = torch.arange(self.adapt_horizon, device=self.device,
                                 dtype=torch.long).unsqueeze(1) - self.adapt_horizon + 1 + self.step_cnt.unsqueeze(0)
        adapt_idx %= self.record_len
        col = torch.arange(self.env_num, device=self.device, dtype=torch.long)
        adapt_obs_ctl = torch.cat([
            # control related parameters
            self.thrust_his.unsqueeze(
                -1)[adapt_idx, col], self.vrpy_target_his[adapt_idx, col],
        ], dim=-1).permute(1, 0, 2, 3).reshape(self.env_num, -1)
        adapt_obs_drones = torch.cat([
            # drone state
            self.xyz_drones_his[adapt_idx, col], self.vxyz_drones_his[
                adapt_idx, col], self.axyz_drones_his[adapt_idx, col], self.jxyz_drones_his[adapt_idx, col],
            self.quat_drones_his[adapt_idx, col], self.vrpy_drones_his[
                adapt_idx, col], self.arpy_drones_his[adapt_idx, col], self.jrpy_drones_his[adapt_idx, col],
        ], dim=-1).permute(1, 0, 2, 3).reshape(self.env_num, -1)
        adapt_obs_obj = torch.cat([
            # object state
            self.xyz_obj_his[adapt_idx, col], self.vxyz_obj_his[adapt_idx,
                                                                col], self.axyz_obj_his[adapt_idx, col], self.jxyz_obj_his[adapt_idx, col],
        ], dim=-1).permute(1, 0, 2).reshape(self.env_num, -1)
        adapt_obs = torch.cat([
            adapt_obs_ctl, adapt_obs_drones, adapt_obs_obj,
        ], dim=-1)
        return {
            'e': expert_info,
            'adapt_obs': adapt_obs,
            'err_x': torch.norm(self.xyz_obj - self.xyz_obj_target, dim=1),
            'err_v': torch.norm((self.vxyz_obj - self.vxyz_obj_target), dim=1),
            'step': self.step_cnt,
        }

    def policy_pid(self):
        # PID controller to command quadrotor to reach the target position

        # run lower level position PID controller
        zero_mass_obj = (self.mass_obj < 1e-6).all(dim=1)

        # Object-level controller
        delta_pos = torch.clip(self.xyz_obj_target - self.xyz_obj, -1.0, 1.0)
        force_obj_pid = self.mass_obj * \
            self.objpos_controller.update(
                delta_pos, self.step_dt)
        target_force_obj = force_obj_pid - self.mass_obj * self.g
        xyz_obj2drone = self.xyz_obj - self.xyz_drones
        z_hat_obj = xyz_obj2drone / \
            torch.norm(xyz_obj2drone, dim=-1, keepdim=True)
        # TODO distribute the force to multiple drones
        target_force_obj_projected = torch.sum(
            target_force_obj * z_hat_obj, dim=-1, keepdim=True) * z_hat_obj / self.drone_num
        target_force_obj_projected[zero_mass_obj, :] = 0.0

        # Drone-level controller
        xyz_drone_target = (self.xyz_obj + target_force_obj /
                            torch.norm(target_force_obj, dim=-1, keepdim=True) *
                            self.rope_length) - self.hook_disp
        # if object mass is too small, use object position as target
        xyz_drone_target[zero_mass_obj] = self.xyz_obj_target[zero_mass_obj].unsqueeze(
            1).repeat(1, self.drone_num, 1)
        delta_pos_drones = xyz_drone_target - self.xyz_drones
        target_force_drone = self.mass_drones*self.pos_controller.update(
            delta_pos_drones, self.step_dt) - (self.mass_drones) * self.g + target_force_obj_projected
        rotmat_drone = geom.quat2rotmat(self.quat_drones)
        thrust_desired = (
            torch.inverse(rotmat_drone)@target_force_drone.unsqueeze(-1)).squeeze(-1)
        thrust = torch.norm(thrust_desired, dim=-1)
        desired_rotvec = torch.zeros(
            [self.env_num, self.drone_num, 3], device=self.device)
        desired_rotvec[:, :, 2] = 1.0

        rot_err = torch.cross(
            desired_rotvec, thrust_desired/torch.norm(thrust_desired, dim=-1, keepdim=True), dim=-1)
        rpy_rate_target = self.attitude_controller.update(
            rot_err, self.step_dt)

        thrust_normed = thrust / (0.5 * self.max_thrust) - 1.0
        vrp_target = rpy_rate_target[..., :2] / self.max_vrp

        return torch.cat([thrust_normed.unsqueeze(-1), vrp_target], dim=-1)

    def reset(self, done=None):
        if done is None:
            done = torch.ones(
                (self.env_num), device=self.device, dtype=torch.bool)
        self.sample_physical_params(done)
        self.sample_state(done)
        
        # reset steps
        self.step_cnt[done] = 0
        self.reach_goal_step[done] = self.max_steps

        self.reset_history(done)

        return self.get_obs(), self.get_info()

    def sample_state(self, done):
        size = torch.sum(done)
        # sample object initial position
        self.xyz_obj[done] = (torch.rand(
            [size, 3], device=self.device)*2.0 - 1.0) * 0.5
        if self.task == 'avoid':
            self.xyz_obj[done] = torch.rand(
                (1, size, 3), device=self.device)*0.2 + 0.3

        # sample target trajectory
        self.xyz_traj[:, done], self.vxyz_traj[:,
                                               done] = self.generate_traj(size)

        # sample goal position
        self.xyz_obj_target[done] = self.xyz_traj[0, done]
        self.vxyz_obj_target[done] = self.vxyz_traj[0, done]
        # sample drone initial position
        thetas = torch.rand([size, self.drone_num],
                            device=self.device) * 2 * np.pi
        phis = torch.rand([size, self.drone_num],
                          device=self.device) * 0.5 * np.pi

        xyz_drones2obj = torch.stack([torch.sin(phis) * torch.cos(thetas),
                                      torch.sin(phis) * torch.sin(thetas),
                                      torch.cos(phis)], dim=-1) * self.rope_length[done] - self.hook_disp[done]
        self.xyz_drones[done] = self.xyz_obj[done].unsqueeze(
            1) + xyz_drones2obj

        # reset drone initial attitude
        self.quat_drones[done] = torch.zeros(
            [size, self.drone_num, 4], device=self.device)
        self.quat_drones[done, :, 3] = 1.0
        # reset drone initial velocity
        self.vxyz_drones[done] = torch.zeros(
            [size, self.drone_num, 3], device=self.device)
        # reset object initial velocity
        self.vxyz_obj[done] = torch.zeros([size, 3], device=self.device)
        # reset drone initial angular velocity
        self.vrpy_drones[done] = torch.zeros(
            [size, self.drone_num, 3], device=self.device)

    def generate_traj(self, size):
        # generate a trajectory for the task. The trajectory is a circle for the track task and a straight line for the hover/avoidance task.

        traj_len = self.max_steps * 2
        delta_t = self.step_dt
        base_w = 2 * np.pi / (40.0 * delta_t)
        t = torch.arange(0, traj_len, 1, device=self.device) * delta_t
        t = torch.tile(t.unsqueeze(-1).unsqueeze(-1), (1, size, 3))
        x = torch.zeros((traj_len, size, 3), device=self.device)
        v = torch.zeros((traj_len, size, 3), device=self.device)
        if self.task == 'track':
            for i in np.arange(0, 2, 1):
                A = 0.5*torch.tile(((torch.rand((1, size, 3),
                                                device=self.device)*0.3+0.7)*(2.0**(-i))), (traj_len, 1, 1))

                w = base_w*(2**i)

                phase = torch.tile(torch.rand(
                    (1, size, 3), device=self.device)*(2*np.pi), (traj_len, 1, 1))

                x += A*torch.cos(t*w+phase)
                v -= w*A*torch.sin(t*w+phase)
        elif self.task == 'hover':
            x = torch.rand((1, size, 3), device=self.device)
        elif self.task == 'avoid':
            x = - torch.rand((1, size, 3), device=self.device)*0.1 - 0.3
        return x, v

    def sample_physical_params(self, done):
        size = torch.sum(done)

        def sample_uni(size):
            if size == 0:
                return (torch.rand(
                    (self.env_num, self.drone_num), device=self.device)*2.0-1.0) * 1.0 * self.curri_param  # DEBUG
            else:
                return (torch.rand(
                    (self.env_num, self.drone_num, size), device=self.device)*2.0-1.0) * 1.0 * self.curri_param  # DEBUG

        self.g = torch.zeros(3, device=self.device)
        self.g[2] = -9.81
        self.mass_drones = torch.zeros(
            (self.env_num, self.drone_num, 3), device=self.device)
        self.mass_drones[..., :] = sample_uni(1) * 0.007 + 0.027
        self.J_drones = torch.zeros(
            (self.env_num, self.drone_num, 3, 3), device=self.device)
        self.J_drones[:, :, 0, 0] = sample_uni(0) * 0.2e-5 + 1.7e-5
        self.J_drones[:, :, 1, 1] = sample_uni(0) * 0.2e-5 + 1.7e-5
        self.J_drones[:, :, 2, 2] = sample_uni(0) * 0.3e-5 + 2.98e-5
        self.hook_disp = sample_uni(3) * torch.tensor(
            [0.01, 0.01, 0.01], device=self.device) + torch.tensor([0.0, 0.0, -0.015], device=self.device)

        if self.disable_hook:
            self.hook_disp *= 0.0 

        self.mass_obj = torch.ones(
            (self.env_num, 3), device=self.device) * 0.02
        self.mass_obj[..., :] = (torch.rand(
            (self.env_num, 1), device=self.device)*2.0-1.0) * 0.005 * self.curri_param + 0.025

        self.rope_length = sample_uni(1) * 0.05 + 0.2
        self.rope_zeta = sample_uni(1) * 0.15 + 1.05
        self.rope_wn = sample_uni(1) * 5 + 50

        self.drone_drag_coef = 0.1
        self.obj_drag_coef = 0.1

        # Update
        self.wall_half_width = 0.05
        self.wall_half_height = 0.07 + \
            np.clip(1-self.curri_param, 0, 1) * 0.1  # from 0.17 -> 0.07
        
        if self.disable_obj:
            self.mass_obj *= 0.0
            self.rope_length *= 0.0
            self.hook_disp *= 0.0

    def set_control_params(self):
        # attitude rate controller
        ones = torch.ones([self.env_num, self.drone_num, 3],
                          device=self.device)
        zeros = torch.zeros(
            [self.env_num, self.drone_num, 3], device=self.device)
        self.attirate_controller = PIDController(
            kp=ones * torch.tensor([4e2, 4e2, 1e2], device=self.device),
            ki=ones * torch.tensor([4e4, 4e4, 2e4], device=self.device),
            kd=zeros,
            ki_max=ones * torch.tensor([1e-1, 1e-1, 1e-1], device=self.device),
            integral=zeros, last_error=zeros
        )
        self.u_attirate = zeros
        self.attirate_controller = L1AdpativeController(
            kp=ones * (100.0),
            As=ones * (-10.0),
            B=ones * 1.0,
            sigma_hat=zeros,
            x_hat=zeros,
            u_ad=zeros,
            filter_co=ones * 1000.0,
            dt=self.ctl_dt
        )
        ones = torch.ones([self.env_num, 3], device=self.device)
        zeros = torch.zeros([self.env_num, 3], device=self.device)
        self.objpos_controller = PIDController(
            kp=ones*8.0, ki=ones*0.0, kd=ones*3.0, ki_max=ones*100.0,
            integral=zeros, last_error=zeros
        )
        # position controller
        ones = torch.ones([self.env_num, self.drone_num, 3],
                          device=self.device)
        zeros = torch.zeros(
            [self.env_num, self.drone_num, 3], device=self.device)
        self.pos_controller = PIDController(
            kp=ones*16.0, ki=ones*0.0, kd=ones*6.0, ki_max=ones*100.0,
            integral=zeros, last_error=zeros
        )
        # attitude controller
        self.attitude_controller = PIDController(
            kp=ones*20.0, ki=ones*0.0, kd=ones*0.0, ki_max=ones*100.0,
            integral=zeros, last_error=zeros
        )
        # thrust limits
        self.max_thrust = 0.6
        self.max_vrp = 12.0
        self.max_rope_force = 20.0
        self.max_torque = torch.tensor([9e-3, 9e-3, 2e-3], device=self.device)
        self.max_angle = np.pi/2.1  # if exceed this angle, reset the environment
        self.panelty_angle = np.pi/3.0  # if exceed this angle, give negative reward

        # auto reset parameters
        self.max_reach_step = 8
        self.reach_threshold = 0.04
        self.max_xyz = 2.0
        self.unmoved_time_limit = 5

    def render(self, mode='human'):
        pass

    def close(self, savepath):
        self.logger.plot(savepath)


class PIDController:
    """PID controller for attitude rate control

    Returns:
        _type_: _description_
    """

    def __init__(self, kp, ki, kd, ki_max, integral, last_error):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.ki_max = ki_max
        self.integral = integral
        self.last_error = last_error
        self.reset()

    def reset(self):
        self.integral *= 0.0
        self.last_error *= 0.0

    def update(self, error, dt):
        self.integral += error * dt
        self.integral = torch.clip(self.integral, -self.ki_max, self.ki_max)
        derivative = (error - self.last_error) / dt
        self.last_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative


class L1AdpativeController:
    def __init__(self, kp, As, B, sigma_hat, x_hat, u_ad, filter_co, dt):
        self.kp = kp
        self.As = As
        self.B = B
        self.dt = dt
        self.sigma_hat = sigma_hat
        self.x_hat = x_hat
        self.u_ad = u_ad

        self.Phi = 1.0/As*(torch.exp(As*dt) - 1)
        self.alpha = torch.exp(-dt * filter_co)

    def reset(self):
        self.x_hat *= 0.0
        self.sigma_hat *= 0.0
        self.u_ad *= 0.0

    def update(self, x, x_desired, u):
        x_hat_dot = self.B * (u + self.sigma_hat) + self.As * (self.x_hat - x)
        self.x_hat += x_hat_dot * self.dt
        self.sigma_hat = - 1.0 / self.B * 1/self.Phi * \
            torch.exp(self.As*self.dt) * (self.x_hat - x)
        self.u_ad = self.u_ad * self.alpha + \
            (-self.sigma_hat) * (1 - self.alpha)
        u_b = self.kp * (x_desired - x)
        return u_b + self.u_ad


class Logger:
    def __init__(self, drone_num=1, enable=True) -> None:
        self.enable = enable
        self.drone_num = drone_num
        self.log_items = ['xyz_drones', 'vxyz_drones', 'rpy_drones', 'quat_drones', 'xyz_obj', 'xyz_obj_err', 'vxyz_obj',
                          'vxyz_obj_err', 'vrpy_drones', 'vrpy_target', 'vrpy_err', 'rope_force_drones', 'thrust_drones', 'torque', 'rope_disp', 'rope_vel', 'xyz_obj_target', 'vxyz_obj_target', 'reward', 'reward_groundtruth']
        self.log_dict = {item: [] for item in self.log_items}

    def log(self, state):
        # ic(state.keys())
        if not self.enable:
            return
        for item in self.log_items:
            value = state[item][0].cpu().numpy()
            if len(value.shape) > 1:
                # flatten the array
                value = value.flatten()
            self.log_dict[item].append(value)

    def plot(self, filename):
        if not self.enable:
            return
        # set seaborn theme
        sns.set_theme()
        # create figure
        fig, axs = plt.subplots(len(self.log_items), 1,
                                figsize=(10, 4*len(self.log_items)))
        # plot
        x_time = np.arange(len(self.log_dict[self.log_items[0]])) * 4e-4
        for i, item in enumerate(self.log_items):
            self.log_dict[item] = np.array(self.log_dict[item])
            if item == 'xyz_obj' and 'xyz_obj_target' in self.log_dict:
                target = self.log_dict['xyz_obj_target']
            else:
                target = None
            if len(self.log_dict[item][0]) > 1:
                # plot each dimension
                for j in range(len(self.log_dict[item][0])):
                    axs[i].plot(x_time, np.array(self.log_dict[item])[
                                :, j], label=f'{item}_{j}')
                    if target is not None:
                        axs[i].plot(x_time, np.array(target)[:, j],
                                    label=f'{item}_{j}_target', linestyle='--')
            else:
                axs[i].plot(x_time, self.log_dict[item])
            axs[i].set_title(item)
            axs[i].legend()
        # save
        fig.savefig(filename+'.png')
        # save all data to csv
        save_dict = {'time': x_time}
        for item in self.log_items:
            if len(self.log_dict[item].shape) == 1:
                save_dict[item] = self.log_dict[item]
            else:
                for i in range(self.log_dict[item].shape[1]):
                    save_dict[item+'_'+str(i)] = self.log_dict[item][:, i]
        df = pd.DataFrame(save_dict)
        df.to_csv(filename+'.csv', index=False)


class MeshVisulizer:
    def __init__(self, env=QuadTransEnv, enable=True) -> None:
        self.enable = enable
        self.drone_num = env.drone_num
        self.env = env
        self.replay_speed = 1.0
        if not enable:
            return
        self.num_frame = 0
        self.real_render_time = 0.0
        self.render_interval = 1.0/60.0
        self.vis = meshcat.Visualizer()
        self.anim = Animation()
        # set camera position
        self.vis["/Cameras/default"].set_transform(
            tf.translation_matrix([0, 0, 0]).dot(
                tf.euler_matrix(0, np.radians(-45), -np.pi/2)))
        self.vis["/Cameras/default/rotated/<object>"].set_transform(
            tf.translation_matrix([1.5, 0, 0]))
        # set quadrotor model
        for i in range(env.drone_num):
            self.vis[f"drone{i}"].set_object(
                g.StlMeshGeometry.from_file('../assets/crazyflie2.stl'))
        # set object model as a sphere
        self.vis["obj"].set_object(g.Sphere(0.01))
        # set target object model as a red sphere
        self.vis["obj_target"].set_object(
            g.Sphere(0.01), material=g.MeshLambertMaterial(color=0xff0000))
        # create arrow for visualize the trajectory
        for i in range(50):
            self.vis[f"traj_x{i}"].set_object(g.StlMeshGeometry.from_file(
                '../assets/arrow.stl'), material=g.MeshLambertMaterial(color=0xf000ff))
        if env.task == 'avoid':
            # set obstacle model as a cube
            self.vis["obstacle1"].set_object(
                g.Box([self.env.wall_half_width*2.0, 0.5, 0.5]))
            self.vis["obstacle2"].set_object(
                g.Box([self.env.wall_half_width*2.0, 0.5, 0.5]))

    def vis_traj(self, traj_x, traj_v):
        for i in range(50):
            self.vis_vector(
                self.vis[f'traj_x{i}'], traj_x[i], traj_v[i], scale=0.5)

    def vis_vector(self, obj, origin, vec, scale=2.0):
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

    def update(self, state):
        if not self.enable:
            return
        # visualize the trajectory
        if state['step'][0] == 0:
            self.vis_traj(state['xyz_traj'][:, 0, :],
                          state['vxyz_traj'][:, 0, :])
        self.real_render_time += 4e-4
        if (self.num_frame * self.render_interval / self.replay_speed) < self.real_render_time:
            # update drone
            for i in range(self.drone_num):
                xyz_drone = state['xyz_drones'][0, i].cpu().numpy()
                quat_drone = state['quat_drones'][0, i].cpu().numpy()
                quat_drone = np.array([quat_drone[3], *quat_drone[:3]])
                transmat = tf.translation_matrix(
                    xyz_drone).dot(tf.quaternion_matrix(quat_drone))
                self.vis[f"drone{i}"].set_transform(transmat)
            # update object
            xyz_obj = state['xyz_obj'][0].cpu().numpy()
            self.vis["obj"].set_transform(tf.translation_matrix(xyz_obj))
            # update target object
            xyz_obj_target = state['xyz_obj_target'][0].cpu().numpy()
            self.vis["obj_target"].set_transform(
                tf.translation_matrix(xyz_obj_target))
            # update obstacle
            if self.env.task == 'avoid':
                # set obstacle position
                self.vis["obstacle1"].set_transform(
                    tf.translation_matrix([0.0, 0.0, self.env.wall_half_height + 0.5 / 2.0]))
                self.vis["obstacle2"].set_transform(
                    tf.translation_matrix([0.0, 0.0, - self.env.wall_half_height - 0.5 / 2.0]))
            self.num_frame += 1
            time.sleep(self.render_interval)


def test_env(env: QuadTransEnv, policy, adaptor=None, compressor=None, save_path=None):
    # make sure the incorperated logger is enabled
    env.logger.enable = True
    state, info = env.reset()
    total_steps = env.max_steps
    for _ in range(total_steps):
        act = policy(state, info)
        state, rew, done, info = env.step(act)
    if save_path == None:
        package_path = os.path.dirname(adaptive_control_gym.__file__)
        save_path = f"{package_path}/../results/test"
    env.close(save_path)


@dataclass
class Args:
    policy_type: str = "pid"  # 'random', 'pid'
    task: str = "track"  # 'track', 'hover', 'avoid'
    policy_path: str = '/home/pcy/rl/policy-adaptation-survey/results/rl/ppo_CurriTrackExpertUnCertain(LimitForce).pt'
    seed: int = 0
    env_num: int = 1
    drone_num: int = 1
    gpu_id: int = -1
    enable_log: bool = True
    enable_vis: bool = True
    disable_obj: bool = False
    disable_hook: bool = False
    disable_loose_rope: bool = False    
    curri_param: float = 1.0

def main(args: Args):
    # set torch print precision as scientific notation and 2 decimal places
    torch.set_printoptions(precision=2, sci_mode=True)
    # get policy
    if args.policy_type == 'ppo':
        if args.gpu_id < 0:
            map_location = 'cpu'
        else:
            map_location = f'cuda:{args.gpu_id}'
        loaded_agent = torch.load(args.policy_path, map_location=map_location)
        actor = loaded_agent['actor']
        compressor = loaded_agent['compressor']
        # disable gradient
        for param in actor.parameters():
            param.requires_grad = False
        for param in compressor.parameters():
            param.requires_grad = False
        def policy(state, info): 
            return actor(state, compressor(info['e']))
    elif args.policy_type == 'random':
        def policy(state, info): return torch.rand(
            [args.env_num, args.drone_num, 3], device=env.device) * 2 - 1
    elif args.policy_type == 'pid':
        def policy(state, info): return env.policy_pid()
    env = QuadTransEnv(env_num=args.env_num, drone_num=args.drone_num, gpu_id=args.gpu_id, seed=args.seed,
                       enable_log=args.enable_log, enable_vis=args.enable_vis, task=args.task, disable_obj = args.disable_obj, disable_hook = args.disable_hook, disable_loose_rope = args.disable_loose_rope)
    env.curri_param = args.curri_param
    test_env(env, policy, save_path='results/test')

if __name__ == '__main__':
    main(tyro.cli(Args))