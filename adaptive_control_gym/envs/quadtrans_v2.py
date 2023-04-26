import gym
import torch
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import time
import pandas as pd
import os

import adaptive_control_gym
from adaptive_control_gym.utils import geom


class QuadTransEnv(gym.Env):
    def __init__(self, env_num: int = 1024, drone_num: int = 1, gpu_id: int = 0, seed: int = 0, enable_log: bool = False, enable_vis: bool = False, **kwargs) -> None:
        super().__init__()
        self.logger = Logger(enable=enable_log)
        self.visualizer = MeshVisulizer(enable=enable_vis)

        # set simulator parameters
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
        torch.manual_seed(self.seed)
        # set torch default float precision
        torch.set_default_dtype(torch.float32)

        # set RL parameters
        self.state_dim = 3 + 3 + (3 + 3 + 4 + 3) * self.drone_num + 3 + 3
        self.expert_dim = 1
        self.adapt_horizon = 1
        self.adapt_dim = 1
        self.action_dim = 3

        # state variables, physical parameters
        self.reset()

    def step(self, action):
        # TODO make action support multi-agent
        action = action.unsqueeze(1)
        thrust = (action[..., 0] + 1.0) * 0.5 * self.max_thrust
        vrp_target = action[..., 1:] * self.max_vrp
        vrpy_target = torch.cat([vrp_target, torch.zeros(
            [*vrp_target.shape[:-1], 1], device=self.device)], dim=-1)

        for _ in range(self.step_substeps):
            self.ctlstep(vrpy_target, thrust)

        reward = self._get_reward()
        # calculate done
        self.step_cnt += 1
        single_done = (self.step_cnt >= self.max_steps)
        done = torch.ones((self.env_num), device=self.device) * single_done
        if single_done:
            self.reset()
        next_obs = self._get_obs()
        next_info = self._get_info()

        return next_obs, reward, done, next_info

    def get_env_params(self):
        return None

    def _get_reward(self):
        # calculate reward
        err_x = torch.norm(self.xyz_obj - self.xyz_obj_target, dim=1)
        err_v = torch.norm(self.vxyz_obj - self.vxyz_obj_target, dim=1)
        reward = 1.0 - torch.clip(err_x, 0, 2)*0.5 - \
            torch.clip(err_v, 0, 2)*0.5
        reward -= torch.clip(torch.log(err_x+1)*5, 0, 1)*0.1  # for 0.2
        reward -= torch.clip(torch.log(err_x+1)*10, 0, 1)*0.1  # for 0.1
        return reward

    def _get_obs(self):
        # calculate observation
        obs = torch.cat([
            self.xyz_obj,
            self.vxyz_obj,
            self.xyz_drones.reshape(self.env_num, -1),
            self.vxyz_drones.reshape(self.env_num, -1) / 2.0,
            self.quat_drones.reshape(self.env_num, -1),
            self.vrpy_drones.reshape(self.env_num, -1) / 15.0,
            self.xyz_obj_target,
            self.vxyz_obj_target
        ], dim=-1)
        return obs

    def _get_info(self):
        return {
            'e': torch.zeros([self.env_num, 1], device=self.device),
            'adapt_obs': torch.zeros([self.env_num, 1], device=self.device),
            'err_x': torch.norm(self.xyz_obj - self.xyz_obj_target, dim=1),
            'err_v': torch.norm((self.vxyz_obj - self.vxyz_obj_target), dim=1),
        }

    def ctlstep(self, vrpy_target: torch.Tensor, thrust: torch.Tensor):
        # run lower level attitude rate PID controller
        self.vrpy_target = vrpy_target
        self.vrpy_err = vrpy_target - self.vrpy_drones
        # torque = (self.J_drones @ self.attirate_controller.update(self.vrpy_err,
        #   self.ctl_dt).unsqueeze(-1)).squeeze(-1)
        torque = (self.J_drones @ self.attirate_controller.update(self.vrpy_drones,
                  self.vrpy_target, self.u_attirate).unsqueeze(-1)).squeeze(-1)
        thrust, torque = torch.clip(thrust, 0.0, self.max_thrust), torch.clip(
            torque, -self.max_torque, self.max_torque)
        self.u_attirate = (torch.inverse(self.J_drones) @
                           torque.unsqueeze(-1)).squeeze(-1)
        for _ in range(self.ctl_substeps):
            state = self.simstep(torque, thrust)
        return state

    def simstep(self, torque: torch.Tensor, thrust: torch.Tensor):
        # state variables
        rotmat_drone = geom.quat2rotmat(self.quat_drones)
        xyz_hook = self.xyz_drones + \
            (rotmat_drone @ self.hook_disp.unsqueeze(-1)).squeeze(-1)

        # analysis the force of the drone
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
        rope_force_drones[loose_rope] *= 0.0
        # total force
        force_drones = gravity_drones + thrust_drones + rope_force_drones
        # TODO set as parameter
        dist2center = torch.norm(self.xyz_drones, dim=-1, keepdim=True)
        force_drones -= 0.1 * (self.xyz_drones - self.xyz_drones / dist2center) * (dist2center > 3.0).float()  
        # total moment
        rope_torque = torch.cross(self.hook_disp, rope_force_drones, dim=-1)
        moment_drones = torque + rope_torque + \
            (self.J_drones @ torch.cross(self.vrpy_drones,
             self.vrpy_drones, axis=-1).unsqueeze(-1)).squeeze(-1)

        # analysis the force of the object
        # gravity
        gravity_obj = self.g * self.mass_obj
        # rope force
        rope_force_obj = -torch.sum(rope_force_drones, dim=1)
        # total force
        force_obj = gravity_obj + rope_force_obj - self.vxyz_obj * 0.01 # TODO set as parameter

        # update the state variables
        # drone
        self.vxyz_drones = self.vxyz_drones + \
            self.sim_dt * force_drones / self.mass_drones
        self.vxyz_drones = torch.clip(self.vxyz_drones, -10, 10)
        self.xyz_drones = self.xyz_drones + self.sim_dt * self.vxyz_drones
        self.vrpy_drones = self.vrpy_drones + self.sim_dt * \
            (torch.inverse(self.J_drones) @ moment_drones.unsqueeze(-1)).squeeze(-1)
        self.vrpy_drones = torch.clip(self.vrpy_drones, -50, 50)
        # integrate the quaternion
        self.quat_drones = geom.integrate_quat(
            self.quat_drones, self.vrpy_drones, self.sim_dt)

        # manually limit the angle
        cos_err_div2 = self.quat_drones[..., 3]
        hit_angle_limit_mask = (
            cos_err_div2 < np.cos(self.max_angle/2)).float()
        self.quat_drones[..., 3] = hit_angle_limit_mask * np.cos(
            self.max_angle/2) + (1 - hit_angle_limit_mask) * self.quat_drones[..., 3]
        self.quat_drones[..., :3] = hit_angle_limit_mask.unsqueeze(-1) * self.quat_drones[..., :3] / torch.norm(
            self.quat_drones[..., :3], dim=-1, keepdim=True) * np.sin(self.max_angle/2) + (1 - hit_angle_limit_mask.unsqueeze(-1)) * self.quat_drones[..., :3]
        self.vrpy_drones = (1 - hit_angle_limit_mask.unsqueeze(-1)) * self.vrpy_drones

        # object
        axyz_obj = force_obj / self.mass_obj
        self.vxyz_obj = self.vxyz_obj + self.sim_dt * axyz_obj
        self.xyz_obj = self.xyz_obj + self.sim_dt * self.vxyz_obj

        # log and visualize for debug purpose
        if self.logger.enable or self.visualizer.enable:
            state = {
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
                'rope_force_obj': rope_force_obj,
                'gravity_drones': gravity_drones,
                'thrust_drones': thrust_drones,
                'rope_disp': rope_disp,
                'rope_vel': rope_vel,
                'torque': torque,
                'xyz_obj_target': self.xyz_obj_target,
                'vxyz_obj_target': self.vxyz_obj_target,
            }
        else:
            state = None
        self.logger.log(state)
        self.visualizer.update(state)

        return state

    def policy_pos(self, pos_target: torch.Tensor):
        # run lower level position PID controller

        # Object-level controller
        delta_pos = torch.clip(pos_target - self.xyz_obj, -1.0, 1.0)
        force_obj_pid = self.mass_obj * \
            self.objpos_controller.update(
                delta_pos, self.step_dt)
        target_force_obj = force_obj_pid - self.mass_obj * self.g
        xyz_obj2drone = self.xyz_obj - self.xyz_drones
        z_hat_obj = xyz_obj2drone / \
            torch.norm(xyz_obj2drone, dim=-1, keepdim=True)
        # TODO distribute the force to multiple drones
        target_force_obj_projected = torch.sum(
            target_force_obj * z_hat_obj, dim=-1) * z_hat_obj

        # Drone-level controller
        xyz_drone_target = (self.xyz_obj + target_force_obj /
                            torch.norm(target_force_obj, dim=-1, keepdim=True) *
                            self.rope_length) - self.hook_disp
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

        return torch.cat([thrust.unsqueeze(-1), rpy_rate_target], dim=-1)

    def reset(self):
        self.sample_physical_params()
        self.sample_control_params()
        self.sample_state()
        # reset steps
        self.step_cnt = 0
        return self._get_obs(), self._get_info()

    def sample_state(self):
        # sample object initial position
        self.xyz_obj = torch.rand(
            [self.env_num, 3], device=self.device) * 2.0 - 1.0
        # sample goal position
        self.xyz_obj_target = torch.rand(
            [self.env_num, 3], device=self.device) * 2.0 - 1.0
        self.vxyz_obj_target = torch.zeros(
            [self.env_num, 3], device=self.device)
        # sample drone initial position
        thetas = torch.rand([self.env_num, self.drone_num],
                            device=self.device) * 2 * np.pi * 0.0
        phis = torch.rand([self.env_num, self.drone_num],
                          device=self.device) * 0.25 * np.pi * 0.0
        xyz_drones2obj = torch.stack([torch.sin(phis) * torch.cos(thetas),
                                      torch.sin(phis) * torch.sin(thetas),
                                      torch.cos(phis)], dim=-1) * self.rope_length - self.hook_disp
        self.xyz_drones = self.xyz_obj.unsqueeze(1) + xyz_drones2obj
        # reset drone initial attitude
        self.quat_drones = torch.zeros(
            [self.env_num, self.drone_num, 4], device=self.device)
        self.quat_drones[:, :, 3] = 1.0
        # reset drone initial velocity
        self.vxyz_drones = torch.zeros(
            [self.env_num, self.drone_num, 3], device=self.device)
        # reset object initial velocity
        self.vxyz_obj = torch.zeros([self.env_num, 3], device=self.device)
        # reset drone initial angular velocity
        self.vrpy_drones = torch.zeros(
            [self.env_num, self.drone_num, 3], device=self.device)

    def sample_physical_params(self):
        self.g = torch.zeros(3, device=self.device)
        self.g[2] = -9.81
        self.mass_drones = torch.ones(
            (self.env_num, self.drone_num, 3), device=self.device) * 0.027
        self.J_drones = torch.zeros(
            (self.env_num, self.drone_num, 3, 3), device=self.device)
        self.J_drones[:, :, 0, 0] = 1.7e-5
        self.J_drones[:, :, 1, 1] = 1.7e-5
        self.J_drones[:, :, 2, 2] = 2.98e-5
        self.hook_disp = torch.zeros(
            (self.env_num, self.drone_num, 3), device=self.device)
        self.hook_disp[:, :, 2] = -0.03
        self.mass_obj = torch.ones(
            (self.env_num, 3), device=self.device) * 0.01
        self.rope_length = torch.ones(
            (self.env_num, self.drone_num, 1), device=self.device) * 0.2
        self.rope_zeta = torch.ones(
            (self.env_num, self.drone_num, 1), device=self.device) * 0.7
        self.rope_wn = torch.ones(
            (self.env_num, self.drone_num, 1), device=self.device) * 1000.0

    def sample_control_params(self):
        # attitude rate controller
        ones = torch.ones([self.env_num, self.drone_num, 3],
                          device=self.device)
        zeros = torch.zeros(
            [self.env_num, self.drone_num, 3], device=self.device)
        # self.attirate_controller = PIDController(
        #     kp=ones * torch.tensor([4e2, 4e2, 1e2], device=self.device),
        #     ki=ones * torch.tensor([4e4, 4e4, 2e4], device=self.device),
        #     kd=zeros,
        #     ki_max=ones * torch.tensor([1e-1, 1e-1, 1e-1], device=self.device),
        #     integral=zeros, last_error=zeros
        # )
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
        self.max_thrust = 0.90
        self.max_vrp = 12.0
        self.max_torque = torch.tensor([9e-3, 9e-3, 2e-3], device=self.device)
        self.max_angle = np.pi/3.0  # manually set max drone angle to 45 degree

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
    def __init__(self, enable=True) -> None:
        self.enable = enable
        self.log_items = ['xyz_drones', 'vxyz_drones', 'rpy_drones', 'quat_drones', 'xyz_obj', 'xyz_obj_err', 'vxyz_obj',
                          'vxyz_obj_err', 'vrpy_drones', 'vrpy_target', 'vrpy_err', 'rope_force_drones', 'thrust_drones', 'torque', 'rope_disp', 'rope_vel', 'xyz_obj_target', 'vxyz_obj_target']
        self.log_dict = {item: [] for item in self.log_items}

    def log(self, state):
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
            if len(self.log_dict[item][0]) > 1:
                # plot each dimension
                for j in range(len(self.log_dict[item][0])):
                    axs[i].plot(x_time, np.array(self.log_dict[item])[
                                :, j], label=f'{item}_{j}')
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
    def __init__(self, enable=True) -> None:
        self.enable = enable
        if not enable:
            return
        self.vis = meshcat.Visualizer()
        # set camera position
        self.vis["/Cameras/default"].set_transform(
            tf.translation_matrix([0, 0, 0]).dot(
                tf.euler_matrix(0, np.radians(-45), -np.pi/2)))
        self.vis["/Cameras/default/rotated/<object>"].set_transform(
            tf.translation_matrix([1.5, 0, 0]))
        # set quadrotor model
        self.vis["drone"].set_object(
            g.StlMeshGeometry.from_file('../assets/crazyflie2.stl'))
        # set object model as a sphere
        self.vis["obj"].set_object(g.Sphere(0.01))

    def update(self, state):
        if not self.enable:
            return
        # update drone
        xyz_drone = state['xyz_drones'][0, 0].cpu().numpy()
        quat_drone = state['quat_drones'][0, 0].cpu().numpy()
        self.vis["drone"].set_transform(tf.translation_matrix(
            xyz_drone).dot(tf.quaternion_matrix(quat_drone)))
        # update object
        xyz_obj = state['xyz_obj'][0].cpu().numpy()
        self.vis["obj"].set_transform(tf.translation_matrix(xyz_obj))
        time.sleep(4e-4)


def test_env(env: QuadTransEnv, policy, adaptor=None, compressor=None, save_path=None):
    # make sure the incorperated logger is enabled
    env.logger.enable = True
    state, info = env.reset()
    total_steps = env.max_steps
    for _ in range(total_steps):
        act = policy(state, None)
        state, rew, done, info = env.step(act)
    if save_path == None:
        package_path = os.path.dirname(adaptive_control_gym.__file__)
        save_path = f"{package_path}/../results/test"
    env.close(save_path)


def main():
    # set torch print precision as scientific notation and 2 decimal places
    torch.set_printoptions(precision=2, sci_mode=True)

    # setup environment
    env_num = 1
    env = QuadTransEnv(env_num=env_num, drone_num=1,
                       gpu_id=-1, enable_log=True, enable_vis=True)
    env.reset()

    target_pos = torch.tensor([[0.5, 0.5, 0.5]], device=env.device)

    all_obs = torch.zeros(
        [env.max_steps, env_num, env.state_dim], device=env.device)
    for i in range(50):
        # policy1: PID
        # action = env.policy_pos(target_pos)

        # policy2: manual
        # total_gravity = env.g * (env.mass_drones + env.mass_obj.unsqueeze(1))
        # vrp_target = torch.zeros([env.env_num, 1, 2], device=env.device)
        # if i%2 < 1:
        #     vrp_target[..., 0] = 10
        # else:
        #     vrp_target[..., 0] = -10
        # action = torch.cat([-total_gravity[..., [2]]/0.6, vrp_target/20.0], dim=-1)

        # policy3: random
        action = torch.rand([env_num, 1, env.action_dim],
                            device=env.device) * 2 - 1

        obs, rew, done, info = env.step(action.squeeze(1))
        all_obs[i] = obs
    ic(all_obs.mean(dim=[0, 1]))
    env.close(savepath='results/test')


if __name__ == '__main__':
    # main()
    loaded_agent = torch.load('/home/pcy/rl/policy-adaptation-survey/results/rl/ppo_transportation.pt', map_location='cpu')
    policy = loaded_agent['actor']
    test_env(QuadTransEnv(env_num=1, drone_num=1, gpu_id=-1, enable_log=True, enable_vis=True), policy, save_path='results/test')