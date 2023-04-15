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

from adaptive_control_gym.utils import geom


class QuadTransEnv(gym.Env):
    def __init__(self, env_num: int = 1024, drone_num: int = 1, gpu_id: int = 0, seed: int = 0, enable_log:bool=False, enable_vis:bool=False, **kwargs) -> None:
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
        self.max_steps = 80
        self.step_dt = self.ctl_dt * self.step_substeps
        self.gpu_id = gpu_id
        self.device = torch.device(
            f"cuda:{self.gpu_id}" if (
                torch.cuda.is_available() & (gpu_id >= 0)) else "cpu"
        )
        torch.manual_seed(self.seed)
        # set torch default float precision
        torch.set_default_dtype(torch.float32)

        # physical parameters
        self.g = torch.zeros((self.env_num, 3), device=self.device)
        self.g[:, 2] = -9.81
        self.mass_drones = torch.ones(
            (self.env_num, self.drone_num, 3), device=self.device) * 0.027
        self.J_drones = torch.zeros(
            (self.env_num, self.drone_num, 3, 3), device=self.device)
        self.J_drones[:, :, 0, 0] = 1.7e-5
        self.J_drones[:, :, 1, 1] = 1.7e-5
        self.J_drones[:, :, 2, 2] = 3.0e-5
        self.hook_disp = torch.zeros(
            (self.env_num, self.drone_num, 3), device=self.device)
        self.hook_disp[:, :, 2] = -0.05
        self.mass_obj = torch.ones(
            (self.env_num, 3), device=self.device) * 0.01
        self.rope_length = torch.ones(
            (self.env_num, self.drone_num), device=self.device) * 0.2
        self.rope_zeta = torch.ones(
            (self.env_num, self.drone_num), device=self.device) * 0.7
        self.rope_wn = torch.ones(
            (self.env_num, self.drone_num), device=self.device) * 1000.0

        # state variables
        self.xyz_drones = torch.zeros(
            (self.env_num, self.drone_num, 3), device=self.device)
        self.vxyz_drones = torch.zeros(
            (self.env_num, self.drone_num, 3), device=self.device)
        self.quat_drones = torch.zeros(
            (self.env_num, self.drone_num, 4), device=self.device)  # [x, y, z, w]
        self.quat_drones[:, :, 3] = 1.0
        self.omega_drones = torch.zeros(
            (self.env_num, self.drone_num, 3), device=self.device)
        self.xyz_targets = torch.zeros((self.env_num, 3), device=self.device)
        self.xyz_obj = torch.zeros((self.env_num, 3), device=self.device)
        self.xyz_obj[..., 2] = -0.25
        self.vxyz_obj = torch.zeros((self.env_num, 3), device=self.device)
        self.steps = 0

        # controller variables
        self.KP = torch.ones((self.env_num, self.drone_num, 3), device=self.device) * \
            torch.tensor([4e2, 4e2, 1e2], device=self.device)
        self.KI = torch.ones((self.env_num, self.drone_num, 3), device=self.device) * \
            torch.tensor([1e3, 1e3, 3e2], device=self.device)
        self.KD = torch.ones((self.env_num, self.drone_num, 3), device=self.device) * \
            torch.tensor([0.0, 0.0, 0.0], device=self.device)
        self.KI_MAX = torch.ones((self.env_num, self.drone_num, 3),
                                 device=self.device) * torch.tensor([1e4, 1e4, 1e4], device=self.device)
        self.atti_rate_controller = PIDController(
            kp=self.KP, ki=self.KI, kd=self.KD, ki_max=self.KI_MAX,
            integral=torch.zeros(
                (self.env_num, self.drone_num, 3), device=self.device),
            last_error=torch.zeros(
                (self.env_num, self.drone_num, 3), device=self.device)
        )
        ones = torch.ones([self.env_num, 3])
        zeros = torch.zeros([self.env_num, 3])
        self.objpos_controller = PIDController(
            kp=ones*8.0,
            ki=ones*0.0,
            kd=ones*3.0,
            ki_max=ones*100.0,
            integral=zeros,
            last_error=zeros
        )
        ones = torch.ones([self.env_num, self.drone_num, 3])
        zeros = torch.zeros([self.env_num, self.drone_num, 3])
        self.pos_controller = PIDController(
            kp=ones*16.0,
            ki=ones*0.0,
            kd=ones*6.0,
            ki_max=ones*100.0,
            integral=zeros,
            last_error=zeros
        )
        self.attitude_controller = PIDController(
            kp=ones*20.0,
            ki=ones*0.0,
            kd=ones*0.0,
            ki_max=ones*100.0,
            integral=zeros,
            last_error=zeros
        )
        self.max_thrust = 0.60
        self.max_torque = torch.tensor([9e-3, 9e-3, 2e-3], device=self.device)

    def step(self, action):
        thrust = action[..., 0]
        omega_target = action[..., 1:]

        for _ in range(self.step_substeps):
            state = self.ctlstep(omega_target, thrust)
        self.steps += 1

        # compute reward
        reward = -torch.norm(self.xyz_obj -
                             self.xyz_targets, dim=-1).mean(dim=-1)
        done = (self.steps >= self.max_steps)
        info = {}

        return state, reward, done, info

    def ctlstep(self, omega_target: torch.Tensor, thrust: torch.Tensor):
        # run lower level attitude rate PID controller
        self.omega_target = omega_target
        omega_error = omega_target - self.omega_drones
        torque = (self.J_drones @ self.atti_rate_controller.update(omega_error,
                  self.ctl_dt).unsqueeze(-1)).squeeze(-1)
        thrust, torque = torch.clip(thrust, 0.0, self.max_thrust), torch.clip(
            torque, -self.max_torque, self.max_torque)
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
        rope_disp = xyz_obj2hook - self.rope_length * xyz_obj2hook_normed
        vxyz_hook = self.vxyz_drones + \
            torch.cross(self.omega_drones, self.hook_disp, dim=-1)
        vxyz_obj2hook = self.vxyz_obj - vxyz_hook
        rope_vel = torch.sum(vxyz_obj2hook * xyz_obj2hook_normed,
                             dim=-1, keepdim=True) * xyz_obj2hook_normed
        mass_joint = self.mass_drones * \
            self.mass_obj.unsqueeze(
                1) / (self.mass_drones + self.mass_obj.unsqueeze(1))
        rope_force_drones = mass_joint * \
            ((self.rope_wn ** 2) * rope_disp + 2 *
             self.rope_zeta * self.rope_wn * rope_vel)
        # total force
        force_drones = gravity_drones + thrust_drones + rope_force_drones
        # total moment
        moment_drones = torque + \
            torch.cross(self.hook_disp, rope_force_drones, dim=-1)

        # analysis the force of the object
        # gravity
        gravity_obj = self.g * self.mass_obj
        # rope force
        rope_force_obj = -torch.sum(rope_force_drones, dim=1)
        # total force
        force_obj = gravity_obj + rope_force_obj

        # update the state variables
        # drone
        self.vxyz_drones = self.vxyz_drones + \
            self.sim_dt * force_drones / self.mass_drones
        self.xyz_drones = self.xyz_drones + self.sim_dt * self.vxyz_drones
        self.omega_drones = self.omega_drones + self.sim_dt * \
            (torch.inverse(self.J_drones) @ moment_drones.unsqueeze(-1)).squeeze(-1)
        # integrate the quaternion
        self.quat_drones = geom.integrate_quat(
            self.quat_drones, self.omega_drones, self.sim_dt)

        # object
        self.vxyz_obj = self.vxyz_obj + self.sim_dt * force_obj / self.mass_obj
        self.xyz_obj = self.xyz_obj + self.sim_dt * self.vxyz_obj

        state = {
            'xyz_drone': self.xyz_drones,
            'vxyz_drone': self.vxyz_drones,
            'quat_drone': self.quat_drones,
            'rpy_drone': geom.quat2rpy(self.quat_drones),
            'omega_drone': self.omega_drones,
            'xyz_obj': self.xyz_obj,
            'vxyz_obj': self.vxyz_obj,
            'force_drones': force_drones,
            'moment_drones': moment_drones,
            'force_obj': force_obj,
            'rope_force_drones': rope_force_drones,
            'gravity_drones': gravity_drones,
            'thrust_drones': thrust_drones,
            'rope_disp': rope_disp,
            'rope_vel': rope_vel,
            'torque': torque,
        }

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
        # DEBUG 
        # xyz_drone_target = pos_target.unsqueeze(1)
        # DEBUG
        # target_force_obj_projected = (-self.mass_obj * self.g).unsqueeze(1)
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

        # DEBUG
        # rpy_target = torch.tensor([[[1.0, 0.0, 0.0]]], device=self.device)
        # quat_target = geom.rpy2quat(rpy_target)
        # quat_error = geom.quat_mul(
        #     quat_target, geom.quat_inv(self.quat_drones))
        # rot_err = quat_error[..., :3]

        rot_err = torch.cross(
            desired_rotvec, thrust_desired/torch.norm(thrust_desired, dim=-1, keepdim=True), dim=-1)
        rpy_rate_target = self.attitude_controller.update(
            rot_err, self.step_dt)

        return torch.cat([thrust.unsqueeze(-1), rpy_rate_target], dim=-1)

    def reset(self):
        pass

    def render(self, mode='human'):
        pass

    def close(self):
        self.logger.plot('results/test.png')


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


class Logger:
    def __init__(self, enable=True) -> None:
        self.enable = enable
        self.log_items = ['xyz_drone', 'vxyz_drone', 'rpy_drone', 'quat_drone', 'omega_drone', 'xyz_obj', 'vxyz_obj', 'force_drones',
                          'moment_drones', 'force_obj', 'rope_force_drones', 'gravity_drones', 'thrust_drones', 'rope_disp', 'rope_vel', 'torque']
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
        fig.savefig(filename)


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
        xyz_drone = state['xyz_drone'][0, 0].cpu().numpy()
        quat_drone = state['quat_drone'][0, 0].cpu().numpy()
        self.vis["drone"].set_transform(tf.translation_matrix(
            xyz_drone).dot(tf.quaternion_matrix(quat_drone)))
        # update object
        xyz_obj = state['xyz_obj'][0].cpu().numpy()
        self.vis["obj"].set_transform(tf.translation_matrix(xyz_obj))
        time.sleep(4e-4)


def main():
    # set torch print precision
    torch.set_printoptions(precision=2)

    # setup environment
    env = QuadTransEnv(env_num=1, drone_num=1, gpu_id=-1, enable_log=True, enable_vis=False)
    env.reset()

    target_pos = torch.tensor([[0.5, 0.5, 0.5]], device=env.device)
    # time.sleep(4)
    for i in range(80):
        action = env.policy_pos(target_pos)
        state, rew, done, info = env.step(action)
    env.close()

if __name__ == '__main__':
    main()
