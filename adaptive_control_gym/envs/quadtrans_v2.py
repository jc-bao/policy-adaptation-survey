import gym
import torch
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic

from adaptive_control_gym.utils.geom import quat2rotmat, integrate_quat

class QuadTransEnv(gym.Env):
    def __init__(self, env_num:int=1024, drone_num:int=1, gpu_id:int=0, seed:int=0, **kwargs) -> None:
        super().__init__()

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
            f"cuda:{self.gpu_id}" if (torch.cuda.is_available() & (gpu_id>=0)) else "cpu"
        )
        torch.manual_seed(self.seed)
        # set torch default float precision
        torch.set_default_dtype(torch.float32)

        # physical parameters
        self.g = torch.zeros((self.env_num, 3), device=self.device)
        self.g[:, 2] = -9.81
        self.mass_drones = torch.ones((self.env_num, self.drone_num, 3), device=self.device) * 0.027
        self.J_drones = torch.zeros((self.env_num, self.drone_num, 3, 3), device=self.device)
        self.J_drones[:, :, 0, 0] = 1.7e-5
        self.J_drones[:, :, 1, 1] = 1.7e-5
        self.J_drones[:, :, 2, 2] = 3.0e-5
        self.hangging_disp = torch.zeros((self.env_num, self.drone_num, 3), device=self.device)
        self.hangging_disp[:, :, 2] = -0.05
        self.mass_obj = torch.ones((self.env_num, 3), device=self.device) * 0.01
        self.rope_length = torch.ones((self.env_num, self.drone_num), device=self.device) * 0.2
        self.rope_zeta = torch.ones((self.env_num, self.drone_num), device=self.device) * 0.7
        self.rope_wn = torch.ones((self.env_num, self.drone_num), device=self.device) * 100.0

        # state variables
        self.xyz_drones = torch.zeros((self.env_num, self.drone_num, 3), device=self.device)
        self.vxyz_drones = torch.zeros((self.env_num, self.drone_num, 3), device=self.device)
        self.quat_drones = torch.zeros((self.env_num, self.drone_num, 4), device=self.device) # [x, y, z, w]
        self.quat_drones[:, :, 3] = 1.0
        self.omega_drones = torch.zeros((self.env_num, self.drone_num, 3), device=self.device)
        self.xyz_targets = torch.zeros((self.env_num, 3), device=self.device)
        self.xyz_obj = torch.zeros((self.env_num, 3), device=self.device)
        self.xyz_obj[..., 2] = -0.25
        self.vxyz_obj = torch.zeros((self.env_num, 3), device=self.device)


    def step(self, action):
        return self.observation_space.sample(), 0, False, {}
    
    def ctlstep(self, vrpy_target:torch.Tensor, thrust:torch.Tensor):
        raise NotImplementedError
    
    def simstep(self, torque:torch.Tensor, thrust:torch.Tensor):
        # state variables
        rotmat_drone = quat2rotmat(self.quat_drones)

        # analysis the force of the drone
        # gravity
        gravity_drones = self.g * self.mass_drones
        # thrust
        thrust_drones = torch.zeros((self.env_num, self.drone_num, 3), device=self.device)
        thrust_drones[:, :, 2] = thrust
        thrust_drones = (rotmat_drone @ thrust_drones.unsqueeze(-1)).squeeze(-1)
        # rope force
        xyz_obj2drone = self.xyz_drones - self.xyz_obj.unsqueeze(1)
        xyz_obj2drone_normed = xyz_obj2drone / torch.norm(xyz_obj2drone, dim=-1, keepdim=True)
        rope_disp = xyz_obj2drone - self.rope_length * xyz_obj2drone_normed
        vxyz_obj2drone = self.vxyz_obj - self.vxyz_drones
        vxyz_obj2drone_projected = torch.sum(vxyz_obj2drone * xyz_obj2drone_normed, dim=-1, keepdim=True) * xyz_obj2drone_normed
        rope_force_drones = self.mass_obj * (self.rope_wn ** 2) * rope_disp + 2 * self.rope_zeta * self.rope_wn * vxyz_obj2drone_projected
        # total force
        force_drones = gravity_drones + thrust_drones + rope_force_drones
        # total moment
        moment_drones = torque + torch.cross(self.hangging_disp, rope_force_drones, dim=-1)

        # analysis the force of the object
        # gravity
        gravity_obj = self.g * self.mass_obj
        # rope force
        rope_force_obj = -torch.sum(rope_force_drones, dim=1)
        # total force
        force_obj = gravity_obj + rope_force_obj

        # update the state variables
        # drone
        self.vxyz_drones = self.vxyz_drones + self.sim_dt * force_drones / self.mass_drones
        self.xyz_drones = self.xyz_drones + self.sim_dt * self.vxyz_drones
        self.omega_drones = self.omega_drones + self.sim_dt * (torch.inverse(self.J_drones) @ moment_drones.unsqueeze(-1)).squeeze(-1)
        # integrate the quaternion
        self.quat_drones = integrate_quat(self.quat_drones, self.omega_drones, self.sim_dt)

        # object
        self.vxyz_obj = self.vxyz_obj + self.sim_dt * force_obj / self.mass_obj
        self.xyz_obj = self.xyz_obj + self.sim_dt * self.vxyz_obj

        return {
            'xyz_drone': self.xyz_drones,
            'vxyz_drone': self.vxyz_drones,
            'quat_drone': self.quat_drones,
            'omega_drone': self.omega_drones,
            'xyz_obj': self.xyz_obj,
            'vxyz_obj': self.vxyz_obj,
        }


    def reset(self):
        pass

    def render(self, mode='human'):
        pass

    def close(self):
        pass

class Logger:
    def __init__(self) -> None:
        self.log_items = ['xyz_drone', 'vxyz_drone', 'quat_drone', 'omega_drone', 'xyz_obj', 'vxyz_obj']
        self.log_dict = {item: [] for item in self.log_items}

    def log(self, state):
        for item in self.log_items:
            value = state[item][0].cpu().numpy()
            if len(value.shape) > 1:
                # flatten the array
                value = value.flatten()
            self.log_dict[item].append(value)

    def plot(self, filename):
        # set seaborn theme
        sns.set_theme()
        # create figure
        fig, axs = plt.subplots(len(self.log_items), 1,
                                figsize=(10, 3*len(self.log_items)))
        # plot
        x_time = np.arange(len(self.log_dict[self.log_items[0]])) * 4e-4
        for i, item in enumerate(self.log_items):
            axs[i].plot(x_time, self.log_dict[item])
            axs[i].set_title(item)
        # save
        fig.savefig(filename)

def main():
    env = QuadTransEnv(env_num=1, drone_num=1, gpu_id=-1)
    env.reset()
    logger = Logger()
    for i in range(100):
        state = env.simstep(torch.zeros((1, 3)), torch.zeros((1, 1)))
        logger.log(state)
    logger.plot('results/test.png')
    env.close()

if __name__ == '__main__':
    main()