import gym
import torch
import numpy as np
from adaptive_control_gym.utils import rpy2rotmat
from icecream import ic

class QuadTrans(gym.Env):
    '''
    This is a class for quadrotor transportation task.
    The control input is the thrust and attitude, i.e. 4 control inputs.
    The state is the position, velocity, attitude, and angular velocity.
    The reward is the object distance to the target position.
    TODO 
    * modelling related: delay of controller, pid control error (especially for attitude control), environment damping ratio
    '''
    def __init__(self, env_num=1024, gpu_id=0, seed=0) -> None:
        super().__init__()

        # environment related parameters
        self.seed = seed
        torch.manual_seed(self.seed)
        self.env_num = env_num
        self.dt = 0.01 # s
        self.g = 9.81 # m/s^2
        self.max_step = 120
        self.gpu_id = gpu_id
        self.device = torch.device(
            f"cuda:{self.gpu_id}" if (torch.cuda.is_available() & (gpu_id>=0)) else "cpu"
        )

        # object related parameters
        self.mass_obj_min, self.mass_obj_max = 0.014, 0.042 # true mass is 0.028
        self.mass_obj_min, self.mass_obj_max = torch.tensor(self.mass_obj_min).to(self.device), torch.tensor(self.mass_obj_max).to(self.device)
        self.mass_obj_mean, self.mass_obj_std = (self.mass_obj_min + self.mass_obj_max)/2, (self.mass_obj_max - self.mass_obj_min)/2
        self.mass_distri = torch.distributions.uniform.Uniform(self.mass_obj_min, self.mass_obj_max)

        self.length_rope_min, self.length_rope_max = 0.1, 0.3
        self.length_rope_min, self.length_rope_max = torch.tensor(self.length_rope_min).to(self.device), torch.tensor(self.length_rope_max).to(self.device)
        self.length_rope_mean, self.length_rope_std = (self.length_rope_min + self.length_rope_max)/2, (self.length_rope_max - self.length_rope_min)/2
        self.length_rope_distri = torch.distributions.uniform.Uniform(self.length_rope_min, self.length_rope_max)

        self.tp_obj_min, self.tp_obj_max = np.array([-np.pi, -np.pi/2]), np.array([np.pi, np.pi/2])
        self.tp_obj_min, self.tp_obj_max = torch.tensor(self.tp_obj_min).to(self.device), torch.tensor(self.tp_obj_max).to(self.device)
        self.tp_obj_mean, self.tp_obj_std = (self.tp_obj_min + self.tp_obj_max)/2, (self.tp_obj_max - self.tp_obj_min)/2
        self.tp_obj_distri = torch.distributions.uniform.Uniform(self.tp_obj_min, self.tp_obj_max)
        
        self.vtp_obj_min, self.vtp_obj_max = np.array([-0.5, -0.5]), np.array([0.5, 0.5])
        self.vtp_obj_min, self.vtp_obj_max = torch.tensor(self.vtp_obj_min).to(self.device), torch.tensor(self.vtp_obj_max).to(self.device)
        self.vtp_obj_mean, self.vtp_obj_std = (self.vtp_obj_min + self.vtp_obj_max)/2, (self.vtp_obj_max - self.vtp_obj_min)/2
        self.vtp_obj_distri = torch.distributions.uniform.Uniform(self.vtp_obj_min, self.vtp_obj_max)

        # drone related parameters
        self.mass_drone_min, self.mass_drone_max = 0.02, 0.04
        self.mass_drone_min, self.mass_drone_max = torch.tensor(self.mass_drone_min).to(self.device), torch.tensor(self.mass_drone_max).to(self.device)
        self.mass_drone_mean, self.mass_drone_std = (self.mass_drone_min + self.mass_drone_max)/2, (self.mass_drone_max - self.mass_drone_min)/2
        self.mass_drone_distri = torch.distributions.uniform.Uniform(self.mass_drone_min, self.mass_drone_max)

        self.xyz_drone_min, self.xyz_drone_max = np.array([-1.0, -1.0, 0.05]), np.array([1.0, 1.0, 1.50])
        self.xyz_drone_min, self.xyz_drone_max = torch.tensor(self.xyz_drone_min).to(self.device), torch.tensor(self.xyz_drone_max).to(self.device)
        self.xyz_drone_mean, self.xyz_drone_std = (self.xyz_drone_min + self.xyz_drone_max)/2, (self.xyz_drone_max - self.xyz_drone_min)/2
        self.xyz_drone_distri = torch.distributions.uniform.Uniform(self.xyz_drone_min, self.xyz_drone_max)

        self.vxyz_drone_min, self.vxyz_drone_max = np.array([-0.5, -0.5, -0.1]), np.array([0.5, 0.5, 0.1])
        self.vxyz_drone_min, self.vxyz_drone_max = torch.tensor(self.vxyz_drone_min).to(self.device), torch.tensor(self.vxyz_drone_max).to(self.device)
        self.vxyz_drone_mean, self.vxyz_drone_std = (self.vxyz_drone_min + self.vxyz_drone_max)/2, (self.vxyz_drone_max - self.vxyz_drone_min)/2
        self.vxyz_drone_distri = torch.distributions.uniform.Uniform(self.vxyz_drone_min, self.vxyz_drone_max)

        self.rp_drone_min, self.rp_drone_max = np.array([-np.pi, -np.pi])/10, np.array([np.pi, np.pi])/10
        self.rp_drone_min, self.rp_drone_max = torch.tensor(self.rp_drone_min).to(self.device), torch.tensor(self.rp_drone_max).to(self.device)
        self.rp_drone_mean, self.rp_drone_std = (self.rp_drone_min + self.rp_drone_max)/2, (self.rp_drone_max - self.rp_drone_min)/2
        self.rp_drone_distri = torch.distributions.uniform.Uniform(self.rp_drone_min, self.rp_drone_max)

        # control related parameters
        self.action_dim = 3
        self.action_min, self.action_max = -1.0, 1.0
        self.action_mean, self.action_std = (self.action_min + self.action_max)/2, (self.action_max - self.action_min)/2
        self.thrust_min, self.thrust_max = 0.0, 1.0 # N
        self.thrust_mean, self.thrust_std = (self.thrust_min + self.thrust_max)/2, (self.thrust_max - self.thrust_min)/2
        self.ctl_row_min, self.ctl_row_max = -1.0, 1.0 # rad
        self.ctl_row_mean, self.ctl_row_std = (self.ctl_row_min + self.ctl_row_max)/2, (self.ctl_row_max - self.ctl_row_min)/2
        self.ctl_pitch_min, self.ctl_pitch_max = -1.0, 1.0 # rad
        self.ctl_pitch_mean, self.ctl_pitch_std = (self.ctl_pitch_min + self.ctl_pitch_max)/2, (self.ctl_pitch_max - self.ctl_pitch_min)/2
        # TBD
        self.damping_rate_drone_min, self.damping_rate_drone_max = 0.0, 0.05
        self.damping_rate_drone_min, self.damping_rate_drone_max = torch.tensor(self.damping_rate_drone_min).to(self.device), torch.tensor(self.damping_rate_drone_max).to(self.device)
        self.damping_rate_drone_mean, self.damping_rate_drone_std = (self.damping_rate_drone_min + self.damping_rate_drone_max)/2, (self.damping_rate_drone_max - self.damping_rate_drone_min)/2
        self.damping_rate_drone_distri = torch.distributions.uniform.Uniform(self.damping_rate_drone_min, self.damping_rate_drone_max)
        # TBD
        self.damping_rate_obj_min, self.damping_rate_obj_max = 0.0, 0.05
        self.damping_rate_obj_min, self.damping_rate_obj_max = torch.tensor(self.damping_rate_obj_min).to(self.device), torch.tensor(self.damping_rate_obj_max).to(self.device)
        self.damping_rate_obj_mean, self.damping_rate_obj_std = (self.damping_rate_obj_min + self.damping_rate_obj_max)/2, (self.damping_rate_obj_max - self.damping_rate_obj_min)/2
        self.damping_rate_obj_distri = torch.distributions.uniform.Uniform(self.damping_rate_obj_min, self.damping_rate_obj_max)

        self.reset()

    def reset(self):
        # sample parameters uniformly using pytorch
        self.mass_obj = self.mass_distri.sample((self.env_num,))
        self.length_rope = self.length_rope_distri.sample((self.env_num,))
        self.mass_drone = self.mass_drone_distri.sample((self.env_num,))
        self.xyz_drone = self.xyz_drone_distri.sample((self.env_num,))
        self.xyz_target = self.xyz_drone_distri.sample((self.env_num,)) / 2.0
        self.vxyz_drone = self.vxyz_drone_distri.sample((self.env_num,))
        self.rpy_drone = torch.zeros((self.env_num, 3)).to(self.device)
        self.rpy_drone[:, :2] = self.rp_drone_distri.sample((self.env_num,))        
        self.tp_obj = self.tp_obj_distri.sample((self.env_num,))
        self.vtp_obj = self.vtp_obj_distri.sample((self.env_num,))
        self.damping_rate_drone = self.damping_rate_drone_distri.sample((self.env_num,))
        self.damping_rate_obj = self.damping_rate_obj_distri.sample((self.env_num,))

        # other parameters
        self.step_cnt = 0

    def step(self, action):
        '''
        action: (env_num, action_dim)
            action=[thrust, ctl_row, ctl_pitch]
        '''
        # convert action to thrust, ctl_row, ctl_pitch TODO assume perfect control
        thrust = (action[:, 0]-self.action_mean)/self.action_std * self.thrust_std + self.thrust_mean
        ctl_row = (action[:, 1]-self.action_mean)/self.action_std * self.ctl_row_std + self.ctl_row_mean
        ctl_pitch = (action[:, 2]-self.action_mean)/self.action_std * self.ctl_pitch_std + self.ctl_pitch_mean

        # analysis two point mass system dynamics which connected with a rope
        # thrust force
        force_thrust_local = torch.zeros((self.env_num, 3), device=self.device)
        force_thrust_local[:, 2] = thrust
        rotmat = rpy2rotmat(self.rpy_drone)
        force_thrust = torch.bmm(rotmat, force_thrust_local.unsqueeze(-1)).squeeze(-1)
        # gravity force
        force_gravity_drone = torch.zeros((self.env_num, 3), device=self.device)
        force_gravity_drone[:, 2] = -self.mass_drone * self.g
        force_gravity_obj = torch.zeros((self.env_num, 3), device=self.device)
        force_gravity_obj[:, 2] = -self.mass_obj * self.g
        # drag
        vxyz_obj_rel = self.get_obj_rel_vel(self.tp_obj, self.vtp_obj, self.length_rope)
        vxyz_obj = self.vxyz_drone + vxyz_obj_rel
        force_drag_drone = -self.damping_rate_drone * self.vxyz_drone
        force_drag_obj = -self.damping_rate_obj * vxyz_obj
        # analysis the center of mass of the two point mass system
        acc_com = (force_thrust + force_gravity_drone + force_gravity_obj + force_drag_drone + force_drag_obj) / (self.mass_drone + self.mass_obj)
        # calculate the acceleration of the object with respect to center of mass
        # object distance to center of mass
        dist2com = self.length_rope * self.mass_drone / (self.mass_drone + self.mass_obj)
        vel2com = vxyz_obj_rel * self.mass_drone / (self.mass_drone + self.mass_obj) * self.mass_drone / (self.mass_drone + self.mass_obj)
        acc_rad_obj = vel2com.square().sum(dim=-1) / dist2com
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
        force_rope = - torch.tensordot(z_hat_obj, obj_external_force) + self.mass_obj * acc_rad_obj
        force_obj_local_x = torch.tensordot(x_hat_obj, obj_external_force)
        force_obj_local_y = torch.tensordot(y_hat_obj, obj_external_force)
        # calculate object angular acceleration
        thetadotdot = force_obj_local_y / self.mass_obj
        phidotdot = -force_obj_local_x / self.mass_obj
        # calculate drone acceleration
        acc_drone = (force_drag_drone + force_gravity_drone + force_rope*z_hat_obj) / self.mass_drone
        # update state
        self.xyz_drone = self.xyz_drone + self.vxyz_drone * self.dt
        self.vxyz_drone = self.vxyz_drone + acc_drone * self.dt
        self.rpy_drone[:, 0] = ctl_row
        self.rpy_drone[:, 1] = ctl_pitch
        self.tp_obj = self.tp_obj + self.vtp_obj * self.dt
        self.vtp_obj[:, 0] = self.vtp_obj[:, 0] + thetadotdot * self.dt
        self.vtp_obj[:, 1] = self.vtp_obj[:, 1] + phidotdot * self.dt

        # calculate reward
        xyz_obj = self.xyz_drone + self.get_obj_disp(self.tp_obj, self.length_rope)
        disp_obj = xyz_obj - self.xyz_target
        disp_drone = self.xyz_drone - self.xyz_target
        disp_drone[..., -1] -= self.length_rope
        err_x = torch.norm(disp_obj,dim=1) * 0.7 + torch.norm(disp_drone,dim=1) * 0.3
        err_v = torch.norm(self.vtp_obj,dim=1) * 0.7 + torch.norm(self.vxyz_drone,dim=1) * 0.3
        reward = 1.0 - torch.clip(err_x, 0, 2)*0.5 - torch.clip(err_v, 0, 1)*0.1
        reward -= torch.clip(torch.log(err_x+1)*5, 0, 1)*0.1 # for 0.2
        reward -= torch.clip(torch.log(err_x+1)*10, 0, 1)*0.1 # for 0.1
        reward -= torch.clip(torch.log(err_x+1)*20, 0, 1)*0.1 # for 0.05
        reward -= torch.clip(torch.log(err_x+1)*50, 0, 1)*0.1 # for 0.02

        # calculate done
        self.step_cnt += 1
        single_done = (self.step_cnt >= self.max_step)
        done = torch.ones((self.env_num), device=self.device) * single_done
        if single_done:
            self.reset()
        next_obs = self._get_obs()
        next_info = self._get_info()

        return next_obs, reward, done, next_info

    def _get_obs(self):
        xyz_drone_normed = (self.xyz_drone - self.xyz_drone_mean) / self.xyz_drone_std
        xyz_target_normed = (self.xyz_target - self.xyz_drone_mean) / self.xyz_drone_std
        vxyz_drone_normed = (self.vxyz_drone - self.vxyz_drone_mean) / self.vxyz_drone_std
        rp_drone_normed = (self.rpy_drone[..., :2] - self.rp_drone_mean) / self.rp_drone_std
        tp_obj_normed = (self.tp_obj - self.tp_obj_mean) / self.tp_obj_std
        vtp_obj_normed = (self.vtp_obj - self.vtp_obj_mean) / self.vtp_obj_std
        return torch.cat([xyz_drone_normed, xyz_target_normed, vxyz_drone_normed, rp_drone_normed, tp_obj_normed, vtp_obj_normed], dim=1)
    
    def _get_info(self):
        return None
        

    def get_obj_disp(self, obj_tp, rope_length):
        '''
        drone_pos: (env_num, 3)
        obj_tp: (env_num, 3)
        rope_length: (env_num, 1)
        '''
        disp = torch.zeros((self.env_num, 3), device=self.device)
        disp[:, 0] = rope_length * torch.cos(obj_tp[:, 0]) * torch.sin(obj_tp[:, 1])
        disp[:, 1] = rope_length * torch.sin(obj_tp[:, 0]) * torch.sin(obj_tp[:, 1])
        disp[:, 2] = rope_length * torch.cos(obj_tp[:, 1])
        return - disp

    def get_obj_rel_vel(self, obj_tp, obj_vtp, rope_length):
        '''
        drone_vxyz: (env_num, 3)
        obj_vtp: (env_num, 3)
        rope_length: (env_num, 1)
        '''
        theta = obj_tp[:, 0]
        phi = obj_tp[:, 1]
        dtheta = obj_vtp[:, 0]
        dphi = obj_vtp[:, 1]

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

        return torch.stack([vx, vy, vz], dim=-1)

def playground():
    '''
    use meshcat to visualize the environment
    '''
    import meshcat
    import meshcat.geometry as g
    import meshcat.transformations as tf
    import time
    env = QuadTrans(env_num=1, gpu_id=-1)
    env.reset()
    vis = meshcat.Visualizer()
    # set camera position
    vis["/Cameras/default"].set_transform(
		tf.translation_matrix([0,0,0]).dot(
		tf.euler_matrix(0,np.radians(-30),-np.pi/2)))
    vis["/Cameras/default/rotated/<object>"].set_transform(
        tf.translation_matrix([1, 0, 0]))
    # set quadrotor position
    vis["drone"].set_object(g.StlMeshGeometry.from_file('../assets/crazyflie2.stl'))
    while True:
        action = torch.rand((env.env_num, env.action_dim), device=env.device) * 2 - 1
        obs, reward, done, info = env.step(action)
        # visualize the drone
        xyz_drone = env.xyz_drone[0].numpy()
        rpy_drone = env.rpy_drone[0].numpy()
        vis["drone"].set_transform(tf.translation_matrix(xyz_drone) @ tf.euler_matrix(*rpy_drone))
        # visualize the target
        xyz_target = env.xyz_target[0].numpy()
        vis["target"].set_transform(tf.translation_matrix(xyz_target))
        # visualize the object
        xyz_obj = env.xyz_drone[0].numpy() + env.get_obj_disp(env.tp_obj, env.length_rope)[0].numpy()
        vis["obj"].set_transform(tf.translation_matrix(xyz_obj))

        time.sleep(1.0)
        
if __name__ == '__main__':
    playground()