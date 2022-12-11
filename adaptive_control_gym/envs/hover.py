from typing import Optional
import gym
import numpy as np
import torch
import control as ct
from icecream import ic
import imageio

from adaptive_control_gym import controllers as ctrl


class HoverEnv(gym.Env):
    def __init__(self, env_num: int = 1, gpu_id: int = 0):
        # parameters
        self.mass_mean, self.mass_std = 1.0, 0.2
        self.disturbance_mean, self.disturbance_std = 0.0, 0.0
        self.init_x_mean, self.init_x_std = 0.0, 1.0
        self.init_v_mean, self.init_v_std = 0.0, 1.0
        self.tau = 1.0/60.0  # seconds between state updates
        self.force_scale = 1.0
        self.max_force = 1.0

        # state
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        self.env_num = env_num
        self.state_dim = 2
        self.action_dim = 1
        self.max_steps = 10
        self.x, self.v, self.mass = self._get_initial_state()
        self.step_cnt = torch.zeros(self.env_num)

        # dynamic
        self.A = np.array([[0, 1], [0, 0]])
        self.B = np.array([[0], [1 / self.mass]])
        self.dynamic_info = self._get_dynamic_info()
        ic(self.dynamic_info)

        self.action_space = gym.spaces.Box(
            low=-self.max_force, high=self.max_force, shape=(1,), dtype=float)

    def _get_dynamic_info(self):
        eig_A = np.linalg.eig(self.A)
        ctrb_AB = ct.ctrb(self.A, self.B)
        ctrb_rank = np.linalg.matrix_rank(ctrb_AB)
        return {
            'eig_A': eig_A,
            'ctrb_AB': ctrb_AB,
            'ctrb_rank': ctrb_rank,
        }
    
    def _get_initial_state(self):
        x = torch.randn(self.env_num) * self.init_x_std + self.init_x_mean
        v = torch.randn(self.env_num) * self.init_v_std + self.init_v_mean
        mass = torch.randn(self.env_num) * self.mass_std + self.mass_mean
        return x, v, mass
    
    def step(self, action):
        disturbance = torch.randn(self.env_num) * self.disturbance_std + self.disturbance_mean
        force = torch.clip(action*self.force_scale, -self.max_force, self.max_force) + disturbance
        self.x += self.x + self.v * self.tau
        self.v += self.v + force / self.mass * self.tau
        return torch.stack([self.x,self.v], dim=-1), 0, False, {}

    def reset(self):
        self.x, self.v, self.mass = self._get_initial_state()
        return torch.stack([self.x,self.v], dim=-1)

def test_cartpole(policy_name = "lqr"):
    np.set_printoptions(precision=3, suppress=True)
    env = HoverEnv(env_num=1)
    state = env.reset()
    vid = []
    if policy_name == "lqr":
        Q = np.array([[50, 0],[0, 1]])
        R = 1
        policy = ctrl.LRQ(env.A, env.B, Q, R)
    elif policy_name == "random":
        policy = ctrl.Random(env.action_space)
    else: 
        raise NotImplementedError
    for _ in range(180):
        vid.append(env.render())
        act = policy(state)
        state, _, _, _ = env.step(act)  # take a random action
    imageio.mimsave(f'../../results/hover_{policy_name}.gif', vid, fps=30)
    env.close()

if __name__ == "__main__":
    test_cartpole()