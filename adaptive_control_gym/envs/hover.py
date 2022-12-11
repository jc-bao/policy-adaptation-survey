from typing import Optional
import gym
import numpy as np
import torch
import control as ct
from icecream import ic
from matplotlib import pyplot as plt

from adaptive_control_gym import controllers as ctrl


class HoverEnv(gym.Env):
    def __init__(self, env_num: int = 1, gpu_id: int = 0):
        # parameters
        self.mass_mean, self.mass_std = 0.1, 0.000
        self.disturbance_mean, self.disturbance_std = 0.0, 0.0
        self.init_x_mean, self.init_x_std = 0.0, 1.0
        self.init_v_mean, self.init_v_std = 0.0, 1.0
        self.tau = 1.0/30.0  # seconds between state updates
        self.force_scale = 1.0
        self.max_force = 1.0

        # state
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        self.env_num = env_num
        self.state_dim = 2
        self.action_dim = 1
        self.max_steps = 40
        self.x, self.v, self.mass = self._get_initial_state()
        self.step_cnt = 0

        # dynamic
        A = np.array([[0, 1], [0, 0]])
        self.A = np.stack([A] * self.env_num, axis=0)
        B = (1 / self.mass).detach().cpu().numpy()
        self.B = np.expand_dims(np.stack([np.zeros_like(B), B], axis=1), axis=2)
        self.dynamic_info = self._get_dynamic_info()
        ic(self.dynamic_info)

        self.action_space = gym.spaces.Box(
            low=-self.max_force, high=self.max_force, shape=(1,), dtype=float)

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
        x = torch.randn(size, device=self.device) * self.init_x_std + self.init_x_mean
        v = torch.randn(size, device=self.device) * self.init_v_std + self.init_v_mean
        mass = torch.randn(size, device=self.device) * self.mass_std + self.mass_mean
        return x, v, mass
    
    def step(self, action):
        disturbance = torch.randn(self.env_num, device=self.device) * self.disturbance_std + self.disturbance_mean
        force = torch.clip(action*self.force_scale, -self.max_force, self.max_force) + disturbance
        force = force.squeeze(1)
        self.x += self.v * self.tau
        self.v += force / self.mass * self.tau
        self.step_cnt += 1
        single_done = self.step_cnt >= self.max_steps
        done = torch.ones(self.env_num, device=self.device)*single_done
        if single_done:
            self.reset()
        reward = 1.0 - torch.abs(self.x)
        return torch.stack([self.x,self.v], dim=-1), reward, done, {}

    def reset(self):
        self.step_cnt = 0
        self.x, self.v, self.mass = self._get_initial_state()
        return torch.stack([self.x,self.v], dim=-1)

def test_cartpole(policy_name = "lqr"):
    env_num = 1
    gpu_id = 0
    np.set_printoptions(precision=3, suppress=True)
    env = HoverEnv(env_num=env_num, gpu_id = gpu_id)
    state = env.reset()
    x_list, v_list, done_list = [], [], []

    if policy_name == "lqr":
        Q = np.array([[50, 0],[0, 1]])
        R = 1
        policy = ctrl.LRQ(env.A, env.B, Q, R, gpu_id = gpu_id)
    elif policy_name == "random":
        policy = ctrl.Random(env.action_space)
    else: 
        raise NotImplementedError

    for t in range(180):
        act = policy(state)
        state, rew, done, info = env.step(act)  # take a random action
        x_list.append(state[0,0].item())
        v_list.append(state[0,1].item())
        if done:
            done_list.append(t)
    # plot x_list, v_list, action_list
    plt.figure()
    plt.plot(x_list, label="x")
    plt.plot(v_list, label="v")
    # draw vertical lines for done
    for t in done_list:
        plt.axvline(t, color="red", linestyle="--")
    plt.legend()
    # save the plot as image
    plt.savefig(f"../../results/hover_{policy_name}.png")
    env.close()

if __name__ == "__main__":
    test_cartpole()