import torch
import gym
import numpy as np
import torch
import control as ct
from icecream import ic
from matplotlib import pyplot as plt

from adaptive_control_gym import controllers as ctrl


class HoverEnv(gym.Env):
    def __init__(self, env_num: int = 1, gpu_id: int = 0, seed:int = 0, expert_mode:bool = False, ood_mode:bool = False, mass_uncertainty_rate:float=0.0, disturb_uncertainty_rate:float=0.0, disturb_period: int = 15):
        torch.manual_seed(seed)

        # parameters
        self.mass_mean, self.mass_std = 0.1, 0.1 * mass_uncertainty_rate
        self.mass_min, self.mass_max = 0.01, 1.0
        self.disturb_mean, self.disturb_std = 0.0, 1.0 * disturb_uncertainty_rate
        self.disturb_period = disturb_period
        self.init_x_mean, self.init_x_std = 0.0, 1.0
        self.init_v_mean, self.init_v_std = 0.0, 1.0
        self.tau = 1.0/30.0  # seconds between state updates
        self.force_scale = 1.0
        self.max_force = 1.0

        # state
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        self.env_num = env_num
        self.ood_mode = ood_mode
        self.expert_mode = expert_mode
        if expert_mode:
            self.state_dim = 4
        else:
            self.state_dim = 2
        self.action_dim = 1
        self.max_steps = 60
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
        if self.ood_mode:
            x = torch.abs(torch.randn(size, device=self.device)) * self.init_x_std + self.init_x_mean
            v = torch.abs(torch.randn(size, device=self.device)) * self.init_v_std + self.init_v_mean
            mass = torch.abs(torch.randn(size, device=self.device)) * self.mass_std + self.mass_mean
        else:
            x = torch.randn(size, device=self.device) * self.init_x_std + self.init_x_mean
            v = torch.randn(size, device=self.device) * self.init_v_std + self.init_v_mean
            mass = torch.randn(size, device=self.device) * self.mass_std + self.mass_mean
        mass = torch.clip(mass, self.mass_min, self.mass_max)
        return x, v, mass
    
    def step(self, action):
        force = torch.clip(action*self.force_scale, -self.max_force, self.max_force) + self.disturb
        self.force = force.squeeze(1)
        self.x += self.v * self.tau
        self.v += self.force / self.mass * self.tau
        self.step_cnt += 1
        single_done = self.step_cnt >= self.max_steps
        done = torch.ones(self.env_num, device=self.device)*single_done
        reward = 1.0 - torch.abs(self.x) - torch.abs(self.v)*0.1
        # update disturb
        if self.step_cnt % self.disturb_period == 0:
            self._set_disturb()
        if single_done:
            self.reset()
        info = {
            'mass': self.mass, 
            'disturb': self.disturb,
        }
        return self._get_obs(), reward, done, info

    def reset(self):
        self.step_cnt = 0
        self._set_disturb()
        self.x, self.v, self.mass = self._get_initial_state()
        return self._get_obs()

    def _get_obs(self):
        if self.expert_mode:
            return torch.stack([self.x,self.v,self.disturb[:,0], self.mass], dim=-1)
        else:
            return torch.stack([self.x,self.v], dim=-1)

    def _set_disturb(self):
        self.disturb = torch.randn((self.env_num,1), device=self.device) * self.disturb_std + self.disturb_mean

def test_cartpole(policy_name = "ppo"):
    env_num = 1
    gpu_id = 0
    np.set_printoptions(precision=3, suppress=True)
    env = HoverEnv(env_num=env_num, gpu_id = gpu_id, seed=2)
    state = env.reset()
    x_list, v_list, a_list, force_list, disturb_list, r_list, done_list = [], [], [], [], [], [], []

    if policy_name == "lqr":
        Q = np.array([[50, 0],[0, 1]])
        R = 1
        policy = ctrl.LRQ(env.A, env.B, Q, R, gpu_id = gpu_id)
    elif policy_name == "random":
        policy = ctrl.Random(env.action_space)
    elif policy_name == "ppo":
        policy = torch.load('../../results/rl/actor_ppo.pt')
    else: 
        raise NotImplementedError

    for t in range(180):
        act = policy(state)
        state, rew, done, info = env.step(act)  # take a random action
        x_list.append(state[0,0].item())
        v_list.append(state[0,1].item()*0.3)
        a_list.append(act[0,0].item())
        r_list.append(rew[0].item())
        force_list.append(env.force[0].item())
        disturb_list.append(env.disturb[0].item())
        if done:
            done_list.append(t)
    # plot x_list, v_list, action_list in three subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    axs[0].plot(x_list, 'g', label="x")
    axs[0].plot(v_list, 'b', label="v*0.3", alpha=0.2)
    axs[1].plot(a_list, label="action", alpha=0.2)
    axs[1].plot(force_list, label="force", alpha=0.2)
    axs[1].plot(disturb_list, label="disturb")
    axs[2].plot(r_list, 'y', label="reward")
    # draw vertical lines for done
    for t in done_list:
        axs[0].axvline(t, color="red", linestyle="--")
    # plot horizontal line for x=0
    axs[0].axhline(0, color="black", linestyle="--")
    for i in range(3):
        axs[i].legend()
    # save the plot as image
    plt.savefig(f"../../results/hover_{policy_name}.png")
    env.close()

if __name__ == "__main__":
    test_cartpole()