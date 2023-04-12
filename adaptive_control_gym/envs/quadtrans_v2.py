import gym
import torch

class QuadTransEnv(gym.Env):
    def __init__(self, env_num:int=1024, gpu_id:int=0, seed:int=0, **kwargs) -> None:
        super().__init__()

        # set simulator parameters
        self.seed = seed
        self.env_num = env_num
        self.sim_dt = 4e-4
        self.ctl_substeps = 5
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

    def step(self, action):
        return self.observation_space.sample(), 0, False, {}
    
    def ctlstep(self, action):
        raise NotImplementedError
    
    def simstep(self):
        raise NotImplementedError

    def reset(self):
        return self.observation_space.sample()

    def render(self, mode='human'):
        pass

    def close(self):
        pass