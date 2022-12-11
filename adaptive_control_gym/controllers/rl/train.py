from adaptive_control_gym.envs import HoverEnv  
from adaptive_control_gym.controllers import PPO

def train():
    env_num = 1
    net_dims = [64, 64]
    state_dim = 2
    action_dim = 1
    gpu_id=1
    
    env = HoverEnv()
    agent = PPO()