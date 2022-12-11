from adaptive_control_gym.envs import HoverEnv  
from adaptive_control_gym.controllers import PPO, ReplayBufferList

def train():
    env_num = 1
    gpu_id = 0
    net_dims = [64, 64]
    
    env = HoverEnv(env_num=env_num, gpu_id=gpu_id)
    agent = PPO(net_dims, env.state_dim, env.action_dim, env_num, gpu_id)

