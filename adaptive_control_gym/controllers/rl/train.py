import torch
import wandb

from adaptive_control_gym.envs import HoverEnv  
from adaptive_control_gym.controllers import PPO

def train():
    use_wandb=False

    env_num = 1
    total_steps = 1e3
    eval_freq = 10
    gpu_id = 0
    net_dims = [64, 64]
    
    env = HoverEnv(env_num=env_num, gpu_id=gpu_id)
    agent = PPO(net_dims, env.state_dim, env.action_dim, env_num, gpu_id)

    if use_wandb:
        wandb.init(project="adaptive-control-gym", entity="jason-zhang")
        wandb.watch(agent.actor)
        wandb.watch(agent.critic)

    for i_ep in range(total_steps//env.max_steps//env_num):
        states, actions, logprobs, rewards, undones = agent.explore_env(env, env.max_steps)
        torch.set_grad_enabled(True)
        logging_tuple = agent.update_net(states, actions, logprobs, rewards, undones)
        torch.set_grad_enabled(False)
        if i_ep % eval_freq == 0:
            states, actions, logprobs, rewards, undones = agent.explore_env(env, env.max_steps, deterministic=True)
            print(f"Episode {i_ep} | {logging_tuple}")