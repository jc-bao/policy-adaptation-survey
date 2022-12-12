import torch
import wandb
from tqdm import trange

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
        wandb.init(project="test-project", entity="adaptive-control") 

    steps_per_ep = env.max_steps*env_num
    with trange(total_steps//steps_per_ep) as t:
        for i_ep in t:
            # train
            total_steps = i_ep * steps_per_ep
            states, actions, logprobs, rewards, undones = agent.explore_env(env, env.max_steps)
            torch.set_grad_enabled(True)
            critic_loss, actor_loss, action_std = agent.update_net(states, actions, logprobs, rewards, undones)
            torch.set_grad_enabled(False)
            if use_wandb:
                wandb.log({
                    'train/rewards': rewards.mean().item(), 
                    'train/actor_loss': actor_loss, 
                    'train/critic_loss': critic_loss,
                    'train/action_std': action_std 
                }, step=total_steps)
            t.set_postfix(actor_loss=actor_loss, critic_loss=critic_loss, rewards=rewards, steps = total_steps)

            # evaluate
            if i_ep % eval_freq == 0:
                states, actions, logprobs, rewards, undones = agent.explore_env(env, env.max_steps, deterministic=True)
                if use_wandb:
                    wandb.log({
                        'eval/rewards': rewards.mean().item(), 
                    }, step=total_steps)
    
    actor_path = '../../../results/rl/actor.pt'
    torch.save(agent.act, actor_path)
    if use_wandb:
        wandb.save(actor_path, policy="now")
