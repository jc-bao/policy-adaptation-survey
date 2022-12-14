import torch
import wandb
from tqdm import trange
from icecream import install
from dataclasses import dataclass
import tyro

install()

from adaptive_control_gym.envs import HoverEnv, test_hover
from adaptive_control_gym.controllers import PPO

@dataclass
class Args:
    use_wandb:bool=False
    program:str='tmp'
    seed:int=0
    dim:int=1
    gpu_id:int=0
    expert_mode:bool=False
    ood_mode:bool=False

def train(args:Args)->None:
    env_num = 1024
    total_steps = 3e6
    eval_freq = 4
    
    env = HoverEnv(
        env_num=env_num, gpu_id=args.gpu_id, dim=args.dim, seed = args.seed, 
        expert_mode=args.expert_mode, ood_mode=args.ood_mode)
    agent = PPO(state_dim=env.state_dim, action_dim=env.action_dim, env_num=env_num, gpu_id=args.gpu_id)
    agent.last_state = env.reset()

    exp_name = f'OOD{args.ood_mode}_EXP{args.expert_mode}_S{args.seed}'
    if args.use_wandb:
        wandb.init(project=args.program, name=exp_name, config=args)
    steps_per_ep = env.max_steps*env_num
    n_ep = int(total_steps//steps_per_ep)
    with trange(n_ep) as t:
        for i_ep in t:
            # train
            total_steps = i_ep * steps_per_ep
            states, actions, logprobs, rewards, undones = agent.explore_env(env, env.max_steps)
            torch.set_grad_enabled(True)
            critic_loss, actor_loss, action_std = agent.update_net(states, actions, logprobs, rewards, undones)
            torch.set_grad_enabled(False)

            # log
            rew_mean = rewards.mean().item()
            rew_final_mean = rewards[-1].mean().item()
            if args.use_wandb:
                wandb.log({
                    'train/rewards_mean': rew_mean, 
                    'train/rewards_final': rew_final_mean,
                    'train/actor_loss': actor_loss, 
                    'train/critic_loss': critic_loss,
                    'train/action_std': action_std 
                }, step=total_steps)
            t.set_postfix(reward_final=rew_final_mean, actor_loss=actor_loss, critic_loss=critic_loss, rewards=rew_mean, steps = total_steps)

            # evaluate
            if i_ep % eval_freq == 0:
                have_ood = hasattr(env, 'ood_mode')
                if have_ood:
                    original_mode = env.ood_mode
                    env.ood_mode = False
                states, actions, logprobs, rewards, undones = agent.explore_env(env, env.max_steps, deterministic=True)
                if have_ood:
                    env.ood_mode = original_mode
                if args.use_wandb:
                    wandb.log({
                        'eval/rewards_mean': rewards.mean().item(), 
                        'eval/rewards_final': rewards[-1].mean().item(),
                    }, step=total_steps)
                else:
                    ic(rewards.mean().item(), rewards[-1].mean().item())
    
    actor_path = f'../../../results/rl/actor_ppo_{exp_name}.pt'
    plt_path = f'../../../results/rl/eval_ppo_{exp_name}.png'
    torch.save(agent.act, actor_path)
    test_hover(HoverEnv(env_num=1, gpu_id =-1, seed=0, expert_mode=args.expert_mode, ood_mode=args.ood_mode, dim = args.dim), agent.act.to('cpu'), save_path=plt_path)
    # evaluate
    if args.use_wandb:
        wandb.save(actor_path, base_path="../../../results/rl", policy="now")
        # save the plot
        wandb.log({"eval/plot": wandb.Image(plt_path)})

if __name__=='__main__':
    train(tyro.cli(Args))