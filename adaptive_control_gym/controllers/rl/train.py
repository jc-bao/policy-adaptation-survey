import torch
import numpy as np
import wandb
from tqdm import trange
from icecream import install
from dataclasses import dataclass
import tyro

install()

from adaptive_control_gym.envs import HoverEnv, test_hover, DodgerEnv, test_dodger, DroneEnv, test_drone
from adaptive_control_gym.controllers import PPO

@dataclass
class Args:
    use_wandb:bool=False
    program:str='tmp'
    seed:int=0
    gpu_id:int=0
    expert_mode:bool=False
    ood_mode:bool=False
    exp_name:str= ''

def train(args:Args)->None:
    env_num = 1024
    total_steps = 6.0e6
    eval_freq = 4
    curri_thereshold = 0.0
    
    args.exp_name = f'EXP{args.expert_mode}_OOD{args.ood_mode}_S{args.seed}'
    env = DroneEnv(
        env_num=env_num, gpu_id=args.gpu_id, seed = args.seed, 
        expert_mode=args.expert_mode, ood_mode=args.ood_mode)
    agent = PPO(state_dim=env.state_dim, action_dim=env.action_dim, env_num=env_num, gpu_id=args.gpu_id)

    # agent.act = torch.load('../../../results/rl/actor_ppo_EXPFalse_OODFalse_S0.pt', map_location='cuda:0')

    if args.use_wandb:
        wandb.init(project=args.program, name=args.exp_name, config=args)
    steps_per_ep = env.max_steps*env_num
    n_ep = int(total_steps//steps_per_ep)
    total_steps = 0
    env.curri_param = 1.0
    with trange(n_ep) as t:
        agent.last_state = env.reset()
        for i_ep in t:
            # train
            explore_steps = int(env.max_steps * np.clip(i_ep/5, 0.2, 1))
            total_steps += explore_steps * env_num
            states, actions, logprobs, rewards, undones, infos = agent.explore_env(env, explore_steps)
            torch.set_grad_enabled(True)
            critic_loss, actor_loss, action_std = agent.update_net(states, actions, logprobs, rewards, undones)
            torch.set_grad_enabled(False)

            # log
            rew_mean = rewards.mean().item()
            rew_final_mean = rewards[-1].mean().item()
            err_x_mean = infos['err_x'].mean().item()
            err_v_mean = infos['err_v'].mean().item()
            err_x_last10_mean = infos['err_x'][-10:].mean().item()
            err_v_last10_mean = infos['err_v'][-10:].mean().item()
            if args.use_wandb:
                wandb.log({
                    'train/rewards_mean': rew_mean, 
                    'train/rewards_final': rew_final_mean,
                    'train/actor_loss': actor_loss, 
                    'train/critic_loss': critic_loss,
                    'train/action_std': action_std, 
                    'train/curri': env.curri_param, 
                    'train/err_x': err_x_mean,
                    'train/err_v': err_v_mean,
                    'train/err_x_last10': err_x_last10_mean,
                    'train/err_v_last10': err_v_last10_mean,
                }, step=total_steps)
            t.set_postfix(reward_final=rew_final_mean, actor_loss=actor_loss, critic_loss=critic_loss, rewards=rew_mean, steps = total_steps)

            # evaluate
            if i_ep % eval_freq == 0:
                have_ood = hasattr(env, 'ood_mode')
                if have_ood:
                    original_mode = env.ood_mode
                    env.ood_mode = False
                states, actions, logprobs, rewards, undones, infos = agent.explore_env(env, env.max_steps, deterministic=True)
                if have_ood:
                    env.ood_mode = original_mode
                rew_mean = rewards.mean().item()
                err_x_mean = infos['err_x'].mean().item()
                err_v_mean = infos['err_v'].mean().item()
                err_x_last10_mean = infos['err_x'][-10:].mean().item()
                err_v_last10_mean = infos['err_v'][-10:].mean().item()
                if args.use_wandb:
                    wandb.log({
                        'eval/rewards_mean': rew_mean,
                        'eval/rewards_final': rewards[-1].mean().item(),
                        'eval/err_x': err_x_mean,
                        'eval/err_v': err_v_mean,
                        'eval/err_x_last10': err_x_last10_mean,
                        'eval/err_v_last10': err_v_last10_mean,
                    }, step=total_steps)
                else:
                    ic(rew_mean, rewards[-1].mean().item())
                if rew_mean > curri_thereshold and env.curri_param < 1.0:
                    env.curri_param+=0.1
            
    
    actor_path = f'../../../results/rl/actor_ppo_{args.exp_name}.pt'
    plt_path = f'../../../results/rl/'
    torch.save(agent.act, actor_path)
    test_drone(DroneEnv(env_num=1, gpu_id =-1, seed=0, expert_mode=args.expert_mode, ood_mode=args.ood_mode), agent.act.cpu(), save_path=plt_path)
    # evaluate
    if args.use_wandb:
        wandb.save(actor_path, base_path="../../../results/rl", policy="now")
        # save the plot
        wandb.log({
            "eval/plot": wandb.Image(plt_path+'/plot.png', caption="plot"), 
            "eval/vis": wandb.Image(plt_path+'/vis.png', caption="vis")
            })

if __name__=='__main__':
    train(tyro.cli(Args))