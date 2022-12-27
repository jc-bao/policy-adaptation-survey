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
    act_expert_mode:int=1
    cri_expert_mode:int=1
    exp_name:str= ''

def train(args:Args)->None:
    env_num = 1024
    total_steps = 1.0e7
    adapt_steps = 0.0e6
    eval_freq = 4
    curri_thereshold = 0.0
    compressor_dim = 0

    if len(args.exp_name) == 0:
        args.exp_name = f'ActEx{args.act_expert_mode}_CriEx{args.cri_expert_mode}_S{args.seed}'
    env = DroneEnv(
        env_num=env_num, gpu_id=args.gpu_id, seed = args.seed) 
    agent = PPO(
        state_dim=env.state_dim, expert_dim=env.expert_dim, action_dim=env.action_dim, 
        adapt_horizon=env.adapt_horizon, 
        act_expert_mode=args.act_expert_mode, cri_expert_mode=args.cri_expert_mode,
        compressor_dim=compressor_dim, 
        env_num=env_num, gpu_id=args.gpu_id)

    # agent.act = torch.load('/home/pcy/rl/policy-adaptation-survey/results/rl/actor_ppo_ActEx0_CriEx0_S0.pt', map_location='cuda:0')
    # agent.adaptor = torch.load('/home/pcy/rl/policy-adaptation-survey/results/rl/adaptor_ppo_ActEx0_CriEx0_S0.pt', map_location='cuda:0')

    if args.use_wandb:
        wandb.init(project=args.program, name=args.exp_name, config=args)
    steps_per_ep = env.max_steps*env_num
    n_ep = int(total_steps//steps_per_ep)
    total_steps = 0
    env.curri_param = 1.0
    with trange(n_ep) as t:
        agent.last_state, agent.last_info = env.reset()
        for i_ep in t:
            # train
            explore_steps = int(env.max_steps * np.clip(i_ep/10, 0.1, 1))
            total_steps += explore_steps * env_num
            states, actions, logprobs, rewards, undones, infos = agent.explore_env(env, explore_steps)
            torch.set_grad_enabled(True)
            critic_loss, actor_loss, action_std = agent.update_net(states, infos['e'], actions, logprobs, rewards, undones)
            torch.set_grad_enabled(False)

            # log
            rew_mean = rewards.mean().item()
            rew_final_mean = rewards[-1].mean().item()
            err_x, err_v = infos['err_x'][:-1], infos['err_v'][:-1]
            err_x_mean = err_x.mean().item()
            err_v_mean = err_v.mean().item()
            err_x_last10_mean = err_x[-10:].mean().item()
            err_v_last10_mean = err_v[-10:].mean().item()
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
            t.set_postfix(e10=err_x_last10_mean, rewards=rew_mean, actor_loss=actor_loss, critic_loss=critic_loss, steps = total_steps)

            # evaluate
            if i_ep % eval_freq == 0:
                log_dict = eval_env(env, agent)
                if args.use_wandb:
                    wandb.log(log_dict, step=total_steps)
                else:
                    ic(log_dict['eval/err_x_last10'])
                if rew_mean > curri_thereshold and env.curri_param < 1.0:
                    env.curri_param+=0.1

    n_ep = int(adapt_steps//steps_per_ep)
    with trange(n_ep) as t:
        agent.last_state, agent.last_info = env.reset()
        for i_ep in t:
            total_steps+=(env.max_steps*env_num)
            states, actions, logprobs, rewards, undones, infos = agent.explore_env(env, env.max_steps)
            torch.set_grad_enabled(True)
            adaptor_loss = agent.update_adaptor(infos['e'], infos['obs_history'])
            torch.set_grad_enabled(False)

            # log
            if args.use_wandb:
                wandb.log({
                    'adapt/adaptor_loss': adaptor_loss, 
                }, step=total_steps)
            else:
                ic(adaptor_loss)
            t.set_postfix(adaptor_loss = adaptor_loss, steps = total_steps)

            # evaluate
            log_dict = eval_env(env, agent, use_adaptor=True)
            if args.use_wandb:
                wandb.log(log_dict, step=total_steps)
            else:
                ic(log_dict['eval/err_x_last10'])
    
    actor_path = f'../../../results/rl/actor_ppo_{args.exp_name}.pt'
    adaptor_path = f'../../../results/rl/adaptor_ppo_{args.exp_name}.pt'
    plt_path = f'../../../results/rl/'
    torch.save(agent.act, actor_path)
    torch.save(agent.adaptor, adaptor_path)
    test_drone(DroneEnv(env_num=1, gpu_id =-1, seed=0), agent.act.cpu(), agent.adaptor.cpu(), save_path=plt_path)
    # evaluate
    if args.use_wandb:
        wandb.save(actor_path, base_path="../../../results/rl", policy="now")
        # save the plot
        wandb.log({
            "eval/plot": wandb.Image(plt_path+'/plot.png', caption="plot"), 
            "eval/vis": wandb.Image(plt_path+'/vis.png', caption="vis")
        })



def eval_env(env, agent, deterministic=True, use_adaptor=False):
    agent.last_state, agent.last_info = env.reset()
    states, actions, logprobs, rewards, undones, infos = agent.explore_env(env, env.max_steps, deterministic=deterministic, use_adaptor=use_adaptor)
    rew_mean = rewards.mean().item()
    err_x, err_v = infos['err_x'][:-1], infos['err_v'][:-1]
    err_x_mean = err_x.mean().item()
    err_v_mean = err_v.mean().item()
    err_x_last10_mean = err_x[-10:].mean().item()
    err_v_last10_mean = err_v[-10:].mean().item()
    log_dict = {
        'eval/rewards_mean': rew_mean,
        'eval/rewards_final': rewards[-1].mean().item(),
        'eval/err_x': err_x_mean,
        'eval/err_v': err_v_mean,
        'eval/err_x_last10': err_x_last10_mean,
        'eval/err_v_last10': err_v_last10_mean,
    }
    return log_dict



if __name__=='__main__':
    train(tyro.cli(Args))