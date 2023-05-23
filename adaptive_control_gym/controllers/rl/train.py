from adaptive_control_gym.controllers import PPO
from adaptive_control_gym.envs import QuadTransEnv, test_env
import adaptive_control_gym
import torch
import numpy as np
import wandb
from tqdm import trange
from icecream import install
from dataclasses import dataclass
import tyro

install()


@dataclass
class Args:
    use_wandb: bool = False
    program: str = 'tmp'
    seed: int = 1
    gpu_id: int = 0
    act_expert_mode: int = 0
    cri_expert_mode: int = 0
    exp_name: str = ''
    compressor_dim: int = 4
    search_dim: int = 0
    res_dyn_param_dim: int = 0


def train(args: Args) -> None:
    env_num = 1024 * 16
    total_steps = 8e7
    adapt_steps = 0.5e7 if ((args.act_expert_mode > 0)
                            | (args.cri_expert_mode > 0)) else 0
    eval_freq = 4
    curri_thereshold = 0.5

    if len(args.exp_name) == 0:
        args.exp_name = f'ActEx{args.act_expert_mode}_CriEx{args.cri_expert_mode}_S{args.seed}'
    env = QuadTransEnv(
        env_num=env_num, gpu_id=args.gpu_id, seed=args.seed,
        res_dyn_param_dim=args.res_dyn_param_dim
    )
    agent = PPO(
        state_dim=env.state_dim, expert_dim=env.expert_dim,
        adapt_dim=env.adapt_dim, action_dim=env.action_dim,
        adapt_horizon=env.adapt_horizon,
        act_expert_mode=args.act_expert_mode, cri_expert_mode=args.cri_expert_mode,
        compressor_dim=args.compressor_dim, search_dim=args.search_dim,
        env_num=env_num, gpu_id=args.gpu_id)

    # loaded_agent = torch.load('/home/pcy/rl/policy-adaptation-survey/results/rl/ppo_TrackAdaptUncertain.pt', map_location=f'cuda:{args.gpu_id}')
    # agent.act.load_state_dict(loaded_agent['actor'].state_dict())
    # agent.cri.load_state_dict(loaded_agent['critic'].state_dict())
    # agent.act.action_std_log = (torch.nn.Parameter(torch.ones((1, 2), device=f'cuda:{args.gpu_id}')*2.0))
    # agent.adaptor.load_state_dict(loaded_agent['adaptor'].state_dict())
    # agent.compressor.load_state_dict(loaded_agent['compressor'].state_dict())

    if args.use_wandb:
        wandb.init(project=args.program, name=args.exp_name, config=args)
    steps_per_ep = env.max_steps*env_num
    n_ep = int(total_steps//steps_per_ep)
    total_steps = 0
    env.curri_param = 0.0
    expert_err_x_final = torch.nan
    with trange(n_ep) as t:
        for i_ep in t:
            # get optimal w
            agent.last_state, agent.last_info = env.reset()
            w, env_params = get_optimal_w(env, agent, args.search_dim)
            # train
            explore_steps = int(np.clip(i_ep / 20, 1.0, 1.0)*env.max_steps)
            total_steps += explore_steps * env_num
            states, actions, logprobs, rewards, undones, infos = agent.explore_env(
                env, explore_steps, use_adaptor=False, w=w)
            agent.last_info['e'] = torch.concat(
                [agent.last_info['e'], w], dim=-1)
            torch.set_grad_enabled(True)
            critic_loss, actor_loss, ada_com_loss, action_std, compressor_std = agent.update_net(
                states, infos['e'], infos['adapt_obs'], actions, logprobs, rewards, undones, update_adaptor=False)
            torch.set_grad_enabled(False)

            # log
            rew_mean = rewards.mean().item()
            rew_final_mean = rewards[-1].mean().item()
            err_x, err_v = infos['err_x'][:-1], infos['err_v'][:-1]
            err_x_mean = err_x.mean().item()
            err_v_mean = err_v.mean().item()
            err_x_last10_mean = err_x[-10:].mean().item()
            err_v_last10_mean = err_v[-10:].mean().item()
            step_mean = infos['step'].float().mean().item()*2.0
            if args.use_wandb:
                wandb.log({
                    'train/rewards_mean': rew_mean,
                    'train/rewards_final': rew_final_mean,
                    'train/actor_loss': actor_loss,
                    'train/critic_loss': critic_loss,
                    'train/ada_com_loss': ada_com_loss,
                    'train/action_std': action_std,
                    'train/compressor_std': compressor_std,
                    'train/curri': env.curri_param,
                    'train/err_x': err_x_mean,
                    'train/err_v': err_v_mean,
                    'train/err_x_last10': err_x_last10_mean,
                    'train/err_v_last10': err_v_last10_mean,
                    'train/step': step_mean,
                }, step=total_steps)
            t.set_postfix(e10=err_x_last10_mean, rewards=rew_mean, actor_loss=actor_loss, critic_loss=critic_loss,
                          compressor_std=compressor_std, ada_com_loss=ada_com_loss, steps=total_steps)

            # evaluate
            if i_ep % eval_freq == 0:
                agent.last_state, agent.last_info = env.reset()
                w, env_params = get_optimal_w(env, agent, args.search_dim)
                log_dict = eval_env(env, agent, use_adaptor=False, w=w)
                if args.use_wandb:
                    wandb.log(log_dict, step=total_steps)
                else:
                    print(
                        f"{log_dict['eval/err_x_last10']:.4f} \pm {log_dict['eval/err_x_last10_std']:.4f}")
                if rew_mean > curri_thereshold and env.curri_param < 1.0:
                    env.curri_param += 0.1
                # log_dict = eval_env(env, agent, use_adaptor=True)
                # print(f"{log_dict['eval/err_x_last10']:.4f} \pm {log_dict['eval/err_x_last10_std']:.4f}")
        expert_err_x_final = log_dict['eval/err_x_last10']

    adapt_err_x_initial = torch.nan
    adapt_err_x_end = torch.nan
    n_ep = int(adapt_steps//steps_per_ep)
    for p in list(agent.cri.parameters())+list(agent.act.parameters())+list(agent.compressor.parameters()):
        p.requires_grad = False
    with trange(n_ep) as t:
        for i_ep in t:
            agent.last_state, agent.last_info = env.reset()
            total_steps += (env.max_steps*env_num)
            states, actions, logprobs, rewards, undones, infos = agent.explore_env(
                env, env.max_steps, use_adaptor=True)
            torch.set_grad_enabled(True)
            # critic_loss, actor_loss, ada_com_loss, action_std, compressor_std = agent.update_net(states, infos['e'], infos['adapt_obs'], actions, logprobs, rewards, undones, update_adaptor=True, update_actor=False, update_critic=False)
            # adaptor_loss, adaptor_err = actor_loss, ada_com_loss
            adaptor_loss, adaptor_err = agent.update_adaptor(
                infos['e'], infos['adapt_obs'])
            torch.set_grad_enabled(False)

            # log
            if args.use_wandb:
                wandb.log({
                    'adapt/adaptor_loss': adaptor_loss,
                    'adapt/adaptor_err': adaptor_err,
                }, step=total_steps)
            else:
                ic(adaptor_loss)
            t.set_postfix(err=adaptor_err,
                          adaptor_loss=adaptor_loss, steps=total_steps)

            # evaluate
            agent.last_state, agent.last_info = env.reset()
            log_dict = eval_env(env, agent, use_adaptor=True)
            if args.use_wandb:
                wandb.log(log_dict, step=total_steps)
            else:
                print(
                    f"{log_dict['eval/err_x_last10']:.4f} \pm {log_dict['eval/err_x_last10_std']:.4f}")
            # log_dict = eval_env(env, agent, use_adaptor=False)
            # print(f"{log_dict['eval/err_x_last10']:.4f} \pm {log_dict['eval/err_x_last10_std']:.4f}")

            if i_ep == 0:
                adapt_err_x_initial = log_dict['eval/err_x_last10']
            if i_ep == (n_ep-1):
                adapt_err_x_end = log_dict['eval/err_x_last10']

    base_path = adaptive_control_gym.__path__[0] + '/../results/rl'
    path = f'{base_path}/ppo_{args.exp_name}.pt'
    plt_path = f'{base_path}/ppo_{args.exp_name}'
    # save agent.act, agent.adaptor, agent.compressor
    torch.save({
        'actor': agent.act,
        'critic': agent.cri,
        'adaptor': agent.adaptor,
        'compressor': agent.compressor,
        'expert_err_x_final': expert_err_x_final,
        'adapt_err_x_initial': adapt_err_x_initial,
        'adapt_err_x_end': adapt_err_x_end,
    }, path)
    test_env(QuadTransEnv(env_num=1, gpu_id=-1, seed=args.seed, res_dyn_param_dim=args.res_dyn_param_dim),
             agent.act.cpu(), agent.adaptor.cpu(), compressor=agent.compressor.cpu(), save_path=plt_path)
    # evaluate
    if args.use_wandb:
        wandb.save(path, policy="now")
        # save the plot
        wandb.log({
            "eval/plot": wandb.Image(f'{plt_path}_plot.png', caption="plot"),
            "eval/vis": wandb.Image(f'{plt_path}_vis.png', caption="vis")
        })
    # print the result
    print(f'{expert_err_x_final:.4f} | {adapt_err_x_initial:.4f} | {adapt_err_x_end:.4f}')


def get_optimal_w(env: QuadTransEnv, agent: PPO, search_dim: int = 0):
    # save initial environment parameters
    env_params = env.get_env_params()
    old_last_state, old_last_info = agent.last_state, agent.last_info
    w = torch.zeros([env.env_num, search_dim], device=env.device)
    if search_dim == 0:
        return w, env_params

    # grid search w
    w_per_dim = 6
    w_single_dim = torch.linspace(-1, 1, w_per_dim, device=env.device)
    w_stacked = torch.stack(torch.meshgrid(
        [w_single_dim]*search_dim), dim=-1).reshape(-1, search_dim)
    w_num = w_per_dim**search_dim
    task_num = env.env_num
    search_env = QuadTransEnv(env_num=task_num*w_num, gpu_id=env.gpu_id,
                              seed=env.seed, res_dyn_param_dim=env.res_dyn_param_dim)

    # set parameters to env params
    search_env_params = search_env.get_env_params()
    for sp, p in zip(search_env_params, env_params):
        for i in range(env.env_num):
            sp[i*w_num:(i+1)*w_num] = p[i]
    # repeat w
    w_repeat = w_stacked.repeat(env.env_num, 1).reshape(-1, search_dim)
    torch.set_grad_enabled(False)
    agent.last_state, agent.last_info = search_env.reset(
        env_params=search_env_params)
    states, actions, logprobs, rewards, undones, infos = agent.explore_env(
        search_env, env.max_steps*20, deterministic=True, use_adaptor=False, w=w_repeat)
    rew_mean = rewards.mean(dim=0)
    torch.set_grad_enabled(True)
    for i in range(task_num):
        # find the best w with max rew_mean
        w_best_idx = torch.argmax(rew_mean[i*w_num:(i+1)*w_num])
        w[i] = w_stacked[w_best_idx]

    agent.last_state, agent.last_info = old_last_state, old_last_info

    return w, env_params


def eval_env(env: QuadTransEnv, agent: PPO, deterministic: bool = True, use_adaptor: bool = False, w=None):

    # env.mass_max = env.mass_mean + env.mass_std*1.0
    # env.mass_min = env.mass_mean - env.mass_std*1.0
    # env.decay_max = env.decay_mean + env.decay_std*1.0
    # env.decay_min = env.decay_mean - env.decay_std*1.0
    # env.disturb_max = env.disturb_mean + env.disturb_std*1.0
    # env.disturb_min = env.disturb_mean - env.disturb_std*1.0
    # env.res_dyn_param_max = env.res_dyn_param_mean + env.res_dyn_param_std*1.0
    # env.res_dyn_param_min = env.res_dyn_param_mean - env.res_dyn_param_std*1.0
    # env.force_scale_max = env.force_scale_mean + env.force_scale_std*1.0
    # env.force_scale_min = env.force_scale_mean - env.force_scale_std*1.0

    # env.res_dyn = env.res_dyn_origin

    # origin_curri_param = env.curri_param
    # env.curri_param = 1.0
    agent.last_state, agent.last_info = env.reset()
    states, actions, logprobs, rewards, undones, infos = agent.explore_env(
        env, env.max_steps, deterministic=deterministic, use_adaptor=use_adaptor, w=w)
    # env.curri_param = origin_curri_param

    # env.res_dyn = env.res_dyn_fit

    # env.mass_max = env.mass_mean + env.mass_std*0.5
    # env.mass_min = env.mass_mean - env.mass_std*0.5
    # env.decay_max = env.decay_mean + env.decay_std*0.5
    # env.decay_min = env.decay_mean - env.decay_std*0.5
    # env.disturb_max = env.disturb_mean + env.disturb_std*0.5
    # env.disturb_min = env.disturb_mean - env.disturb_std*0.5
    # env.res_dyn_param_max = env.res_dyn_param_mean + env.res_dyn_param_std*0.5
    # env.res_dyn_param_min = env.res_dyn_param_mean - env.res_dyn_param_std*0.5
    # env.force_scale_max = env.force_scale_mean + env.force_scale_std*0.5
    # env.force_scale_min = env.force_scale_mean - env.force_scale_std*0.5

    rew_mean = rewards.mean().item()
    err_x, err_v = infos['err_x'][:-1], infos['err_v'][:-1]
    err_x_mean = err_x.mean().item()
    err_v_mean = err_v.mean().item()
    err_x_last10_mean = err_x[-10:].mean().item()
    err_x_last10_std = err_x[-10:].mean(dim=0).std().item()
    err_v_last10_mean = err_v[-10:].mean().item()
    err_v_last10_std = err_v[-10:].mean(dim=0).std().item()
    step_mean = infos['step'].float().mean().item()*2.0
    log_dict = {
        'eval/rewards_mean': rew_mean,
        'eval/rewards_final': rewards[-1].mean().item(),
        'eval/err_x': err_x_mean,
        'eval/err_v': err_v_mean,
        'eval/err_x_last10': err_x_last10_mean,
        'eval/err_x_last10_std': err_x_last10_std,
        'eval/err_v_last10': err_v_last10_mean,
        'eval/err_v_last10_std': err_v_last10_std,
        'eval/step': step_mean,
    }
    return log_dict


if __name__ == '__main__':
    train(tyro.cli(Args))
    # env_num = 256
    # env = QuadTrans(env_num=env_num, gpu_id =0, seed=0, res_dyn_param_dim=0)
    # agent = PPO(
    #     state_dim=env.state_dim, expert_dim=env.expert_dim,
    #     adapt_dim=env.adapt_dim, action_dim=env.action_dim,
    #     adapt_horizon=env.adapt_horizon,
    #     act_expert_mode=1, cri_expert_mode=1,
    #     compressor_dim=4, search_dim=2,
    #     env_num=env_num, gpu_id=0)
    # w, env_params = get_optimal_w(env, agent, search_dim=2)