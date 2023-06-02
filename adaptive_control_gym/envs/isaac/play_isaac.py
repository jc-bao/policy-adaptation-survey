import numpy as np
import torch
import hydra
from omegaconf import DictConfig

from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict

from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames

@hydra.main(config_name="config", config_path="config")
def parse_hydra_configs(cfg: DictConfig):

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    headless = cfg.headless
    render = not headless
    enable_viewport = "enable_cameras" in cfg.task.sim and cfg.task.sim.enable_cameras

    env = VecEnvRLGames(headless=headless, sim_device=cfg.device_id, enable_livestream=cfg.enable_livestream, enable_viewport=enable_viewport)
    # sets seed. if seed is -1 will pick a random one
    from omni.isaac.core.utils.torch.maths import set_seed
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    cfg_dict['seed'] = cfg.seed

    from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
    sim_config = SimConfig(cfg_dict)
    from adaptive_control_gym.envs.isaac.crazyflie import CrazyflieTask
    task = CrazyflieTask(
        name=cfg["task_name"], sim_config=sim_config, env=env
    )
    env.set_task(task=task, sim_params=sim_config.get_physics_params(), backend="torch", init_sim=True)

    while env._simulation_app.is_running():
        if env._world.is_playing():
            if env._world.current_time_step_index == 0:
                env._world.reset(soft=True)
            actions = torch.tensor(np.array([env.action_space.sample() for _ in range(env.num_envs)]), device=task.rl_device)
            env._task.pre_physics_step(actions)
            env._world.step(render=render)
            env.sim_frame_count += 1
            env._task.post_physics_step()
        else:
            env._world.step(render=render)

    env._simulation_app.close()

if __name__ == '__main__':
    parse_hydra_configs()
