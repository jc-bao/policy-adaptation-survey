import functools
from datetime import datetime
import jax
from jax import numpy as jp
import matplotlib.pyplot as plt
from brax.training.agents.ppo import train as ppo
from brax.io import model

from adaptive_control_gym.envs.brax.quadbrax import QuadBrax
from adaptive_control_gym.envs.brax.play_brax import Args, play_main

def main():
    env = QuadBrax()
    rng = jax.random.PRNGKey(seed=1)
    state = jax.jit(env.reset)(rng=rng)

    train_fn = functools.partial(ppo.train, num_timesteps=4_000_000, num_evals=20, reward_scaling=1.0, episode_length=100, normalize_observations=True, action_repeat=1, unroll_length=5, num_minibatches=32, num_updates_per_batch=4, discounting=0.97, learning_rate=3e-4, entropy_cost=1e-2, num_envs=2048, batch_size=1024, seed=1)

    xdata, ydata = [], []
    times = [datetime.now()]

    def progress(num_steps, metrics):
        times.append(datetime.now())
        xdata.append(num_steps)
        ydata.append(metrics['eval/episode_reward'])
        plt.xlabel('# environment steps')
        plt.ylabel('reward per episode')
        plt.plot(xdata, ydata)
        plt.savefig('../results/brax_train.png')

    make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)

    print(f'time to jit: {times[1] - times[0]}')
    print(f'time to train: {times[-1] - times[1]}')

    model.save_params('../results/params', params)

    agrs= Args
    agrs.policy_type = "ppo"
    agrs.policy_path = '../results/params'
    play_main(args=agrs)

if __name__ == '__main__':
    main()