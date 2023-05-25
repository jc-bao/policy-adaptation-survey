import brax
import jax
from brax import envs
from brax.io import html, model
from brax.training.agents.ppo import networks as ppo_networks
from flask import Flask
from tqdm import trange
import matplotlib.pyplot as plt
import numpy as np
import tyro
from dataclasses import dataclass
from brax.training.acme import running_statistics
from typing import Optional

from adaptive_control_gym.envs.brax.quadbrax import QuadBrax


def vis_html(file):
    app = Flask(__name__)

    @app.route('/')
    def hello_world():
        return file
    app.run()


def plot_rollout(rollout):
    q = np.array([s.pipeline_state.q for s in rollout])
    qd = np.array([s.pipeline_state.qd for s in rollout])
    rews = np.array([s.reward for s in rollout])
    # plot in 2 subplots
    fig, axs = plt.subplots(3)
    axs[0].plot(q)
    # set y limit
    axs[0].set_ylim([-1.5, 1.5])
    axs[0].set_title('q')
    axs[1].plot(qd)
    axs[1].set_ylim([-15, 15])
    axs[1].set_title('qd')
    axs[2].plot(rews)
    axs[2].set_title('rews')
    # save it
    fig.savefig('../results/brax.png')


def random_policy(obs, rng):
    act = jax.random.uniform(rng, shape=(2,), minval=-1.0, maxval=1.0) * 1.0
    return act, ()


@dataclass
class Args:
    policy_type: str = "random"  # 'random', 'ppo'
    policy_path: Optional[str] = None


def play_main(args: Args):
    env = QuadBrax()
    rng = jax.random.PRNGKey(seed=1)
    state = jax.jit(env.reset)(rng=rng)
    # html_file = html.render(env.sys, [state.pipeline_state])

    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)
    if args.policy_type == "random":
        inference_fn = random_policy()
    elif args.policy_type == "ppo":
        params = model.load_params(args.policy_path)
        inference_fn = ppo_networks.make_inference_fn(
            ppo_networks.make_ppo_networks(
                6, 2,
                preprocess_observations_fn=running_statistics.normalize))(params)
    jit_inference_fn = jax.jit(inference_fn)
    rollout = []
    state = jit_env_reset(rng=rng)
    for i in trange(1000):
        rollout.append(state)
        act_rng, rng = jax.random.split(rng)
        act, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_env_step(state, act)
        if state.done or (i % 100 == 99):
            state = jit_env_reset(rng=rng)

    plot_rollout(rollout)
    html_file = html.render(env.sys.replace(dt=env.dt), [
                            r.pipeline_state for r in rollout], height=720)
    # save html_file to a file
    with open('../results/brax.html', 'w') as f:
        f.write(html_file)
    vis_html(html_file)


if __name__ == '__main__':
    play_main(tyro.cli(Args))
