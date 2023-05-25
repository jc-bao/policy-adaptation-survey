import brax
import jax
from brax import envs
from brax.io import html
from flask import Flask
from tqdm import trange
import matplotlib.pyplot as plt
import numpy as np

from adaptive_control_gym.envs.brax.quadbrax import InvertedPendulum

def vis_html(file):
    app = Flask(__name__)
    @app.route('/')
    def hello_world():
        return file
    app.run()

def plot_rollout(rollout):
    q = np.array([s.q for s in rollout])
    qd = np.array([s.qd for s in rollout])
    # plot in 2 subplots
    fig, axs = plt.subplots(2)
    axs[0].plot(q)
    # set y limit
    axs[0].set_ylim([-1.5, 1.5])
    axs[0].set_title('q')
    axs[1].plot(qd)
    axs[1].set_ylim([-15, 15])
    axs[1].set_title('qd')
    # save it
    fig.savefig('../results/brax.png')

def inference_fn(obs, rng):
    act = jax.random.uniform(rng, shape=(2,), minval=-1.0, maxval=1.0) * 1.0
    return act, ()

def main():
    env = InvertedPendulum()
    rng = jax.random.PRNGKey(seed=1)
    state = jax.jit(env.reset)(rng=rng)
    # html_file = html.render(env.sys, [state.pipeline_state])

    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)
    jit_inference_fn = jax.jit(inference_fn)
    rollout = []
    state = jit_env_reset(rng=rng)
    for _ in trange(100):
        rollout.append(state.pipeline_state)
        act_rng, rng = jax.random.split(rng)
        act, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_env_step(state, act)

    plot_rollout(rollout)
    html_file = html.render(env.sys.replace(dt=env.dt), rollout, height=720)
    vis_html(html_file)

if __name__ == '__main__':
    main()