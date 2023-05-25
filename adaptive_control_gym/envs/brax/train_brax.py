import brax
import jax
from brax import envs
from brax.io import html
from flask import Flask

from adaptive_control_gym.envs.brax.inverted_pendulum import InvertedPendulum

def vis_html(file):
    app = Flask(__name__)
    @app.route('/')
    def hello_world():
        return file
    app.run()

def inference_fn(obs, rng):
    act = jax.random.uniform(rng, shape=(3,), minval=-1.0, maxval=1.0)
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
    for _ in range(100):
        rollout.append(state.pipeline_state)
        act_rng, rng = jax.random.split(rng)
        act, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_env_step(state, act)

    html_file = html.render(env.sys.replace(dt=env.dt), rollout)
    vis_html(html_file)

if __name__ == '__main__':
    main()