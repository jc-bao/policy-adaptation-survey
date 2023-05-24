import brax
import jax
from brax.io import html
from flask import Flask

from adaptive_control_gym.envs.brax.inverted_pendulum import InvertedPendulum

def vis_html(file):
    app = Flask(__name__)
    @app.route('/')
    def hello_world():
        return file
    app.run()

def main():
    backend = 'spring'
    env = InvertedPendulum(backend=backend)
    state = jax.jit(env.reset)(rng=jax.random.PRNGKey(0))
    html_file = html.render(env.sys, [state.pipeline_state])
    vis_html(html_file)

if __name__ == '__main__':
    main()
    