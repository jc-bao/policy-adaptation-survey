from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath
import jax
from jax import numpy as jp


class QuadBrax(PipelineEnv):
    def __init__(self, backend='generalized', **kwargs):
        path = epath.resource_path(
            'adaptive_control_gym') / 'assets/quadbrax.xml'
        sys = mjcf.load(path)

        n_frames = 2

        if backend in ['spring', 'positional']:
            sys = sys.replace(dt=0.005)
            n_frames = 4

        kwargs['n_frames'] = kwargs.get('n_frames', n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)

    def reset(self, rng: jp.ndarray) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        q_bound = jp.array([2.0, 2.0, jp.pi/3])
        q = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=-q_bound, maxval=q_bound
        )
        qd = jax.random.uniform(
            rng2, (self.sys.qd_size(),), minval=-0.01, maxval=0.01
        )
        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)
        reward, done, reached_steps = jp.zeros(3)
        metrics = {
            'reached_steps': reached_steps,
        }

        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jp.ndarray) -> State:
        """Run one timestep of the environment's dynamics."""

        thrust = (action[..., 0]+1.0)/2.0
        torque = action[..., 1]
        angle = state.pipeline_state.q[2]
        force_x = thrust * jp.sin(angle)
        force_y = thrust * jp.cos(angle)
        action_applied = jp.stack([force_x, force_y, torque], axis=-1)
        pipeline_state = self.pipeline_step(
            state.pipeline_state, action_applied)

        obs = self._get_obs(pipeline_state)
        theta = pipeline_state.q[2]
        err_x = jp.linalg.norm(pipeline_state.q[:2])
        vel = jp.linalg.norm(pipeline_state.qd[:2])
        omega = jp.abs(pipeline_state.qd[2])
        reward = 1.5 - err_x - vel * 0.2 - \
            omega * 0.1 - jp.clip((theta-jp.pi*0.45)*5.0, 0.0, 1.0)
        state.metrics.update(
            reached_steps=jp.where(err_x<0.1, state.metrics['reached_steps']+1.0, 0.0)
        )
        done = jp.where(
            (err_x > 3.0) | (theta > jp.pi/2), 1.0, 0.0)

        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

    @property
    def action_size(self):
        return 2

    def _get_obs(self, pipeline_state: base.State) -> jp.ndarray:
        """Observe cartpole body position and velocities."""
        return jp.concatenate([pipeline_state.q, pipeline_state.qd])
