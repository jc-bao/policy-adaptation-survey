import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces
from typing import Tuple, Optional
import chex
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from functools import partial

from adaptive_control_gym.envs.jax_env.dynamics.utils import angle_normalize, EnvParams, EnvState, Action
from adaptive_control_gym.envs.jax_env.dynamics.taut import get_taut_dynamics
from adaptive_control_gym.envs.jax_env.dynamics.loose import get_loose_dynamics
from adaptive_control_gym.envs.jax_env.dynamics.trans import get_dynamic_transfer


class Quad2D(environment.Environment):
    """
    JAX Compatible version of Quad2D-v0 OpenAI gym environment. Source:
    github.com/openai/gym/blob/master/gym/envs/classic_control/Quad2D.py
    """

    def __init__(self):
        super().__init__()
        # self.obs_shape = (24,)
        self.taut_dynamics = get_taut_dynamics()
        self.loose_dynamics = get_loose_dynamics()
        self.dynamic_transfer = get_dynamic_transfer()

    @property
    def default_params(self) -> EnvParams:
        """Default environment parameters for Quad2D-v0."""
        return EnvParams()
    
    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: float,
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        thrust = (action[0] + 1.0) / 2.0 * params.max_thrust
        tau = action[1] * params.max_torque
        err_pos = jnp.sqrt((state.y_tar - state.y_obj) ** 2 + (state.z_tar - state.z_obj) ** 2)
        err_vel = jnp.sqrt((state.y_dot_tar - state.y_obj_dot) ** 2 + (state.z_dot_tar - state.z_obj_dot) ** 2) 
        reward = 1.0 - 0.8 * err_pos - 0.05 * err_vel
        reward = reward.squeeze()
        env_action = Action(thrust=thrust, tau=tau)

        old_loose_state = state.l_rope < (params.l - params.rope_taut_therehold)
        taut_state = self.taut_dynamics(params, state, env_action)
        loose_state = self.loose_dynamics(params, state, env_action)
        new_state = self.dynamic_transfer(params, loose_state, taut_state, old_loose_state)

        done = self.is_terminal(state, params)
        return (
            lax.stop_gradient(self.get_obs(new_state)),
            lax.stop_gradient(new_state),
            reward,
            done,
            {
                "discount": self.discount(new_state, params), 
                "err_pos": err_pos,
                "err_vel": err_vel,
            },
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Reset environment state by sampling theta, theta_dot."""
        # generate reference trajectory by adding a few sinusoids together
        y_traj, z_traj, y_dot_traj, z_dot_traj = self.generate_traj(key)

        high = jnp.array([1, 1, jnp.pi / 3])
        y, z, theta = jax.random.uniform(key, shape=(3,), minval=-high, maxval=high)
        y_hook = y + params.delta_yh * jnp.cos(theta) - params.delta_zh * jnp.sin(theta) 
        z_hook = z + params.delta_yh * jnp.sin(theta) + params.delta_zh * jnp.cos(theta)
        state = EnvState(
            y=y, z=z, theta=theta, 
            y_dot=0.0, z_dot=0.0, theta_dot=0.0,last_thrust=0.0,last_tau=0.0,time=0,
            y_traj=y_traj,z_traj=z_traj,y_dot_traj=y_dot_traj,z_dot_traj=z_dot_traj,y_tar=y_traj[0],z_tar=z_traj[0],y_dot_tar=y_dot_traj[0],z_dot_tar=z_dot_traj[0],
            phi=0.0,phi_dot=0.0,
            y_hook=y_hook,z_hook=z_hook,y_hook_dot=0.0,z_hook_dot=0.0,
            y_obj=y_hook + params.l * jnp.sin(theta), z_obj = z_hook - params.l * jnp.cos(theta), y_obj_dot=0.0, z_obj_dot=0.0,
            f_rope=0.0, f_rope_y=0.0, f_rope_z=0.0,l_rope=params.l,
        )
        return self.get_obs(state), state
    
    @partial(jax.jit, static_argnums=(0,))
    def sample_params(self, key: chex.PRNGKey) -> EnvParams:
        """Sample environment parameters."""
        
        key, key1, key2, key3, key4, key5, key6 = jax.random.split(key, 7)
        
        m = jax.random.uniform(key1, shape=(), minval=0.025, maxval=0.04)
        I = jax.random.uniform(key2, shape=(), minval=2.5e-5, maxval=3.5e-5)
        mo = jax.random.uniform(key3, shape=(), minval=0.003, maxval=0.01)
        l = jax.random.uniform(key4, shape=(), minval=0.2, maxval=0.4)
        delta_yh = jax.random.uniform(key5, shape=(), minval=-0.04, maxval=0.04)
        delta_zh = jax.random.uniform(key6, shape=(), minval=0.0, maxval=0.06)
        
        return EnvParams(m=m, I=I, mo=mo, l=l, delta_yh=delta_yh, delta_zh=delta_zh)

    def generate_traj(self, key: chex.PRNGKey) -> chex.Array:
        # get random attitude and phase
        key_amp_y, key_phase_y, key_amp_z, key_phase_z = jax.random.split(key, 4)
        rand_amp_y = jax.random.uniform(key_amp_y, shape=(2,), minval=-1.0, maxval=1.0)
        rand_amp_z = jax.random.uniform(key_amp_z, shape=(2,), minval=-1.0, maxval=1.0)
        rand_phase_y = jax.random.uniform(
            key_phase_y, shape=(2,), minval=-jnp.pi, maxval=jnp.pi
        )
        rand_phase_z = jax.random.uniform(
            key_phase_z, shape=(2,), minval=-jnp.pi, maxval=jnp.pi
        )
        # get trajectory
        scale = 0.8
        ts = jnp.arange(
            0, self.default_params.max_steps_in_episode + 50, self.default_params.dt
        )  # NOTE: do not use params for jax limitation
        w1 = 2 * jnp.pi * 0.25
        w2 = 2 * jnp.pi * 0.5
        y_traj = scale * rand_amp_y[0] * jnp.sin(
            w1 * ts + rand_phase_y[0]
        ) + scale * rand_amp_y[1] * jnp.sin(w2 * ts + rand_phase_y[1])
        z_traj = scale * rand_amp_z[0] * jnp.sin(
            w1 * ts + rand_phase_z[0]
        ) + scale * rand_amp_z[1] * jnp.sin(w2 * ts + rand_phase_z[1])
        y_dot_traj = scale * rand_amp_y[0] * w1 * jnp.cos(
            w1 * ts + rand_phase_y[0]
        ) + scale * rand_amp_y[1] * w2 * jnp.cos(w2 * ts + rand_phase_y[1])
        z_dot_traj = scale * rand_amp_z[0] * w1 * jnp.cos(
            w1 * ts + rand_phase_z[0]
        ) + scale * rand_amp_z[1] * w2 * jnp.cos(w2 * ts + rand_phase_z[1])
        return y_traj, z_traj, y_dot_traj, z_dot_traj

    def get_obs(self, state: EnvState, params: EnvParams) -> chex.Array:
        """Return angle in polar coordinates and change."""
        
        obs_elements = [
            state.y,
            state.z,
            state.theta,
            state.y_dot / 4.0,
            state.z_dot / 4.0,
            state.theta_dot / 40.0,
            state.y_tar,
            state.z_tar,
            state.y_dot_tar / 4.0,
            state.z_dot_tar / 4.0,
            state.y_tar - state.y,
            state.z_tar - state.z,
            (state.y_dot_tar - state.y_dot) / 4.0,
            (state.z_dot_tar - state.z_dot) / 4.0,
            state.phi,
            state.phi_dot / 10.0,
            state.y_obj,
            state.z_obj,
            state.y_obj_dot / 4.0,
            state.z_obj_dot / 4.0,
            state.y_hook,
            state.z_hook,
            state.y_hook_dot / 4.0,
            state.z_hook_dot / 4.0,
        ]
        # future trajectory observation 
        # start arange from step+1, end at step+params.traj_obs_len*params.traj_obs_gap+1, with gap params.traj_obs_gap as step
        future_traj_idx = jnp.arange(state.time+1, state.time+params.traj_obs_len*params.traj_obs_gap+1, params.traj_obs_gap)
        obs_elements.append(*state.y_traj[future_traj_idx])
        obs_elements.append(*state.z_traj[future_traj_idx])
        obs_elements.append(*state.y_dot_traj[future_traj_idx] / 4.0)
        obs_elements.append(*state.z_dot_traj[future_traj_idx] / 4.0)

        # parameter observation
        param_elements = [
            (params.m-0.025)/(0.04-0.025) * 2.0 - 1.0,
            (params.I-2.5e-4)/(3.5e-4 - 2.5e-4) * 2.0 - 1.0,
            (params.mo-0.003)/(0.01-0.003) * 2.0 - 1.0,
            (params.l-0.2)/(0.4-0.2) * 2.0 - 1.0,
            (params.delta_yh-0.04)/(0.04-(-0.04)) * 2.0 - 1.0,
            (params.delta_zh-0.0)/(0.06-0.0) * 2.0 - 1.0,
        ]

        
        return jnp.array(obs_elements+param_elements).squeeze()

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        # Check number of steps in episode termination condition
        done = (
            (state.time >= params.max_steps_in_episode)
            | (jnp.abs(state.y) > 2.0)
            | (jnp.abs(state.z) > 2.0)
            | (jnp.abs(state.theta_dot) > 100.0)
            | (jnp.abs(state.phi_dot) > 100.0)
        )
        return done

    @property
    def name(self) -> str:
        """Environment name."""
        return "Quad2D-v1"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 2

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        """Action space of the environment."""
        if params is None:
            params = self.default_params
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=jnp.float32,
        )

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(-1.0, 1.0, shape=(24+params.traj_obs_len*4+6,), dtype=jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "y": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (),
                    jnp.float32,
                ),
                "z": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (),
                    jnp.float32,
                ),
                "y_dot": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (),
                    jnp.float32,
                ),
                "z_dot": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (),
                    jnp.float32,
                ),
                "theta": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (),
                    jnp.float32,
                ),
                "theta_dot": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (),
                    jnp.float32,
                ),
                "last_thrust": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (),
                    jnp.float32,
                ),
                "last_tau": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (),
                    jnp.float32,
                ),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )

def test_env(env: Quad2D, policy, render_video=False):
    rng = jax.random.PRNGKey(1)
    rng, rng_params = jax.random.split(rng)
    env_params = env.sample_params(rng_params)
    # env_params = env.default_params

    state_seq, obs_seq, reward_seq = [], [], []
    rng, rng_reset = jax.random.split(rng)
    obs, env_state = env.reset(rng_reset, env_params)
    n_dones = 0
    while True:
        state_seq.append(env_state)
        rng, rng_act, rng_step = jax.random.split(rng, 3)
        action = policy(obs, rng_act)
        next_obs, next_env_state, reward, done, info = env.step(
            rng_step, env_state, action, env_params
        )
        if done:
            rng, rng_params = jax.random.split(rng)
            env_params = env.sample_params(rng_params)
        reward_seq.append(reward)
        obs_seq.append(obs)
        if done:
            n_dones += 1
        obs = next_obs
        env_state = next_env_state
        if n_dones >= 1:
            break

    # plot trajectory
    def update_plot(frame_num):
        plt.gca().clear()
        frame_list = np.arange(np.max([0, frame_num - 200]), frame_num + 1)
        plt.plot([s.y_obj for s in state_seq[frame_list[0]:frame_list[-1]]], [s.z_obj for s in state_seq[frame_list[0]:frame_list[-1]]], alpha=0.5)
        plt.plot([s.y_tar for s in state_seq], [s.z_tar for s in state_seq], "--", alpha=0.3)
        
        start = max(0, frame_num)
        for i in range(start, frame_num + 1):
            num_steps = max(frame_num - start, 1)
            alpha = 1 if i == frame_num else ((i-start) / num_steps * 0.1)
            plt.arrow(
                state_seq[i].y,
                state_seq[i].z,
                -0.1 * jnp.sin(state_seq[i].theta),
                0.1 * jnp.cos(state_seq[i].theta),
                width=0.01,
                color="b",
                alpha=alpha,
            )
            # plot object as point
            plt.plot(state_seq[i].y_obj, state_seq[i].z_obj, "o", color="b", alpha=alpha)
            # plot hook as cross
            plt.plot(state_seq[i].y_hook, state_seq[i].z_hook, "x", color="r", alpha=alpha)
            # plot rope as line (gree if slack, red if taut)
            plt.arrow(
                state_seq[i].y_hook,
                state_seq[i].z_hook,
                state_seq[i].y_obj - state_seq[i].y_hook,
                state_seq[i].z_obj - state_seq[i].z_hook,
                width=0.01,
                color="r" if state_seq[i].l_rope > (env_params.l - env_params.rope_taut_therehold) else "g",
                alpha=alpha,
            )
            # plot y_tar and z_tar with red dot
            plt.plot(state_seq[i].y_tar, state_seq[i].z_tar, "ro", alpha=alpha)
        plt.xlabel("y")
        plt.ylabel("z")
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])

    if render_video:
        plt.figure(figsize=(4, 4))
        anim = FuncAnimation(plt.gcf(), update_plot, frames=len(state_seq), interval=20)
        anim.save(filename="../results/anim.gif", writer="imagemagick", fps=int(1.0/env_params.dt))

    num_figs = len(state_seq[0].__dict__) + 2
    time = [s.time * env_params.dt for s in state_seq]

    # calculate number of rows needed
    num_rows = int(jnp.ceil(num_figs / 6))

    # create num_figs subplots
    plt.subplots(num_rows, 6, figsize=(4 * 6, 2 * num_rows))

    # plot reward
    plt.subplot(num_rows, 6, 1)
    plt.plot(time, reward_seq)
    plt.ylabel("reward")

    # plot obs
    plt.subplot(num_rows, 6, 2)
    for i in range(len(obs_seq[0])):
        plt.plot(time, [o[i] for o in obs_seq], label=f"obs[{i}]")
    plt.ylabel("obs")
    plt.legend(fontsize=3)

    # plot state
    current_fig = 3
    for i, (name, value) in enumerate(state_seq[0].__dict__.items()):
        if name in ["y_traj", "z_traj", "y_dot_traj", "z_dot_traj", "theta_traj"]:
            continue
        current_fig += 1
        plt.subplot(num_rows, 6, current_fig)
        plt.plot(time, [getattr(s, name) for s in state_seq])
        if name in ["y", "z", "y_dot", "z_dot"]:
            plt.plot(time, [s.__dict__[name + "_tar"] for s in state_seq], "--")
            plt.legend(["actual", "target"], fontsize=3)
        plt.ylabel(name)

    plt.xlabel("time")
    plt.savefig("../results/plot.png")

if __name__=='__main__':
    env = Quad2D()

    def pid_policy(obs, rng):
        y = obs[0]
        z = obs[1]
        theta = obs[2]
        y_dot = obs[3] * 4.0
        z_dot = obs[4] * 4.0
        theta_dot = obs[5] * 40.0
        y_tar = obs[6] * 0.0 # DEBUG
        z_tar = obs[7] * 0.0
        y_dot_tar = obs[8] * 4.0 * 0.0
        z_dot_tar = obs[9] * 4.0 * 0.0

        w0 = 10.0
        zeta = 0.95
        kp = env.default_params.m * (w0**2)
        kd = env.default_params.m * 2.0 * zeta * w0
        target_force_y = kp * (y_tar - y) + kd * (y_dot_tar - y_dot)
        target_force_z = (
            kp * (z_tar - z)
            + kd * (z_dot_tar - z_dot)
            + (env.default_params.m + env.default_params.mo) * env.default_params.g
        )
        thrust = -target_force_y * jnp.sin(theta) + target_force_z * jnp.cos(theta)
        target_theta = -jnp.arctan2(target_force_y, target_force_z)

        w0 = 30.0
        zeta = 0.95
        tau = env.default_params.I * (
            (w0**2) * (target_theta - theta) + 2.0 * zeta * w0 * (0.0 - theta_dot)
        )

        # convert into action space
        thrust_normed = jnp.clip(
            thrust / env.default_params.max_thrust * 2.0 - 1.0, -1.0, 1.0
        )
        tau_normed = jnp.clip(tau / env.default_params.max_torque, -1.0, 1.0)
        return jnp.array([thrust_normed, tau_normed])

    random_policy = lambda obs, rng: env.action_space(env.default_params).sample(rng)

    # with jax.disable_jit():
    print('starting test...')
    test_env(env, policy=pid_policy, render_video=True)