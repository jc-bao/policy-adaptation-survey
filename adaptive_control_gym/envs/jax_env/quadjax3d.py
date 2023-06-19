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
from dataclasses import dataclass as pydataclass
import tyro

from adaptive_control_gym.envs.jax_env.dynamics.utils import get_hit_penalty, EnvParams3D, EnvState3D, Action3D
from adaptive_control_gym.envs.jax_env.dynamics.loose import get_loose_dynamics_3d


class Quad3D(environment.Environment):
    """
    JAX Compatible version of Quad3D-v0 OpenAI gym environment. Source:
    github.com/openai/gym/blob/master/gym/envs/classic_control/Quad3D.py
    """

    def __init__(self, task: str = "tracking_zigzag"):
        super().__init__()
        self.task = task
        # reference trajectory function
        if task == "tracking":
            self.generate_traj = self.generate_lissa_traj
        elif task == "tracking_zigzag":
            self.generate_traj = self.generate_zigzag_traj
        elif task in ["jumping", "hovering"]:
            self.generate_traj = self.generate_fixed_traj
        else:
            raise NotImplementedError
        # dynamics
        self.taut_dynamics = None
        self.loose_dynamics = get_loose_dynamics_3d
        self.dynamic_transfer = None
        # controllers

    @property
    def default_params(self) -> EnvParams3D:
        """Default environment parameters for Quad3D-v0."""
        return EnvParams3D()

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState3D,
        action: float,
        params: EnvParams3D,
    ) -> Tuple[chex.Array, EnvState3D, float, bool, dict]:
        thrust = (action[0] + 1.0) / 2.0 * params.max_thrust
        torque = action[1:4] * params.max_torque
        err_pos = jnp.linalg.norm(state.pos_tar - state.pos)
        err_vel = jnp.linalg.norm(state.vel_tar - state.vel)
        if self.task == "jumping":
            raise NotImplementedError
            drone_panelty = get_hit_penalty(state.y, state.z) * 3.0
            obj_panelty = get_hit_penalty(state.y_obj, state.z_obj) * 3.0
            reward = 1.0 - 0.6 * err_pos - 0.15 * err_vel \
                + (drone_panelty + obj_panelty)
        elif self.task == 'hovering':
            reward = 1.0 - 0.6 * err_pos - 0.1 * err_vel
        else:
            raise NotImplementedError
            reward = 1.0 - 0.8 * err_pos - 0.05 * err_vel
        reward = reward.squeeze()
        env_action = Action3D(thrust=thrust, torque=torque)

        # old_loose_state = state.l_rope < (
        #     params.l - params.rope_taut_therehold)
        # taut_state = self.taut_dynamics(params, state, env_action)
        # loose_state = self.loose_dynamics(params, state, env_action)
        # new_state = self.dynamic_transfer(
        #     params, loose_state, taut_state, old_loose_state)

        new_state = self.loose_dynamics(params, state, env_action)

        done = self.is_terminal(state, params)
        return (
            lax.stop_gradient(self.get_obs(new_state, params)),
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
        self, key: chex.PRNGKey, params: EnvParams3D
    ) -> Tuple[chex.Array, EnvState3D]:
        """Reset environment state by sampling theta, theta_dot."""
        # generate reference trajectory by adding a few sinusoids together
        pos_traj, vel_traj = self.generate_traj(key)
        # sample initial position
        pos_key = jax.random.split(key)[0]

        if self.task == 'jumping':
            pos = jnp.array([-1.0, 0.0, 0.0])
        else:
            pos = jax.random.uniform(
                pos_key, shape=(3,), minval=-1.0, maxval=1.0)
        pos_hook = pos + params.hook_offset
        pos_obj = pos + jnp.array([0.0, -params.l, 0.0])
        zeros3 = jnp.zeros(3)
        state = EnvState3D(
            # drone
            pos=pos, vel=zeros3, omega=zeros3, quat=jnp.concatenate(
                [zeros3, jnp.array([1.0])]),
            # object
            pos_obj=pos_obj, vel_obj=zeros3,
            # hook
            pos_hook=pos_hook, vel_hook=zeros3,
            # rope
            l_rope=params.l, zeta=jnp.array([0.0, 0.0, -1.0]), zeta_dot=zeros3,
            f_rope=zeros3, f_rope_norm=0.0,
            # trajectory
            pos_tar=pos_traj[0], vel_tar=vel_traj[0],
            pos_traj=pos_traj, vel_traj=vel_traj,
            # debug value
            last_thrust=0.0, last_torque=zeros3,
            # step
            time=0,
        )
        return self.get_obs(state, params), state

    @partial(jax.jit, static_argnums=(0,))
    def sample_params(self, key: chex.PRNGKey) -> EnvParams3D:
        """Sample environment parameters."""

        key, key1, key2, key3, key4, key5, key6 = jax.random.split(key, 7)

        m = jax.random.uniform(key1, shape=(), minval=0.025, maxval=0.04)
        I = jax.random.uniform(key2, shape=(), minval=2.5e-5, maxval=3.5e-5)
        mo = jax.random.uniform(key3, shape=(), minval=0.003, maxval=0.01)
        l = jax.random.uniform(key4, shape=(), minval=0.2, maxval=0.4)
        delta_yh = jax.random.uniform(
            key5, shape=(), minval=-0.04, maxval=0.04)
        delta_zh = jax.random.uniform(
            key6, shape=(), minval=-0.06, maxval=0.00)

        return EnvParams3D(m=m, I=I, mo=mo, l=l, delta_yh=delta_yh, delta_zh=delta_zh)

    @partial(jax.jit, static_argnums=(0,))
    def generate_fixed_traj(self, key: chex.PRNGKey) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        ts = jnp.arange(
            0, self.default_params.max_steps_in_episode + 50, self.default_params.dt
        )
        key_y, key_z, key_sign = jax.random.split(key, 3)
        sign = jax.random.choice(key_sign, jnp.array([-1.0, 1.0]))
        y = jax.random.uniform(key_y, shape=(), minval=0.12, maxval=1.0)
        z = jax.random.uniform(key_z, shape=(), minval=-1.0, maxval=1.0)
        y_traj = jnp.zeros_like(ts) + sign * y
        z_traj = jnp.zeros_like(ts) + z
        y_dot_traj = jnp.zeros_like(ts)
        z_dot_traj = jnp.zeros_like(ts)
        return y_traj, z_traj, y_dot_traj, z_dot_traj

    @partial(jax.jit, static_argnums=(0,))
    def generate_lissa_traj(self, key: chex.PRNGKey) -> chex.Array:
        # get random amplitude and phase
        key_amp, key_phase = jax.random.split(key, 2)
        rand_amp = jax.random.uniform(
            key_amp, shape=(3, 2), minval=-1.0, maxval=1.0)
        rand_phase = jax.random.uniform(
            key_phase, shape=(3, 2), minval=-jnp.pi, maxval=jnp.pi
        )
        # get trajectory
        scale = 0.8
        ts = jnp.arange(
            0, self.default_params.max_steps_in_episode + 50, self.default_params.dt
        )  # NOTE: do not use params for jax limitation
        w1 = 2 * jnp.pi * 0.25
        w2 = 2 * jnp.pi * 0.5

        pos_traj = scale * jnp.stack([
            rand_amp[i, 0] * jnp.sin(w1 * ts + rand_phase[i, 0]) +
            rand_amp[i, 1] * jnp.sin(w2 * ts + rand_phase[i, 1])
            for i in range(3)
        ])

        vel_traj = scale * jnp.stack([
            rand_amp[i, 0] * w1 * jnp.cos(w1 * ts + rand_phase[i, 0]) +
            rand_amp[i, 1] * w2 * jnp.cos(w2 * ts + rand_phase[i, 1])
            for i in range(3)
        ])

        return pos_traj, vel_traj

    @partial(jax.jit, static_argnums=(0,))
    def generate_zigzag_traj(self, key: chex.PRNGKey) -> chex.Array:
        point_per_seg = 40
        num_seg = self.default_params.max_steps_in_episode // point_per_seg + 1

        key_keypoints = jax.random.split(key, num_seg)
        key_angles = jax.random.split(key, num_seg)

        # sample from 3d -1.5 to 1.5
        prev_point = jax.random.uniform(
            key_keypoints[0], shape=(3,), minval=-1.5, maxval=1.5)

        def update_fn(carry, i):
            key_keypoint, key_angle, prev_point = carry

            # Calculate the unit vector pointing to the center
            vec_to_center = - prev_point / jnp.linalg.norm(prev_point)

            # Sample random rotation angles for theta and phi from [-pi/3, pi/3]
            delta_theta, delta_phi = jax.random.uniform(
                key_angle, shape=(2,), minval=-jnp.pi/3, maxval=jnp.pi/3)

            # Calculate new direction
            theta = jnp.arccos(vec_to_center[2]) + delta_theta
            phi = jnp.arctan2(vec_to_center[1], vec_to_center[0]) + delta_phi
            new_direction = jnp.array(
                [jnp.sin(theta) * jnp.cos(phi), jnp.sin(theta) * jnp.sin(phi), jnp.cos(theta)])

            # Sample the distance from [1.5, 2.5]
            distance = jax.random.uniform(key_keypoint, minval=1.5, maxval=2.5)

            # Calculate the new point
            next_point = prev_point + distance * new_direction

            point_traj_seg = jnp.stack([jnp.linspace(
                prev, next_p, point_per_seg, endpoint=False) for prev, next_p in zip(prev_point, next_point)], axis=-1)
            point_dot_traj_seg = (next_point - prev_point) / \
                (point_per_seg + 1) * jnp.ones((point_per_seg, 3))

            carry = (key_keypoints[i+1], key_angles[i+1], next_point)
            return carry, (point_traj_seg, point_dot_traj_seg)

        initial_carry = (key_keypoints[1], key_angles[1], prev_point)
        _, (point_traj_segs, point_dot_traj_segs) = lax.scan(
            update_fn, initial_carry, jnp.arange(1, num_seg))

        return jnp.concatenate(point_traj_segs, axis=-1), jnp.concatenate(point_dot_traj_segs, axis=-1)

    def get_obs(self, state: EnvState3D, params: EnvParams3D) -> chex.Array:
        """Return angle in polar coordinates and change."""

        env_state = env_state.replace(
            # drone
            pos=pos, vel=vel, omega=omega, quat=quat,
            # object
            pos_obj=pos_obj, vel_obj=vel_obj,
            # hook
            pos_hook=pos_hook, vel_hook=vel_hook,
            # rope
            l_rope=l_rope, zeta=zeta, zeta_dot=zeta_dot,
            f_rope=f_rope, f_rope_norm=f_rope_norm,
            # trajectory
            pos_tar=pos_tar, vel_tar=vel_tar,
            # debug value
            last_thrust=last_thrust, last_torque=last_torque,
            # step
            time=time,
        )

        obs_elements = [
            # drone
            state.pos, state.vel/4.0, state.quat, state.omega/40.0,  # 3*3+4=13
            # object
            state.pos_obj, state.vel_obj/4.0,  # 3*2=6
            # hook
            state.pos_hook, state.vel_hook/4.0,  # 3*2=6
            # rope
            state.l_rope, state.zeta, state.zeta_dot,  # 3*3=9
            state.f_rope, jnp.expand_dims(state.f_rope_norm, axis=0),  # 3+1=4
            # trajectory
            state.pos_tar, state.vel_tar/4.0  # 3*2=6
        ]  # 13+6+6+9+4+6=44
        # future trajectory observation
        # NOTE: use statis default value rather than params for jax limitation
        traj_obs_len, traj_obs_gap = self.default_params.traj_obs_len, self.default_params.traj_obs_gap
        for i in range(traj_obs_len):  # 6*traj_obs_len
            idx = state.time + 1 + i * traj_obs_gap
            obs_elements.append(state.pos_traj[idx]),
            obs_elements.append(state.vel_traj[idx]/4.0)

        # parameter observation
        param_elements = [
            jnp.array([
                (params.m-0.025)/(0.04-0.025) * 2.0 - 1.0,
                (params.I-2.5e-5)/(3.5e-5 - 2.5e-5) * 2.0 - 1.0,
                (params.mo-0.003)/(0.01-0.003) * 2.0 - 1.0,
                (params.l-0.2)/(0.4-0.2) * 2.0 - 1.0]),  # 4
            (params.hook_offset-(-0.06))/(0.0-(-0.06)) * 2.0 - 1.0,  # 3
        ]  # 4+3=7

        return jnp.concatenate(obs_elements+param_elements).squeeze()

    def is_terminal(self, state: EnvState3D, params: EnvParams3D) -> bool:
        """Check whether state is terminal."""
        # Check number of steps in episode termination condition
        done = (
            (state.time >= params.max_steps_in_episode)
            | (jnp.abs(state.pos) > 2.0).any()
            | (jnp.abs(state.omega) > 100.0).any()
        )
        return done

    @property
    def name(self) -> str:
        """Environment name."""
        return "Quad3D-v1"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 2

    def action_space(self, params: Optional[EnvParams3D] = None) -> spaces.Box:
        """Action3D space of the environment."""
        if params is None:
            params = self.default_params
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=jnp.float32,
        )

    def observation_space(self, params: EnvParams3D) -> spaces.Box:
        """Observation space of the environment."""
        # NOTE: use default params for jax limitation
        return spaces.Box(-1.0, 1.0, shape=(44+self.default_params.traj_obs_len*6+7,), dtype=jnp.float32)


def test_env(env: Quad3D, policy, render_video=False):
    rng = jax.random.PRNGKey(1)
    rng, rng_params = jax.random.split(rng)
    env_params = env.sample_params(rng_params)

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
        plt.plot([s.y_obj for s in state_seq[frame_list[0]:frame_list[-1]]],
                 [s.z_obj for s in state_seq[frame_list[0]:frame_list[-1]]], alpha=0.5)
        plt.plot([s.y_tar for s in state_seq], [
                 s.z_tar for s in state_seq], "--", alpha=0.3)

        if env.task == 'jumping':
            hy, hz = 0.05, 0.3
            square1 = [[hy, hz], [hy, 2.0], [-hy, 2.0], [-hy, hz], [hy, hz]]
            square2 = [[hy, -hz], [hy, -2.0],
                       [-hy, -2.0], [-hy, -hz], [hy, -hz]]
            for square in [square1, square2]:
                x, y = zip(*square)
                plt.plot(x, y, linestyle='-')

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
            plt.plot(state_seq[i].y_obj, state_seq[i].z_obj,
                     "o", color="b", alpha=alpha)
            # plot hook as cross
            plt.plot(state_seq[i].y_hook, state_seq[i].z_hook,
                     "x", color="r", alpha=alpha)
            # plot rope as line (gree if slack, red if taut)
            plt.arrow(
                state_seq[i].y_hook,
                state_seq[i].z_hook,
                state_seq[i].y_obj - state_seq[i].y_hook,
                state_seq[i].z_obj - state_seq[i].z_hook,
                width=0.01,
                color="r" if state_seq[i].l_rope > (
                    env_params.l - env_params.rope_taut_therehold) else "g",
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
        anim = FuncAnimation(plt.gcf(), update_plot,
                             frames=len(state_seq), interval=20)
        anim.save(filename="../results/anim.gif",
                  writer="imagemagick", fps=int(1.0/env_params.dt))

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
    # plot 10 obs in a subplot
    current_fig = 2
    for i in range(len(obs_seq[0])//10+1):
        plt.subplot(num_rows, 6, current_fig)
        current_fig += 1
        for j in range(10):
            idx = i*10+j
            plt.plot(time, [o[idx] for o in obs_seq], label=f"{idx}")
        plt.ylabel("obs")
        plt.legend(fontsize=6, ncol=2)

    # plot state
    for i, (name, value) in enumerate(state_seq[0].__dict__.items()):
        if name in ["y_traj", "z_traj", "y_dot_traj", "z_dot_traj", "theta_traj"]:
            continue
        current_fig += 1
        plt.subplot(num_rows, 6, current_fig)
        plt.plot(time, [getattr(s, name) for s in state_seq])
        if name in ["y", "z", "y_dot", "z_dot"]:
            plt.plot(time, [s.__dict__[name + "_tar"]
                     for s in state_seq], "--")
            plt.legend(["actual", "target"], fontsize=3)
        plt.ylabel(name)

    plt.xlabel("time")
    plt.savefig("../results/plot.png")


@pydataclass
class Args:
    task: str = "tracking"
    render: bool = False


def main(args: Args):
    env = Quad3D(task=args.task)

    def pid_policy(obs, rng):
        y = obs[0]
        z = obs[1]
        theta = obs[2]
        y_dot = obs[3] * 4.0
        z_dot = obs[4] * 4.0
        theta_dot = obs[5] * 40.0
        y_obj = obs[6]
        y_tar = obs[6]  # DEBUG
        z_tar = obs[7]
        y_dot_tar = obs[8] * 4.0
        z_dot_tar = obs[9] * 4.0
        y_obj = obs[16]
        z_obj = obs[17]
        y_obj_dot = obs[18] * 4.0
        z_obj_dot = obs[19] * 4.0
        m = obs[25+5*4+0] * (0.04-0.025)/2.0 + (0.04+0.025)/2.0
        I = obs[25+5*4+1] * (3.5e-5 - 2.5e-5)/2.0 + (3.5e-5 + 2.5e-5)/2.0
        mo = obs[25+5*4+2] * (0.01-0.003)/2.0 + (0.01+0.003)/2.0
        l = obs[25+5*4+3] * (0.4-0.2)/2.0 + (0.4+0.2)/2.0
        delta_yh = obs[25+5*4+4] * (0.04-(-0.04))/2.0 + (0.04+(-0.04))/2.0
        delta_zh = obs[25+5*4+5] * (0.0-(-0.06))/2.0 + (0.0+(-0.06))/2.0

        # get object target force
        w0 = 8.0
        zeta = 0.95
        kp = mo * (w0**2)
        kd = mo * 2.0 * zeta * w0
        target_force_y_obj = kp * (y_tar - y_obj) + \
            kd * (y_dot_tar - y_obj_dot)
        target_force_z_obj = kp * (z_tar - z_obj) + \
            kd * (z_dot_tar - z_obj_dot) + mo * 9.81
        phi_tar = -jnp.arctan2(target_force_y_obj, target_force_z_obj)
        y_drone_tar = y_obj - l * jnp.sin(phi_tar) - delta_yh
        z_drone_tar = z_obj + l * jnp.cos(phi_tar) - delta_zh

        # get drone target force
        w0 = 8.0
        zeta = 0.95
        kp = m * (w0**2)
        kd = m * 2.0 * zeta * w0
        target_force_y = kp * (y_drone_tar - y) + kd * \
            (y_dot_tar - y_dot) + target_force_y_obj
        target_force_z = (
            kp * (z_drone_tar - z)
            + kd * (z_dot_tar - z_dot)
            + m * 9.81
        ) + target_force_z_obj
        thrust = -target_force_y * \
            jnp.sin(theta) + target_force_z * jnp.cos(theta)
        thrust = jnp.sqrt(target_force_y**2 + target_force_z**2)
        target_theta = -jnp.arctan2(target_force_y, target_force_z)

        w0 = 30.0
        zeta = 0.95
        tau = env.default_params.I * (
            (w0**2) * (target_theta - theta) +
            2.0 * zeta * w0 * (0.0 - theta_dot)
        )

        # convert into action space
        thrust_normed = jnp.clip(
            thrust / env.default_params.max_thrust * 2.0 - 1.0, -1.0, 1.0
        )
        tau_normed = jnp.clip(tau / env.default_params.max_torque, -1.0, 1.0)
        return jnp.array([thrust_normed, tau_normed])

    def random_policy(obs, rng): return env.action_space(
        env.default_params).sample(rng)

    print('starting test...')
    # with jax.disable_jit():
    test_env(env, policy=pid_policy, render_video=args.render)


if __name__ == "__main__":
    main(tyro.cli(Args))
