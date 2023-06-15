from jax import numpy as jnp

from adaptive_control_gym.envs.jax_env.dynamics.utils import angle_normalize, EnvParams, EnvState, Action

def get_loose_dynamics():

    # dynamics (params, states) -> states_dot
    def loose_dynamics(env_params:EnvParams, env_state:EnvState, env_action:Action):
        params = [env_params.m, env_params.I, env_params.g, env_params.l, env_params.mo, env_params.delta_yh, env_params.delta_zh]
        states = [env_state.y, env_state.z, env_state.theta, env_state.phi, env_state.y_dot, env_state.z_dot, env_state.theta_dot, env_state.phi_dot]
        action = [env_action.thrust, env_action.tau]

        y_ddot = -env_action.thrust * jnp.sin(env_state.theta) / env_params.m
        z_ddot = env_action.thrust * jnp.cos(env_state.theta) / env_params.m - env_params.g
        theta_ddot = env_action.tau / env_params.I

        new_y_dot = env_state.y_dot + env_params.dt * y_ddot
        new_z_dot = env_state.z_dot + env_params.dt * z_ddot
        new_theta_dot = env_state.theta_dot + env_params.dt * theta_ddot
        new_y = env_state.y + env_params.dt * new_y_dot
        new_z = env_state.z + env_params.dt * new_z_dot
        new_theta = angle_normalize(env_state.theta + env_params.dt * new_theta_dot)

        # states = [new_y, new_z, new_theta, env_state.phi, new_y_dot, new_z_dot, new_theta_dot, env_state.phi_dot]

        delta_y_hook = env_params.delta_yh * jnp.cos(new_theta) - env_params.delta_zh * jnp.sin(new_theta)
        delta_z_hook = env_params.delta_yh * jnp.sin(new_theta) + env_params.delta_zh * jnp.cos(new_theta)
        y_hook = new_y + delta_y_hook
        z_hook = new_z + delta_z_hook
        y_hook_dot = new_y_dot - new_theta_dot * delta_z_hook
        z_hook_dot = new_z_dot + new_theta_dot * delta_y_hook

        new_y_obj_dot = env_state.y_obj_dot
        new_z_obj_dot = env_state.z_obj_dot - env_params.g * env_params.dt
        new_y_obj = env_state.y_obj + env_params.dt * new_y_obj_dot
        new_z_obj = env_state.z_obj + env_params.dt * new_z_obj_dot

        phi_th = -jnp.arctan2(y_hook - new_y_obj, z_hook - new_z_obj)
        new_phi = angle_normalize(phi_th - new_theta)

        y_obj2hook_dot = new_y_obj_dot - y_hook_dot
        z_obj2hook_dot = new_z_obj_dot - z_hook_dot
        phi_th_dot = y_obj2hook_dot * jnp.cos(phi_th) + z_obj2hook_dot * jnp.sin(phi_th)
        new_phi_dot = phi_th_dot - new_theta_dot

        new_l_rope = jnp.sqrt((y_hook - new_y_obj) ** 2 + (z_hook - new_z_obj) ** 2)

        env_state = env_state.replace(
            y=new_y,
            z=new_z,
            theta=new_theta,
            phi=new_phi,
            y_dot=new_y_dot,
            z_dot=new_z_dot,
            theta_dot=new_theta_dot,
            phi_dot=new_phi_dot,
            y_hook=y_hook,
            z_hook=z_hook,
            y_hook_dot=y_hook_dot,
            z_hook_dot=z_hook_dot,
            y_obj=new_y_obj,
            z_obj=new_z_obj,
            y_obj_dot=new_y_obj_dot,
            z_obj_dot=new_z_obj_dot,
            l_rope=new_l_rope,
            f_rope=0.0,
            f_rope_y=0.0,
            f_rope_z=0.0,
            last_thrust=env_action.thrust,
            last_tau=env_action.tau,
            time=env_state.time + 1,
            y_tar=env_state.y_traj[env_state.time],
            z_tar=env_state.z_traj[env_state.time],
            y_dot_tar=env_state.y_dot_traj[env_state.time],
            z_dot_tar=env_state.z_dot_traj[env_state.time],
        )

        return env_state
    
    return loose_dynamics