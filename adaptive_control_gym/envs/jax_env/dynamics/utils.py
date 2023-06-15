from flax import struct
from jax import numpy as jnp

@struct.dataclass
class EnvState:
    y: float
    z: float
    theta: float
    phi: float
    y_dot: float
    z_dot: float
    theta_dot: float
    phi_dot: float
    last_thrust: float  # Only needed for rendering
    last_tau: float  # Only needed for rendering
    time: int
    y_traj: jnp.ndarray
    z_traj: jnp.ndarray
    y_dot_traj: jnp.ndarray
    z_dot_traj: jnp.ndarray
    y_tar: float
    z_tar: float
    y_dot_tar: float
    z_dot_tar: float
    y_hook: float
    z_hook: float
    y_hook_dot: float
    z_hook_dot: float
    y_obj: float
    z_obj: float
    y_obj_dot: float
    z_obj_dot: float
    f_rope: float
    f_rope_y: float
    f_rope_z: float
    l_rope: float


@struct.dataclass
class EnvParams:
    max_speed: float = 8.0
    max_torque: float = 0.012
    max_thrust: float = 0.8
    dt: float = 0.02
    g: float = 9.81  # gravity
    m: float = 0.03  # mass
    I: float = 2.0e-5  # moment of inertia
    mo: float = 0.005  # mass of the object attached to the rod
    l: float = 0.3  # length of the rod
    delta_yh: float = 0.03  # y displacement of the hook from the quadrotor center
    delta_zh: float = -0.06  # z displacement of the hook from the quadrotor center
    max_steps_in_episode: int = 300
    rope_taut_therehold: float = 1e-4

@struct.dataclass
class Action:
    thrust: float
    tau: float

def angle_normalize(x: float) -> float:
    """Normalize the angle - radians."""
    return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi