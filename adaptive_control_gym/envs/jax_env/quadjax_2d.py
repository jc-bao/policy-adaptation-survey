import sympy as sp
from sympy.physics.mechanics import (
    dynamicsymbols,
    ReferenceFrame,
    Point,
    Particle,
    RigidBody,
    inertia,
    KanesMethod,
)
from jax import numpy as jnp

def get_taut_dynamics():
    # Define symbols
    t = sp.Symbol("t")  # time
    m = sp.Symbol("m", positive=True)  # mass of the quadrotor
    I = sp.Symbol("I", positive=True)  # moment of inertia
    g = sp.Symbol("g", positive=True)  # gravitational acceleration
    l = sp.Symbol("l", positive=True)  # length of the rod
    mo = sp.Symbol("mo", positive=True)  # mass of the object attached to the rod
    thrust = sp.Function("thrust")(t)  # thrust force
    tau = sp.Function("tau")(t)  # torque
    f_rope = sp.Symbol("f_rope")  # force in the rope
    delta_yh = sp.Symbol("delta_yh")  # y displacement of the hook from the quadrotor center
    delta_zh = sp.Symbol("delta_zh")  # z displacement of the hook from the quadrotor center

    # Define state variables and their derivatives
    y, z, theta, phi = dynamicsymbols("y z theta phi")
    y_dot, z_dot, theta_dot, phi_dot = y.dot(t), z.dot(t), theta.dot(t), phi.dot(t)
    y_ddot, z_ddot, theta_ddot, phi_ddot = y_dot.diff(t), z_dot.diff(t), theta_dot.diff(t), phi_dot.diff(t)

    # intermediate variables
    delta_yh_global = delta_yh * sp.cos(theta) - delta_yh * sp.sin(theta)
    delta_zh_global = delta_zh * sp.sin(theta) + delta_yh * sp.cos(theta)
    f_rope_y = f_rope * sp.sin(theta+phi)
    f_rope_z = -f_rope * sp.cos(theta+phi)
    y_hook = y + delta_yh_global
    z_hook = z + delta_zh_global
    y_obj = y_hook + l * sp.sin(theta+phi)
    z_obj = z_hook - l * sp.cos(theta+phi)
    y_obj_dot = y_dot.diff(t)
    z_obj_dot = z_dot.diff(t)
    y_obj_ddot = y_ddot.diff(t)
    z_obj_ddot = z_ddot.diff(t)

    # Define inertial reference frame
    N = ReferenceFrame("N")
    N_origin = Point('N_origin')
    A = N.orientnew("A", "Axis", [theta, N.x])
    B = A.orientnew("B", "Axis", [phi, A.x])

    # Define point
    drone = Point("drone")
    drone.set_pos(N_origin, y * N.y + z * N.z)
    hook = drone.locatenew("hook", delta_yh * A.y + delta_zh * A.z)
    obj = hook.locatenew("obj", -l * B.z)
    drone.set_vel(N, y_dot * N.y + z_dot * N.z)

    # Inertia
    inertia_quadrotor = inertia(N, I, 0, 0)
    quadrotor = RigidBody("quadrotor", drone, A, m, (inertia_quadrotor, drone))
    obj_particle = Particle("obj_particle", obj, mo)

    # Newton's law
    eq_quad_y = -thrust * sp.sin(theta) + f_rope_y - m * y_ddot
    eq_quad_z = thrust * sp.cos(theta) + f_rope_z + m * g - m * z_ddot
    eq_quad_theta = tau + delta_yh_global * f_rope_z - delta_zh_global * f_rope_y - I * theta_ddot
    eq_obj_y = -f_rope_y - mo * y_obj_ddot
    eq_obj_z = -f_rope_z - mo * g - mo * z_obj_ddot

    eqs = [eq_quad_y, eq_quad_z, eq_quad_theta, eq_obj_y, eq_obj_z]

    # Solve for the acceleration
    coeffs = [phi_ddot, theta_ddot, y_ddot, z_ddot, f_rope]
    A = sp.zeros(4, 4)
    b = sp.zeros(4, 1)
    for i in range(4):
        for j in range(4):
            A[i, j] = eqs[i].coeff(coeffs[j])
        b[i] = -eqs[i].subs([(coeffs[j], 0) for j in range(4)])
