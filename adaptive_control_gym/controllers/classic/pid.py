import numpy as np
import torch

class PID:
    def __init__(self) -> None:
        self.Kp = 2.0
        self.Kd = 2.0
        self.Krotp = 30.0
        self.Krotd = 8.0

    def __call__(self, state, info):
        decay_x, decay_y, decay_w = info['decay'].numpy()[0]
        disturb_x, disturb_y, disturb_w = info['disturb'].numpy()[0]
        mass_x, mass_y, moment = info['mass'].numpy()[0]
        x, y, theta = info['pos'].numpy()[0]
        vx, vy, w = info['vel'].numpy()[0]

        target_force_x = -mass_x * self.Kp * x - mass_x * (self.Kd-decay_x) * vx - disturb_x
        target_force_y = -mass_y * self.Kp * y - mass_y * (self.Kd-decay_y) * vy - disturb_y + mass_y * 9.8
        thrust = target_force_y * np.cos(theta) + target_force_x * np.sin(theta)

        target_angle = np.arctan2(target_force_x, target_force_y)
        torque = - moment * self.Krotp * (theta-target_angle) - moment * (self.Krotd - decay_w) * w - disturb_w


        return torch.Tensor([[thrust, torque]])