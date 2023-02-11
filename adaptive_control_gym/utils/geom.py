import torch

def rpy2quat(rpy: torch.Tensor) -> torch.Tensor:
    # convert roll-pitch-yaw to quaternion with torch
    # rpy: (batch_size, 3)
    # quat: (batch_size, 4)
    roll, pitch, yaw = rpy[:, 0], rpy[:, 1], rpy[:, 2]
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    q = torch.zeros(rpy.shape[0], 4, device=rpy.device)
    q[:, 0] = cy * cp * cr + sy * sp * sr
    q[:, 1] = cy * cp * sr - sy * sp * cr
    q[:, 2] = sy * cp * sr + cy * sp * cr
    q[:, 3] = sy * cp * cr - cy * sp * sr
    return q

def quat2rpy(quad: torch.Tensor) -> torch.Tensor:
    # convert quaternion to roll-pitch-yaw with torch
    # quad: (batch_size, 4)
    # rpy: (batch_size, 3)
    x, y, z, w = quad[:, 0], quad[:, 1], quad[:, 2], quad[:, 3]
    rpy = torch.zeros(quad.shape[0], 3, device=quad.device)
    rpy[:, 0] = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    rpy[:, 1] = torch.asin(2 * (w * y - z * x))
    rpy[:, 2] = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    return rpy

def quat2rotmat(quad: torch.Tensor) -> torch.Tensor:
    # convert quaternion to rotation matrix with torch
    # quad: (batch_size, 4)
    # rotmat: (batch_size, 3, 3)
    x, y, z, w = quad[:, 0], quad[:, 1], quad[:, 2], quad[:, 3]
    rotmat = torch.zeros(quad.shape[0], 3, 3, device=quad.device)
    rotmat[:, 0, 0] = 1 - 2 * y**2 - 2 * z**2
    rotmat[:, 0, 1] = 2 * x * y - 2 * z * w
    rotmat[:, 0, 2] = 2 * x * z + 2 * y * w
    rotmat[:, 1, 0] = 2 * x * y + 2 * z * w
    rotmat[:, 1, 1] = 1 - 2 * x**2 - 2 * z**2
    rotmat[:, 1, 2] = 2 * y * z - 2 * x * w
    rotmat[:, 2, 0] = 2 * x * z - 2 * y * w
    rotmat[:, 2, 1] = 2 * y * z + 2 * x * w
    rotmat[:, 2, 2] = 1 - 2 * x**2 - 2 * y**2
    return rotmat