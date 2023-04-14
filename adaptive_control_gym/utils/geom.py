import torch

def rpy2quat(rpy: torch.Tensor) -> torch.Tensor:
    # convert roll-pitch-yaw to quaternion with torch
    # rpy: (batch_size, 3)
    # quat: (batch_size, 4) [x, y, z, w]
    roll, pitch, yaw = rpy[..., 0], rpy[..., 1], rpy[..., 2]
    q = torch.zeros([*rpy.shape[:-1], 4], device=rpy.device)
    x = torch.sin(roll / 2) * torch.cos(pitch / 2) * torch.cos(yaw / 2) - torch.cos(roll / 2) * torch.sin(pitch / 2) * torch.sin(yaw / 2)
    y = torch.cos(roll / 2) * torch.sin(pitch / 2) * torch.cos(yaw / 2) + torch.sin(roll / 2) * torch.cos(pitch / 2) * torch.sin(yaw / 2)
    z = torch.cos(roll / 2) * torch.cos(pitch / 2) * torch.sin(yaw / 2) - torch.sin(roll / 2) * torch.sin(pitch / 2) * torch.cos(yaw / 2)
    w = torch.cos(roll / 2) * torch.cos(pitch / 2) * torch.cos(yaw / 2) + torch.sin(roll / 2) * torch.sin(pitch / 2) * torch.sin(yaw / 2)
    q[..., 0] = x
    q[..., 1] = y
    q[..., 2] = z
    q[..., 3] = w
    return q

def rpy2rotmat(rpy: torch.Tensor) -> torch.Tensor:
    # convert roll-pitch-yaw to rotation matrix with torch
    # rpy: (batch_size, 3)
    # rotmat: (batch_size, 3, 3)
    roll, pitch, yaw = rpy[..., 0], rpy[..., 1], rpy[..., 2]
    rotmat = torch.zeros([*rpy.shape[:-1], 3, 3], device=rpy.device)
    rotmat[..., 0, 0] = torch.cos(yaw) * torch.cos(pitch)
    rotmat[..., 0, 1] = torch.cos(yaw) * torch.sin(pitch) * torch.sin(roll) - torch.sin(yaw) * torch.cos(roll)
    rotmat[..., 0, 2] = torch.cos(yaw) * torch.sin(pitch) * torch.cos(roll) + torch.sin(yaw) * torch.sin(roll)
    rotmat[..., 1, 0] = torch.sin(yaw) * torch.cos(pitch)
    rotmat[..., 1, 1] = torch.sin(yaw) * torch.sin(pitch) * torch.sin(roll) + torch.cos(yaw) * torch.cos(roll)
    rotmat[..., 1, 2] = torch.sin(yaw) * torch.sin(pitch) * torch.cos(roll) - torch.cos(yaw) * torch.sin(roll)
    rotmat[..., 2, 0] = -torch.sin(pitch)
    rotmat[..., 2, 1] = torch.cos(pitch) * torch.sin(roll)
    rotmat[..., 2, 2] = torch.cos(pitch) * torch.cos(roll)
    return rotmat

def quat2rpy(quat: torch.Tensor) -> torch.Tensor:
    # convert quaternion to roll-pitch-yaw with torch
    # quat: (batch_size, 4) 
    # rpy: (batch_size, 3)
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    rpy = torch.zeros([*quat.shape[:-1], 3], device=quat.device)
    rpy[..., 0] = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    rpy[..., 1] = torch.asin(2 * (w * y - z * x))
    rpy[..., 2] = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    return rpy

def quat2rotmat(quat: torch.Tensor) -> torch.Tensor:
    # convert quaternion to rotation matrix with torch
    # quat: (batch_size, 4)
    # rotmat: (batch_size, 3, 3)
    bs = quat.shape[:-1]
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    rotmat = torch.zeros([*bs, 3, 3], device=quat.device)
    rotmat[..., 0, 0] = 1 - 2 * y**2 - 2 * z**2
    rotmat[..., 0, 1] = 2 * x * y - 2 * z * w
    rotmat[..., 0, 2] = 2 * x * z + 2 * y * w
    rotmat[..., 1, 0] = 2 * x * y + 2 * z * w
    rotmat[..., 1, 1] = 1 - 2 * x**2 - 2 * z**2
    rotmat[..., 1, 2] = 2 * y * z - 2 * x * w
    rotmat[..., 2, 0] = 2 * x * z - 2 * y * w
    rotmat[..., 2, 1] = 2 * y * z + 2 * x * w
    rotmat[..., 2, 2] = 1 - 2 * x**2 - 2 * y**2
    return rotmat

def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    # multiply two quaternions with torch
    # q1: (batch_size, 4) x, y, z, w
    # q2: (batch_size, 4)
    # q: (batch_size, 4)
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    q = torch.zeros(q1.shape[0], 4, device=q1.device)
    q[..., 0] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    q[..., 1] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    q[..., 2] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    q[..., 3] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return q

def quat_inv(q: torch.Tensor) -> torch.Tensor:
    # invert quaternion with torch
    # q: (batch_size, 4) x, y, z, w
    # q_inv: (batch_size, 4)
    q_inv = torch.zeros([*q.shape[:-1], 4], device=q.device)
    q_inv[..., 0] = -q[..., 0]
    q_inv[..., 1] = -q[..., 1]
    q_inv[..., 2] = -q[..., 2]
    q_inv[..., 3] = q[..., 3]
    return q_inv

def integrate_quat(quat, omega, dt):
    # integrate quaternion with angular velocity
    # quat: (batch_size, 4) with x, y, z, w
    # omega: (batch_size, 3) with x, y, z
    # dt: (batch_size, 1)
    # quat: (batch_size, 4) with x, y, z, w

    bs = quat.shape[:-1]
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    wx, wy, wz = omega[..., 0], omega[..., 1], omega[..., 2]
    q = torch.zeros([*bs, 4], device=quat.device)
    q[..., 0] = x + 0.5 * dt * (w * wx + y * wz - z * wy)
    q[..., 1] = y + 0.5 * dt * (w * wy - x * wz + z * wx)
    q[..., 2] = z + 0.5 * dt * (w * wz + x * wy - y * wx)
    q[..., 3] = w + 0.5 * dt * (-x * wx - y * wy - z * wz)
    return q
