import control as ct
import torch

class LRQ:
    def __init__(self, A, B, Q, R, gpu_id:int = 0):
        self.A = torch.tensor(A).to(gpu_id)
        self.B = torch.tensor(B).to(gpu_id)
        self.Q = torch.tensor(Q).to(gpu_id)
        self.R = torch.tensor(R).to(gpu_id)
        K = [torch.tensor(ct.lqr(A[i], B[i], Q, R)[0], dtype=torch.float32) for i in range(A.shape[0])]
        self.K = torch.stack(K, dim=0).to(gpu_id) # Tensor(env_num, action_dim, state_dim)

    def __call__(self, state):
        act = -torch.einsum('eas,es->ea', self.K, state)
        return act