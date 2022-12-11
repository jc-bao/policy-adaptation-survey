import torch

class ReplayBufferList(list):  # for on-policy
    def __init__(self):
        list.__init__(self)  # (buf_state, buf_reward, buf_mask, buf_action, buf_noise) = self[:]

    def update_buffer(self, traj_list):
        cur_items = [map(list, zip(*traj_list))]
        self[:] = [torch.cat(item, dim=-1) for item in cur_items]

        steps = self[0].shape[0]
        r_exp = self[0].mean().item()
        return steps, r_exp