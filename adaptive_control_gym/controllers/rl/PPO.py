import torch
from typing import List, Iterable, Union
import math

from adaptive_control_gym.controllers.rl.net import ActorPPO, CriticPPO

class PPO:
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int, env_num: int, gpu_id: int = 0):
        # env
        self.env_num = env_num
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.last_state = None  # last state of the trajectory for training
        self.reward_scale = 1.0
        # update network
        self.gamma = 0.99
        self.batch_size = 128
        self.repeat_times = 1
        self.learning_rate = 1e-3
        self.clip_grad_norm = 3.0
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        # network
        self.act = ActorPPO(net_dims, state_dim, action_dim).to(self.device)
        self.cri = CriticPPO(net_dims, state_dim, action_dim).to(self.device)
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), self.learning_rate)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), self.learning_rate)
        # ppo
        self.ratio_clip = 0.25  # `ratio.clamp(1 - clip, 1 + clip)`
        self.lambda_gae_adv = 0.95  # could be 0.50~0.99 # GAE for sparse reward
        self.lambda_entropy = 0.01  # could be 0.00~0.20
        self.lambda_entropy = torch.tensor(self.lambda_entropy, dtype=torch.float32, device=self.device)

    def explore_env(self, env, horizon_len: int) -> List[torch.Tensor]:
        states = torch.zeros((horizon_len, self.env_num, self.state_dim), dtype=torch.float32).to(self.device)
        actions = torch.zeros((horizon_len, self.env_num, 1), dtype=torch.int32).to(self.device) if self.if_discrete \
            else torch.zeros((horizon_len, self.env_num, self.action_dim), dtype=torch.float32).to(self.device)
        logprobs = torch.zeros((horizon_len, self.env_num), dtype=torch.float32).to(self.device)
        rewards = torch.zeros((horizon_len, self.env_num), dtype=torch.float32).to(self.device)
        dones = torch.zeros((horizon_len, self.env_num), dtype=torch.bool).to(self.device)

        state = self.last_state  # shape == (env_num, state_dim) for a vectorized env.

        get_action = self.act.get_action
        convert = self.act.convert_action_for_env
        for t in range(horizon_len):
            action, logprob = get_action(state)
            states[t] = state

            state, reward, done, _ = env.step(convert(action))  # next_state
            actions[t] = action
            logprobs[t] = logprob
            rewards[t] = reward
            dones[t] = done

        self.last_state = state

        rewards *= self.reward_scale
        undones = 1.0 - dones.type(torch.float32)
        return states, actions, logprobs, rewards, undones

    def update_net(self, buffer):
        with torch.no_grad():
            states, actions, logprobs, rewards, undones = buffer
            buffer_size = states.shape[0]

            '''get advantages reward_sums'''
            bs = 2 ** 10  # set a smaller 'batch_size' when out of GPU memory.
            values = [self.cri(states[i:i + bs]) for i in range(0, buffer_size, bs)]
            values = torch.cat(values, dim=0).squeeze(1)  # values.shape == (buffer_size, )

            advantages = self.get_advantages(rewards, undones, values)  # advantages.shape == (buffer_size, )
            reward_sums = advantages + values  # reward_sums.shape == (buffer_size, )
            del rewards, undones, values

            advantages = (advantages - advantages.mean()) / (advantages.std(dim=0) + 1e-5)
        assert logprobs.shape == advantages.shape == reward_sums.shape == (buffer_size,)

        '''update network'''
        obj_critics = 0.0
        obj_actors = 0.0

        update_times = int(buffer_size * self.repeat_times / self.batch_size)
        assert update_times >= 1
        for _ in range(update_times):
            indices = torch.randint(buffer_size, size=(self.batch_size,), requires_grad=False)
            state = states[indices]
            action = actions[indices]
            logprob = logprobs[indices]
            advantage = advantages[indices]
            reward_sum = reward_sums[indices]

            value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, reward_sum)
            self.optimizer_update(self.cri_optimizer, obj_critic)

            new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)
            ratio = (new_logprob - logprob.detach()).exp()
            surrogate1 = advantage * ratio
            surrogate2 = advantage * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = torch.min(surrogate1, surrogate2).mean()

            obj_actor = obj_surrogate + obj_entropy.mean() * self.lambda_entropy
            self.optimizer_update(self.act_optimizer, -obj_actor)

            obj_critics += obj_critic.item()
            obj_actors += obj_actor.item()
        action_std = getattr(self.act, 'action_std_log', torch.zeros(1)).exp().mean()
        return obj_critics / update_times, obj_actors / update_times, action_std.item()

    def get_advantages(self, rewards: torch.Tensor, undones: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        advantages = torch.empty_like(values)  # advantage value

        masks = undones * self.gamma
        horizon_len = rewards.shape[0]

        next_value = self.cri(self.last_state).detach().squeeze(1)

        advantage = torch.zeros_like(next_value)  # last advantage value by GAE (Generalized Advantage Estimate)
        for t in range(horizon_len - 1, -1, -1):
            next_value = rewards[t] + masks[t] * next_value
            advantages[t] = advantage = next_value - values[t] + masks[t] * self.lambda_gae_adv * advantage
            next_value = values[t]
        return advantages

    def optimizer_update(self, optimizer, objective):
        optimizer.zero_grad()
        objective.backward()
        clip_grad_norm_(parameters=optimizer.param_groups[0]["params"], max_norm=self.clip_grad_norm)
        optimizer.step()


def clip_grad_norm_(
        parameters: Union[torch.Tensor, Iterable[torch.Tensor]], max_norm: float, norm_type: float = 2.0,
        error_if_nonfinite: bool = False) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.)
    device = grads[0].device
    if norm_type == math.inf:
        norms = [g.detach().abs().max().to(device) for g in grads]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]), norm_type)
    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f'The total norm of order {norm_type} for gradients from '
            '`parameters` is non-finite, so it cannot be clipped. To disable '
            'this error and scale the gradients by the non-finite norm anyway, '
            'set `error_if_nonfinite=False`')
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for g in grads:
        g.detach().mul_(clip_coef_clamped.to(g.device))
    return total_norm