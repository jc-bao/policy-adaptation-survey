import torch
from typing import List, Iterable, Union
import math

from adaptive_control_gym.controllers.rl.net import ActorPPO, CriticPPO
from adaptive_control_gym.controllers.rl.buffer import ReplayBufferList
from adaptive_control_gym.controllers.rl.adaptor import AdaptorMLP

class PPO:
    def __init__(self, 
        state_dim: int, expert_dim: int, action_dim: int, adapt_horizon: int,
        act_expert_mode: int, cri_expert_mode: int,
        env_num: int, gpu_id: int = 0):
        # env
        self.env_num = env_num
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.expert_dim = expert_dim
        self.last_state, self.last_info = None, None  # last state of the trajectory for training
        self.reward_scale = 1.0
        # update network
        self.gamma = 0.95
        self.batch_size = 1024
        self.repeat_times = 4
        self.learning_rate = 1e-4
        self.clip_grad_norm = 3.0
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        # network
        self.act = ActorPPO(state_dim, expert_dim, action_dim, act_expert_mode).to(self.device)
        self.cri = CriticPPO(state_dim, expert_dim, action_dim, cri_expert_mode).to(self.device)
        # self.adaptor = Adaptor(adapt_dim, adapt_mode).to(self.device)
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), self.learning_rate)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), self.learning_rate)
        self.criterion = torch.nn.SmoothL1Loss(reduction="mean")
        # ppo
        self.ratio_clip = 0.25  # `ratio.clamp(1 - clip, 1 + clip)`
        self.lambda_gae_adv = 0.95  # could be 0.50~0.99 # GAE for sparse reward
        self.lambda_entropy = 0.005  # could be 0.00~0.20
        self.lambda_entropy = torch.tensor(self.lambda_entropy, dtype=torch.float32, device=self.device)
        # buffer
        self.buffer = ReplayBufferList()
        # adaptor
        self.adapt_horizon = adapt_horizon
        self.adaptor = AdaptorMLP(state_dim, adapt_horizon, expert_dim).to(self.device)
        self.adaptor_optimizer = torch.optim.Adam(self.adaptor.parameters(), self.learning_rate)

    def explore_env(self, env, horizon_len: int, deterministic: bool=False, use_adaptor: bool=False) -> List[torch.Tensor]:
        states = torch.zeros((horizon_len, self.env_num, self.state_dim), dtype=torch.float32).to(self.device)
        actions = torch.zeros((horizon_len, self.env_num, self.action_dim), dtype=torch.float32).to(self.device)
        logprobs = torch.zeros((horizon_len, self.env_num), dtype=torch.float32).to(self.device)
        rewards = torch.zeros((horizon_len, self.env_num), dtype=torch.float32).to(self.device)
        dones = torch.zeros((horizon_len, self.env_num), dtype=torch.bool).to(self.device)
        err_xs = torch.zeros((horizon_len, self.env_num), dtype=torch.float32).to(self.device)
        err_vs = torch.zeros((horizon_len, self.env_num), dtype=torch.float32).to(self.device)
        es = torch.zeros((horizon_len, self.env_num, self.expert_dim), dtype=torch.float32).to(self.device)
        obs_his = torch.zeros((horizon_len, self.env_num, self.state_dim*self.adapt_horizon), dtype=torch.float32).to(self.device)

        state, info = self.last_state, self.last_info  # shape == (env_num, state_dim) for a vectorized env.
        e, obs_history = info['e'], info['obs_history']

        if deterministic:
            get_action = lambda x,e: (self.act(x,e), 0.0)
            convert = lambda x: x
        else:
            get_action = self.act.get_action
            convert = self.act.convert_action_for_env
        for t in range(horizon_len):
            if use_adaptor:
                e = self.adaptor(obs_history)
            action, logprob = get_action(state, e)
            states[t] = state

            state, reward, done, info = env.step(convert(action))  # next_state
            e = info['e']
            actions[t] = action
            logprobs[t] = logprob
            rewards[t] = reward
            dones[t] = done
            es[t] = e
            err_xs[t] = info["err_x"]
            err_vs[t] = info["err_v"]
            obs_his[t] = info["obs_history"]

        self.last_state, self.last_info = state, info

        rewards *= self.reward_scale
        undones = 1.0 - dones.type(torch.float32)
        infos = {"err_x": err_xs, "err_v": err_vs, 'e': es, 'obs_history': obs_his}
        return states, actions, logprobs, rewards, undones, infos

    def update_net(self, states, es, actions, logprobs, rewards, undones):
        with torch.no_grad():
            buffer_size = states.shape[0]
            buffer_num = states.shape[1]

            '''get advantages and reward_sums'''
            values = torch.empty_like(rewards)  # values.shape == (buffer_size, buffer_num)
            for i in range(0, buffer_size, self.batch_size):
                for j in range(buffer_num):
                    values[i:i + self.batch_size, j] = self.cri(states[i:i + self.batch_size, j], es[i:i+self.batch_size, j]).squeeze(1)

            advantages = self.get_advantages(rewards, undones, values)  # shape == (buffer_size, buffer_num)
            reward_sums = advantages + values  # shape == (buffer_size, buffer_num)
            del rewards, undones, values

            advantages = (advantages - advantages.mean()) / (advantages.std(dim=0) + 1e-4)

        '''update network'''
        obj_critics = 0.0
        obj_actors = 0.0
        sample_len = buffer_size - 1

        update_times = int(buffer_size * buffer_num * self.repeat_times / self.batch_size)
        assert update_times >= 1
        for _ in range(update_times):
            ids = torch.randint(sample_len * buffer_num, size=(self.batch_size,), requires_grad=False)
            ids0 = torch.fmod(ids, sample_len)  # ids % sample_len
            ids1 = torch.div(ids, sample_len, rounding_mode='floor')  # ids // sample_len

            state = states[ids0, ids1]
            e = es[ids0, ids1]
            action = actions[ids0, ids1]
            logprob = logprobs[ids0, ids1]
            advantage = advantages[ids0, ids1]
            reward_sum = reward_sums[ids0, ids1]

            value = self.cri(state, e).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, reward_sum)
            self.optimizer_update(self.cri_optimizer, obj_critic)


            # judge if self.x contains nan
            new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action, e)
            ratio = (new_logprob - logprob.detach()).exp()
            surrogate1 = advantage * ratio
            surrogate2 = advantage * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = torch.min(surrogate1, surrogate2).mean()

            obj_actor = obj_surrogate + obj_entropy.mean() * self.lambda_entropy
            self.optimizer_update(self.act_optimizer, -obj_actor)

            obj_critics += obj_critic.item()
            obj_actors += obj_actor.item()
        a_std_log = self.act.action_std_log.mean()
        return obj_critics / update_times, obj_actors / update_times, a_std_log.item()

    def update_adaptor(self, es, obs_histories):
        adaptor_loss = 0.0
        buffer_size = es.shape[0]
        buffer_num = es.shape[1]
        update_times = int(buffer_size * buffer_num * self.repeat_times / self.batch_size)
        sample_len = buffer_size - 1
        assert update_times >= 1
        for _ in range(update_times):
            ids = torch.randint(sample_len * buffer_num, size=(self.batch_size,), requires_grad=False)
            ids0 = torch.fmod(ids, sample_len)  # ids % sample_len
            ids1 = torch.div(ids, sample_len, rounding_mode='floor')  # ids // sample_len

            obs_history = obs_histories[ids0, ids1]
            e = es[ids0, ids1]


            # predict e with obs_history and adaptor
            e_pred = self.adaptor(obs_history)
            # calculate loss and update adaptor
            obj_adaptor = self.criterion(e_pred, e)
            self.optimizer_update(self.adaptor_optimizer, obj_adaptor)

            adaptor_loss += obj_adaptor.item()
        return adaptor_loss / update_times


    def get_advantages(self, rewards: torch.Tensor, undones: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        advantages = torch.empty_like(values)  # advantage value

        masks = undones * self.gamma
        horizon_len = rewards.shape[0]

        next_value = self.cri(self.last_state, self.last_info['e']).detach().squeeze(1)

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
