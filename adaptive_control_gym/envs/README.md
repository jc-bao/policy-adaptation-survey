# Environment setup instructions

## Required implementations

* gym.Env class
  * info must include `err_x`, `err_v`, `e` and `adapt_obs` keys
  * function `get_env_params` to return environment parameters in dict format (mainly for later search)
  * rl related parameters
```
        # set RL parameters
        self.state_dim = 3 + 3 + (3 + 3 + 4 + 3) * self.drone_num + 3 + 3
        self.expert_dim = 1
        self.adapt_horizon = 1
        self.adapt_dim = 1
        self.action_dim = 3
```
* `test_env` function