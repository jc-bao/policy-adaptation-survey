# 🚁 Policy Adaptation Survey

## 🤖 Environment

|Quadrotor transportation| Cartpole                                                     | Hover                                                        |
|-| ------------------------------------------------------------ | ------------------------------------------------------------ |
|![quadrotor](https://user-images.githubusercontent.com/60093981/236766584-83b40cdb-bc8e-4c64-8562-f4261ac5af68.gif)| ![cartpole](https://tva1.sinaimg.cn/large/008vxvgGly1h8whypx7rig305k02s41b.gif) | ![hover](https://tva1.sinaimg.cn/large/008vxvgGly1h8whraypc1g301e05kq3r.gif) |


## 🐣 Train

```shell
cd controller/rl
python train.py
```

## 🕹 Play with environment

```shell
# go to environment folder
cd adaptive_control_gym/envs

# run environment
python quadtrans_v2.py 
    --policy_type pid  # 'random', 'pid'
    --task "avoid"  # 'track', 'hover', 'avoid'
    --policy_path 'ppo.pt' # for PPO only
    --seed 0
    --env_num 1
    --drone_num 1
    --gpu_id -1 # use CPU
    --enable_log  true # log parameter to csv and plot
    --enable_vis  true # use meshcat to visualize
    --curri_param 1.0 # 0.0 for simple case

```

| `task=aviod curri_param=1.0`                                 |    `task=aviod curri_param=0.0`    |  `task=track`    | `task=hover` |
| ------------------------------------------------------------ | ---- | ---- | - |
| ![image](https://github.com/jc-bao/policy-adaptation-survey/assets/60093981/2db622a8-23cc-4fc7-8c82-94c54cc154e0)
|   ![image](https://github.com/jc-bao/policy-adaptation-survey/assets/60093981/c5178f3c-29e6-4c98-a09f-4028cf70e496)
   |  ![image](https://github.com/jc-bao/policy-adaptation-survey/assets/60093981/1a3546ce-6527-451a-8c0a-975f256964e0)
    | ![image](https://github.com/jc-bao/policy-adaptation-survey/assets/60093981/76c82417-b7bd-4ea4-8988-f8e2b2fc1134)
 |


## 🐒 Policy
`
* Classic
  - [x] LQR
  - [x] PID
  - [ ] MPC
* RL
  - [x] PPO 
