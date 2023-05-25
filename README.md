# 🚁 Policy Adaptation Survey

## 🤖 Environment

|Quadrotor transportation| Cartpole                                                     | Hover                                                        |
|-| ------------------------------------------------------------ | ------------------------------------------------------------ |
|![quadrotor](https://user-images.githubusercontent.com/60093981/236766584-83b40cdb-bc8e-4c64-8562-f4261ac5af68.gif)| ![cartpole](https://tva1.sinaimg.cn/large/008vxvgGly1h8whypx7rig305k02s41b.gif) | ![hover](https://tva1.sinaimg.cn/large/008vxvgGly1h8whraypc1g301e05kq3r.gif) |


## 🐣 Train

### Torch-based environemnt

```shell
cd adaptive_control_gym/controller/rl
# Run the train function with the parsed arguments
python train.py \
    --use_wandb $USE_WANDB \
    --program $PROGRAM \
    --seed $SEED \
    --gpu_id $GPU_ID \
    --act_expert_mode $ACT_EXPERT_MODE \
    --cri_expert_mode $CRI_EXPERT_MODE \
    --exp_name $EXP_NAME \
    --compressor_dim $COMPRESSOR_DIM \
    --search_dim $SEARCH_DIM \
    --res_dyn_param_dim $RES_DYN_PARAM_DIM \
    --task $TASK \
    --resume_path $RESUME_PATH \
    --drone_num $DRONE_NUM \
    --env_num $ENV_NUM \
    --total_steps $TOTAL_STEPS \
    --adapt_steps $ADAPT_STEPS \
    --curri_thereshold $CURRI_THERESHOLD
```

* use_wandb: A boolean flag indicating whether to use the Weights & Biases service for logging and visualization. Default is False.
* program: A string specifying the name of the program. Default is 'tmp'.
* seed: An integer specifying the random seed to use. Default is 1.
* gpu_id: An integer specifying the ID of the GPU to use. Default is 0.
* act_expert_mode: An integer specifying the expert mode for the actor network. Default is 0.
* cri_expert_mode: An integer specifying the expert mode for the critic network. Default is 0.
* exp_name: A string specifying the name of the experiment. Default is an empty string.
* compressor_dim: An integer specifying the dimension of the compressor network. Default is 4.
* search_dim: An integer specifying the dimension of the search network. Default is 0.
* res_dyn_param_dim: An integer specifying the dimension of the residual dynamic parameter network. Default is 0.
* task: A string specifying the task to perform. Can be 'track', 'hover', or 'avoid'. Default is 'track'.
* resume_path: A string specifying the path to a saved checkpoint to resume training from. Default is None.
* drone_num: An integer specifying the number of drones to use. Default is 1.
* env_num: An integer specifying the number of environments to use. Default is 16384.
* total_steps: An integer specifying the total number of training steps to perform. Default is 8e7.
* adapt_steps: An integer specifying the number of adaptation steps to perform. Default is 5e6.
* curri_thereshold: A float specifying the curriculum threshold. Default is 0.2.

### Brax-based environemnt

```shell
cd adaptive_control_gym/envs/brax
python train_brax.py
python play_brax.py --policy-type ppo --policy-path '../results/params' # visualize
```

### Examples

```shell
# train with RMA
python train.py --exp-name "TrackRMA"  --task track --act-expert-mode 1 --cri-expert-mode 1 --use-wandb  --gpu-id 0
# train Robust policy
python train.py --exp-name "TrackRobust"  --task track --use-wandb --gpu-id 0
```

## 🕹 Play with environment

```shell
# go to environment folder
cd adaptive_control_gym/envs

# run environment
python quadtrans.py 
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

| `task=aviod curri_param=1.0`                                 |    `task=aviod curri_param=0.0`    |  `task=track`    | `task=hover` | `Brax` |
| ------------------------------------------------------------ | ---- | ---- | - | - |
| ![image](https://github.com/jc-bao/policy-adaptation-survey/assets/60093981/2db622a8-23cc-4fc7-8c82-94c54cc154e0) |   ![image](https://github.com/jc-bao/policy-adaptation-survey/assets/60093981/c5178f3c-29e6-4c98-a09f-4028cf70e496) |  ![image](https://github.com/jc-bao/policy-adaptation-survey/assets/60093981/1a3546ce-6527-451a-8c0a-975f256964e0)  | ![image](https://github.com/jc-bao/policy-adaptation-survey/assets/60093981/76c82417-b7bd-4ea4-8988-f8e2b2fc1134)|  |


## 🐒 Policy

* Classic
  - [x] LQR
  - [x] PID
  - [ ] MPC
* RL
  - [x] PPO 
  - [x] RMA
