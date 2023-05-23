wandb offline
srun -n 1 -N 1 -p gpu -w gnode01 python train.py --exp-name "DroneEnv1024RMA" --act-expert-mode 1 --cri-expert-mode 1 --use-wandb --gpu-id 0 --env-num 1024

# srun -n 1 -N 1 -p gpu python train.py --exp-name "DroneRobust" --act-expert-mode 0 --cri-expert-mode 0 --use-wandb --gpu-id 3