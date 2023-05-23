wandb offline
srun -n 1 -N 1 -p gpu python train.py --exp-name "DroneRMA" --act-expert-mode 1 --cri-expert-mode 1 --use-wandb --gpu-id 2

# srun -n 1 -N 1 -p gpu python train.py --exp-name "DroneRobust" --act-expert-mode 0 --cri-expert-mode 0 --use-wandb --gpu-id 3