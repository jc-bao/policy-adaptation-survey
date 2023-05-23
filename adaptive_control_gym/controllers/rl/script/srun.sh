wandb offline
srun -n 1 -N 1 -p gpu python train.py --exp-name "TrackRMA"  --task track --act-expert-mode 1 --cri-expert-mode 1 --use-wandb --curri-param-init 0.3