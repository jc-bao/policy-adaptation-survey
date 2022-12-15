gpu=2
python train.py --use-wandb --exp-name 'dyndim=16,expert' --expert-mode --gpu-id $gpu &
python train.py --use-wandb --exp-name 'dyndim=16' --gpu-id $gpu &