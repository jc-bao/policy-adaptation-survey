seed=0

python train.py --act-expert-mode 3 --cri-expert-mode 1 --use-wandb --seed $seed --gpu-id 0 --exp-name "act3_cri1" &
python train.py --act-expert-mode 0 --cri-expert-mode 0 --use-wandb --seed $seed --gpu-id 1 --exp-name "act0_cri0" &
