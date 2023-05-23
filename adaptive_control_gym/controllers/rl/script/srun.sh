wandb login "local-9bea0da32236f3205f9a534bbf0dedfff2ad5dac" --host=http://192.168.1.1
export WANDB_BASE_URL = "http://192.168.1.1"
srun -n 1 -N 1 -p gpu python train.py --exp-name "TrackRMA"  --task track --act-expert-mode 1 --cri-expert-mode 1 --use-wandb --curri-param-init 0.3