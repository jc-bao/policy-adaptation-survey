python train.py --exp-name mlp_expert_Dw0 --res-dyn-param-dim 0
python train.py --exp-name mlp_expert_Dw1 --res-dyn-param-dim 1
python train.py --exp-name mlp_expert_Dw2 --res-dyn-param-dim 2
python train.py --exp-name mlp_expert_Dw1_C4  --res-dyn-param-dim 1 --compressor-dim 4
python train.py --exp-name mlp_vanilla_Dw1 --act-expert-mode 0 --cri-expert-mode 0 --res-dyn-param-dim 1