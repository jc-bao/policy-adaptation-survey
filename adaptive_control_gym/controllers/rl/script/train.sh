res_dyn_param_dim=0
gpu_id=0

for compressor_dim in 3
do
python train.py --res-dyn-param-dim $res_dyn_param_dim --compressor-dim $compressor_dim --exp-name "ood-only-disturb-center-100-com$compressor_dim" --gpu-id $gpu_id --use-wandb
done