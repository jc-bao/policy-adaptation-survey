seed=0
gpu_id=2
am=3

for cm in {0..4}
do
    python train.py --act-expert-mode $am --cri-expert-mode $cm --use-wandb --seed $seed --gpu-id $gpu_id
done