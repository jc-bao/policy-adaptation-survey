gpu_id=0

# run 3 seeds
for seed in {1..3} 
do
    python train.py --exp-name ""  --program "may" --use-wandb --seed $seed
done