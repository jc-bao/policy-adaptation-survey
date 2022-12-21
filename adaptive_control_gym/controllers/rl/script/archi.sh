seed=0
gpu_id=0
expert_mode=1

for ss in {1..3}
do
    for mm in ${mass_uncertainty_rates[@]}
    do
        for dd in ${disturb_uncertainty_rates[@]}
        do
            for pp in ${disturb_periods[@]}
            do
                python train.py --use_wandb --program $program --expert_mode --ood_mode --mass_uncertainty_rate $mm --disturb_uncertainty_rate $dd --disturb_period $pp --seed $ss
                python train.py --use_wandb --program $program --ood_mode --mass_uncertainty_rate $mm --disturb_uncertainty_rate $dd --disturb_period $pp --seed $ss
                # python train.py --use_wandb --program $program --expert_mode --mass_uncertainty_rate $mm --disturb_uncertainty_rate $dd --disturb_period $pp --seed $ss
                # python train.py --use_wandb --program $program --mass_uncertainty_rate $mm --disturb_uncertainty_rate $dd --disturb_period $pp --seed $ss
                # echo "$mm, $dd, $pp"
            done
        done
    done
done