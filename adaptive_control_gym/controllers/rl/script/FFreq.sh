program='FFreq'
mass_uncertainty_rates=(0.0)
disturb_uncertainty_rates=(0.5) #(0.0 0.2 0.5 1.0)
disturb_periods=(1 5 15 30 60)

for mm in ${mass_uncertainty_rates[@]}
do
    for dd in ${disturb_uncertainty_rates[@]}
    do
        for pp in ${disturb_periods[@]}
        do
            python train.py --use_wandb --program $program --expert_mode --ood_mode --mass_uncertainty_rate $mm --disturb_uncertainty_rate $dd --disturb_period $pp
            python train.py --use_wandb --program $program --ood_mode --mass_uncertainty_rate $mm --disturb_uncertainty_rate $dd --disturb_period $pp
        done
    done
done