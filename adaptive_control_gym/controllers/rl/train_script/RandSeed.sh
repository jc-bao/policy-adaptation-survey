program='RandSeed'
mass_uncertainty_rates=(0.0)
disturb_uncertainty_rates=(0.5) #(0.0 0.2 0.5 1.0)
disturb_periods=(15) #[1, 5, 15, 30, 60]

for ss in {0..3}
do
    for mm in ${mass_uncertainty_rates[@]}
    do
        for dd in ${disturb_uncertainty_rates[@]}
        do
            for pp in ${disturb_periods[@]}
            do
                python train.py --use_wandb --program $program --export_mode --ood_mode --mass_uncertainty_rate $mm --disturb_uncertainty_rate $dd --disturb_period $pp --seed $ss
                python train.py --use_wandb --program $program --ood_mode --mass_uncertainty_rate $mm --disturb_uncertainty_rate $dd --disturb_period $pp --seed $ss
                # echo "$mm, $dd, $pp"
            done
        done
    done
done