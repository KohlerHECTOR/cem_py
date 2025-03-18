#!/bin/bash

# Arrays of parameters to sweep
seeds=(0 1 2)
n_pops=(8 16 32)
elite_props=(0.125 0.250)

# Create a logs directory if it doesn't exist
mkdir -p logs

# Loop through all combinations
for seed in "${seeds[@]}"
do
    for n_pop in "${n_pops[@]}"
    do
        for elite_prop in "${elite_props[@]}"
        do
            # Create a unique job name
            job_name="cem_s${seed}_e${n_pop}_elitep${elite_prop}"
            
            oarsub "source cem_env/bin/activate; python3 cem_beta_pendulum.py \
                --seed ${seed} \
                --n_pop ${n_pop} \
                --elite_prop ${elite_prop}" \
                -l nodes=2,walltime=4:00:00 \
                -p grvingt \
                -q production \
                -n "${job_name}" \
                -O "logs/${job_name}.%jobid%.out" \
                -E "logs/${job_name}.%jobid%.err"
            
            # Add a small delay between submissions
            sleep 1.2
        done
    done
done