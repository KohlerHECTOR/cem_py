#!/bin/bash

# Arrays of parameters to sweep
seeds=(3 4 5 6 7 8 9 10 11 12 13 14 15 16 17)

# Create a logs directory if it doesn't exist
mkdir -p logs

# Loop through all combinations
for seed in "${seeds[@]}"
do
    job_name="es_best_hp"
    
    oarsub "source cem_env/bin/activate; python3 es_beta_pendulum.py \
        --seed ${seed} \
        --n_pop 8 \
        --lr 0.01 \
        --n_evals 50 \
        --noise_std 0.1" \
        -l nodes=2,walltime=4:00:00 \
        -p grvingt \
        -q production \
        -n "${job_name}" \
        -O "logs/${job_name}.%jobid%.out" \
        -E "logs/${job_name}.%jobid%.err"
    
    # Add a small delay between submissions
    sleep 1.2
done