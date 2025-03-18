#!/bin/bash

# Arrays of parameters to sweep
seeds=(3 4 5 6 7 8 9 10 11 12 13 14 15 16 17)

# Create a logs directory if it doesn't exist
mkdir -p logs

# Loop through all combinations
for seed in "${seeds[@]}"
do
    job_name="cem_best_hp"
    
    oarsub "source cem_env/bin/activate; python3 cem_beta_pendulum.py \
        --seed ${seed} \
        --n_evals 50 \
        --elite_prop 0.125" \
        -l nodes=2,walltime=4:00:00 \
        -p grvingt \
        -q production \
        -n "${job_name}" \
        -O "logs/${job_name}.%jobid%.out" \
        -E "logs/${job_name}.%jobid%.err"
    
    # Add a small delay between submissions
    sleep 1.2

    job_name="reinforce_best_hp"
    
    oarsub "source cem_env/bin/activate; python3 reinforce_beta_pendulum.py \
        --seed ${seed} \
        --n_evals 1 \
        --lr 0.001 \
        --batch_size 64" \
        -l nodes=2,walltime=4:00:00 \
        -p grvingt \
        -q production \
        -n "${job_name}" \
        -O "logs/${job_name}.%jobid%.out" \
        -E "logs/${job_name}.%jobid%.err"
    
    # Add a small delay between submissions
    sleep 1.2

    job_name="reinforce_baseline_best_hp"
    
    oarsub "source cem_env/bin/activate; python3 reinforce_beta_pendulum_sub_baseline.py \
        --seed ${seed} \
        --n_evals 1 \
        --lr 0.001 \
        --batch_size 64" \
        -l nodes=2,walltime=4:00:00 \
        -p grvingt \
        -q production \
        -n "${job_name}" \
        -O "logs/${job_name}.%jobid%.out" \
        -E "logs/${job_name}.%jobid%.err"
    
    # Add a small delay between submissions
    sleep 1.2
done