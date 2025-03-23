#!/bin/bash

# Arrays of parameters to sweep
seeds=(3 4 5 6 7 8 9 10 11 12)
batch_sizes=(8 16 32 64 128 256)
learning_rates=(0.0001 0.001 0.01 0.1)

# Create a logs directory if it doesn't exist
mkdir -p logs

# Loop through all combinations
for seed in "${seeds[@]}"
do
    for batch_size in "${batch_sizes[@]}"
    do
        for lr in "${learning_rates[@]}"
        do
            echo "Launching training with seed ${seed}, n_evals ${batch_size}, lr ${lr}"
            
            # Create a unique job name
            job_name="reinforce_s${seed}_e${batch_size}_lr${lr}"
            
            oarsub "source cem_env/bin/activate; python3 reinforce_full.py \
                --seed ${seed} \
                --batch_size ${batch_size} \
                --lr ${lr}" \
                -l nodes=4,walltime=4:00:00 \
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