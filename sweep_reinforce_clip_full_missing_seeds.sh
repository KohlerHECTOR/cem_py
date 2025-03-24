#!/bin/bash

# Arrays of parameters to sweep
seeds=(5 6 7 8 9 10 11 12 13 14)
batch_sizes=(1 2 4 8 16 32)
learning_rates=(0.0001 0.001 0.01)

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
                --clip \
                --lr ${lr}" \
                -l nodes=1,walltime=4:00:00 \
                -p nancy \
                -q production \
                -n "${job_name}" \
                -O "logs/${job_name}.%jobid%.out" \
                -E "logs/${job_name}.%jobid%.err"
            
            # Add a small delay between submissions
            sleep 1.2
        done
    done
done
