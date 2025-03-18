#!/bin/bash

# Arrays of parameters to sweep
seeds=(0 1 2)
n_evals=(1 2 50)
batch_sizes=(8 16 32 64)
learning_rates=(0.0001 0.001 0.01 0.1)

# Create a logs directory if it doesn't exist
mkdir -p logs

# Loop through all combinations
for seed in "${seeds[@]}"
do
    for n_eval in "${n_evals[@]}"
    do
        for lr in "${learning_rates[@]}"
        do
            for batch_size in "${batch_sizes[@]}"
            do
                echo "Launching training with seed ${seed}, n_evals ${n_eval}, lr ${lr}"
                
                # Create a unique job name
                job_name="reinforce_s${seed}_e${n_eval}_lr${lr}_bs${batch_size}"
                
                oarsub "source cem_env/bin/activate; python3 reinforce_beta_pendulum_sub_baseline.py \
                    --seed ${seed} \
                    --n_evals ${n_eval} \
                    --batch_size ${batch_size} \
                    --lr ${lr}" \
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
done