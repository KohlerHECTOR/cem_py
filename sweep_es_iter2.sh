#!/bin/bash

# Arrays of parameters to sweep
seeds=(0 1 2)
n_pops=(64 128 256)
stds=(0.1 0.01)
lrs=(0.000001 0.00001 0.0001)

# Create a logs directory if it doesn't exist
mkdir -p logs

# Loop through all combinations
for seed in "${seeds[@]}"
do
    for n_pop in "${n_pops[@]}"
    do
        for std in "${stds[@]}"
        do
            for lr in "${lrs[@]}"
            do
                # Create a unique job name
                job_name="es_s${seed}_e${n_pop}_lr${lr}_std${std}"
                
                oarsub "source cem_env/bin/activate; python3 es.py \
                    --seed ${seed} \
                    --batch_size ${n_pop} \
                    --lr ${lr} \
                    --noise_std ${std}" \
                    -l nodes=12,walltime=4:00:00 \
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