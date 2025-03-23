import os
import glob

# Get all .npy files in the results_sweep directory that contain 'reinforce'
files = glob.glob("results_sweep/results_reinforce*.npy")

for file in files:
    # Only process files that don't contain 'clip'
    if 'clip' not in file:
        # Split the filename at '_seed'
        parts = file.split('_seed')
        if len(parts) == 2:
            # Create new filename by inserting 'clipFalse' before '_seed'
            new_filename = f"{parts[0]}_clipFalse_seed{parts[1]}"
            try:
                os.rename(file, new_filename)
                print(f"Renamed: {file} -> {new_filename}")
            except Exception as e:
                print(f"Error renaming {file}: {e}")