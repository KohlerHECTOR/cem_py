import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from matplotlib.lines import Line2D

def plot_data(data_paths, line_style, color_map, alpha=0.3):
    # Store data for calculating mean trend
    all_x_data = []
    all_y_data = []
    
    # Plot individual seeds
    for data_path in data_paths:
        try:
            data = np.load(data_path, allow_pickle=True).item()
            train_scores = data["train_scores"]
            train_samples = data["walltime"]
            
            num_points = 100
            indices = np.linspace(0, train_scores.shape[0] - 1, num_points, dtype=int)
            
            train_scores = train_scores[indices]
            train_samples_cumsum = train_samples[indices]
            
            # Store data for mean calculation
            all_x_data.append(train_samples_cumsum)
            all_y_data.append(train_scores)
            
            # Plot individual seed
            sns.lineplot(x=train_samples_cumsum, y=train_scores, linestyle=line_style, 
                         color=color_map, linewidth=1, alpha=alpha)
        except FileNotFoundError:
            continue
    
    return all_x_data, all_y_data

def compute_mean_performance(x_data_list, y_data_list):
    """Compute mean performance for a set of runs with the same hyperparameters."""
    if not x_data_list or not y_data_list or len(x_data_list) == 0:
        return None, None, float('-inf')
    
    # Find common x range
    valid_runs = [(x, y) for x, y in zip(x_data_list, y_data_list) if len(x) > 0 and len(y) > 0]
    if not valid_runs:
        return None, None, float('-inf')
    
    x_arrays, y_arrays = zip(*valid_runs)
    
    min_x = max([min(x) for x in x_arrays if len(x) > 0])
    max_x = min([max(x) for x in x_arrays if len(x) > 0])
    common_x = np.linspace(min_x, max_x, 100)
    
    # Interpolate all data to common x points
    interpolated_y = []
    for x, y in valid_runs:
        if len(x) > 1:  # Only interpolate if we have at least 2 points
            interpolated_y.append(np.interp(common_x, x, y))
    
    if not interpolated_y:
        return None, None, float('-inf')
    
    # Calculate mean across all data
    mean_y = np.mean(interpolated_y, axis=0)
    
    # Return mean final score
    mean_final_score = mean_y[-1] if len(mean_y) > 0 else float('-inf')
    
    return common_x, mean_y, mean_final_score

# Set Seaborn style
sns.set(style="whitegrid")

# Create figure with appropriate size
plt.figure(figsize=(10, 7))

# Store data for global algorithm trends
cem_x_data = []
cem_y_data = []
reinforce_x_data = []
reinforce_y_data = []
reinforce_baseline_x_data = []
reinforce_baseline_y_data = []
new_baseline_x_data = []
new_baseline_y_data = []

# Store best hyperparameter configurations
cem_best_config_x = None
cem_best_config_y = None
cem_best_config_score = float('-inf')

reinforce_best_config_x = None
reinforce_best_config_y = None
reinforce_best_config_score = float('-inf')

reinforce_baseline_best_config_x = None
reinforce_baseline_best_config_y = None
reinforce_baseline_best_config_score = float('-inf')

new_baseline_best_config_x = None
new_baseline_best_config_y = None
new_baseline_best_config_score = float('-inf')

# Plot CEM data
c_idx = 0
combs = np.linspace(0, 1, 1*4*2)
for i, std_ in enumerate([True]):
    for j, n_pop in enumerate([8, 16, 32, 64]):
        for k, elite in enumerate([0.125, 0.25]):
            data_paths = [
                f'beta_cem_data_Pendulum-v1_{"" if not std_ else "std_"}pop{n_pop}_size16_evals50_elite{elite}_seed{seed}/training_data_iter.npy'
                for seed in range(3)
            ]
            # Use only the second half of the inferno colormap
            color = sns.color_palette("OrRd", n_colors=len(combs)*2)[len(combs) + c_idx]
            x_data, y_data = plot_data(data_paths, '-', color)
            
            # Compute mean performance for this hyperparameter configuration
            config_x, config_y, config_score = compute_mean_performance(x_data, y_data)
            
            # Check if this is the best configuration
            if config_score > cem_best_config_score:
                cem_best_config_score = config_score
                cem_best_config_x = config_x
                cem_best_config_y = config_y
            
            cem_x_data.extend(x_data)
            cem_y_data.extend(y_data)
            c_idx += 1

# Plot REINFORCE data
c_idx = 0
combs = np.linspace(0, 1, 4*3*4)
for i, lr in enumerate([0.0001, 0.001, 0.01, 0.1]):
    for j, evals in enumerate([1, 2, 50]):
        for k, batch_size in enumerate([8, 16, 32, 64]):
            data_paths = [
                f'beta_reinforce_data_Pendulum-v1_batch{batch_size}_lr{lr}_evals{evals}_seed{seed}/training_data_iter.npy'
                for seed in range(3)
            ]
            color = sns.color_palette("GnBu", n_colors=len(combs)*2)[len(combs) + c_idx]
            x_data, y_data = plot_data(data_paths, '-', color)
            
            # Compute mean performance for this hyperparameter configuration
            config_x, config_y, config_score = compute_mean_performance(x_data, y_data)
            
            # Check if this is the best configuration
            if config_score > reinforce_best_config_score:
                reinforce_best_config_score = config_score
                reinforce_best_config_x = config_x
                reinforce_best_config_y = config_y
            
            reinforce_x_data.extend(x_data)
            reinforce_y_data.extend(y_data)
            c_idx += 1

# Plot REINFORCE with baseline data
c_idx = 0
combs = np.linspace(0, 1, 4*3*4)  # Adjusted for the new parameter combinations
for i, lr in enumerate([0.0001, 0.001, 0.01, 0.1]):
    for j, evals in enumerate([1, 50]):  # From the shell script
        for k, batch_size in enumerate([8, 16, 32, 64]):  # From the shell script
            data_paths = [
                f'beta_reinforce_sub_baseline_data_Pendulum-v1_batch{batch_size}_lr{lr}_evals{evals}_seed{seed}/training_data_iter.npy'
                for seed in range(3)  # From the shell script
            ]
            color = sns.color_palette("Greens", n_colors=len(combs)*2)[len(combs) + c_idx]
            x_data, y_data = plot_data(data_paths, '-', color)
            
            # Compute mean performance for this hyperparameter configuration
            config_x, config_y, config_score = compute_mean_performance(x_data, y_data)
            
            # Check if this is the best configuration
            if config_score > reinforce_baseline_best_config_score:
                reinforce_baseline_best_config_score = config_score
                reinforce_baseline_best_config_x = config_x
                reinforce_baseline_best_config_y = config_y
            
            reinforce_baseline_x_data.extend(x_data)
            reinforce_baseline_y_data.extend(y_data)
            c_idx += 1

# Plot new baseline data
c_idx = 0
combs = np.linspace(0, 1, 4*3*4)  # Adjust based on your parameter combinations
for i, lr in enumerate([0.0001, 0.001, 0.01, 0.1]):  # Adjust parameters as needed
    for j, noise_std in enumerate([0.1, 1, 0.01]):  # Adjust parameters as needed
        for k, n_pop in enumerate([8, 16, 32, 64]):  # Adjust parameters as needed
            data_paths = [
                f'beta_es_data_Pendulum-v1_pop{n_pop}_size16_evals50_lr{lr}_std{noise_std}_seed{seed}/training_data_iter.npy'
                for seed in range(3)  # Adjust seed count as needed
            ]
            color = sns.color_palette("Purples", n_colors=len(combs)*2)[len(combs) + c_idx]
            x_data, y_data = plot_data(data_paths, '-', color)
            
            # Compute mean performance for this hyperparameter configuration
            config_x, config_y, config_score = compute_mean_performance(x_data, y_data)
            
            # Check if this is the best configuration
            if config_score > new_baseline_best_config_score:
                new_baseline_best_config_score = config_score
                new_baseline_best_config_x = config_x
                new_baseline_best_config_y = config_y
            
            new_baseline_x_data.extend(x_data)
            new_baseline_y_data.extend(y_data)
            c_idx += 1

# Plot global trends for each algorithm
def plot_global_trend(x_data, y_data, color, label):
    if len(x_data) > 1:
        # Find common x range
        min_x = max([min(x) for x in x_data if len(x) > 0])
        max_x = min([max(x) for x in x_data if len(x) > 0])
        common_x = np.linspace(min_x, max_x, 100)
        
        # Interpolate all data to common x points
        interpolated_y = []
        for x, y in zip(x_data, y_data):
            if len(x) > 1:  # Only interpolate if we have at least 2 points
                interpolated_y.append(np.interp(common_x, x, y))
        
        if interpolated_y:
            # Calculate mean across all data
            mean_y = np.mean(interpolated_y, axis=0)
            
            # Plot global trend with thick line
            plt.plot(common_x, mean_y, color=color, linewidth=5, label=label)

# Plot global trends
plot_global_trend(cem_x_data, cem_y_data, '#e85c47', 'CEM (global)')
plot_global_trend(reinforce_x_data, reinforce_y_data, '#4878cf', 'REINFORCE (global)')
plot_global_trend(reinforce_baseline_x_data, reinforce_baseline_y_data, '#2ca02c', 'REINFORCE+baseline (global)')
plot_global_trend(new_baseline_x_data, new_baseline_y_data, '#9467bd', 'ES (global)')

# Plot best hyperparameter configurations for each algorithm with dotted lines
if cem_best_config_x is not None and cem_best_config_y is not None:
    plt.plot(cem_best_config_x, cem_best_config_y, linestyle='dashed', color='#e85c47', linewidth=4, label='CEM (best config)')

if reinforce_best_config_x is not None and reinforce_best_config_y is not None:
    plt.plot(reinforce_best_config_x, reinforce_best_config_y, linestyle='dashed', color='#4878cf', linewidth=4, label='REINFORCE (best config)')

if reinforce_baseline_best_config_x is not None and reinforce_baseline_best_config_y is not None:
    plt.plot(reinforce_baseline_best_config_x, reinforce_baseline_best_config_y, linestyle='dashed', color='#2ca02c', linewidth=4, label='REINFORCE+baseline (best config)')

if new_baseline_best_config_x is not None and new_baseline_best_config_y is not None:
    plt.plot(new_baseline_best_config_x, new_baseline_best_config_y, linestyle='dashed', color='#9467bd', linewidth=4, label='ES (best config)')

# Update legend with best configurations
legend_elements = [
    Line2D([0], [0], color='#e85c47', lw=3, label='CEM (global)'),
    Line2D([0], [0], color='#e85c47', linestyle='dashed', lw=3, label='CEM (best config)'),
    Line2D([0], [0], color='#4878cf', lw=3, label='REINFORCE (global)'),
    Line2D([0], [0], color='#4878cf', linestyle='dashed', lw=3, label='REINFORCE (best config)'),
    Line2D([0], [0], color='#2ca02c', lw=3, label='REINFORCE+baseline (global)'),
    Line2D([0], [0], color='#2ca02c', linestyle='dashed', lw=3, label='REINFORCE+baseline (best config)'),
    Line2D([0], [0], color='#9467bd', lw=3, label='ES (global)'),
    Line2D([0], [0], color='#9467bd', linestyle='dashed', lw=3, label='ES (best config)')
]
plt.legend(handles=legend_elements, fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4)

# Increase font sizes for better readability
plt.xlabel("Wall time (s)", fontsize=22)
plt.ylabel("RL Objective", fontsize=22)

# Increase tick label sizes
plt.xticks(fontsize=14)
plt.yticks(fontsize=20)

# Add more bottom margin to accommodate the legend
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)
plt.savefig("pendulum_plot_walltime.png")
