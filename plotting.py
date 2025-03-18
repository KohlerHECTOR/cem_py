import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from matplotlib.lines import Line2D

def plot_data(data_paths, line_style, color_map, smooth_sigma=2):
    all_train_scores = []
    all_train_samples = []

    for data_path in data_paths:
        try:
            data = np.load(data_path, allow_pickle=True).item()
            all_train_scores.append(data["train_scores"])
            all_train_samples.append(data["total_samples"])
        except FileNotFoundError:
            continue

    if all_train_scores:
        train_scores = np.array(all_train_scores)
        train_samples = np.array(all_train_samples)

        num_points = 100
        indices = np.linspace(0, train_scores.shape[1] - 1, num_points, dtype=int)

        train_mean = np.mean(train_scores, axis=0)[indices]
        train_std = np.std(train_scores, axis=0)[indices]
        train_samples_mean = np.cumsum(np.mean(train_samples, axis=0))[indices]
        
        # Apply Gaussian smoothing to the mean curve
        if smooth_sigma > 0:
            train_mean = gaussian_filter1d(train_mean, sigma=smooth_sigma)

        sns.lineplot(x=train_samples_mean, y=train_mean, linestyle=line_style, color=color_map, linewidth=2)

# Set Seaborn style
sns.set(style="whitegrid")

# Create figure with appropriate size
plt.figure(figsize=(10, 7))

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
            plot_data(data_paths, '-', color)
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
            plot_data(data_paths, '-', color)
            c_idx += 1

# Plot REINFORCE with baseline data
c_idx = 0
combs = np.linspace(0, 1, 4*2*4)  # Adjusted for the new parameter combinations
for i, lr in enumerate([0.0001, 0.001, 0.01, 0.1]):
    for j, evals in enumerate([1, 50]):  # From the shell script
        for k, batch_size in enumerate([8, 16, 32, 64]):  # From the shell script
            data_paths = [
                f'beta_reinforce_sub_baseline_data_Pendulum-v1_batch{batch_size}_lr{lr}_evals{evals}_seed{seed}/training_data_iter.npy'
                for seed in range(3)  # From the shell script
            ]
            color = sns.color_palette("Greens", n_colors=len(combs)*2)[len(combs) + c_idx]
            plot_data(data_paths, '-', color)
            c_idx += 1

# Add legend with representative colors for all algorithms
legend_elements = [
    Line2D([0], [0], color='#e85c47', lw=3, label='CEM'),  # Red color for CEM
    Line2D([0], [0], color='#4878cf', lw=3, label='REINFORCE'),  # Blue color for REINFORCE
    Line2D([0], [0], color='#2ca02c', lw=3, label='REINFORCE+baseline')  # Green color for REINFORCE with baseline
]
plt.legend(handles=legend_elements, fontsize=20, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3)

# Increase font sizes for better readability
plt.xlabel("Total Samples", fontsize=22)
plt.ylabel("RL Objective", fontsize=22)

# Increase tick label sizes
plt.xticks(fontsize=14)
plt.yticks(fontsize=20)

# Add more bottom margin to accommodate the legend
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)
plt.savefig("pendulum_plot.png")
