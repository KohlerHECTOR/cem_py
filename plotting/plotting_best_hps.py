import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_data(data_paths, line_style, color, label, ax1, ax2):
    all_train_scores = []
    all_train_samples = []
    all_walltimes = []

    for data_path in data_paths:
        try:
            data = np.load(data_path, allow_pickle=True).item()
            all_train_scores.append(data["train_scores"])
            all_train_samples.append(data["total_samples"])
            all_walltimes.append(
                data.get("walltime", np.zeros_like(data["train_scores"]))
            )
        except FileNotFoundError:
            continue

    if all_train_scores:
        train_scores = np.array(all_train_scores)
        train_samples = np.array(all_train_samples)
        walltimes = np.array(all_walltimes)

        num_points = 100
        indices = np.linspace(0, train_scores.shape[1] - 1, num_points, dtype=int)

        train_mean = np.mean(train_scores, axis=0)[indices]
        train_std = np.std(train_scores, axis=0)[indices]
        train_samples_mean = np.cumsum(np.mean(train_samples, axis=0))[indices]
        walltime_mean = np.mean(walltimes, axis=0)[indices]

        # Plot mean line on first subplot (samples)
        ax1.plot(
            train_samples_mean,
            train_mean,
            linestyle=line_style,
            color=color,
            linewidth=4,
            label=label,
        )

        # Add shaded area for standard deviation on first subplot
        ax1.fill_between(
            train_samples_mean,
            train_mean - train_std,
            train_mean + train_std,
            color=color,
            alpha=0.2,
        )

        # Plot mean line on second subplot (walltime)
        ax2.plot(
            walltime_mean,
            train_mean,
            linestyle=line_style,
            color=color,
            linewidth=4,
            label=label,
        )

        # Add shaded area for standard deviation on second subplot
        ax2.fill_between(
            walltime_mean,
            train_mean - train_std,
            train_mean + train_std,
            color=color,
            alpha=0.2,
        )


# Set Seaborn style
sns.set(style="whitegrid")

# Create figure with two subplots sharing y-axis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7), sharey=True)

# Plot only the best hyperparameters for each algorithm

# Best CEM hyperparameters (n_pop=16, elite_prop=0.125)
cem_data_paths = [
    f"beta_cem_data_Pendulum-v1_std_pop16_size16_evals50_elite0.125_seed{seed}/training_data_iter.npy"
    for seed in range(3, 3 + 15)
]
plot_data(cem_data_paths, "-", "#e85c47", "CEM", ax1, ax2)

# Best REINFORCE hyperparameters (lr=0.001, batch_size=64)
reinforce_data_paths = [
    f"beta_reinforce_data_Pendulum-v1_batch64_lr0.001_evals1_seed{seed}/training_data_iter.npy"
    for seed in range(3, 3 + 15)
]
plot_data(reinforce_data_paths, "-", "#4878cf", "REINFORCE", ax1, ax2)

# Best REINFORCE with baseline hyperparameters (lr=0.001, batch_size=64)
reinforce_baseline_data_paths = [
    f"beta_reinforce_sub_baseline_data_Pendulum-v1_batch64_lr0.001_evals1_seed{seed}/training_data_iter.npy"
    for seed in range(3, 3 + 15)
]
plot_data(reinforce_baseline_data_paths, "-", "#2ca02c", "REINFORCE+baseline", ax1, ax2)

es_data_paths = [
    f"beta_es_data_Pendulum-v1_pop8_size16_evals50_lr0.01_std0.1_seed{seed}/training_data_iter.npy"
    for seed in range(3, 3 + 15)
]
plot_data(es_data_paths, "-", "purple", "ES", ax1, ax2)

# Set labels for first subplot
ax1.set_xlabel("Total Samples", fontsize=22)
ax1.set_ylabel("RL Objective", fontsize=22)
ax1.tick_params(axis="both", which="major", labelsize=18)
ax1.tick_params(axis="y", which="major", labelsize=20)

# Set labels for second subplot
ax2.set_xlabel("Wall Time (seconds)", fontsize=22)
ax2.tick_params(axis="both", which="major", labelsize=18)
ax2.tick_params(axis="y", which="major", labelsize=20)

# Add a single legend at the bottom of the figure
# Add more bottom margin to accommodate the legend
plt.legend(fontsize=25)
plt.tight_layout()
plt.savefig("pendulum_best_hp_plot.png")
