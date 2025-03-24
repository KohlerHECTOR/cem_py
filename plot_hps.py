import numpy as np
import matplotlib.pyplot as plt

# Create figures for each algorithm
fig_cem, axes_cem = plt.subplots(1, 2, figsize=(12, 5))
fig_reinforce, axes_reinforce = plt.subplots(1, 2, figsize=(12, 5))

# Style function for axes
def style_axes(ax):
    ax.tick_params(axis='both', which='major', labelsize=14)  # Increase tick label size
    ax.tick_params(axis='both', which='both', width=2)        # Make tick marks thicker
    ax.tick_params(axis='both', which='major', length=6)      # Make tick marks longer

# CEM analysis
batch_sizes_cem = [8, 16, 32, 64, 128, 256]
elite_props = [0.05, 0.1, 0.125, 0.25, 0.5]

# For batch size
best_returns_bs = []
for bs in batch_sizes_cem:
    returns_for_bs = []
    for prop in elite_props:
        # Get average return across seeds for this combination
        returns_this_combo = []
        valid_seeds = []
        for s in [0, 1, 2]:
            try:
                a = np.load(f"results_sweep/results_cem_elite_prop{prop}_batch_size{bs}_seed{s}.npy")
                if np.all(~np.isnan(a[:, 1]) & ~np.isinf(a[:, 1]) & (a[:, 1] < 0)):
                    returns_this_combo.append(a[-1, 1])
                    valid_seeds.append(s)
            except FileNotFoundError:
                continue
        if len(returns_this_combo) > 0:
            returns_for_bs.append(np.mean(returns_this_combo))
    best_returns_bs.append(max(returns_for_bs) if returns_for_bs else np.nan)

axes_cem[0].plot(batch_sizes_cem, best_returns_bs, 'o-', linewidth=3)
axes_cem[0].set_xlabel('Batch Size', fontsize=16)
axes_cem[0].set_ylabel('Best Final Return', fontsize=16)
axes_cem[0].set_xscale('log', base=2)
axes_cem[0].grid(True)

# For elite proportion
best_returns_ep = []
for ep in elite_props:
    returns_for_ep = []
    for bs in batch_sizes_cem:
        # Get average return across seeds for this combination
        returns_this_combo = []
        for s in [0, 1, 2]:
            try:
                a = np.load(f"results_sweep/results_cem_elite_prop{ep}_batch_size{bs}_seed{s}.npy")
                if np.all(~np.isnan(a[:, 1]) & ~np.isinf(a[:, 1]) & (a[:, 1] < 0)):
                    returns_this_combo.append(a[-1, 1])
            except FileNotFoundError:
                continue
        if len(returns_this_combo) > 0:
            returns_for_ep.append(np.mean(returns_this_combo))
    best_returns_ep.append(max(returns_for_ep) if returns_for_ep else np.nan)

axes_cem[1].plot(elite_props, best_returns_ep, 'o-', linewidth=3)
axes_cem[1].set_xlabel('Elite Proportion', fontsize=16)
axes_cem[1].set_ylabel('Best Final Return', fontsize=16)
axes_cem[1].grid(True)

# REINFORCE analysis
batch_sizes_reinforce = [1, 2, 4, 8, 16, 32, 64, 128]
learning_rates = [0.0001, 0.001, 0.01, 0.1]

# For batch size
best_returns_bs = []
for bs in batch_sizes_reinforce:
    returns_for_bs = []
    for lr in learning_rates:
        for clip in [True, False]:
            returns_this_combo = []
            for s in [0, 1, 2]:
                try:
                    filename = f"results_sweep/results_reinforce_lr{lr}_batch_size{bs}_clip{clip}_seed{s}.npy"
                    a = np.load(filename)
                    if np.all(~np.isnan(a[:, 1]) & ~np.isinf(a[:, 1]) & (a[:, 1] < 0)):
                        returns_this_combo.append(a[-1, 1])
                except FileNotFoundError:
                    continue
            if len(returns_this_combo) > 0:
                returns_for_bs.append(np.mean(returns_this_combo))
    best_returns_bs.append(max(returns_for_bs) if returns_for_bs else np.nan)

axes_reinforce[0].plot(batch_sizes_reinforce, best_returns_bs, 'o-', linewidth=3)
axes_reinforce[0].set_xlabel('Batch Size', fontsize=16)
axes_reinforce[0].set_ylabel('Best Final Return', fontsize=16)
axes_reinforce[0].set_xscale('log', base=2)
axes_reinforce[0].grid(True)

# For learning rate
best_returns_lr = []
for lr in learning_rates:
    returns_for_lr = []
    for bs in batch_sizes_reinforce:
        for clip in [True, False]:
            returns_this_combo = []
            for s in [0, 1, 2]:
                try:
                    filename = f"results_sweep/results_reinforce_lr{lr}_batch_size{bs}_clip{clip}_seed{s}.npy"
                    a = np.load(filename)
                    if np.all(~np.isnan(a[:, 1]) & ~np.isinf(a[:, 1]) & (a[:, 1] < 0)):
                        returns_this_combo.append(a[-1, 1])
                except FileNotFoundError:
                    continue
            if len(returns_this_combo) > 0:
                returns_for_lr.append(np.mean(returns_this_combo))
    best_returns_lr.append(max(returns_for_lr) if returns_for_lr else np.nan)

axes_reinforce[1].plot(learning_rates, best_returns_lr, 'o-', linewidth=3)
axes_reinforce[1].set_xlabel('Learning Rate', fontsize=16)
axes_reinforce[1].set_ylabel('Best Final Return', fontsize=16)
axes_reinforce[1].set_xscale('log')
axes_reinforce[1].grid(True)

# Style function for axes
style_axes(axes_cem[0])
style_axes(axes_cem[1])
style_axes(axes_reinforce[0])
style_axes(axes_reinforce[1])

# Save the figures
plt.figure(fig_cem.number)
plt.tight_layout()
plt.savefig("cem_hyperparameters.pdf", bbox_inches='tight', dpi=300)

plt.figure(fig_reinforce.number)
plt.tight_layout()
plt.savefig("reinforce_hyperparameters.pdf", bbox_inches='tight', dpi=300)