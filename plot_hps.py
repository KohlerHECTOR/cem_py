import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams, rc
from matplotlib.lines import Line2D

# Initialize list to store all results
all_results = []
success, fail = 0, 0

# Set up plotting style
sns.set_style("white")
rcParams['legend.loc'] = 'best'
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rc('text', usetex=False)

# Create figure with subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
axes = [ax1, ax2, ax3]

# Process CEM results with updated styling
cem_curves = []  # Store (final_return, plot_data) for alpha calculation
success, fail = 0, 0
for p in [8, 16, 32, 64, 128, 256]:
    for prop in [0.05, 0.1, 0.125, 0.25, 0.5]:
        to_plot_x = []
        to_plot_y = []
        for s in [0, 1, 2]:
            try:
                a = np.load(f"results_sweep/results_cem_elite_prop{prop}_batch_size{p}_seed{s}.npy")
                valid_mask = ~(np.isnan(a[:, 1]) | np.isinf(a[:, 1])) & (a[:, 1] < 0)
                if not np.all(valid_mask):
                    raise FileNotFoundError
                success += 1
                to_plot_y.append(gaussian_filter1d(a[:,1], sigma=1e-4))
                to_plot_x.append(a[:,0])
            except FileNotFoundError:
                fail += 1
                continue
        if to_plot_x and to_plot_y:
            y_mean = np.mean(to_plot_y, axis=0)
            final_return = y_mean[-1]  # Get the last point's return
            plot_data = (np.mean(to_plot_x, axis=0), y_mean)
            cem_curves.append((final_return, plot_data))
print(success / (fail+success))

# Plot CEM curves with normalized alphas based on final returns
if cem_curves:
    returns = np.array([r for r, _ in cem_curves])
    min_return = np.min(returns)
    max_return = np.max(returns)
    for final_return, (x, y) in cem_curves:
        alpha = 0.05 + 0.95 * (final_return - min_return) / (max_return - min_return) if max_return != min_return else 0.75
        linewidth = 0.5 + 2.5 * (final_return - min_return) / (max_return - min_return) if max_return != min_return else 1.5
        ax1.plot(x, y, c='#FF7F7F', alpha=min(1, alpha), linewidth=linewidth)

# Process REINFORCE results with updated styling
reinforce_curves = []
success, fail = 0, 0

for bs in [1, 2, 4, 8, 16, 32, 64, 128]:
    for lr in [0.0001, 0.001, 0.01, 0.1]:
        for clip in [False, True]:
            to_plot_x = []
            to_plot_y = []
            for s in [0, 1, 2]:
                try:
                    a = np.load(f"results_sweep/results_reinforce_lr{lr}_batch_size{bs}_clip{clip}_seed{s}.npy")
                    valid_mask = ~(np.isnan(a[:, 1]) | np.isinf(a[:, 1])) & (a[:, 1] < 0)
                    if not np.all(valid_mask):
                        raise FileNotFoundError
                    success += 1
                    to_plot_y.append(gaussian_filter1d(a[:,1], sigma=1e-4))
                    to_plot_x.append(a[:,0])
                except FileNotFoundError:
                    fail += 1
                    continue
            if to_plot_x and to_plot_y:
                y_mean = np.mean(to_plot_y, axis=0)
                final_return = y_mean[-1]  # Get the last point's return
                plot_data = (np.mean(to_plot_x, axis=0), y_mean)
                reinforce_curves.append((final_return, plot_data))
print(success / (fail+success))

# Plot REINFORCE curves with normalized alphas based on final returns
if reinforce_curves:
    returns = np.array([r for r, _ in reinforce_curves])
    min_return = np.min(returns)
    max_return = np.max(returns)
    for final_return, (x, y) in reinforce_curves:
        alpha = 0.05 + 0.95 * (final_return - min_return) / (max_return - min_return) if max_return != min_return else 0.75
        linewidth = 0.5 + 2.5 * (final_return - min_return) / (max_return - min_return) if max_return != min_return else 1.5
        ax2.plot(x, y, c='#7F7FFF', alpha=min(1, alpha), linewidth=linewidth)

# Process ES results with updated styling
success, fail = 0, 0
es_curves = []
for bs in [8, 16, 32, 64, 128, 256]:
    for lr in [0.000001, 0.00001, 0.0001, 0.001, 0.01]:
        for noise_std in [1, 0.1, 0.01]:
            to_plot_x = []
            to_plot_y = []
            for s in [0, 1, 2]:
                try:
                    a = np.load(f"results_sweep/results_es_lr{lr}_noise_std{noise_std}_batch_size{bs}_seed{s}.npy")
                    valid_mask = ~(np.isnan(a[:, 1]) | np.isinf(a[:, 1])) & (a[:, 1] < 0)
                    if not np.all(valid_mask):
                        raise FileNotFoundError
                    success += 1
                    to_plot_y.append(gaussian_filter1d(a[:,1], sigma=1e-4))
                    to_plot_x.append(a[:,0])
                except FileNotFoundError:

                    fail += 1
                    continue
            if to_plot_x and to_plot_y:
                y_mean = np.mean(to_plot_y, axis=0)
                final_return = y_mean[-1]  # Get the last point's return
                plot_data = (np.mean(to_plot_x, axis=0), y_mean)
                es_curves.append((final_return, plot_data))

print(success / (fail+success))

# Plot ES curves with normalized alphas based on final returns
if es_curves:
    returns = np.array([r for r, _ in es_curves])
    min_return = np.min(returns)
    max_return = np.max(returns)
    for final_return, (x, y) in es_curves:
        alpha = 0.05 + 0.95 * (final_return - min_return) / (max_return - min_return) if max_return != min_return else 0.75
        linewidth = 0.5 + 2.5 * (final_return - min_return) / (max_return - min_return) if max_return != min_return else 1.5
        ax3.plot(x, y, c='#AF7FAF', alpha=min(1, alpha), linewidth=linewidth)

# Style each subplot
for ax, title in zip(axes, ['CEM', 'REINFORCE', 'ES']):
    ax.grid(True, alpha=0.7)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.set_xlabel('Samples', fontsize=14)
    ax.set_title(title, fontsize=16)

# Add shared y-label
fig.text(-0.01, 0.5, 'Return', va='center', rotation='vertical', fontsize=16)

# Create manual legend below the plots
legend_elements = [
    Line2D([0], [0], color='#FF7F7F', label='CEM', linewidth=2),
    Line2D([0], [0], color='#7F7FFF', label='REINFORCE', linewidth=2),
    Line2D([0], [0], color='#AF7FAF', label='ES', linewidth=2)
]

plt.tight_layout()
# Adjust layout to make room for legend
plt.savefig("algorithm_comparison.pdf", bbox_inches='tight', dpi=300)

# Create figures for each algorithm
fig_cem, axes_cem = plt.subplots(1, 2, figsize=(12, 5))
fig_reinforce, axes_reinforce = plt.subplots(1, 3, figsize=(15, 5))
fig_es, axes_es = plt.subplots(1, 3, figsize=(15, 5))

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

axes_cem[0].plot(batch_sizes_cem, best_returns_bs, 'o-')
axes_cem[0].set_xlabel('Batch Size')
axes_cem[0].set_ylabel('Best Final Return')
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

axes_cem[1].plot(elite_props, best_returns_ep, 'o-')
axes_cem[1].set_xlabel('Elite Proportion')
axes_cem[1].set_ylabel('Best Final Return')
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

axes_reinforce[0].plot(batch_sizes_reinforce, best_returns_bs, 'o-')
axes_reinforce[0].set_xlabel('Batch Size')
axes_reinforce[0].set_ylabel('Best Final Return')
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

axes_reinforce[1].plot(learning_rates, best_returns_lr, 'o-')
axes_reinforce[1].set_xlabel('Learning Rate')
axes_reinforce[1].set_ylabel('Best Final Return')
axes_reinforce[1].set_xscale('log')
axes_reinforce[1].grid(True)

# Add new analysis for clipping
clip_options = [False, True]
best_returns_clip = []
for clip in clip_options:
    returns_for_clip = []
    for bs in batch_sizes_reinforce:
        for lr in learning_rates:
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
                returns_for_clip.append(np.mean(returns_this_combo))
    best_returns_clip.append(max(returns_for_clip) if returns_for_clip else np.nan)

axes_reinforce[2].plot(['No Clip', 'Clip'], best_returns_clip, 'o-')
axes_reinforce[2].set_xlabel('Gradient Clipping')
axes_reinforce[2].set_ylabel('Best Final Return')
axes_reinforce[2].grid(True)

# ES analysis
batch_sizes_es = [8, 16, 32, 64, 128, 256]
learning_rates_es = [0.000001, 0.00001, 0.0001, 0.001, 0.01]
noise_stds = [1, 0.1, 0.01]

# For batch size
best_returns_bs = []
for bs in batch_sizes_es:
    returns_for_bs = []
    for lr in learning_rates_es:
        for std in noise_stds:
            returns_this_combo = []
            for s in [0, 1, 2]:
                try:
                    a = np.load(f"results_sweep/results_es_lr{lr}_noise_std{std}_batch_size{bs}_seed{s}.npy")
                    if np.all(~np.isnan(a[:, 1]) & ~np.isinf(a[:, 1]) & (a[:, 1] < 0)):
                        returns_this_combo.append(a[-1, 1])
                except FileNotFoundError:
                    continue
            if len(returns_this_combo) > 0:
                returns_for_bs.append(np.mean(returns_this_combo))
    best_returns_bs.append(max(returns_for_bs) if returns_for_bs else np.nan)

axes_es[0].plot(batch_sizes_es, best_returns_bs, 'o-')
axes_es[0].set_xlabel('Batch Size')
axes_es[0].set_ylabel('Best Final Return')
axes_es[0].set_xscale('log', base=2)
axes_es[0].grid(True)

# For learning rate
best_returns_lr = []
for lr in learning_rates_es:
    returns_for_lr = []
    for bs in batch_sizes_es:
        for std in noise_stds:
            returns_this_combo = []
            for s in [0, 1, 2]:
                try:
                    a = np.load(f"results_sweep/results_es_lr{lr}_noise_std{std}_batch_size{bs}_seed{s}.npy")
                    if np.all(~np.isnan(a[:, 1]) & ~np.isinf(a[:, 1]) & (a[:, 1] < 0)):
                        returns_this_combo.append(a[-1, 1])
                except FileNotFoundError:
                    continue
            if len(returns_this_combo) > 0:
                returns_for_lr.append(np.mean(returns_this_combo))
    best_returns_lr.append(max(returns_for_lr) if returns_for_lr else np.nan)

axes_es[1].plot(learning_rates_es, best_returns_lr, 'o-')
axes_es[1].set_xlabel('Learning Rate')
axes_es[1].set_ylabel('Best Final Return')
axes_es[1].set_xscale('log')
axes_es[1].grid(True)

# For noise std
best_returns_std = []
for std in noise_stds:
    returns_for_std = []
    for bs in batch_sizes_es:
        for lr in learning_rates_es:
            returns_this_combo = []
            for s in [0, 1, 2]:
                try:
                    a = np.load(f"results_sweep/results_es_lr{lr}_noise_std{std}_batch_size{bs}_seed{s}.npy")
                    if np.all(~np.isnan(a[:, 1]) & ~np.isinf(a[:, 1]) & (a[:, 1] < 0)):
                        returns_this_combo.append(a[-1, 1])
                except FileNotFoundError:
                    continue
            if len(returns_this_combo) > 0:
                returns_for_std.append(np.mean(returns_this_combo))
    best_returns_std.append(max(returns_for_std) if returns_for_std else np.nan)

axes_es[2].plot(noise_stds, best_returns_std, 'o-')
axes_es[2].set_xlabel('Noise Std')
axes_es[2].set_ylabel('Best Final Return')
axes_es[2].set_xscale('log')
axes_es[2].grid(True)

# Save the figures
plt.figure(fig_cem.number)
plt.tight_layout()
plt.savefig("cem_hyperparameters.pdf", bbox_inches='tight', dpi=300)

plt.figure(fig_reinforce.number)
plt.tight_layout()
plt.savefig("reinforce_hyperparameters.pdf", bbox_inches='tight', dpi=300)

plt.figure(fig_es.number)
plt.tight_layout()
plt.savefig("es_hyperparameters.pdf", bbox_inches='tight', dpi=300)