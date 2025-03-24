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
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
axes = [ax1, ax2]

# Add downsampling function after the initial imports
def downsample_curve(x, y, n_points=50):
    if len(x) <= n_points:
        return x, y
    indices = np.linspace(0, len(x) - 1, n_points, dtype=int)
    return x[indices], y[indices]

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
                to_plot_y.append(a[:,1])
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
        x_down, y_down = downsample_curve(x, y)
        ax1.plot(x_down, y_down, c='#FF7F7F', alpha=0.8, linewidth=1)

# Process REINFORCE results with updated styling
reinforce_curves = []
success, fail = 0, 0

for bs in [1, 2, 4, 8, 16, 32, 64, 128]:
    for lr in [0.0001, 0.001, 0.01, 0.1]:
        for clip in [True]:
            to_plot_x = []
            to_plot_y = []
            for s in [0, 1, 2]:
                try:
                    a = np.load(f"results_sweep/results_reinforce_lr{lr}_batch_size{bs}_clip{clip}_seed{s}.npy")
                    valid_mask = ~(np.isnan(a[:, 1]) | np.isinf(a[:, 1])) & (a[:, 1] < 0)
                    if not np.all(valid_mask):
                        raise FileNotFoundError
                    success += 1
                    to_plot_y.append(a[:,1])
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
        x_down, y_down = downsample_curve(x, y)
        ax2.plot(x_down, y_down, c='#7F7FFF', alpha=0.8, linewidth=1)

# Style each subplot
for ax, title in zip(axes, ['CEM', 'REINFORCE']):
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
    Line2D([0], [0], color='#7F7FFF', label='REINFORCE', linewidth=2)
]

plt.tight_layout()
# Adjust layout to make room for legend
plt.savefig("algorithm_comparison.pdf", bbox_inches='tight', dpi=300)

# Create figures for each algorithm
fig_cem, axes_cem = plt.subplots(1, 2, figsize=(12, 5))
fig_reinforce, axes_reinforce = plt.subplots(1, 3, figsize=(15, 5))

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

axes_reinforce[2].plot(['No Clip', 'Clip'], best_returns_clip, 'o-', linewidth=3)
axes_reinforce[2].set_xlabel('Gradient Clipping', fontsize=16)
axes_reinforce[2].set_ylabel('Best Final Return', fontsize=16)
axes_reinforce[2].grid(True)

# Style function for axes
style_axes(axes_cem[0])
style_axes(axes_cem[1])
style_axes(axes_reinforce[0])
style_axes(axes_reinforce[1])
style_axes(axes_reinforce[2])

# Save the figures
plt.figure(fig_cem.number)
plt.tight_layout()
plt.savefig("cem_hyperparameters.pdf", bbox_inches='tight', dpi=300)

plt.figure(fig_reinforce.number)
plt.tight_layout()
plt.savefig("reinforce_hyperparameters.pdf", bbox_inches='tight', dpi=300)