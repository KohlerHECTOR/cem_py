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
    for prop in [0.125, 0.25, 0.5]:
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
for bs in [8, 16, 32, 64, 128]:
    for lr in [0.0001, 0.001, 0.01, 0.1]:
        to_plot_x = []
        to_plot_y = []
        for s in [0, 1, 2]:
            try:
                a = np.load(f"results_sweep/results_reinforce_lr{lr}_batch_size{bs}_seed{s}.npy")
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


for bs in [8, 16, 32, 64, 128]:
    for lr in [0.0001, 0.001, 0.01, 0.1]:
        to_plot_x = []
        to_plot_y = []
        for s in [0, 1, 2]:
            try:
                a = np.load(f"results_sweep/results_reinforce_lr{lr}_batch_size{bs}_clip{True}_seed{s}.npy")
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
for bs in [8, 16, 32, 64]:
    for lr in [0.0001, 0.001, 0.01]:
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
                    print(lr, noise_std, bs)

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