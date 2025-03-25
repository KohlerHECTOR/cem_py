import numpy as np
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
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
axes = [ax1, ax2, ax3]

# Process CEM results with updated styling
cem_curves = []  # Store (final_return, plot_data) for alpha calculation
success, fail = 0, 0
for p in [16, 32, 64, 128]:
    for prop in [0.125, 0.25]:
        to_plot_x = []
        to_plot_y = []
        for s in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            try:
                a = np.load(f"results_sweep/full_results_cem_elite_prop{prop}_batch_size{p}_seed{s}.npy")
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
            n = len(to_plot_y)
            y_mean = np.mean(to_plot_y, axis=0)
            y_std = np.std(to_plot_y, axis=0)
            y_sem = y_std / np.sqrt(n)
            # For 90% CI with normal distribution, use z=1.645
            z = 1.645
            y_lower = y_mean - z * y_sem
            y_upper = y_mean + z * y_sem
            final_return = y_mean[-1]
            plot_data = (np.mean(to_plot_x, axis=0), y_mean, y_lower, y_upper)
            cem_curves.append((final_return, plot_data))
            if p == 128 and prop == 0.125:  # This is our target configuration
                best_cem_data = (np.mean(to_plot_x, axis=0), y_mean, y_lower, y_upper)
print(success / (fail+success))


# Plot CEM curves with normalized alphas based on final returns
if cem_curves:
    returns = np.array([r for r, _ in cem_curves])
    min_return = np.min(returns)
    max_return = np.max(returns)
    for final_return, (x, y, y_lower, y_upper) in cem_curves:
        # Downsample to 100 points
        indices = np.linspace(0, len(x) - 1, 100).astype(int)
        x_downsampled = x[indices]
        y_downsampled = y[indices]
        y_lower_downsampled = y_lower[indices]
        y_upper_downsampled = y_upper[indices]
        ax1.plot(x_downsampled, y_downsampled, c='#FF7F7F', alpha=1, linewidth=2)
        ax1.fill_between(x_downsampled, y_lower_downsampled, y_upper_downsampled, color='#FF7F7F', alpha=0.2)

# Process REINFORCE results with updated styling
reinforce_curves = []
test_curves = []
success, fail = 0, 0

for bs in [2, 4, 8, 16, 32]:
    for lr in [0.0001, 0.001, 0.01]:
        for clip in [True]:
            to_plot_x = []
            to_plot_y = []
            to_plot_test = []
            for s in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]:
                try:
                    a = np.load(f"results_sweep/full_results_reinforce_lr{lr}_batch_size{bs}_clip{clip}_seed{s}.npy")
                    b = np.array(list(np.load(f"test_scores/full_results_reinforce_lr{lr}_batch_size{bs}_clip{clip}_seed{s}_weights_test_scores.npy", allow_pickle=True).item()))[:,0]
                    valid_mask = ~(np.isnan(a[:, 1]) | np.isinf(a[:, 1])) & (a[:, 1] < 0)
                    if not np.all(valid_mask):
                        print("oops")
                        raise FileNotFoundError
                    success += 1
                    to_plot_y.append(a[:,1])
                    to_plot_x.append(a[:,0])
                    to_plot_test.append(b)
                except FileNotFoundError:
                    fail += 1
                    continue
            if to_plot_x and to_plot_y:
                y_mean = np.mean(to_plot_y, axis=0)
                test_mean = np.mean(to_plot_test, axis=0)
                n = len(to_plot_test)
                y_std = np.std(to_plot_test, axis=0) 
                y_sem = y_std / np.sqrt(n)
                # For 90% CI with normal distribution, use z=1.645
                z = 1.645
                y_lower = test_mean - z * y_sem
                y_upper = test_mean + z * y_sem
                final_return = test_mean[-1]
                plot_data = (np.mean(to_plot_x, axis=0), y_mean, y_lower, y_upper, test_mean)
                reinforce_curves.append((final_return, plot_data))
                if bs == 32 and lr == 0.01:  # This is our target configuration
                    best_reinforce_data = (np.mean(to_plot_x, axis=0), y_mean, y_lower, y_upper, test_mean)
print(success / (fail+success))

# Plot REINFORCE curves with normalized alphas based on final returns
if reinforce_curves:
    returns = np.array([r for r, _ in reinforce_curves])
    min_return = np.min(returns)
    max_return = np.max(returns)
    for final_return, (x, y, y_lower, y_upper, test_mean) in reinforce_curves:
        # Downsample to 100 points
        indices = np.linspace(0, len(x) - 1, 100).astype(int)
        x_downsampled = x[indices]
        y_downsampled = y[indices]
        # y_lower_downsampled = y_lower[indices]
        # y_upper_downsampled = y_upper[indices]
        # ax2.plot(x_downsampled, y_downsampled, c='#7F7FFF', alpha=1, linewidth=2, linestyle="dotted")
        ax2.plot(x_downsampled, test_mean, c='#7F7FFF', alpha=1, linewidth=2)
        ax2.fill_between(x_downsampled, y_lower, y_upper, color='#7F7FFF', alpha=0.2)

# Plot best curves in third subplot
if best_cem_data:
    x, y, y_lower, y_upper = best_cem_data
    indices = np.linspace(0, len(x) - 1, 100).astype(int)
    x_downsampled = x[indices]
    y_downsampled = y[indices]
    y_lower_downsampled = y_lower[indices]
    y_upper_downsampled = y_upper[indices]
    ax3.plot(x_downsampled, y_downsampled, c='#FF7F7F', alpha=1, linewidth=2, label='CEM (b=128, e=0.125)')
    ax3.fill_between(x_downsampled, y_lower_downsampled, y_upper_downsampled, color='#FF7F7F', alpha=0.2)

if best_reinforce_data:
    x, y, y_lower, y_upper, test_mean = best_reinforce_data
    indices = np.linspace(0, len(x) - 1, 100).astype(int)
    x_downsampled = x[indices]
    ax3.plot(x_downsampled, test_mean, c='#7F7FFF', alpha=1, linewidth=2, label='REINFORCE (b=32, lr=0.01)')
    ax3.fill_between(x_downsampled, y_lower, y_upper, color='#7F7FFF', alpha=0.2)

# Style each subplot
for ax, title in zip(axes, ['CEM', 'REINFORCE', 'Best Configurations']):
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
]

# Add legend to the third subplot
ax3.legend(fontsize=10)

plt.tight_layout()
# Adjust layout to make room for legend
plt.savefig("algorithm_full_comparison.png", bbox_inches='tight')