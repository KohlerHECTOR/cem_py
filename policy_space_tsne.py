import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.cm as cm
from scipy.interpolate import griddata

# Best CEM hyperparameters (n_pop=16, elite_prop=0.125)
cem_data_paths = [
    f'beta_cem_data_Pendulum-v1_std_pop16_size16_evals50_elite0.125_seed{seed}/training_data_iter.npy'
    for seed in range(3, 3 + 15)
]
all_data_cem = [np.load(p, allow_pickle=True).item() for p in cem_data_paths]
all_seeds_weights_cem = [d["population_weights"] for d in all_data_cem]
best_weights_per_seed_cem = []
for s, weights_iters in enumerate(all_seeds_weights_cem):
    per_s = []
    for i, w in enumerate(weights_iters):
        per_s.append(w[all_data_cem[s]["best_pop_idx"][i]])
    best_weights_per_seed_cem.append(per_s)


weights_cem = np.array(best_weights_per_seed_cem)

# Best REINFORCE hyperparameters (lr=0.001, batch_size=64)
reinforce_data_paths = [
    f'beta_reinforce_data_Pendulum-v1_batch64_lr0.001_evals1_seed{seed}/training_data_iter.npy'
    for seed in range(3, 3 + 15)
]
all_data_reinforce = [np.load(p, allow_pickle=True).item() for p in reinforce_data_paths]
all_seeds_weights_reinforce = np.array([d["params"] for d in all_data_reinforce])
weights_per_seed_reinforce = []
for s in all_seeds_weights_reinforce:
    idxs = np.linspace(0, len(s) - 1, 400, dtype=np.uint32)
    weights_per_seed_reinforce.append(s[idxs])

weights_reinforce = np.array(weights_per_seed_reinforce)

# Best REINFORCE with baseline hyperparameters (lr=0.001, batch_size=64)
reinforce_baseline_data_paths = [
    f'beta_reinforce_sub_baseline_data_Pendulum-v1_batch64_lr0.001_evals1_seed{seed}/training_data_iter.npy'
    for seed in range(3, 3 + 15)
]

all_data_reinforce_baseline = [np.load(p, allow_pickle=True).item() for p in reinforce_baseline_data_paths]
all_seeds_weights_reinforce_baseline = [d["params"] for d in all_data_reinforce_baseline]
weights_per_seed_reinforce_bs = []
for s in all_seeds_weights_reinforce_baseline:
    idxs = np.linspace(0, len(s) - 1, 400, dtype=np.uint32)
    weights_per_seed_reinforce_bs.append(s[idxs])

weights_reinforce_bs = np.array(weights_per_seed_reinforce_bs)

# Combine all weights into a single dataset for t-SNE
# Reshape the weights arrays to combine all seeds and iterations
# Each row will be a single policy parameter vector
cem_reshaped = weights_cem.reshape(-1, weights_cem.shape[-1])

# Add all CEM population weights for better score interpolation
all_cem_pop_weights = []
all_cem_pop_scores = []
for s in range(len(all_data_cem)):
    for i in range(len(all_data_cem[s]["population_weights"])):
        pop_weights = all_data_cem[s]["population_weights"][i]
        pop_scores = all_data_cem[s]["population_scores"][i]
        all_cem_pop_weights.append(pop_weights)
        all_cem_pop_scores.append(pop_scores)

all_cem_pop_weights = np.vstack(all_cem_pop_weights)
all_cem_pop_scores = np.concatenate(all_cem_pop_scores)

reinforce_reshaped = weights_reinforce.reshape(-1, weights_reinforce.shape[-1])
reinforce_bs_reshaped = weights_reinforce_bs.reshape(-1, weights_reinforce_bs.shape[-1])

# Extract the scores for each policy
cem_scores = np.array([[all_data_cem[s]["train_scores"][all_data_cem[s]["best_pop_idx"][i]] for i in range(len(all_data_cem[s]["best_pop_idx"]))] 
                      for s in range(len(all_data_cem))]).flatten()
reinforce_scores = np.array([d["train_scores"] for d in all_data_reinforce])
reinforce_scores = np.array([r[idxs] for r, idxs in zip(reinforce_scores, 
                            [np.linspace(0, len(s) - 1, 400, dtype=np.uint32) for s in all_seeds_weights_reinforce])]).flatten()
reinforce_bs_scores = np.array([d["train_scores"] for d in all_data_reinforce_baseline])
reinforce_bs_scores = np.array([r[idxs] for r, idxs in zip(reinforce_bs_scores, 
                               [np.linspace(0, len(s) - 1, 400, dtype=np.uint32) for s in all_seeds_weights_reinforce_baseline])]).flatten()

# Combine all scores and weights for t-SNE and interpolation
all_scores_for_interp = np.concatenate([cem_scores, reinforce_scores, reinforce_bs_scores, all_cem_pop_scores])
all_weights_for_interp = np.vstack([cem_reshaped, reinforce_reshaped, reinforce_bs_reshaped, all_cem_pop_weights])

# Combine only the algorithm trajectories for visualization (without the extra CEM population)
all_weights = np.vstack([cem_reshaped, reinforce_reshaped, reinforce_bs_reshaped])
all_scores = np.concatenate([cem_scores, reinforce_scores, reinforce_bs_scores])
print(all_scores)
# Apply t-SNE to project to 2D for all weights (including extra CEM population)
print("Applying t-SNE projection...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
all_weights_2d_interp = tsne.fit_transform(all_weights_for_interp)

# Create a grid for the contour plot
grid_x, grid_y = np.mgrid[
    all_weights_2d_interp[:, 0].min():all_weights_2d_interp[:, 0].max():200j,
    all_weights_2d_interp[:, 1].min():all_weights_2d_interp[:, 1].max():200j
]

# Interpolate the scores onto the grid
grid_z = griddata(all_weights_2d_interp, all_scores_for_interp, (grid_x, grid_y), 
                 method='linear', 
                 fill_value=all_scores_for_interp.min())

# Extract only the trajectory points for visualization (without extra CEM population)
n_trajectory_points = len(all_weights)
trajectory_points_2d = all_weights_2d_interp[:n_trajectory_points]

# Split the projected data back into the original algorithms
n_cem = cem_reshaped.shape[0]
n_reinforce = reinforce_reshaped.shape[0]
n_reinforce_bs = reinforce_bs_reshaped.shape[0]

cem_2d = trajectory_points_2d[:n_cem].reshape(weights_cem.shape[0], weights_cem.shape[1], 2)
reinforce_2d = trajectory_points_2d[n_cem:n_cem+n_reinforce].reshape(weights_reinforce.shape[0], weights_reinforce.shape[1], 2)
reinforce_bs_2d = trajectory_points_2d[n_cem+n_reinforce:].reshape(weights_reinforce_bs.shape[0], weights_reinforce_bs.shape[1], 2)

# Plot the score surface and learning trajectories
plt.figure(figsize=(10, 8))

# Plot the score surface as a contour plot with explicit levels
levels = np.linspace(all_scores_for_interp.min(), all_scores_for_interp.max(), 10)
contour = plt.contourf(grid_x, grid_y, grid_z, levels=levels, cmap='Greys', alpha=0.8)
plt.colorbar(contour, label='Extrapolated objective values')

# Plot CEM trajectories
for i in range(cem_2d.shape[0]):
    plt.plot(cem_2d[i, :, 0], cem_2d[i, :, 1], 'r-', alpha=1, linewidth=2)
    plt.plot(cem_2d[i, 0, 0], cem_2d[i, 0, 1], 'ro', markersize=6)  # Start point
    plt.plot(cem_2d[i, -1, 0], cem_2d[i, -1, 1], 'rx', markersize=10)  # End point

# Plot REINFORCE trajectories
for i in range(reinforce_2d.shape[0]):
    plt.plot(reinforce_2d[i, :, 0], reinforce_2d[i, :, 1], 'b-', alpha=1, linewidth=2)
    plt.plot(reinforce_2d[i, 0, 0], reinforce_2d[i, 0, 1], 'bo', markersize=6)  # Start point
    plt.plot(reinforce_2d[i, -1, 0], reinforce_2d[i, -1, 1], 'bx', markersize=10)  # End point

# Plot REINFORCE with baseline trajectories
for i in range(reinforce_bs_2d.shape[0]):
    plt.plot(reinforce_bs_2d[i, :, 0], reinforce_bs_2d[i, :, 1], 'g-', alpha=1, linewidth=2)
    plt.plot(reinforce_bs_2d[i, 0, 0], reinforce_bs_2d[i, 0, 1], 'go', markersize=6)  # Start point
    plt.plot(reinforce_bs_2d[i, -1, 0], reinforce_bs_2d[i, -1, 1], 'gx', markersize=10)  # End point

# Add legend and labels
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='r', lw=2, label='CEM'),
    Line2D([0], [0], color='b', lw=2, label='REINFORCE'),
    Line2D([0], [0], color='g', lw=2, label='REINFORCE+Baseline'),
    Line2D([0], [0], marker='o', color='k', label='Init Weight', markersize=8, linestyle='None'),
    Line2D([0], [0], marker='X', color='k', label='Learned Weight', markersize=8, linestyle='None')
]
plt.legend(handles=legend_elements, loc='best', fontsize=11)

plt.xticks([])
plt.yticks([])

# Save the figure
plt.savefig('policy_space_tsne.png', dpi=300, bbox_inches='tight')

