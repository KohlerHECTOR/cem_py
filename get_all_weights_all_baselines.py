import numpy as np

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

weights_reinforce = np.array(all_seeds_weights_reinforce)

# Best REINFORCE with baseline hyperparameters (lr=0.001, batch_size=64)
reinforce_baseline_data_paths = [
    f'beta_reinforce_sub_baseline_data_Pendulum-v1_batch64_lr0.001_evals1_seed{seed}/training_data_iter.npy'
    for seed in range(3, 3 + 15)
]

all_data_reinforce_baseline = [np.load(p, allow_pickle=True).item() for p in reinforce_baseline_data_paths]
all_seeds_weights_reinforce_baseline = [d["params"] for d in all_data_reinforce_baseline]

weights_reinforce_bs = np.array(all_seeds_weights_reinforce_baseline)

# Best ES hyperparameters
es_data_paths = [
    f'beta_es_data_Pendulum-v1_pop8_size16_evals50_lr0.01_std0.1_seed{seed}/training_data_iter.npy'
    for seed in range(5, 3 + 15)
]

all_data_es = [np.load(p, allow_pickle=True).item() for p in es_data_paths]
all_seeds_weights_es = [d["theta"] for d in all_data_es]

weights_es = np.array(all_seeds_weights_es)

# Flatten and stack all weights
all_weights = []

# Flatten CEM weights
flattened_cem = weights_cem.reshape(weights_cem.shape[0] * weights_cem.shape[1], -1)
all_weights.append(flattened_cem)

# Flatten REINFORCE weights
flattened_reinforce = weights_reinforce.reshape(weights_reinforce.shape[0] * weights_reinforce.shape[1], -1)
all_weights.append(flattened_reinforce)

# Flatten REINFORCE with baseline weights
flattened_reinforce_bs = weights_reinforce_bs.reshape(weights_reinforce_bs.shape[0] * weights_reinforce_bs.shape[1], -1)
all_weights.append(flattened_reinforce_bs)

# Flatten ES weights
flattened_es = weights_es.reshape(weights_es.shape[0] * weights_es.shape[1], -1)
all_weights.append(flattened_es)

# Stack all flattened weights
stacked_weights = np.vstack(all_weights)
cvx_hull = [np.min(stacked_weights, axis=0), np.max(stacked_weights, axis=0)]
np.save("cvx_hull_weights_baselines.npy", np.array(cvx_hull))






