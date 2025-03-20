import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from matplotlib.lines import Line2D
import os
from collections import defaultdict


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

        sns.lineplot(
            x=train_samples_mean,
            y=train_mean,
            linestyle=line_style,
            color=color_map,
            linewidth=2,
        )


def find_best_hyperparameters():
    # Dictionaries to store results for each algorithm
    cem_results = {}
    reinforce_results = {}
    reinforce_baseline_results = {}
    es_results = {}  # Added ES results dictionary

    # Process CEM data
    for std_ in [True]:
        for n_pop in [8, 16, 32, 64]:
            for elite in [0.125, 0.25]:
                key = f"std_{std_}_pop{n_pop}_elite{elite}"
                final_scores = []

                for seed in range(3):
                    data_path = f'beta_cem_data_Pendulum-v1_{"" if not std_ else "std_"}pop{n_pop}_size16_evals50_elite{elite}_seed{seed}/training_data_iter.npy'
                    try:
                        data = np.load(data_path, allow_pickle=True).item()
                        final_scores.append(data["train_scores"][-1])
                    except FileNotFoundError:
                        continue

                if final_scores:
                    cem_results[key] = np.mean(final_scores)

    # Process REINFORCE data
    for lr in [0.0001, 0.001, 0.01, 0.1]:
        for evals in [1, 2, 50]:
            for batch_size in [8, 16, 32, 64]:
                key = f"lr{lr}_evals{evals}_batch{batch_size}"
                final_scores = []

                for seed in range(3):
                    data_path = f"beta_reinforce_data_Pendulum-v1_batch{batch_size}_lr{lr}_evals{evals}_seed{seed}/training_data_iter.npy"
                    try:
                        data = np.load(data_path, allow_pickle=True).item()
                        final_scores.append(data["train_scores"][-1])
                    except FileNotFoundError:
                        continue

                if final_scores:
                    reinforce_results[key] = np.mean(final_scores)

    # Process REINFORCE with baseline data
    for lr in [0.0001, 0.001, 0.01, 0.1]:
        for evals in [1, 50]:
            for batch_size in [8, 16, 32, 64]:
                key = f"lr{lr}_evals{evals}_batch{batch_size}"
                final_scores = []

                for seed in range(3):
                    data_path = f"beta_reinforce_sub_baseline_data_Pendulum-v1_batch{batch_size}_lr{lr}_evals{evals}_seed{seed}/training_data_iter.npy"
                    try:
                        data = np.load(data_path, allow_pickle=True).item()
                        final_scores.append(data["train_scores"][-1])
                    except FileNotFoundError:
                        continue

                if final_scores:
                    reinforce_baseline_results[key] = np.mean(final_scores)

    # Process ES data
    for lr in [0.0001, 0.001, 0.01, 0.1]:
        for noise_std in [0.01, 0.1, 1]:
            for n_pop in [8, 16, 32, 64]:
                key = f"lr{lr}_std{noise_std}_pop{n_pop}"
                final_scores = []

                for seed in range(3):
                    data_path = f"beta_es_data_Pendulum-v1_pop{n_pop}_size16_evals50_lr{lr}_std{noise_std}_seed{seed}/training_data_final.npy"
                    try:
                        data = np.load(data_path, allow_pickle=True).item()
                        final_scores.append(data["train_scores"][-1])
                    except FileNotFoundError:
                        continue

                if final_scores:
                    es_results[key] = np.mean(final_scores)

    # Find best hyperparameters for each algorithm
    best_cem = max(cem_results.items(), key=lambda x: x[1]) if cem_results else None
    best_reinforce = (
        max(reinforce_results.items(), key=lambda x: x[1])
        if reinforce_results
        else None
    )
    best_reinforce_baseline = (
        max(reinforce_baseline_results.items(), key=lambda x: x[1])
        if reinforce_baseline_results
        else None
    )
    best_es = max(es_results.items(), key=lambda x: x[1]) if es_results else None

    print("Best Hyperparameters (based on final training score):")
    print(
        f"CEM: {best_cem[0]} with score {best_cem[1]:.4f}"
        if best_cem
        else "No CEM data found"
    )
    print(
        f"REINFORCE: {best_reinforce[0]} with score {best_reinforce[1]:.4f}"
        if best_reinforce
        else "No REINFORCE data found"
    )
    print(
        f"REINFORCE+baseline: {best_reinforce_baseline[0]} with score {best_reinforce_baseline[1]:.4f}"
        if best_reinforce_baseline
        else "No REINFORCE+baseline data found"
    )
    print(
        f"ES: {best_es[0]} with score {best_es[1]:.4f}"
        if best_es
        else "No ES data found"
    )

    return {
        "CEM": best_cem,
        "REINFORCE": best_reinforce,
        "REINFORCE+baseline": best_reinforce_baseline,
        "ES": best_es,
    }


if __name__ == "__main__":
    find_best_hyperparameters()
