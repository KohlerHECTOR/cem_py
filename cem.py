from common_utils import (
    init_policy_params,
    collect_trajectory,
    get_score_batch,
    trajectory_dtype,
    total_iters_budget,
)
import numpy as np
from gymnasium import make
from joblib import Parallel, delayed
import argparse


def cem(elite_prop=0.125, batch_size=32, seed=0):
    np.random.seed(seed)
    num_iterations = total_iters_budget // (batch_size * 50)
    envs = [make("Pendulum-v1") for _ in range(batch_size)]

    population_weights = [init_policy_params() for _ in range(batch_size)]
    population_weights_flatten = np.array(
        [
            np.concatenate([param.flatten() for param in pop])
            for pop in population_weights
        ]
    )
    # Reshape flattened weights back to original parameter shapes
    param_shapes, param_sizes = zip(
        *[(param.shape, param.size) for param in population_weights[0]]
    )
    total_flat_size = len(population_weights_flatten[0])

    all_trajs = np.zeros(
        (num_iterations, batch_size, 50), dtype=trajectory_dtype
    )  # 50 evals per traj to get good estimate of score
    topK = int(elite_prop * batch_size)

    results = []
    for iteration in range(num_iterations):
        # Collect n_evals trajectories for each environment in parallel
        all_trajs[iteration] = np.array(
            Parallel(n_jobs=-1)(
                delayed(collect_trajectory)(
                    envs[b], population_weights[b], deterministic=True
                )
                for b in range(batch_size)
                for e in range(50)
            )
        ).reshape(batch_size, 50)

        scores_pop = np.array(
            Parallel(n_jobs=-1)(
                delayed(get_score_batch)(all_trajs[iteration][b])
                for b in range(batch_size)
            )
        )

        best_idxs = np.argsort(scores_pop)[-topK:]
        tot_steps, train_score = (iteration + 1) * batch_size * 50 * 200, scores_pop[
            best_idxs[-1]
        ]
        results.append([tot_steps, train_score])
        print(f"Steps: {tot_steps} - score: {train_score}")

        new_pop_mean = population_weights_flatten[best_idxs].mean(axis=0)
        new_pop_std = population_weights_flatten[best_idxs].std(axis=0)

        # Sample new population weights from normal distribution
        population_weights_flatten = np.random.normal(
            loc=new_pop_mean,
            scale=new_pop_std + 1e-6,  # Add small constant to avoid zero std
            size=(batch_size, total_flat_size),
        )

        population_weights = []
        for i in range(batch_size):
            params = []
            start_idx = 0
            for j, shape in enumerate(param_shapes):
                size = param_sizes[j]
                param = population_weights_flatten[
                    i, start_idx : start_idx + size
                ].reshape(shape)
                params.append(param)
                start_idx += size
            population_weights.append(params)

    np.save(
        f"results_cem_elite_prop{elite_prop}_batch_size{batch_size}_seed{seed}.npy",
        results
    )
    for env in envs:
        env.close()


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Cross-Entropy Method for Pendulum-v1')
    parser.add_argument('--elite_prop', type=float, default=0.125,
                        help='Proportion of elite samples (default: 0.125)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for evaluation (default: 16)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default: 0)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run CEM with parsed arguments
    cem(elite_prop=args.elite_prop, batch_size=args.batch_size, seed=args.seed)
