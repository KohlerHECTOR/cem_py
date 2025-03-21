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


def es(lr=0.01, noise_std=0.1, batch_size=32, seed=0):
    np.random.seed(seed)
    num_iterations = total_iters_budget // (batch_size * 50)

    envs = [make("Pendulum-v1") for _ in range(batch_size)]
    theta = init_policy_params()
    all_trajs = np.zeros(
        (num_iterations, batch_size, 50), dtype=trajectory_dtype
    )  # 50 evals per traj to get good estimate of score

    results = []
    for iteration in range(num_iterations):
        # Generate a single epsilon for each member of the population

        # Create perturbed versions of theta
        perturbed_thetas = []
        epsilons = []
        for _ in range(batch_size - 1):
            eps = []
            perturbed_params = []
            for param in theta:
                eps_param = np.random.normal(0, 1, size=param.shape)
                eps.append(eps_param)
                perturbed_params.append(param + noise_std * eps_param)

            # For each epsilon, perturb all parameters in theta
            perturbed_thetas.append(perturbed_params)
            epsilons.append(eps)

        # Collect trajectories for each perturbed policy
        # Collect n_evals trajectories for each environment in parallel
        all_trajs[iteration][:-1] = np.array(
            Parallel(n_jobs=-1)(
                delayed(collect_trajectory)(
                    envs[b], perturbed_thetas[b], deterministic=True
                )
                for b in range(batch_size - 1)
                for e in range(50)
            )
        ).reshape(batch_size - 1, 50)

        scores_pop = np.array(
            Parallel(n_jobs=-1)(
                delayed(get_score_batch)(all_trajs[iteration][b])
                for b in range(batch_size)
            )
        )

        # Calculate and apply gradient updates
        # Create a list to hold the gradient updates for each parameter
        grad_updates = []
        for param_idx, param in enumerate(theta):
            # Initialize gradient update with zeros in the shape of the parameter
            param_grad = np.zeros_like(param)

            # Accumulate gradient contributions from each perturbation
            for b in range(batch_size - 1):
                # Add the weighted contribution to the gradient
                param_grad += scores_pop[b] * epsilons[b][param_idx]

            # Normalize and scale the gradient
            param_grad = (lr / ((batch_size - 1) * noise_std)) * param_grad
            grad_updates.append(param_grad)

        # Update parameters
        theta = [param + update for param, update in zip(theta, grad_updates)]

        # Evaluate current policy
        all_trajs[iteration][-1] = np.array(
            [collect_trajectory(envs[-1], theta, deterministic=True) for _ in range(50)]
        )

        tot_steps, train_score = (
            iteration + 1
        ) * batch_size * 50 * 200, get_score_batch(all_trajs[iteration][-1])
        results.append([tot_steps, train_score])
        print(f"Steps: {tot_steps} - score: {train_score}")

    np.save(
        f"results_es_lr{lr}_noise_std{noise_std}_batch_size{batch_size}_seed{seed}.npy",
        results,
    )
    for env in envs:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ES algorithm for Pendulum-v1")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--noise_std", type=float, default=0.1, help="noise")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()
    es(lr=args.lr, batch_size=args.batch_size, noise_std=args.noise_std, seed=args.seed)
