from gymnasium import make
import numpy as np
from joblib import Parallel, delayed
import os
from time import perf_counter

def init_policy_params(layer_sizes):
    """Initialize policy parameters for network with Beta distribution output"""
    # Use He initialization for ReLU networks: sqrt(2/n_in)
    scale_w1 = np.sqrt(2.0 / layer_sizes[0])
    scale_w2 = np.sqrt(2.0 / layer_sizes[1])
    
    # For the final layer, use a smaller initialization to start close to alpha=beta=1
    scale_w3 = 0.01  # Small initial weights for more stable start
    
    # Draw parameters with He initialization for hidden layers
    w1 = np.random.normal(0, scale_w1, (layer_sizes[0], layer_sizes[1]))
    b1 = np.zeros(layer_sizes[1])
    
    w2 = np.random.normal(0, scale_w2, (layer_sizes[1], layer_sizes[2]))
    b2 = np.zeros(layer_sizes[2])
    
    # Initialize final layer to produce values close to 0 before softplus
    w3 = np.random.normal(0, scale_w3, (layer_sizes[2], layer_sizes[-1]))
    b3 = np.zeros(layer_sizes[-1])

    return w1, b1, w2, b2, w3, b3

def sample_action(state, params, action_bounds):
    """Sample action from Beta distribution for a single state"""
    w1, b1, w2, b2, w3, b3 = params
    
    # Linear layers for single state
    h1 = np.maximum(0, state @ w1 + b1)  # ReLU
    h2 = np.maximum(0, h1 @ w2 + b2)     # ReLU
    
    # Final layer outputs log(alpha) and log(beta)
    log_params = h2 @ w3 + b3
    
    # Transform to valid alpha, beta parameters (now works with single state)
    alpha = np.exp(np.clip(log_params[0], -10, 2)) + 1e-6
    beta = np.exp(np.clip(log_params[1], -10, 2)) + 1e-6
    
    # stochastic : action = np.random.beta(alpha, beta)
    action = alpha / (alpha + beta) # mean of beta distrib
    # Scale action from [0,1] to action bounds
    scaled_action = action_bounds[0] + (action_bounds[1] - action_bounds[0]) * action
    return scaled_action, alpha, beta, h1, h2

def evaluate_weights(
    n_evals,
    params_sizes,
    action_bounds,
    env,
    flat_weights,
    gamma,
):
    evaluations = np.empty(n_evals, dtype=np.float64)
    episode_lengths = np.empty(n_evals, dtype=np.int32)
    input_, size, output, w1_end, b1_end, w2_end, b2_end, w3_end = params_sizes
    w1 = flat_weights[0:w1_end].reshape(input_, size)
    b1 = flat_weights[w1_end:b1_end]
    w2 = flat_weights[b1_end:w2_end].reshape(size, size)
    b2 = flat_weights[w2_end:b2_end]
    w3 = flat_weights[b2_end:w3_end].reshape(size, output)
    b3 = flat_weights[w3_end:]
    params = (w1, b1, w2, b2, w3, b3)
    for episode in range(n_evals):
        s, _ = env.reset()
        done = False
        episode_reward = 0.0
        steps = 0
        discount = 1.0

        while not done:
            # Forward pass without using out parameter
            action, _, _, _, _ = sample_action(s, params, action_bounds)
            steps += 1
            s, r, term, trunc, _ = env.step(action)
            episode_reward += discount * r
            discount *= gamma
            done = term or trunc

        evaluations[episode] = episode_reward
        episode_lengths[episode] = steps

    return np.mean(evaluations), np.sum(episode_lengths)


def cem(
    n_pop=64,
    n_evals=50,
    size=16,
    elite_prop=1 / 8,
    seed=0,
    gamma=0.99,  # Added gamma parameter
):
    # Create a unique folder name based on parameters
    folder_name = f"temp_beta_cem_data_Pendulum-v1_std_pop{n_pop}_size{size}_evals{n_evals}_elite{elite_prop}_seed{seed}"
    os.makedirs(folder_name, exist_ok=True)
    base_steps = 100 * 64 * 50  # baseline: 100 iterations with batch_size=64, n_evals=50
    n_iters = int(base_steps / (n_pop * n_evals))

    envs = [make("Pendulum-v1") for _ in range(n_pop)]
    input_ = envs[0].observation_space.shape[0]
    output = envs[0].action_space.shape[0] * 2 #2 params alpha beta
    action_bounds = (envs[0].action_space.low.copy(), envs[1].action_space.high.copy())
    layer_sizes = [input_, size, size, output]
    params_pop = [init_policy_params(layer_sizes) for _ in range(n_pop)]
    w1, b1, w2, b2, w3, b3 = params_pop[0]
    w1_end = w1.size
    b1_end = w1_end + b1.size
    w2_end = b1_end + w2.size
    b2_end = w2_end + b2.size
    w3_end = b2_end + w3.size
    params_sizes = (input_, size, output, w1_end, b1_end, w2_end, b2_end, w3_end)
    total_weight_size = w3_end + b3.size
    population = np.zeros((n_pop, total_weight_size))
    for p in range(n_pop):
        population[p] = np.concatenate([param.flatten() for param in params_pop[p]]) 

    # Pre-allocate arrays for training data
    training_data = {
        "iterations": np.arange(n_iters),
        "total_samples": np.empty(n_iters),
        "population_weights": np.empty((n_iters, n_pop, total_weight_size)),
        "population_scores": np.empty((n_iters, n_pop)),
        "best_pop_idx": np.empty(n_iters, dtype=np.int64),
        "train_scores": np.empty(n_iters),
        "walltime": np.empty(n_iters),
    }
    t = perf_counter()
    n_elite = int(n_pop * elite_prop)  # Changed to use elite_prop

    for iters in range(n_iters):
        # Collect more detailed evaluation results
        eval_results = Parallel(n_jobs=n_pop)(
            delayed(evaluate_weights)(
                n_evals,
                params_sizes,
                action_bounds,
                envs[i],
                population[i],
                gamma,  # Pass gamma to evaluate_weights
            )
            for i in range(n_pop)
        )

        # Unpack results for each pop element
        pop_mean_rewards, pop_episode_lengths = zip(*eval_results)
        # Elite selection
        elite_indices = np.argsort(pop_mean_rewards)[-n_elite:]

        # Directly compute mean and std of elite population members
        mean_new_pop = np.mean(population[elite_indices], axis=0)
        std_new_pop = np.std(population[elite_indices], axis=0)

        # cov_new_pop = np.cov(population[elite_indices], rowvar=False)
        # Update all training data at end of iteration
        training_data["total_samples"][iters] = np.sum(pop_episode_lengths)
        training_data["population_weights"][iters] = population
        training_data["population_scores"][iters] = pop_mean_rewards
        training_data["best_pop_idx"][iters] = elite_indices[-1]
        training_data["train_scores"][iters] = pop_mean_rewards[elite_indices[-1]]
        print(training_data["train_scores"][iters])
        # population = np.random.multivariate_normal(
        #     mean=mean_new_pop, cov=cov_new_pop, size=n_pop
        # )
        population = np.random.multivariate_normal(
            mean=mean_new_pop, cov=np.diag(std_new_pop), size=n_pop
        )

        training_data["walltime"][iters] = perf_counter() - t

        # Save training data periodically in the dedicated folder
    save_path = os.path.join(folder_name, f"training_data_iter.npy")
    np.save(save_path, training_data)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run CEM optimization")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument(
        "--n_pop", type=int, default=64, help="Population size (default: 64)"
    )
    parser.add_argument(
        "--n_evals",
        type=int,
        default=50,
        help="Number of evaluations per member (default: 50)",
    )
    parser.add_argument(
        "--size", type=int, default=16, help="Size of hidden layers (default: 16)"
    )
    parser.add_argument(
        "--elite_prop",
        type=float,
        default=1 / 8,
        help="Proportion of elite population members (default: 1/8)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor for rewards (default: 0.99)",
    )

    args = parser.parse_args()

    cem(
        seed=args.seed,
        n_pop=args.n_pop,
        n_evals=args.n_evals,
        size=args.size,
        elite_prop=args.elite_prop,
        gamma=args.gamma,
    )
