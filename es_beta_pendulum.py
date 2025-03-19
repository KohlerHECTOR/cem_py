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

def es(
    n_pop=64,
    n_evals=50,
    size=16,
    learning_rate=0.01,
    noise_std=0.1,
    seed=0,
    gamma=0.99,
):
    """
    Evolution Strategy implementation for Pendulum-v1
    
    Args:
        n_pop: Number of perturbations/population size
        n_evals: Number of evaluations per perturbation
        size: Size of hidden layers
        learning_rate: Learning rate for parameter updates
        noise_std: Standard deviation of the noise for perturbations
        seed: Random seed
        gamma: Discount factor for rewards
    """
    # Create a unique folder name based on parameters
    folder_name = f"beta_es_data_Pendulum-v1_pop{n_pop}_size{size}_evals{n_evals}_lr{learning_rate}_std{noise_std}_seed{seed}"
    os.makedirs(folder_name, exist_ok=True)
    base_steps = 100 * 64 * 50  # baseline: 100 iterations with batch_size=64, n_evals=50
    n_iters = int(base_steps / (n_pop * n_evals))

    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Create environments
    envs = [make("Pendulum-v1") for _ in range(n_pop)]
    input_ = envs[0].observation_space.shape[0]
    output = envs[0].action_space.shape[0] * 2  # 2 params alpha beta
    action_bounds = (envs[0].action_space.low.copy(), envs[0].action_space.high.copy())
    layer_sizes = [input_, size, size, output]
    
    # Initialize policy parameters (theta)
    w1, b1, w2, b2, w3, b3 = init_policy_params(layer_sizes)
    theta = np.concatenate([param.flatten() for param in (w1, b1, w2, b2, w3, b3)])
    
    # Calculate parameter sizes for reshaping
    w1_end = w1.size
    b1_end = w1_end + b1.size
    w2_end = b1_end + w2.size
    b2_end = w2_end + b2.size
    w3_end = b2_end + w3.size
    params_sizes = (input_, size, output, w1_end, b1_end, w2_end, b2_end, w3_end)
    total_weight_size = w3_end + b3.size
    
    # Pre-allocate arrays for training data
    training_data = {
        "iterations": np.arange(n_iters),
        "total_samples": np.empty(n_iters),
        "theta": np.empty((n_iters, total_weight_size)),
        "perturbation_scores": np.empty((n_iters, n_pop)),
        "train_scores": np.empty(n_iters),
        "walltime": np.empty(n_iters),
    }
    
    t = perf_counter()
    total_samples = 0
    
    for iters in range(n_iters):
        # Sample perturbations (noise vectors)
        epsilons = np.random.normal(0, 1, (n_pop, total_weight_size))
        
        # Create perturbed policies
        perturbed_thetas = [theta + noise_std * epsilon for epsilon in epsilons]
        
        # Evaluate all perturbed policies in parallel
        eval_results = Parallel(n_jobs=n_pop)(
            delayed(evaluate_weights)(
                n_evals,
                params_sizes,
                action_bounds,
                envs[i],
                perturbed_thetas[i],
                gamma,
            )
            for i in range(n_pop)
        )
        
        # Unpack results for each perturbed policy
        rewards, episode_lengths = zip(*eval_results)
        rewards = np.array(rewards)
        
        # Normalize rewards for more stable updates (optional)
        train_score = np.mean(rewards)
        normalized_rewards = (rewards - train_score) / (np.std(rewards) + 1e-6)
        
        # Compute the gradient estimate
        # θt+1 ← θt + α * (1/nσ) * sum(Fi * ei)
        # Replace loop with vectorized operation
        gradient_estimate = np.sum(normalized_rewards[:,np.newaxis] * epsilons, axis=0) / (n_pop * noise_std)
        
        # Update policy parameters
        training_data["theta"][iters] = theta
        theta += learning_rate * gradient_estimate
        # Update training data
        training_data["total_samples"][iters] = sum(episode_lengths)
        training_data["perturbation_scores"][iters] = rewards
        training_data["train_scores"][iters] = train_score
        training_data["walltime"][iters] = perf_counter() - t
    
    # Save final training data
    save_path = os.path.join(folder_name, f"training_data_iter.npy")
    np.save(save_path, training_data)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ES optimization")
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
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate for parameter updates (default: 0.01)",
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=0.1,
        help="Standard deviation of the noise for perturbations (default: 0.1)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor for rewards (default: 0.99)",
    )

    args = parser.parse_args()

    es(
        seed=args.seed,
        n_pop=args.n_pop,
        n_evals=args.n_evals,
        size=args.size,
        learning_rate=args.lr,
        noise_std=args.noise_std,
        gamma=args.gamma,
    )