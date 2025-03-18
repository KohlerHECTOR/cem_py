import numpy as np
from gymnasium import make
from joblib import Parallel, delayed
import os
from time import perf_counter
import argparse
from scipy.special import digamma, gammaln


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
    w3 = np.random.normal(0, scale_w3, (layer_sizes[2], 2))
    b3 = np.zeros(2)

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
    
    action = np.random.beta(alpha, beta)
    # Scale action from [0,1] to action bounds
    scaled_action = action_bounds[0] + (action_bounds[1] - action_bounds[0]) * action
    return scaled_action, alpha, beta, h1, h2


def compute_returns(rewards, gamma):
    """Compute returns for a single episode efficiently"""
    returns = np.empty_like(rewards)
    G = 0
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns


def collect_trajectory(env, params, gamma, action_bounds):
    """Collect a single trajectory and compute returns (to be run in parallel)"""
    states, actions, rewards, alphas, betas, h1s, h2s = [], [], [], [], [], [], []
    state, _ = env.reset()
    done = False

    while not done:
        action_taken, alpha, beta, h1, h2 = sample_action(state, params, action_bounds)
        next_state, reward, term, trunc, _ = env.step(action_taken)

        states.append(state)
        actions.append(action_taken)
        rewards.append(reward)
        alphas.append(alpha)
        betas.append(beta)
        h1s.append(h1)
        h2s.append(h2)

        state = next_state
        done = term or trunc

    # Convert to numpy arrays
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    alphas = np.array(alphas)
    betas = np.array(betas)
    h1s = np.array(h1s)
    h2s = np.array(h2s)

    # Compute returns for this trajectory
    returns = compute_returns(rewards, gamma)

    return states, actions, returns, alphas, betas, h1s, h2s

def clip_by_norm(grad):
    """Clip gradients by norm with robust handling of numerical issues.
    
    Args:
        grad: numpy array of gradients
    Returns:
        Clipped gradients with same shape as input
    """
    # Handle empty arrays
    if grad.size == 0:
        return grad
        
    # Check for any NaN or inf values
    if np.any(np.isnan(grad)) or np.any(np.isinf(grad)):
        print(f"Warning: Found {np.sum(np.isnan(grad))} NaN and {np.sum(np.isinf(grad))} inf values in gradient")
        # Replace NaN/inf with zeros
        grad = np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Compute norm safely with a small epsilon to avoid division by zero
    grad_norm = np.linalg.norm(grad) + 1e-6
    
    # Check if norm is too small (indicates possible numerical issues)
    if grad_norm < 1e-6:
        print("Warning: Gradient norm near zero, returning small random gradient")
        return np.random.normal(0, 1e-6, size=grad.shape)
    
    # Clip if norm is too large
    if grad_norm > 1.0:
        return grad * (1.0 / grad_norm)
    
    return grad

def update_policy(batch_trajectories, params, learning_rate, action_bounds):
    """Update policy parameters using proper Beta distribution gradients"""
    w1, b1, w2, b2, w3, b3 = params
    low, high = action_bounds

    # Concatenate all episodes
    all_states = np.concatenate([traj[0] for traj in batch_trajectories])
    all_actions = np.concatenate([traj[1] for traj in batch_trajectories])
    all_returns = np.concatenate([traj[2] for traj in batch_trajectories])
    alpha = np.concatenate([traj[3] for traj in batch_trajectories])
    beta = np.concatenate([traj[4] for traj in batch_trajectories])
    h1 = np.concatenate([traj[5] for traj in batch_trajectories])
    h2 = np.concatenate([traj[6] for traj in batch_trajectories])

    # Normalize returns for stability
    all_returns = (all_returns - all_returns.mean()) / (all_returns.std() + 1e-6)
    
    # Scale actions to [0, 1]
    scaled_actions = (all_actions - low) / (high - low)
    scaled_actions = scaled_actions.squeeze()
    
    # Clip actions to [epsilon, 1-epsilon] for numerical stability
    epsilon = 1e-6
    scaled_actions = np.clip(scaled_actions, epsilon, 1 - epsilon)

    # Compute log probability gradients for Beta distribution
    common_term = digamma(alpha + beta)
    grad_alpha = (common_term - digamma(alpha) + np.log(scaled_actions)) * all_returns
    grad_beta = (common_term - digamma(beta) + np.log(1 - scaled_actions)) * all_returns

    # Account for exponential transformation from log params
    grad_log_alpha = grad_alpha * alpha
    grad_log_beta = grad_beta * beta

    # Combine gradients - reshape to match dimensions
    grad_params = np.column_stack([grad_log_alpha, grad_log_beta])

    # Backpropagation
    dh2 = (grad_params @ w3.T) * (h2 > 0)
    dh1 = (dh2 @ w2.T) * (h1 > 0)

    # Compute all gradients with proper shapes
    grad_w3 = h2.T @ grad_params  # (16, 2)
    grad_b3 = grad_params.sum(axis=0)  # (2,)
    grad_w2 = h1.T @ dh2  # (16, 16)
    grad_b2 = dh2.sum(axis=0)  # (16,)
    grad_w1 = all_states.T @ dh1  # (3, 16)
    grad_b1 = dh1.sum(axis=0)  # (16,)

    # Apply gradient clipping to all gradients
    grad_w1 = clip_by_norm(grad_w1)
    grad_b1 = clip_by_norm(grad_b1)
    grad_w2 = clip_by_norm(grad_w2)
    grad_b2 = clip_by_norm(grad_b2)
    grad_w3 = clip_by_norm(grad_w3)
    grad_b3 = clip_by_norm(grad_b3)

    # Store gradients in the correct order and shape
    grads = np.concatenate([
        grad_w1.flatten(),  # 3 * 16 = 48
        grad_b1.flatten(),  # 16
        grad_w2.flatten(),  # 16 * 16 = 256
        grad_b2.flatten(),  # 16
        grad_w3.flatten(),  # 16 * 2 = 32
        grad_b3.flatten()   # 2
    ])  # Total: 370 parameters

    # Parameter updates
    w3 += learning_rate * grad_w3
    b3 += learning_rate * grad_b3
    w2 += learning_rate * grad_w2
    b2 += learning_rate * grad_b2
    w1 += learning_rate * grad_w1
    b1 += learning_rate * grad_b1

    return w1, b1, w2, b2, w3, b3, grads

def reinforce(
    size=16,
    learning_rate=0.01,
    gamma=0.99,
    batch_size=64,
    n_evals=50,
    seed=0,
):
    """Train policy using parallel batch updates"""
    # Create a unique folder name based on parameters
    folder_name = f"beta_reinforce_data_Pendulum-v1_batch{batch_size}_lr{learning_rate}_evals{n_evals}_seed{seed}"
    os.makedirs(folder_name, exist_ok=True)

    # Calculate number of iterations to maintain similar total steps
    base_steps = 100 * 64 * 50  # baseline: 100 iterations with batch_size=64, n_evals=50
    num_iterations = int(base_steps / (batch_size * n_evals))

    # Create environments
    envs = [make("Pendulum-v1") for _ in range(batch_size)]
    state_dim = envs[0].observation_space.shape[0]
    action_dim = envs[0].action_space.shape[0]
    action_bounds = (envs[0].action_space.low.copy(), envs[0].action_space.high.copy())

    # Initialize policy parameters (simplified for 1D action space)
    layer_sizes = [state_dim, size, size, action_dim]  # Last dimension is 1 since we know action_dim=1
    params = init_policy_params(layer_sizes)

    # Calculate total parameter size for tracking
    w1, b1, w2, b2, w3, b3 = params
    total_param_size = sum(p.size for p in params)

    # Pre-allocate arrays for training data
    training_data = {
        "iterations": np.arange(num_iterations),
        "total_samples": np.empty(num_iterations),
        "train_scores": np.empty(num_iterations),
        "walltime": np.empty(num_iterations),
        "params": np.empty((num_iterations, total_param_size)),
        "grads": np.empty((num_iterations, total_param_size)),
    }

    t = perf_counter()
    for iteration in range(num_iterations):
        # Collect n_evals trajectories for each environment in parallel
        batch_trajectories = []
        for _ in range(n_evals):
            trajectories = Parallel(n_jobs=batch_size)(
                delayed(collect_trajectory)(env, params, gamma, action_bounds)
                for env in envs
            )
            batch_trajectories.extend(trajectories)

        # Calculate episode information
        discounted_returns = [traj[2][0] for traj in batch_trajectories]

        training_data["total_samples"][iteration] = sum(len(traj[2]) for traj in batch_trajectories)
        training_data["train_scores"][iteration] = np.mean(discounted_returns)
        training_data["params"][iteration] = np.concatenate([p.flatten() for p in params])
        training_data["walltime"][iteration] = perf_counter() - t

        # Update policy using all episodes in the batch
        params_update = update_policy(batch_trajectories, params, learning_rate, action_bounds)
        params = params_update[:-1]  # First 6 elements are the parameter updates
        grads = params_update[-1]  # Last element is the gradients array
        training_data["grads"][iteration] = grads
        
    save_path = os.path.join(folder_name, f"training_data_iter.npy")
    np.save(save_path, training_data)

    # At the end of reinforce function
    for env in envs:
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run REINFORCE optimization')

    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed (default: 0)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size (default: 64)')
    parser.add_argument('--n_evals', type=int, default=50,
                       help='Number of evaluation episodes (default: 50)')
    parser.add_argument('--size', type=int, default=16,
                       help='Hidden layer dimensions (default: 16)')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate (default: 0.01)')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor (default: 0.99)')
    
    args = parser.parse_args()
    
    reinforce(
        size=args.size,
        learning_rate=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        n_evals=args.n_evals,
        seed=args.seed,
    )
