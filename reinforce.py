from common_utils import (
    init_policy_params,
    collect_trajectory,
    get_score_batch,
    trajectory_dtype,
    total_iters_budget_full as total_iters_budget,
)
import numpy as np
from gymnasium import make
from joblib import Parallel, delayed
from scipy.special import digamma
import argparse

def clip_grad_norm(grad):
    grad_norm = np.sqrt(np.sum(grad ** 2))
    if grad_norm > 1.0:
        return grad * (1.0 / (grad_norm + 1e-8))
    return grad

def reinforce(lr=0.01, batch_size=64, clip=False, seed=0):
    np.random.seed(seed)
    num_iterations = total_iters_budget // batch_size
    envs = [make("Pendulum-v1") for _ in range(batch_size)]
    w1, b1, w2, b2, w3, b3 = init_policy_params()
    all_trajs = np.zeros((num_iterations, batch_size), dtype=trajectory_dtype)

    results = []
    test_scores = np.zeros(num_iterations)
    for iteration in range(num_iterations):
        # Collect n_evals trajectories for each environment in parallel
        all_trajs[iteration] = Parallel(n_jobs=-1)(
            delayed(collect_trajectory)(
                envs[b], (w1, b1, w2, b2, w3, b3), deterministic=False
            )
            for b in range(batch_size)
        )
        tot_steps, train_score = (iteration + 1) * batch_size * 200, get_score_batch(
            all_trajs[iteration]
        )
        results.append([tot_steps, train_score])
        print(f"Steps: {tot_steps} - score: {train_score}")

        # Concatenate all episodes
        all_states = np.concatenate([traj["states"] for traj in all_trajs[iteration]])
        all_actions = np.concatenate([traj["actions"] for traj in all_trajs[iteration]])
        all_returns = np.concatenate([traj["returns"] for traj in all_trajs[iteration]])
        alphas = np.concatenate([traj["alphas"] for traj in all_trajs[iteration]])
        betas = np.concatenate([traj["betas"] for traj in all_trajs[iteration]])
        h1s = np.concatenate([traj["h1s"] for traj in all_trajs[iteration]])
        h2s = np.concatenate([traj["h2s"] for traj in all_trajs[iteration]])
        pre_softplus_outputs = np.concatenate(
            [traj["pre_softplus"] for traj in all_trajs[iteration]]
        )

        # Normalize returns for stability
        all_returns = (all_returns - all_returns.mean()) / (all_returns.std() + 1e-8)

        # Scale actions to [0, 1]
        scaled_actions = (all_actions - (-2)) / (2 - (-2))

        # Clip actions to [epsilon, 1-epsilon] for numerical stability, Beta distrib is defined on ]0, 1[
        epsilon = 1e-8
        scaled_actions = np.clip(scaled_actions, epsilon, 1 - epsilon)

        # Compute log probability gradients for Beta distribution
        common_term = digamma(alphas + betas)
        grad_alpha = (
            common_term - digamma(alphas) + np.log(scaled_actions)
        ) * all_returns
        grad_beta = (
            common_term - digamma(betas) + np.log(1 - scaled_actions)
        ) * all_returns

        # Account for exponential transformation from log params
        grad_log_alpha = grad_alpha * alphas
        grad_log_beta = grad_beta * betas

        # Apply softplus gradient (sigmoid) with numerical stability
        softplus_grad = np.zeros_like(pre_softplus_outputs)
        # Apply conditional gradient based on the same condition in the forward pass
        mask_small = pre_softplus_outputs < 20

        # For values < 20, use numerically stable sigmoid gradient
        # For very negative values, sigmoid(x) ≈ 0, so gradient ≈ 0
        # For very positive values, sigmoid(x) ≈ 1, so gradient ≈ 1
        # For values in between, compute normally
        safe_range = (pre_softplus_outputs > -20) & mask_small
        very_negative = pre_softplus_outputs <= -20

        # For values in safe range, compute sigmoid normally
        softplus_grad[safe_range] = 1 / (1 + np.exp(-pre_softplus_outputs[safe_range]))
        # For very negative values, gradient is approximately 0
        softplus_grad[very_negative] = 1e-8
        # For values >= 20, gradient is 1 (since we directly use the value)
        softplus_grad[~mask_small] = 1.0

        grad_pre_softplus = np.column_stack(
            [grad_log_alpha * softplus_grad[:, 0], grad_log_beta * softplus_grad[:, 1]]
        )

        # Backpropagation
        dh2 = (grad_pre_softplus @ w3.T) * (h2s > 0)
        dh1 = (dh2 @ w2.T) * (h1s > 0)

        # Compute all gradients with proper shapes
        grad_w3 = h2s.T @ grad_pre_softplus  # (16, 2)
        grad_b3 = grad_pre_softplus.sum(axis=0)  # (2,)
        grad_w2 = h1s.T @ dh2  # (16, 16)
        grad_b2 = dh2.sum(axis=0)  # (16,)
        grad_w1 = all_states.T @ dh1  # (3, 16)
        grad_b1 = dh1.sum(axis=0)  # (16,)

        # Parameter updates
        # Add gradient norm clipping
        if clip:            
            # Apply gradient norm clipping
            grad_w1 = clip_grad_norm(grad_w1)
            grad_b1 = clip_grad_norm(grad_b1)
            grad_w2 = clip_grad_norm(grad_w2)
            grad_b2 = clip_grad_norm(grad_b2)
            grad_w3 = clip_grad_norm(grad_w3)
            grad_b3 = clip_grad_norm(grad_b3)
            
        w3 = w3 + lr * grad_w3
        b3 = b3 + lr * grad_b3
        w2 = w2 + lr * grad_w2
        b2 = b2 + lr * grad_b2
        w1 = w1 + lr * grad_w1
        b1 = b1 + lr * grad_b1

        eval_trajs = Parallel(n_jobs=-1)(
            delayed(collect_trajectory)(
                make("Pendulum-v1"), (w1, b1, w2, b2, w3, b3), deterministic=True
            )
            for e in range(50)
        )
        test_scores[iteration] = get_score_batch(eval_trajs)
    np.save(f"full_results_reinforce_lr{lr}_batch_size{batch_size}_clip{clip}_seed{seed}_train_scores.npy", results)
    np.save(f"full_results_reinforce_lr{lr}_batch_size{batch_size}_clip{clip}_seed{seed}_test_scores.npy", test_scores)
    
    for env in envs:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="REINFORCE algorithm for Pendulum-v1")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--clip", action="store_true", help="Whether to clip gradients")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()
    reinforce(lr=args.lr, batch_size=args.batch_size, clip=args.clip, seed=args.seed)
