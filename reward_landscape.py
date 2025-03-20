from gymnasium import make
import numpy as np
from joblib import Parallel, delayed
import os
from cem_beta_pendulum import sample_action, init_policy_params, evaluate_weights

def sample_and_evaluate_weights(
    n_samples=500000,      # Number of weight samples
    n_evals=50,              # Evaluations per weight set
    size=16,                # Hidden layer size
    gamma=0.99,            # Discount factor
    seed=0                  # Random seed
):
    # Initialize environment and get dimensions
    cvx_hull = np.load("cvx_hull_weights_baselines.npy")
    mins, maxs = cvx_hull[0] - 0.5, cvx_hull[1] + 0.5

    env = make("Pendulum-v1")
    input_ = env.observation_space.shape[0]
    output = env.action_space.shape[0] * 2  # 2 params: alpha, beta
    action_bounds = (env.action_space.low.copy(), env.action_space.high.copy())
    layer_sizes = [input_, size, size, output]
    
    # Get initial parameters as the mean
    w1, b1, w2, b2, w3, b3 = init_policy_params(layer_sizes)
    
    # Calculate sizes for flattening/reshaping
    w1_end = w1.size
    b1_end = w1_end + b1.size
    w2_end = b1_end + w2.size
    b2_end = w2_end + b2.size
    w3_end = b2_end + w3.size
    params_sizes = (input_, size, output, w1_end, b1_end, w2_end, b2_end, w3_end)
    total_weight_size = w3_end + b3.size
    

    all_weights = np.zeros((n_samples, total_weight_size))
    for dim in range(total_weight_size):
        all_weights[:,dim] = np.random.uniform(mins[dim], maxs[dim], n_samples)
    
    # Evaluate all weight sets in parallel
    eval_res = Parallel(n_jobs=-1)(
        delayed(evaluate_weights)(
            n_evals,
            params_sizes,
            action_bounds,
            env,
            weights,
            gamma,
        ) # Only take the mean reward, not episode lengths
        for weights in all_weights[:n_samples//2]
    )
    

    all_scores, _ = zip(*eval_res)
    # Convert to numpy array
    all_scores = np.array(all_scores)

    # Save all weights and scores
    output_file = f"landscape_pendul_first_half.npz"
    np.savez(output_file, weights=all_weights, scores=all_scores)


    eval_res = Parallel(n_jobs=-1)(
        delayed(evaluate_weights)(
            n_evals,
            params_sizes,
            action_bounds,
            env,
            weights,
            gamma,
        ) # Only take the mean reward, not episode lengths
        for weights in all_weights[n_samples//2:]
    )
    

    all_scores, _ = zip(*eval_res)
    # Convert to numpy array
    all_scores = np.array(all_scores)

    # Save all weights and scores
    output_file = f"landscape_pendul_second_half.npz"
    np.savez(output_file, weights=all_weights, scores=all_scores)

sample_and_evaluate_weights()