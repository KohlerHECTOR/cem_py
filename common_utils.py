import numpy as np

total_iters_budget = int(64 * 50 * 100)

# Define trajectory dtype
trajectory_dtype = np.dtype(
    [
        ("states", np.float64, (200, 3)),
        ("actions", np.float64, 200),
        ("rewards", np.float64, 200),
        ("returns", np.float64, 200),
        ("alphas", np.float64, 200),
        ("betas", np.float64, 200),
        ("h1s", np.float64, (200, 16)),
        ("h2s", np.float64, (200, 16)),
        ("pre_softplus", np.float64, (200, 2)),
    ]
)


def init_policy_params():
    """
    Initialize policy parameters for network with Beta distribution output.
    For Pendulum the state is 3-dimensional, and the action 1-dimensional.
    We fix the ReLU net size to two hidden layers of 16 parameters.
    We use the He et al., 2015 init for ReLU net.
    """

    scale_w1 = np.sqrt(2.0 / 3)
    scale_w2 = np.sqrt(2.0 / 16)
    scale_w3 = np.sqrt(2.0 / 16)

    # Draw parameters with He initialization for hidden layers
    w1 = np.random.normal(0, scale_w1, (3, 16))
    b1 = np.zeros(16)

    w2 = np.random.normal(0, scale_w2, (16, 16))
    b2 = np.zeros(16)

    # Initialize final layer to produce values close to 0 before softplus
    w3 = np.random.normal(0, scale_w3, (16, 2))
    b3 = np.zeros(2)

    return w1, b1, w2, b2, w3, b3


def sample_action(state, params, deterministic=True):
    """Sample action from Beta distribution for a single state"""
    w1, b1, w2, b2, w3, b3 = params

    # Linear layers for single state
    h1 = np.maximum(0, state @ w1 + b1)  # ReLU
    h2 = np.maximum(0, h1 @ w2 + b2)  # ReLU
    # Store pre-softplus values
    pre_softplus = h2 @ w3 + b3

    # alpha beta are softplus(out) with numerical stability.
    alpha, beta = pre_softplus
    alpha = np.log1p(np.exp(alpha)) if alpha < 20 else alpha
    beta = np.log1p(np.exp(beta)) if beta < 20 else beta

    # Clip values to prevent overflow and div/0
    # Somehow np.log1p can return 0
    alpha = np.clip(alpha, 1e-8, 1e6)
    beta = np.clip(beta, 1e-8, 1e6)

    if deterministic:
        action = alpha / (alpha + beta)  # mean of beta distrib
    else:
        action = np.random.beta(alpha, beta)
    # Scale action from [0,1] to action bounds
    # The action range of Pendulum is [-2, 2]
    scaled_action = -2 + ((2) - (-2)) * action
    return scaled_action, alpha, beta, h1, h2, pre_softplus


def collect_trajectory(env, params, deterministic=True):
    """Collect a single trajectory and compute returns (to be run in parallel)"""

    # Create a single structured array for the trajectory
    trajectory = np.zeros(1, dtype=trajectory_dtype)[0]

    state, _ = env.reset()

    # Use a for loop since we know exact number of steps
    for t in range(200):
        action_taken, alpha, beta, h1, h2, pre_softplus = sample_action(
            state, params, deterministic
        )
        next_state, reward, term, trunc, _ = env.step([action_taken])

        trajectory["states"][t] = state
        trajectory["actions"][t] = action_taken
        trajectory["rewards"][t] = reward
        trajectory["alphas"][t] = alpha
        trajectory["betas"][t] = beta
        trajectory["h1s"][t] = h1
        trajectory["h2s"][t] = h2
        trajectory["pre_softplus"][t] = pre_softplus

        state = next_state
        if term or trunc:
            break

    # Compute returns for this trajectory
    future_return = 0
    for t in reversed(range(len(trajectory["rewards"]))):
        future_return = trajectory["rewards"][t] + future_return  # No discount factor
        trajectory["returns"][t] = future_return

    return trajectory


def get_score_batch(traj_batch):
    return np.mean([traj["returns"][0] for traj in traj_batch])
