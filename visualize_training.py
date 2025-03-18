import numpy as np
from gymnasium import make
import argparse
from cem_beta_pendulum import sample_action

def load_best_policy(file_path):
    """Load the best policy from the saved training data"""
    training_data = np.load(file_path, allow_pickle=True).item()
    
    # Get the last iteration data
    last_iter = len(training_data["iterations"]) - 1
    
    # Get the best population index from the last iteration
    best_idx = training_data["best_pop_idx"][last_iter]
    
    # Get the weights of the best policy
    best_weights = training_data["population_weights"][last_iter][best_idx]
    
    # Get the score of the best policy
    best_score = training_data["population_scores"][last_iter][best_idx]
    
    print(f"Loaded best policy from iteration {last_iter}")
    print(f"Best score: {best_score}")
    
    return best_weights

def reconstruct_policy_params(flat_weights, input_size, hidden_size, output_size):
    """Reconstruct policy parameters from flattened weights"""
    # Calculate indices for reshaping
    w1_size = input_size * hidden_size
    b1_size = hidden_size
    w2_size = hidden_size * hidden_size
    b2_size = hidden_size
    w3_size = hidden_size * output_size
    b3_size = output_size
    
    w1_end = w1_size
    b1_end = w1_end + b1_size
    w2_end = b1_end + w2_size
    b2_end = w2_end + b2_size
    w3_end = b2_end + w3_size
    
    # Reshape weights
    w1 = flat_weights[0:w1_end].reshape(input_size, hidden_size)
    b1 = flat_weights[w1_end:b1_end]
    w2 = flat_weights[b1_end:w2_end].reshape(hidden_size, hidden_size)
    b2 = flat_weights[w2_end:b2_end]
    w3 = flat_weights[b2_end:w3_end].reshape(hidden_size, output_size)
    b3 = flat_weights[w3_end:]
    
    return w1, b1, w2, b2, w3, b3

def episodes(weights_file, hidden_size=16, num_episodes=100):
    """Visualize the best policy from the saved weights"""
    # Load the best policy
    best_weights = load_best_policy(weights_file)
    
    # Create environment
    env = make("Pendulum-v1")
    
    # Get dimensions
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0] * 2  # alpha and beta parameters
    action_bounds = (env.action_space.low.copy(), env.action_space.high.copy())
    
    # Reconstruct policy parameters
    params = reconstruct_policy_params(best_weights, input_size, hidden_size, output_size)
    
    # Run episodes
    total_reward = 0
    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Sample action using the policy
            action, _, _, _, _ = sample_action(state, params, action_bounds)
            
            # Take step in environment
            next_state, reward, term, trunc, _ = env.step(action)
            
            episode_reward += reward
            state = next_state
            done = term or trunc
        
        print(f"Episode {ep+1} reward: {episode_reward}")
        total_reward += episode_reward
    
    print(f"Average reward over {num_episodes} episodes: {total_reward/num_episodes}")
    env.close()

def plot_policy(weights_file, hidden_size=16, resolution=100):
    """
    Plot the policy mean actions over the 2D state space of the pendulum.
    
    Args:
        weights_file: Path to the .npy file containing training data
        hidden_size: Size of hidden layers
        resolution: Number of points to sample in each dimension
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    
    # Load the best policy
    best_weights = load_best_policy(weights_file)
    
    # Create environment to get action space bounds
    env = make("Pendulum-v1")
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0] * 2  # alpha and beta parameters
    action_bounds = (env.action_space.low.copy(), env.action_space.high.copy())
    
    # Reconstruct policy parameters
    params = reconstruct_policy_params(best_weights, input_size, hidden_size, output_size)
    
    # Create a grid of states using theta and theta_dot
    theta = np.linspace(-np.pi, np.pi, resolution)  # Angle in radians
    theta_dot = np.linspace(-8, 8, resolution)      # Angular velocity
    
    # Initialize action grid
    action_grid = np.zeros((resolution, resolution))
    
    # Compute actions for each state
    for i, th in enumerate(theta):
        for j, th_dot in enumerate(theta_dot):
            # Convert theta to cos(theta) and sin(theta) for the observation space
            cos_th = np.cos(th)
            sin_th = np.sin(th)
            
            # Create state
            state = np.array([cos_th, sin_th, th_dot])
            
            # Get mean action (alpha parameter determines the mean)
            action, _, _, _, _ = sample_action(state, params, action_bounds)
            
            # Store the mean action
            action_grid[j, i] = action[0]  # Use first element if alpha is an array
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Use a fixed color range from -2 to 2
    norm = Normalize(vmin=-2, vmax=2)
    
    # Create the heatmap with fixed color scale
    plt.imshow(action_grid, extent=[-np.pi, np.pi, -8, 8], origin='lower', 
               aspect='auto', cmap='inferno', norm=norm)
    
    # Remove x-axis ticks
    plt.xticks([])
    plt.yticks([])

    
    # Make axis labels bigger
    plt.xlabel('θ (radians)', fontsize=22)
    plt.ylabel('θ̇*(angular velocity)', fontsize=22)
    
    # Add horizontal line at theta_dot = 0
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.5)
    # Add vertical line at theta = 0
    plt.axvline(x=0, color='r', linestyle='-', alpha=0.5)
    
    # Mark the upright equilibrium point
    plt.plot(0, 0, 'ro', markersize=15)
    
    # Add grid for better readability
    plt.grid(alpha=0.5, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('policy_visualization.png')
    
    env.close()

def plot_policy_evolution(weights_file, hidden_size=16, resolution=100, output_gif="policy_evolution.gif"):
    """
    Create a GIF showing how the policy evolves throughout training.
    
    Args:
        weights_file: Path to the .npy file containing training data
        hidden_size: Size of hidden layers
        resolution: Number of points to sample in each dimension
        output_gif: Path to save the output GIF
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import imageio.v2 as imageio
    import os
    from tqdm import tqdm
    
    # Load all training data
    training_data = np.load(weights_file, allow_pickle=True).item()
    
    # Create environment to get action space bounds
    env = make("Pendulum-v1")
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0] * 2  # alpha and beta parameters
    action_bounds = (env.action_space.low.copy(), env.action_space.high.copy())
    
    # Create a grid of states using theta and theta_dot
    theta = np.linspace(-np.pi, np.pi, resolution)  # Angle in radians
    theta_dot = np.linspace(-8, 8, resolution)      # Angular velocity
    
    # Create a temporary directory to store frames
    import tempfile
    temp_dir = tempfile.mkdtemp()
    
    # Get total number of iterations
    num_iterations = len(training_data["iterations"])
    
    # Get the total samples data directly
    cumulative_samples = training_data["total_samples"]
    
    # Create frames for each iteration
    frames = []
    
    print(f"Creating policy evolution GIF with {num_iterations} frames...")
    for iter_idx in tqdm(range(num_iterations)):
        # Get the best population index from this iteration
        best_idx = training_data["best_pop_idx"][iter_idx]
        
        # Get the weights of the best policy
        best_weights = training_data["population_weights"][iter_idx][best_idx]
        
        # Get the score of the best policy
        best_score = training_data["population_scores"][iter_idx][best_idx]
        
        # Get total samples so far
        total_samples = np.sum(cumulative_samples[:iter_idx])
        
        # Reconstruct policy parameters
        params = reconstruct_policy_params(best_weights, input_size, hidden_size, output_size)
        
        # Initialize action grid
        action_grid = np.zeros((resolution, resolution))
        
        # Compute actions for each state
        for i, th in enumerate(theta):
            for j, th_dot in enumerate(theta_dot):
                # Convert theta to cos(theta) and sin(theta) for the observation space
                cos_th = np.cos(th)
                sin_th = np.sin(th)
                
                # Create state
                state = np.array([cos_th, sin_th, th_dot])
                
                # Get mean action
                action, _, _, _, _ = sample_action(state, params, action_bounds)
                
                # Store the mean action
                action_grid[j, i] = action[0]
        
        # Create the plot
        plt.figure(figsize=(10, 8))
        
        # Use a fixed color range from -2 to 2
        norm = Normalize(vmin=-2, vmax=2)
        
        # Create the heatmap with fixed color scale
        plt.imshow(action_grid, extent=[-np.pi, np.pi, -8, 8], origin='lower', 
                   aspect='auto', cmap='inferno', norm=norm)
        
        # Remove x-axis ticks
        plt.xticks([])
        plt.yticks([])
        
        # Make axis labels bigger
        plt.xlabel('θ (radians)', fontsize=22)
        plt.ylabel('θ̇ (angular velocity)', fontsize=22)
        
        # Add horizontal line at theta_dot = 0
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.5)
        # Add vertical line at theta = 0
        plt.axvline(x=0, color='r', linestyle='-', alpha=0.5)
        
        # Mark the upright equilibrium point
        plt.plot(0, 0, 'ro', markersize=15)
        
        # Add grid for better readability
        plt.grid(alpha=0.5, linestyle='--')
        
        # Add total samples and score information
        plt.title(f"Total Samples: {total_samples}, Score: {best_score:.2f}", fontsize=18)
        
        # Save the frame
        frame_path = os.path.join(temp_dir, f"frame_{iter_idx:04d}.png")
        plt.tight_layout()
        plt.savefig(frame_path)
        plt.close()
        
        frames.append(frame_path)
    
    # Create GIF
    print(f"Creating GIF at {output_gif}...")
    with imageio.get_writer(output_gif, mode='I', duration=0.2) as writer:
        for frame_path in frames:
            image = imageio.imread(frame_path)
            writer.append_data(image)
    
    # Clean up temporary files
    for frame_path in frames:
        os.remove(frame_path)
    os.rmdir(temp_dir)
    
    print(f"GIF created successfully at {output_gif}")
    env.close()

def plot_policy_evolution_reinforce(weights_file, hidden_size=16, resolution=100, output_gif="policy_evolution_reinforce.gif"):
    """
    Create a GIF showing how the policy evolves throughout training.
    
    Args:
        weights_file: Path to the .npy file containing training data
        hidden_size: Size of hidden layers
        resolution: Number of points to sample in each dimension
        output_gif: Path to save the output GIF
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import imageio.v2 as imageio
    import os
    from tqdm import tqdm
    
    # Load all training data
    training_data = np.load(weights_file, allow_pickle=True).item()
    
    # Create environment to get action space bounds
    env = make("Pendulum-v1")
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0] * 2  # alpha and beta parameters
    action_bounds = (env.action_space.low.copy(), env.action_space.high.copy())
    
    # Create a grid of states using theta and theta_dot
    theta = np.linspace(-np.pi, np.pi, resolution)  # Angle in radians
    theta_dot = np.linspace(-8, 8, resolution)      # Angular velocity
    
    # Create a temporary directory to store frames
    import tempfile
    temp_dir = tempfile.mkdtemp()
    
    # Get total number of iterations
    num_iterations = len(training_data["iterations"])
    
    # Get the total samples data directly
    cumulative_samples = training_data["total_samples"]
    
    # Create frames for each iteration
    frames = []
    
    print(f"Creating policy evolution GIF with 100 frames...")
    for iter_idx in tqdm(range(0, num_iterations, num_iterations // 100)):
        # Get total samples so far
        best_score = training_data["train_scores"][iter_idx]
        total_samples = np.sum(cumulative_samples[:iter_idx])
        
        # Reconstruct policy parameters
        params = reconstruct_policy_params(training_data["params"][iter_idx], input_size, hidden_size, output_size)
        
        # Initialize action grid
        action_grid = np.zeros((resolution, resolution))
        
        # Compute actions for each state
        for i, th in enumerate(theta):
            for j, th_dot in enumerate(theta_dot):
                # Convert theta to cos(theta) and sin(theta) for the observation space
                cos_th = np.cos(th)
                sin_th = np.sin(th)
                
                # Create state
                state = np.array([cos_th, sin_th, th_dot])
                
                # Get mean action
                action, _, _, _, _ = sample_action(state, params, action_bounds)
                
                # Store the mean action
                action_grid[j, i] = action[0]
        
        # Create the plot
        plt.figure(figsize=(10, 8))
        
        # Use a fixed color range from -2 to 2
        norm = Normalize(vmin=-2, vmax=2)
        
        # Create the heatmap with fixed color scale
        plt.imshow(action_grid, extent=[-np.pi, np.pi, -8, 8], origin='lower', 
                   aspect='auto', cmap='inferno', norm=norm)
        
        # Remove x-axis ticks
        plt.xticks([])
        plt.yticks([])
        
        # Make axis labels bigger
        plt.xlabel('θ (radians)', fontsize=22)
        plt.ylabel('θ̇ (angular velocity)', fontsize=22)
        
        # Add horizontal line at theta_dot = 0
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.5)
        # Add vertical line at theta = 0
        plt.axvline(x=0, color='r', linestyle='-', alpha=0.5)
        
        # Mark the upright equilibrium point
        plt.plot(0, 0, 'ro', markersize=15)
        
        # Add grid for better readability
        plt.grid(alpha=0.5, linestyle='--')
        
        # Add total samples and score information
        plt.title(f"Total Samples: {total_samples}, Score: {best_score:.2f}", fontsize=18)
        
        # Save the frame
        frame_path = os.path.join(temp_dir, f"frame_{iter_idx:04d}.png")
        plt.tight_layout()
        plt.savefig(frame_path)
        plt.close()
        
        frames.append(frame_path)
    
    # Create GIF
    print(f"Creating GIF at {output_gif}...")
    with imageio.get_writer(output_gif, mode='I', duration=0.2) as writer:
        for frame_path in frames:
            image = imageio.imread(frame_path)
            writer.append_data(image)
    
    # Clean up temporary files
    for frame_path in frames:
        os.remove(frame_path)
    os.rmdir(temp_dir)
    
    print(f"GIF created successfully at {output_gif}")
    env.close()

def plot_policy_evolution_comparison(cem_weights_file, reinforce_weights_file, hidden_size=16, resolution=100, output_gif="policy_comparison.gif"):
    """
    Create a GIF showing how both CEM and REINFORCE policies evolve side by side.
    
    Args:
        cem_weights_file: Path to the .npy file containing CEM training data
        reinforce_weights_file: Path to the .npy file containing REINFORCE training data
        hidden_size: Size of hidden layers
        resolution: Number of points to sample in each dimension
        output_gif: Path to save the output GIF
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import imageio.v2 as imageio
    import os
    from tqdm import tqdm
    
    # Load training data
    cem_data = np.load(cem_weights_file, allow_pickle=True).item()
    reinforce_data = np.load(reinforce_weights_file, allow_pickle=True).item()
    
    # Create environment to get action space bounds
    env = make("Pendulum-v1")
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0] * 2  # alpha and beta parameters
    action_bounds = (env.action_space.low.copy(), env.action_space.high.copy())
    
    # Create a grid of states using theta and theta_dot
    theta = np.linspace(-np.pi, np.pi, resolution)  # Angle in radians
    theta_dot = np.linspace(-8, 8, resolution)      # Angular velocity
    
    # Create a temporary directory to store frames
    import tempfile
    temp_dir = tempfile.mkdtemp()
    
    # Get total number of iterations
    cem_iterations = len(cem_data["iterations"])
    reinforce_iterations = len(reinforce_data["iterations"])
    
    # Get the total samples data
    cem_samples = cem_data["total_samples"]
    reinforce_samples = reinforce_data["total_samples"]
    
    # Create frames for each iteration
    frames = []
    
    # Determine how many frames to create (100 frames total)
    num_frames = 100
    cem_step = max(1, cem_iterations // num_frames)
    reinforce_step = max(1, reinforce_iterations // num_frames)
    
    print(f"Creating policy comparison GIF with {num_frames} frames...")
    for frame_idx in tqdm(range(num_frames)):
        # Calculate indices for each algorithm
        cem_idx = min(frame_idx * cem_step, cem_iterations - 1)
        reinforce_idx = min(frame_idx * reinforce_step, reinforce_iterations - 1)
        
        # Get CEM data for this frame
        cem_best_idx = cem_data["best_pop_idx"][cem_idx]
        cem_best_weights = cem_data["population_weights"][cem_idx][cem_best_idx]
        cem_best_score = cem_data["population_scores"][cem_idx][cem_best_idx]
        cem_total_samples = np.sum(cem_samples[:cem_idx])
        
        # Get REINFORCE data for this frame
        reinforce_best_score = reinforce_data["train_scores"][reinforce_idx]
        reinforce_weights = reinforce_data["params"][reinforce_idx]
        reinforce_total_samples = np.sum(reinforce_samples[:reinforce_idx])
        
        # Use the smaller of the two sample counts for the common title
        total_samples = min(cem_total_samples, reinforce_total_samples)
        
        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Use a fixed color range from -2 to 2
        norm = Normalize(vmin=-2, vmax=2)
        
        # Plot CEM policy
        cem_params = reconstruct_policy_params(cem_best_weights, input_size, hidden_size, output_size)
        cem_action_grid = np.zeros((resolution, resolution))
        
        # Compute actions for each state for CEM
        for i, th in enumerate(theta):
            for j, th_dot in enumerate(theta_dot):
                cos_th = np.cos(th)
                sin_th = np.sin(th)
                state = np.array([cos_th, sin_th, th_dot])
                action, _, _, _, _ = sample_action(state, cem_params, action_bounds)
                cem_action_grid[j, i] = action[0]
        
        # Plot REINFORCE policy
        reinforce_params = reconstruct_policy_params(reinforce_weights, input_size, hidden_size, output_size)
        reinforce_action_grid = np.zeros((resolution, resolution))
        
        # Compute actions for each state for REINFORCE
        for i, th in enumerate(theta):
            for j, th_dot in enumerate(theta_dot):
                cos_th = np.cos(th)
                sin_th = np.sin(th)
                state = np.array([cos_th, sin_th, th_dot])
                action, _, _, _, _ = sample_action(state, reinforce_params, action_bounds)
                reinforce_action_grid[j, i] = action[0]
        
        # Plot CEM
        im1 = ax1.imshow(cem_action_grid, extent=[-np.pi, np.pi, -8, 8], origin='lower', 
                         aspect='auto', cmap='inferno', norm=norm)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_xlabel('θ (radians)', fontsize=18)
        ax1.set_ylabel('θ̇ (angular velocity)', fontsize=18)
        ax1.axhline(y=0, color='r', linestyle='-', alpha=0.5)
        ax1.axvline(x=0, color='r', linestyle='-', alpha=0.5)
        ax1.plot(0, 0, 'ro', markersize=12)
        ax1.grid(alpha=0.5, linestyle='--')
        ax1.set_title(f"CEM: Score = {cem_best_score:.2f}", fontsize=20)
        
        # Plot REINFORCE
        im2 = ax2.imshow(reinforce_action_grid, extent=[-np.pi, np.pi, -8, 8], origin='lower', 
                         aspect='auto', cmap='inferno', norm=norm)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_xlabel('θ (radians)', fontsize=18)
        ax2.set_ylabel('θ̇ (angular velocity)', fontsize=18)
        ax2.axhline(y=0, color='r', linestyle='-', alpha=0.5)
        ax2.axvline(x=0, color='r', linestyle='-', alpha=0.5)
        ax2.plot(0, 0, 'ro', markersize=12)
        ax2.grid(alpha=0.5, linestyle='--')
        ax2.set_title(f"REINFORCE: Score = {reinforce_best_score:.2f}", fontsize=20)
        
        # Add common suptitle
        plt.suptitle(f"Total Samples: {total_samples}", fontsize=20)
        
        # Save the frame
        frame_path = os.path.join(temp_dir, f"frame_{frame_idx:04d}.png")
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
        plt.savefig(frame_path)
        plt.close()
        
        frames.append(frame_path)
    
    # Create GIF
    print(f"Creating GIF at {output_gif}...")
    with imageio.get_writer(output_gif, mode='I', duration=0.2) as writer:
        for frame_path in frames:
            image = imageio.imread(frame_path)
            writer.append_data(image)
    
    # Clean up temporary files
    for frame_path in frames:
        os.remove(frame_path)
    os.rmdir(temp_dir)
    
    print(f"Comparison GIF created successfully at {output_gif}")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize CEM trained policy")
    parser.add_argument("--weights", type=str, required=True, 
                        help="Path to the .npy file containing training data")

    parser.add_argument("--weights_reinforce", type=str, default=None, 
                        help="Path to the .npy file containing training data")

    parser.add_argument("--hidden_size", type=int, default=16,
                        help="Size of hidden layers (default: 16)")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of episodes to run (default: 100)")
    parser.add_argument("--gif", action="store_true",
                        help="Create a GIF of policy evolution")
    parser.add_argument("--output_gif", type=str, default="policy_evolution.gif",
                        help="Path to save the output GIF (default: policy_evolution.gif)")
    parser.add_argument("--compare", action="store_true",
                        help="Create a side-by-side comparison GIF of CEM and REINFORCE")
    
    args = parser.parse_args()
    
    if args.compare and args.weights_reinforce is not None:
        plot_policy_evolution_comparison(
            args.weights,
            args.weights_reinforce,
            hidden_size=args.hidden_size,
            output_gif=args.output_gif
        )
    elif args.gif:
        if args.weights_reinforce is not None:
            plot_policy_evolution_reinforce(
                args.weights_reinforce
            )
        else:
            plot_policy_evolution(
                args.weights,
                hidden_size=args.hidden_size,
                output_gif=args.output_gif
            )
    else:
        plot_policy(args.weights)