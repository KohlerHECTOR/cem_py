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

def visualize_policy(weights_file, hidden_size=16, num_episodes=100):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize CEM trained policy")
    parser.add_argument("--weights", type=str, required=True, 
                        help="Path to the .npy file containing training data")
    parser.add_argument("--hidden_size", type=int, default=16,
                        help="Size of hidden layers (default: 16)")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of episodes to run (default: 100)")
    
    args = parser.parse_args()
    
    visualize_policy(
        args.weights,
        hidden_size=args.hidden_size,
        num_episodes=args.episodes
    )
