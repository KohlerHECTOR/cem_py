from pathlib import Path
from common_utils import trajectory_dtype, get_score_batch, collect_trajectory, total_iters_budget_full
import numpy as np
from gymnasium import make
from joblib import Parallel, delayed

def find_weight_files(directory):
    """
    Find all files ending with 'weights.npy' in the given directory.
    
    Args:
        directory (str): Path to the directory to search
    
    Returns:
        list: List of Path objects for weight files
    """
    path = Path(directory)
    weight_files = list(path.glob('*weights.npy'))
    return weight_files

# Example usage
directory = ""
weight_files = find_weight_files(directory)
nb_weights = 100
p_shape = [(3, 16), 16, (16, 16), 16, (16, 2), 2]
sizes = [np.prod(p_s) for p_s in p_shape]
envs = [make("Pendulum-v1") for _ in range(50)]

for weight_file in weight_files:
    weights = np.load(weight_file)
    samples_per_iter = int(200 * total_iters_budget_full) // len(weights)
    idxs = np.linspace(0, len(weights) - 1, nb_weights, dtype=np.int32)
    scores = np.zeros(nb_weights)
    samples = np.zeros(nb_weights)

    for k, i in enumerate(idxs):
        params = []
        start_idx = 0
        for j, shape in enumerate(p_shape):
            size = sizes[j]
            param = weights[
                i, start_idx : start_idx + size
            ].reshape(shape)
            params.append(param)
            start_idx += size

        all_trajs = np.array(Parallel(n_jobs=-1)(
            delayed(collect_trajectory)(envs[e], params, deterministic=True) for e in range(50)
        ), dtype=trajectory_dtype)
        scores[k] =  get_score_batch(all_trajs)
        samples[k] = (k + 1) * samples_per_iter
    
    # Create new filename by adding 'test_scores' before the extension
    save_path = weight_file.with_name(weight_file.stem + '_test_scores' + weight_file.suffix)
    np.save(save_path, zip(scores, samples))
    print(save_path)
