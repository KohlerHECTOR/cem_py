import numpy as np
# Function to get top N combinations
def get_top_n_combinations(algorithm='cem', n=10):
    results = []
    
    if algorithm.lower() == 'cem':
        for p in [8, 16, 32, 64, 128, 256]:
            for prop in [0.05, 0.1, 0.125, 0.25, 0.5]:
                returns = []
                for s in [0, 1, 2]:
                    try:
                        a = np.load(f"results_sweep/results_cem_elite_prop{prop}_batch_size{p}_seed{s}.npy")
                        if np.all(~np.isnan(a[:, 1]) & ~np.isinf(a[:, 1]) & (a[:, 1] < 0)):
                            returns.append(a[-1, 1])
                    except FileNotFoundError:
                        continue
                if returns:
                    mean_return = np.mean(returns)
                    std_return = np.std(returns) if len(returns) > 1 else float('inf')
                    results.append({
                        'batch_size': p,
                        'elite_prop': prop,
                        'mean_return': mean_return,
                        'std_return': std_return,
                        'n_seeds': len(returns)
                    })
    else:  # REINFORCE
        for bs in [1, 2, 4, 8, 16, 32, 64, 128]:
            for lr in [0.0001, 0.001, 0.01, 0.1]:
                for clip in [True]:
                    returns = []
                    for s in [0, 1, 2]:
                        try:
                            a = np.load(f"results_sweep/results_reinforce_lr{lr}_batch_size{bs}_clip{clip}_seed{s}.npy")
                            if np.all(~np.isnan(a[:, 1]) & ~np.isinf(a[:, 1]) & (a[:, 1] < 0)):
                                returns.append(a[-1, 1])
                        except FileNotFoundError:
                            continue
                    if returns:
                        mean_return = np.mean(returns)
                        std_return = np.std(returns) if len(returns) > 1 else float('inf')
                        results.append({
                            'batch_size': bs,
                            'learning_rate': lr,
                            'clip': clip,
                            'mean_return': mean_return,
                            'std_return': std_return,
                            'n_seeds': len(returns)
                        })
    
    # Sort by mean return (higher is better)
    sorted_results = sorted(results, key=lambda x: x['mean_return'], reverse=True)
    return sorted_results[:n]

# Print top combinations
print("\nTop 10 CEM combinations:")
for i, result in enumerate(get_top_n_combinations('cem', 10), 1):
    print(f"{i}. Batch Size: {result['batch_size']}, Elite Prop: {result['elite_prop']:.3f}, "
          f"Return: {result['mean_return']:.2f} ± {result['std_return']:.2f} (n={result['n_seeds']})")

print("\nTop 10 REINFORCE combinations:")
for i, result in enumerate(get_top_n_combinations('reinforce', 10), 1):
    print(f"{i}. Batch Size: {result['batch_size']}, Learning Rate: {result['learning_rate']} "
          f"Return: {result['mean_return']:.2f} ± {result['std_return']:.2f} (n={result['n_seeds']})")
