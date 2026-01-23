import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# Configurations for different env variations
configs = {
    '10-env': {
        'search': 'logs_res_rm/lottery_ticket_results/can/envs10_samples5000_seed999_ddim8_20251130_223053_nenvs_10/ranking.txt',
        'eval': 'logs_res_rm/noise_eval_results/can/envs10_samples5000_seed999_ddim8_20251130_223053_nenvs_10',
        'eval_ticket0': 'logs_res_rm/noise_eval_results/can/envs10_samples5000_seed999_ddim8_20251130_223053_nenvs_10'
    },
    '50-env': {
        'search': 'logs_res_rm/lottery_ticket_results/can/envs50_samples5000_seed999_ddim8_20251130_223116_nenvs_50/ranking.txt',
        'eval': 'logs_res_rm/noise_eval_results/can/envs50_samples5000_seed999_ddim8_20251130_223116_nenvs_50',
        'eval_ticket0': 'logs_res_rm/noise_eval_results/can/envs50_samples5000_seed999_ddim8_20251130_223116_nenvs_50'
    },
    '100-env': {
        'search': 'logs_res_rm/lottery_ticket_results/can/envs100_samples5000_seed999_ddim8_20251130_221846_ddim8/ranking.txt',
        'eval': 'logs_res_rm/noise_eval_results/can/envs100_samples5000_seed999_ddim8_20251130_221846_ddim8',
        'eval_ticket0': 'logs_res_rm/noise_eval_results/can/envs100_samples5000_seed999_20251122_233346_ddim8'
    }
}

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for idx, (name, paths) in enumerate(configs.items()):
    print(f"\n=== {name} ===")
    
    # Load search results - success_rates.npy is already ranked
    search_dir = os.path.dirname(paths['search'])
    success_rates_file = os.path.join(search_dir, 'success_rates.npy')
    print(f"Search success rates file: {success_rates_file}")
    
    success_rates = np.load(success_rates_file, allow_pickle=True)
    print(f"Success rates shape: {success_rates.shape}")
    
    # Mean across environments for each rank (0-50 = 51 tickets)
    search_perf = {}
    for rank in range(51):
        search_perf[rank] = success_rates[rank].mean()
    
    print(f"Search success rates (first 5): {[search_perf[i] for i in range(5)]}")
    
    # Find eval results file
    eval_path = paths['eval']
    if os.path.isdir(eval_path):
        # Find the eval_idx1-50 subdirectory with aggregate_results.json
        subdirs = [d for d in os.listdir(eval_path) if 'eval_idx1-50' in d and os.path.isdir(os.path.join(eval_path, d))]
        if subdirs:
            eval_path = os.path.join(eval_path, subdirs[0], 'aggregate_results.json')
    
    print(f"Eval file (tickets 1-50): {eval_path}")
    
    # Load eval results for tickets 1-50
    with open(eval_path) as f:
        data = json.load(f)
    
    # Load ticket 0 separately from eval_idx0-2 directory
    eval_ticket0_dir = paths['eval_ticket0']
    ticket0_subdirs = [d for d in os.listdir(eval_ticket0_dir) if 'eval_idx0-2' in d and os.path.isdir(os.path.join(eval_ticket0_dir, d))]
    ticket0_path = os.path.join(eval_ticket0_dir, ticket0_subdirs[0], 'aggregate_results.json')
    print(f"Eval file (ticket 0): {ticket0_path}")
    with open(ticket0_path) as f:
        ticket0_data = json.load(f)
    ticket0_success_mean = ticket0_data['per_index_results']['0']['success_mean']
    ticket0_success_std = ticket0_data['per_index_results']['0']['success_std']
    
    eval_mean = []
    eval_std = []
    search_mean = []
    
    # Add ticket 0 if available
    search_mean.append(search_perf[0])  # ticket 0 = rank 0
    eval_mean.append(ticket0_success_mean)
    eval_std.append(ticket0_success_std)
    
    # Add tickets 1-50 (which are ranks 1-50)
    for ticket in range(1, 51):
        search_mean.append(search_perf[ticket])
        eval_mean.append(data['per_index_results'][str(ticket)]['success_mean'])
        eval_std.append(data['per_index_results'][str(ticket)]['success_std'])
    
    print(f"Ticket 0 - Search: {search_perf[0]:.3f}, Eval: {ticket0_success_mean:.3f} +/- {ticket0_success_std:.3f}" if ticket0_success_mean else "Ticket 0 not available")
    print(f"Eval success rates (tickets 1-5): {[data['per_index_results'][str(i)]['success_mean'] for i in range(1, 6)]}")
    
    # Plot
    ax = axes[idx]
    ax.errorbar(search_mean, eval_mean, yerr=eval_std, fmt='o', capsize=3,
                alpha=0.6, markersize=4)
    ax.set_xlabel('Search Performance', fontsize=11)
    ax.set_ylabel('Eval Performance', fontsize=11)
    ax.set_title(f'{name}', fontsize=12)
    
    # Diagonal reference line
    lim = [min(min(search_mean), min(eval_mean)) - 0.05,
           max(max(search_mean), max(eval_mean)) + 0.05]
    ax.plot(lim, lim, 'k--', alpha=0.3, linewidth=1)
    ax.grid(True, alpha=0.2)
    
    print(f"{name}: Search [{min(search_mean):.3f}, {max(search_mean):.3f}], "
          f"Eval [{min(eval_mean):.3f}, {max(eval_mean):.3f}]")

plt.tight_layout()
plt.savefig('regression_to_mean_all.png', dpi=300, bbox_inches='tight')
plt.show()

print("Saved plot to regression_to_mean_all.png")

# Export data to CSV
csv_data = []
for name in configs.keys():
    # Re-extract data for CSV (already computed above)
    pass

# Collect all data for CSV export
all_data = {'ticket': [], 'env_variation': [], 'search_success_rate': [], 
            'eval_success_rate': [], 'eval_success_std': []}

fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.5))

for idx, (name, paths) in enumerate(configs.items()):
    # Load search results
    search_dir = os.path.dirname(paths['search'])
    success_rates_file = os.path.join(search_dir, 'success_rates.npy')
    success_rates = np.load(success_rates_file, allow_pickle=True)
    
    search_perf = {}
    for rank in range(51):
        search_perf[rank] = success_rates[rank].mean()
    
    # Load eval results
    eval_path = paths['eval']
    if os.path.isdir(eval_path):
        subdirs = [d for d in os.listdir(eval_path) if 'eval_idx1-50' in d and os.path.isdir(os.path.join(eval_path, d))]
        if subdirs:
            eval_path = os.path.join(eval_path, subdirs[0], 'aggregate_results.json')
    
    with open(eval_path) as f:
        data = json.load(f)
    
    # Load ticket 0
    eval_ticket0_dir = paths['eval_ticket0']
    ticket0_subdirs = [d for d in os.listdir(eval_ticket0_dir) if 'eval_idx0-2' in d and os.path.isdir(os.path.join(eval_ticket0_dir, d))]
    ticket0_path = os.path.join(eval_ticket0_dir, ticket0_subdirs[0], 'aggregate_results.json')
    with open(ticket0_path) as f:
        ticket0_data = json.load(f)
    
    # Add ticket 0
    all_data['ticket'].append(0)
    all_data['env_variation'].append(name)
    all_data['search_success_rate'].append(search_perf[0])
    all_data['eval_success_rate'].append(ticket0_data['per_index_results']['0']['success_mean'])
    all_data['eval_success_std'].append(ticket0_data['per_index_results']['0']['success_std'])
    
    # Add tickets 1-50
    for ticket in range(1, 51):
        all_data['ticket'].append(ticket)
        all_data['env_variation'].append(name)
        all_data['search_success_rate'].append(search_perf[ticket])
        all_data['eval_success_rate'].append(data['per_index_results'][str(ticket)]['success_mean'])
        all_data['eval_success_std'].append(data['per_index_results'][str(ticket)]['success_std'])

df = pd.DataFrame(all_data)
df.to_csv('regression_to_mean_success_rates.csv', index=False)
print("Saved data to regression_to_mean_success_rates.csv")
