import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# Configurations for different env variations
configs = {
    '10-env': {
        'search': 'logs_res_rm/lottery_ticket_results/can/envs10_samples5000_seed999_ddim8_20251130_223053_nenvs_10',
        'eval': 'logs_res_rm/noise_eval_results/can/envs10_samples5000_seed999_ddim8_20251130_223053_nenvs_10',
        'eval_ticket0': 'logs_res_rm/noise_eval_results/can/envs10_samples5000_seed999_ddim8_20251130_223053_nenvs_10'
    },
    '50-env': {
        'search': 'logs_res_rm/lottery_ticket_results/can/envs50_samples5000_seed999_ddim8_20251130_223116_nenvs_50',
        'eval': 'logs_res_rm/noise_eval_results/can/envs50_samples5000_seed999_ddim8_20251130_223116_nenvs_50',
        'eval_ticket0': 'logs_res_rm/noise_eval_results/can/envs50_samples5000_seed999_ddim8_20251130_223116_nenvs_50'
    },
    '100-env': {
        'search': 'logs_res_rm/lottery_ticket_results/can/envs100_samples5000_seed999_ddim8_20251130_221846_ddim8',
        'eval': 'logs_res_rm/noise_eval_results/can/envs100_samples5000_seed999_ddim8_20251130_221846_ddim8',
        'eval_ticket0': 'logs_res_rm/noise_eval_results/can/envs100_samples5000_seed999_20251122_233346_ddim8'
    }
}

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for idx, (name, paths) in enumerate(configs.items()):
    print(f"\n=== {name} ===")
    
    # Load search results - rewards.npy is already ranked
    rewards_file = os.path.join(paths['search'], 'rewards.npy')
    print(f"Search rewards file: {rewards_file}")
    
    rewards_array = np.load(rewards_file, allow_pickle=True)
    print(f"Rewards array shape: {rewards_array.shape}")
    
    # Mean across environments for each rank (0-50 = 51 tickets)
    search_rewards = {}
    for rank in range(51):
        search_rewards[rank] = rewards_array[rank].mean()
    
    print(f"Search rewards (first 5): {[search_rewards[i] for i in range(5)]}")
    
    # Find eval results file
    eval_path = paths['eval']
    subdirs = [d for d in os.listdir(eval_path) if 'eval_idx1-50' in d and os.path.isdir(os.path.join(eval_path, d))]
    eval_json = os.path.join(eval_path, subdirs[0], 'aggregate_results.json')
    print(f"Eval file (tickets 1-50): {eval_json}")
    
    # Load eval results for tickets 1-50
    with open(eval_json) as f:
        data = json.load(f)
    
    # Load ticket 0 separately from eval_idx0-2 directory
    eval_ticket0_dir = paths['eval_ticket0']
    ticket0_subdirs = [d for d in os.listdir(eval_ticket0_dir) if 'eval_idx0-2' in d and os.path.isdir(os.path.join(eval_ticket0_dir, d))]
    if ticket0_subdirs:
        ticket0_path = os.path.join(eval_ticket0_dir, ticket0_subdirs[0], 'aggregate_results.json')
        print(f"Eval file (ticket 0): {ticket0_path}")
        with open(ticket0_path) as f:
            ticket0_data = json.load(f)
        ticket0_reward_mean = ticket0_data['per_index_results']['0']['reward_mean']
        ticket0_reward_std = ticket0_data['per_index_results']['0']['reward_std']
    
    eval_mean = []
    eval_std = []
    search_mean = []
    
    # Add ticket 0 if available
    search_mean.append(search_rewards[0])  # ticket 0 = rank 0
    eval_mean.append(ticket0_reward_mean)
    eval_std.append(ticket0_reward_std)
    
    # Add tickets 1-50 (which are ranks 1-50)
    for ticket in range(1, 51):
        search_mean.append(search_rewards[ticket])
        eval_mean.append(data['per_index_results'][str(ticket)]['reward_mean'])
        eval_std.append(data['per_index_results'][str(ticket)]['reward_std'])
    
    print(f"Ticket 0 - Search: {search_rewards[0]:.1f}, Eval: {ticket0_reward_mean:.1f} +/- {ticket0_reward_std:.1f}" if ticket0_reward_mean else "Ticket 0 not available")
    print(f"Eval rewards (tickets 1-5): {[data['per_index_results'][str(i)]['reward_mean'] for i in range(1, 6)]}")
    
    # Plot
    ax = axes[idx]
    ax.errorbar(search_mean, eval_mean, yerr=eval_std, fmt='o', capsize=3,
                alpha=0.6, markersize=4)
    ax.set_xlabel('Search Reward', fontsize=11)
    ax.set_ylabel('Eval Reward', fontsize=11)
    ax.set_title(f'{name}', fontsize=12)
    
    # Diagonal reference line
    all_vals = search_mean + eval_mean
    lim = [min(all_vals) - 10, max(all_vals) + 10]
    ax.plot(lim, lim, 'k--', alpha=0.3, linewidth=1)
    ax.grid(True, alpha=0.2)
    
    print(f"{name}: Search [{min(search_mean):.1f}, {max(search_mean):.1f}], "
          f"Eval [{min(eval_mean):.1f}, {max(eval_mean):.1f}]")

plt.tight_layout()
plt.savefig('regression_to_mean_rewards.png', dpi=300, bbox_inches='tight')
plt.show()

print("Saved plot to regression_to_mean_rewards.png")

# Export data to CSV
all_data = {'ticket': [], 'env_variation': [], 'search_reward': [], 
            'eval_reward': [], 'eval_reward_std': []}

for idx, (name, paths) in enumerate(configs.items()):
    # Load search results
    rewards_file = os.path.join(paths['search'], 'rewards.npy')
    rewards_array = np.load(rewards_file, allow_pickle=True)
    
    search_rewards = {}
    for rank in range(51):
        search_rewards[rank] = rewards_array[rank].mean()
    
    # Load eval results
    eval_path = paths['eval']
    subdirs = [d for d in os.listdir(eval_path) if 'eval_idx1-50' in d and os.path.isdir(os.path.join(eval_path, d))]
    eval_json = os.path.join(eval_path, subdirs[0], 'aggregate_results.json')
    
    with open(eval_json) as f:
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
    all_data['search_reward'].append(search_rewards[0])
    all_data['eval_reward'].append(ticket0_data['per_index_results']['0']['reward_mean'])
    all_data['eval_reward_std'].append(ticket0_data['per_index_results']['0']['reward_std'])
    
    # Add tickets 1-50
    for ticket in range(1, 51):
        all_data['ticket'].append(ticket)
        all_data['env_variation'].append(name)
        all_data['search_reward'].append(search_rewards[ticket])
        all_data['eval_reward'].append(data['per_index_results'][str(ticket)]['reward_mean'])
        all_data['eval_reward_std'].append(data['per_index_results'][str(ticket)]['reward_std'])

df = pd.DataFrame(all_data)
df.to_csv('regression_to_mean_rewards.csv', index=False)
print("Saved data to regression_to_mean_rewards.csv")
