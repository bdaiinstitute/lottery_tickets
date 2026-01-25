import json
import os

import numpy as np
from scipy.stats import spearmanr

def generate_noise_samples(dataset_train, cfg, num_noises, batch_size=64):
    """Generate noise samples in the shape of actions."""
    import torch
    from dppo.agent.pretrain.train_agent import batch_to_device
    
    dataset = dataset_train
    trajs = dataset.get_trajectory(0)
    inp_traj = trajs[0]
    if dataset.device == "cpu":
        inp_traj = batch_to_device(inp_traj, device=cfg.device)
    
    batch_actions = inp_traj.actions.unsqueeze(0).repeat(batch_size, 1, 1)
    action_shape = batch_actions.shape
    device = cfg.device
    
    noise_list = []
    total_noises = 0
    while total_noises < num_noises:
        noise = torch.randn(action_shape, device=device)
        noise_list.append(noise)
        total_noises += batch_size
    
    # Flatten to individual samples
    all_noises = []
    for noise_batch in noise_list:
        all_noises.extend(noise_batch.detach().cpu().numpy())
    
    return all_noises[:num_noises]


def load_episodes_from_path(episodes_dir):
    """Load all episode npy files from a directory."""
    # If episodes are stored under a subfolder (recorded_episodes), use it
    ep_dir = episodes_dir
    subdir = os.path.join(episodes_dir, "recorded_episodes")
    if os.path.isdir(subdir):
        ep_dir = subdir
    print(f"Loading episodes from {ep_dir}...")
    episode_files = sorted([f for f in os.listdir(ep_dir) if f.startswith('episode_') and f.endswith('.npy')])
    all_episodes = []
    for f in episode_files:
        episode_data = np.load(os.path.join(ep_dir, f), allow_pickle=True).item()
        idx = int(f.replace('episode_', '').replace('.npy', ''))
        all_episodes.append((idx, episode_data))
    print(f"Loaded {len(all_episodes)} episodes")
    return all_episodes


def get_succ_fail_eps(all_episodes):
    """Partition episodes by success flag."""
    def episode_success(ep):
        if ep.get('success'):
            return True
        infos = ep.get('infos') or ep.get('info')
        if isinstance(infos, (list, tuple)) and infos and isinstance(infos[-1], dict):
            return bool(infos[-1].get('is_success'))
        return False

    # Partition episodes by success flag
    success_eps = [(idx, ep) for idx, ep in all_episodes if episode_success(ep)]
    fail_eps = [(idx, ep) for idx, ep in all_episodes if not episode_success(ep)]
    print(f"Episode partition: success={len(success_eps)}, fail={len(fail_eps)}")
    return success_eps, fail_eps


def get_succ_fail_samples(all_episodes):
    """Extract samples from success and fail episodes."""
    success_eps, fail_eps = get_succ_fail_eps(all_episodes)
    
    def extract_samples(episodes):
        samples = []
        for ep_idx, ep_data in episodes:
            num_timesteps = ep_data['states'].shape[0]
            for t in range(num_timesteps):
                sample_data = {
                    'state': ep_data['states'][t],
                    'actions': ep_data['actions'][t],  # (action_chunks, action_dim)
                    'reward': ep_data['rewards'][t]
                }
                samples.append((ep_idx, t, sample_data))
        return samples

    success_samples_all = extract_samples(success_eps)
    fail_samples_all = extract_samples(fail_eps)
    print(f"Extracted samples: success={len(success_samples_all)}, fail={len(fail_samples_all)}")
    return success_samples_all, fail_samples_all


def filter_samples_based_on_gripper(all_episodes, max_samples=10000, span=3, use_span_filter=False):
    """Filter samples based on gripper state changes, and return (success_samples, fail_samples)."""
    success_eps, fail_eps = get_succ_fail_eps(all_episodes)

    def filter_gripper(ep_idx, ep_data):
        """Extract samples from episode and filter based on gripper state."""
        # First, extract all samples from this episode
        num_timesteps = ep_data['states'].shape[0]
        episode_samples = []
        for t in range(num_timesteps):
            sample_data = {
                'state': ep_data['states'][t],
                'actions': ep_data['actions'][t],  # (action_chunks, action_dim)
                'reward': ep_data['rewards'][t]
            }
            episode_samples.append((ep_idx, t, sample_data))
        
        filtered = []
        if use_span_filter:
            # Sliding window approach: find spans with both open and close
            for i in range(max(0, len(episode_samples) - span + 1)):
                window = episode_samples[i:i+span]
                has_open = any(np.any(np.isclose(s[2]['actions'][:, -1], -1.0, atol=1e-1)) for s in window)
                has_close = any(np.any(np.isclose(s[2]['actions'][:, -1], 1.0, atol=1e-1)) for s in window)
                if has_open and has_close:
                    for s in window:
                        if s not in filtered:
                            filtered.append(s)
        else:
            # Simple filter: keep samples with gripper close
            for sample in episode_samples:
                ep_idx, t, sample_data = sample
                gripper_values = sample_data['actions'][:, -1]
                has_close = np.any(np.isclose(gripper_values, 1.0, atol=1e-1))
                has_open = np.any(np.isclose(gripper_values, -1.0, atol=1e-1))
                if has_close:
                    filtered.append(sample)
        return filtered

    # Apply filter to each episode and flatten results
    success_filtered = []
    for ep_idx, ep_data in success_eps:
        success_filtered.extend(filter_gripper(ep_idx, ep_data))
        if len(success_filtered) >= max_samples:
            success_filtered = success_filtered[:max_samples]
            break
    
    fail_filtered = []
    for ep_idx, ep_data in fail_eps:
        fail_filtered.extend(filter_gripper(ep_idx, ep_data))
        if len(fail_filtered) >= max_samples:
            fail_filtered = fail_filtered[:max_samples]
            break
    
    print(f"Filtered samples: success={len(success_filtered)}, fail={len(fail_filtered)}")
    return success_filtered, fail_filtered


def save_noise_losses(noises, losses, save_dir):
    """Save noises and losses to files."""
    import json
    
    os.makedirs(save_dir, exist_ok=True)
    
    noises_array = np.array(noises)
    losses_array = np.array(losses)
    
    np.save(os.path.join(save_dir, "noises.npy"), noises_array)
    np.save(os.path.join(save_dir, "losses.npy"), losses_array)
    
    # Also save as JSON for easy inspection
    results = {
        "noises": noises_array.tolist(),
        "losses": losses_array.tolist(),
        "num_noises": len(losses)
    }
    with open(os.path.join(save_dir, "noise_losses.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved noises and losses to {save_dir}")


def load_noise_samples(noise_path, num_noises=None):
    """    
    Args:
        noise_path: Path to .npy file containing noise samples or directory containing noise_samples.npy
        num_noises: Number of top samples to use        
    Returns:
        List of noise samples
    """
    # If noise_path is a directory, look for noise_samples.npy inside it
    if os.path.isdir(noise_path):
        noise_path = os.path.join(noise_path, "noise_samples.npy")
    
    print(f"Loading noises from {noise_path}")
    eval_noises = np.load(noise_path)
    if num_noises is not None and num_noises < len(eval_noises):
        eval_noises = eval_noises[:num_noises]
    
    # Convert to list for compatibility
    eval_noises = list(eval_noises)
    print(f"Loaded {len(eval_noises)} noise samples from file")
    return eval_noises

def save_summary_statistics(results, num_noises, save_dir):
    """Save summary statistics in a format similar to lottery ticket results, using means.
    
    Args:
        results: Dict with keys as (diffusion_t, traj_t, traj_idx) tuples,
                 values as dicts containing 'losses' and 'dotprods' lists
        num_noises: Number of noise samples
        save_dir: Directory to save summary
    """
    # Calculate mean loss for each noise across all trajectories
    losses_per_noise = [[] for _ in range(num_noises)]
    dotprods_per_noise = [[] for _ in range(num_noises)]
    
    for key, value_dict in results.items():
        losses = value_dict['losses']
        dotprods = value_dict['dotprods']
        for noise_idx in range(num_noises):
            losses_per_noise[noise_idx].append(losses[noise_idx])
            dotprods_per_noise[noise_idx].append(dotprods[noise_idx])
    
    # Calculate mean for each noise
    mean_losses = [np.mean(losses) for losses in losses_per_noise]
    mean_dotprods = [np.mean(dotprods) for dotprods in dotprods_per_noise]
    
    # Get sorted indices (ascending for losses - lower is better)
    sorted_indices = np.argsort(mean_losses)
    
    # Calculate Spearman correlation between noise index and mean loss
    noise_indices = np.arange(num_noises)
    spearman_corr, spearman_p = spearmanr(noise_indices, mean_losses)
    
    # Create summary similar to lottery ticket format
    summary = {
        "num_noise_samples": num_noises,
        "num_trajectories": len(set(k[2] for k in results.keys())),
        "loss_mean_mean": float(np.mean(mean_losses)),
        "loss_mean_std": float(np.std(mean_losses)),
        "best_loss_mean": float(np.min(mean_losses)),
        "best_original_index": int(sorted_indices[0]),
        "worst_loss_mean": float(np.max(mean_losses)),
        "worst_original_index": int(sorted_indices[-1]),
        "mean_losses_sorted": [float(mean_losses[i]) for i in sorted_indices],
        "original_mean_losses": [float(x) for x in mean_losses],
        "mean_dotprods_sorted": [float(mean_dotprods[i]) for i in sorted_indices],
        "original_mean_dotprods": [float(x) for x in mean_dotprods],
        "sorted_indices": sorted_indices.tolist(),
        "spearman_correlation": float(spearman_corr),
        "spearman_p_value": float(spearman_p)
    }
    
    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    # Save mean losses and dotprods as numpy arrays
    np.save(os.path.join(save_dir, "mean_losses.npy"), np.array(mean_losses))
    np.save(os.path.join(save_dir, "mean_dotprods.npy"), np.array(mean_dotprods))
    
    # Create ranking file similar to lottery ticket format
    with open(os.path.join(save_dir, "ranking.txt"), "w") as f:
        for rank, idx in enumerate(sorted_indices):
            f.write(f"rank={rank}\tidx={idx}\tmean_loss={mean_losses[idx]:.6f}\tmean_dotprod={mean_dotprods[idx]:.6f}\n")
    
    print(f"Summary statistics saved to {save_dir}/summary.json")
    print(f"Ranking saved to {save_dir}/ranking.txt")
    print(f"Spearman correlation (noise_idx vs mean_loss): ρ={spearman_corr:.4f}, p={spearman_p:.4e}")


def sort_noises_by_mean_metric(noises, results, save_dir=None, metric='mse_losses'):
    """Sort noise samples by their mean metric value across samples."""
    num_noises = len(noises)
    
    # Collect metric values for each noise across all samples
    metrics_per_noise = [[] for _ in range(num_noises)]
    
    # results is a sample-wise loss/dotprod dict for all eval noises
    for key, value_dict in results.items():
        metric_values = value_dict[metric]
        for noise_idx, value in enumerate(metric_values):
            metrics_per_noise[noise_idx].append(value)
    
    # Calculate mean metric for each noise
    mean_metrics = [np.mean(values) for values in metrics_per_noise]
    
    # Get sorted indices (ascending order)
    sorted_indices = np.argsort(mean_metrics)
    
    # Reorder noises
    sorted_noises = [noises[i] for i in sorted_indices]
    
    # Reorder results to match sorted noises
    sorted_results = {}
    for key, value_dict in results.items():
        sorted_dict = {}
        for metric_name, metric_values in value_dict.items():
            sorted_dict[metric_name] = [metric_values[i] for i in sorted_indices]
        sorted_results[key] = sorted_dict
    
    # Save sorted noise order if save_dir provided
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        sorted_means = [mean_metrics[i] for i in sorted_indices]
        
        with open(os.path.join(save_dir, "noise_order.txt"), "w") as f:
            f.write(f"# Noises sorted by mean {metric} across trajectories (ascending)\n")
            f.write(f"# sorted_idx\toriginal_idx\tmean_{metric}\n")
            for sorted_idx, orig_idx in enumerate(sorted_indices):
                f.write(f"{sorted_idx}\t{orig_idx}\t{sorted_means[sorted_idx]:.6f}\n")
        
        print(f"Noise order saved to {os.path.join(save_dir, 'noise_order.txt')}")
    
    return sorted_noises, sorted_results, sorted_indices


def save_correlation_summary_statistics(results_success, num_noises, save_dir):
    """Save summary statistics for MSE correlation for successful ep samples."""
    # Extract success MSE data
    success_keys = sorted(results_success.keys())
    success_mse = np.array([results_success[k]['mse_losses'] for k in success_keys])
    
    # Calculate statistics for success
    noise_indices = np.arange(num_noises)
    stats = ['mean', 'max', 'min']
    mse_success = {stat: getattr(success_mse, stat)(axis=0) for stat in stats}
    corr_success = {stat: spearmanr(noise_indices, mse_success[stat]) for stat in stats}
    sorted_indices = {stat: np.argsort(mse_success[stat]) for stat in stats}
    
    # Build summary dictionary
    summary = {
        "num_noise_samples": num_noises,
        "num_success_samples": len(success_keys),
    }
    
    # Add success statistics
    for stat in stats:
        summary[f"{stat}_mse_success_avg"] = float(np.mean(mse_success[stat]))
        summary[f"spearman_correlation_{stat}_success"] = float(corr_success[stat][0])
        summary[f"spearman_p_value_{stat}_success"] = float(corr_success[stat][1])
        summary[f"{stat}_mse_success"] = mse_success[stat].tolist()
        summary[f"sorted_indices_by_{stat}_success"] = sorted_indices[stat].tolist()
    
    # Save summary JSON
    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    # Create ranking file
    with open(os.path.join(save_dir, "ranking.txt"), "w") as f:
        for stat in stats:
            f.write(f"# Ranking by {stat.title()} MSE Success\n")
            f.write(f"rank\tidx\t{stat}_succ\n")
            for rank, idx in enumerate(sorted_indices[stat]):
                f.write(f"{rank}\t{idx}\t{mse_success[stat][idx]:.6f}\n")
            f.write("\n")
    
    # Print results
    print(f"Summary statistics saved to {save_dir}/summary.json")
    print(f"Ranking saved to {save_dir}/ranking.txt")
    print(f"\nCorrelations (noise_idx vs MSE Success):")
    print("=" * 80)
    print(f"{'Metric':<30} {'Spearman ρ':>15} {'p-value':>15}")
    print("-" * 80)
    
    for stat in stats:
        print(f"{stat.title() + ' MSE Success':<30} {corr_success[stat][0]:>15.4f} {corr_success[stat][1]:>15.4e}")
    
    print("=" * 80)


def save_gte_correlation_summary_statistics(results_success, num_noises, save_dir):
    """Save summary statistics for GTE correlation for successful ep samples.
    
    This function handles the three GTE metrics:
    - ld_gt_mse_losses: Low-dimensional denoised vs ground truth
    - hd_gt_mse_losses: High-dimensional denoised vs ground truth  
    - gte_losses: Global Truncation Error (HD - LD denoised)
    """
    # Extract success data for all three metrics
    success_keys = sorted(results_success.keys())
    success_ld_gt = np.array([results_success[k]['ld_gt_mse_losses'] for k in success_keys])
    success_hd_gt = np.array([results_success[k]['hd_gt_mse_losses'] for k in success_keys])
    success_gte = np.array([results_success[k]['gte_losses'] for k in success_keys])
    
    # Calculate statistics for success
    noise_indices = np.arange(num_noises)
    stats = ['mean', 'max', 'min']
    
    ld_gt_success = {stat: getattr(success_ld_gt, stat)(axis=0) for stat in stats}
    hd_gt_success = {stat: getattr(success_hd_gt, stat)(axis=0) for stat in stats}
    gte_success = {stat: getattr(success_gte, stat)(axis=0) for stat in stats}
    
    corr_ld_gt_success = {stat: spearmanr(noise_indices, ld_gt_success[stat]) for stat in stats}
    corr_hd_gt_success = {stat: spearmanr(noise_indices, hd_gt_success[stat]) for stat in stats}
    corr_gte_success = {stat: spearmanr(noise_indices, gte_success[stat]) for stat in stats}
    
    sorted_indices_gte = {stat: np.argsort(gte_success[stat]) for stat in stats}
    
    # Build summary dictionary
    summary = {
        "num_noise_samples": num_noises,
        "num_success_samples": len(success_keys),
    }
    
    # Add success statistics for all three metrics
    for stat in stats:
        # LD-GT
        summary[f"{stat}_ld_gt_success_avg"] = float(np.mean(ld_gt_success[stat]))
        summary[f"spearman_correlation_{stat}_ld_gt_success"] = float(corr_ld_gt_success[stat][0])
        summary[f"spearman_p_value_{stat}_ld_gt_success"] = float(corr_ld_gt_success[stat][1])
        summary[f"{stat}_ld_gt_success"] = ld_gt_success[stat].tolist()
        
        # HD-GT
        summary[f"{stat}_hd_gt_success_avg"] = float(np.mean(hd_gt_success[stat]))
        summary[f"spearman_correlation_{stat}_hd_gt_success"] = float(corr_hd_gt_success[stat][0])
        summary[f"spearman_p_value_{stat}_hd_gt_success"] = float(corr_hd_gt_success[stat][1])
        summary[f"{stat}_hd_gt_success"] = hd_gt_success[stat].tolist()
        
        # GTE
        summary[f"{stat}_gte_success_avg"] = float(np.mean(gte_success[stat]))
        summary[f"spearman_correlation_{stat}_gte_success"] = float(corr_gte_success[stat][0])
        summary[f"spearman_p_value_{stat}_gte_success"] = float(corr_gte_success[stat][1])
        summary[f"{stat}_gte_success"] = gte_success[stat].tolist()
        summary[f"sorted_indices_by_{stat}_gte_success"] = sorted_indices_gte[stat].tolist()
    
    # Save summary JSON
    with open(os.path.join(save_dir, "gte_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    # Create ranking file
    with open(os.path.join(save_dir, "gte_ranking.txt"), "w") as f:
        for stat in stats:
            f.write(f"# Ranking by {stat.title()} GTE Success\n")
            f.write("rank\tidx\t{}_gte_succ\n".format(stat))
            for rank, idx in enumerate(sorted_indices_gte[stat]):
                f.write(f"{rank}\t{idx}\t{gte_success[stat][idx]:.6f}\n")
            f.write("\n")
    
    print(f"GTE ranking saved to {save_dir}/gte_ranking.txt")
    print(f"\nGTE Correlations (noise_idx vs GTE Success):")
    print("=" * 80)
    print(f"{'Metric':<30} {'Spearman ρ':>15} {'p-value':>15}")
    print("-" * 80)
    
    for stat in stats:
        print(f"{stat.title() + ' GTE Success':<30} {corr_gte_success[stat][0]:>15.4f} {corr_gte_success[stat][1]:>15.4e}")
    
    print("=" * 80)


def save_approx_loss_statistics(results_success, num_noises, save_dir):
    """Save summary statistics for approximation loss metrics."""
    success_keys = sorted(results_success.keys())
    ld_clip = np.array([results_success[k]['ld_clip_freq'] for k in success_keys])
    ld_recon = np.array([results_success[k]['ld_recon_loss'] for k in success_keys])
    ld_clipped_recon = np.array([results_success[k]['ld_clipped_recon_loss'] for k in success_keys])
    hd_clip = np.array([results_success[k]['hd_clip_freq'] for k in success_keys])
    hd_recon = np.array([results_success[k]['hd_recon_loss'] for k in success_keys])
    hd_clipped_recon = np.array([results_success[k]['hd_clipped_recon_loss'] for k in success_keys])
    
    noise_indices = np.arange(num_noises)
    stats = ['mean', 'max', 'min']
    
    ld_clip_stats = {stat: getattr(ld_clip, stat)(axis=0) for stat in stats}
    ld_recon_stats = {stat: getattr(ld_recon, stat)(axis=0) for stat in stats}
    ld_clipped_recon_stats = {stat: getattr(ld_clipped_recon, stat)(axis=0) for stat in stats}
    hd_clip_stats = {stat: getattr(hd_clip, stat)(axis=0) for stat in stats}
    hd_recon_stats = {stat: getattr(hd_recon, stat)(axis=0) for stat in stats}
    hd_clipped_recon_stats = {stat: getattr(hd_clipped_recon, stat)(axis=0) for stat in stats}
    
    corr_ld_clip = {stat: spearmanr(noise_indices, ld_clip_stats[stat]) for stat in stats}
    corr_ld_recon = {stat: spearmanr(noise_indices, ld_recon_stats[stat]) for stat in stats}
    corr_ld_clipped_recon = {stat: spearmanr(noise_indices, ld_clipped_recon_stats[stat]) for stat in stats}
    corr_hd_clip = {stat: spearmanr(noise_indices, hd_clip_stats[stat]) for stat in stats}
    corr_hd_recon = {stat: spearmanr(noise_indices, hd_recon_stats[stat]) for stat in stats}
    corr_hd_clipped_recon = {stat: spearmanr(noise_indices, hd_clipped_recon_stats[stat]) for stat in stats}
    
    summary = {
        "num_noise_samples": num_noises,
        "num_success_samples": len(success_keys),
    }
    
    for stat in stats:
        summary[f"{stat}_ld_clip_freq_avg"] = float(np.mean(ld_clip_stats[stat]))
        summary[f"spearman_corr_{stat}_ld_clip_freq"] = float(corr_ld_clip[stat][0])
        summary[f"spearman_p_{stat}_ld_clip_freq"] = float(corr_ld_clip[stat][1])
        
        summary[f"{stat}_ld_recon_loss_avg"] = float(np.mean(ld_recon_stats[stat]))
        summary[f"spearman_corr_{stat}_ld_recon_loss"] = float(corr_ld_recon[stat][0])
        summary[f"spearman_p_{stat}_ld_recon_loss"] = float(corr_ld_recon[stat][1])
        
        summary[f"{stat}_ld_clipped_recon_loss_avg"] = float(np.mean(ld_clipped_recon_stats[stat]))
        summary[f"spearman_corr_{stat}_ld_clipped_recon_loss"] = float(corr_ld_clipped_recon[stat][0])
        summary[f"spearman_p_{stat}_ld_clipped_recon_loss"] = float(corr_ld_clipped_recon[stat][1])
        
        summary[f"{stat}_hd_clip_freq_avg"] = float(np.mean(hd_clip_stats[stat]))
        summary[f"spearman_corr_{stat}_hd_clip_freq"] = float(corr_hd_clip[stat][0])
        summary[f"spearman_p_{stat}_hd_clip_freq"] = float(corr_hd_clip[stat][1])
        
        summary[f"{stat}_hd_recon_loss_avg"] = float(np.mean(hd_recon_stats[stat]))
        summary[f"spearman_corr_{stat}_hd_recon_loss"] = float(corr_hd_recon[stat][0])
        summary[f"spearman_p_{stat}_hd_recon_loss"] = float(corr_hd_recon[stat][1])
        
        summary[f"{stat}_hd_clipped_recon_loss_avg"] = float(np.mean(hd_clipped_recon_stats[stat]))
        summary[f"spearman_corr_{stat}_hd_clipped_recon_loss"] = float(corr_hd_clipped_recon[stat][0])
        summary[f"spearman_p_{stat}_hd_clipped_recon_loss"] = float(corr_hd_clipped_recon[stat][1])
    
    with open(os.path.join(save_dir, "approx_loss_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    sorted_idx = np.argsort(ld_recon_stats['mean'])
    with open(os.path.join(save_dir, "approx_loss_ranking.txt"), "w") as f:
        f.write("Noise Ranking by Mean LD Recon Loss (Pre-Clamp):\n")
        f.write("Rank\tNoise_Idx\tLD_Recon\tLD_Clipped_Recon\tLD_Clip_Freq\tHD_Recon\tHD_Clipped_Recon\tHD_Clip_Freq\n")
        for rank, idx in enumerate(sorted_idx):
            f.write(f"{rank}\t{idx}\t{ld_recon_stats['mean'][idx]:.6f}\t{ld_clipped_recon_stats['mean'][idx]:.6f}\t{ld_clip_stats['mean'][idx]:.2f}\t{hd_recon_stats['mean'][idx]:.6f}\t{hd_clipped_recon_stats['mean'][idx]:.6f}\t{hd_clip_stats['mean'][idx]:.2f}\n")
    
    print(f"Approximation loss statistics saved to {save_dir}/approx_loss_summary.json")
    print(f"Approximation loss ranking saved to {save_dir}/approx_loss_ranking.txt")
