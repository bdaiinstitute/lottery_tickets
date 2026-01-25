"""
Plotting used for approx error analysis. 
Contains functions for all-steps DDIM analysis of reconstruction loss and clip frequency.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def plot_all_steps_analysis(all_steps_data, save_dir, exp_name='', task_name='', 
                            low_ddim=8, high_ddim=20, num_noises=448, plot_clip_freq=False):
    """
    Plot per-step reconstruction loss and clip frequency for all DDIM steps.
    Creates separate plots for LD and HD models.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    title_prefix = f"{task_name.upper()} - {exp_name}" if task_name and exp_name else (task_name.upper() if task_name else exp_name or "Approx Loss")
    
    # Process LD model
    ld_all_steps = all_steps_data['ld_all_steps']
    if ld_all_steps:
        plot_model_all_steps(ld_all_steps, save_dir, title_prefix, 'LD', low_ddim, num_noises, plot_clip_freq)
    
    # Process HD model
    hd_all_steps = all_steps_data['hd_all_steps']
    if hd_all_steps:
        plot_model_all_steps(hd_all_steps, save_dir, title_prefix, 'HD', high_ddim, num_noises, plot_clip_freq)


def plot_model_all_steps(steps_data, save_dir, title_prefix, model_name, ddim_steps, num_noises, plot_clip_freq):
    """
    Plot all-steps analysis for a single model (LD or HD).
    
    Creates:
    1. Per-step recon loss (pre-clamp) curves (one line per noise)
    2. Per-step clipped recon loss (post-clamp) curves
    3. Per-step sum and max of both recon losses across noises
    4. Optional: Per-step clip freq curves and aggregations
    """
    # Sort timesteps
    timesteps = sorted(steps_data.keys())
    num_steps = len(timesteps)
    
    if num_steps == 0:
        print(f"No steps data for {model_name} model")
        return
    
    # Prepare data matrices: [num_noises x num_steps]
    # Data is currently [num_samples][num_noises] per timestep, need to aggregate across samples
    recon_loss_matrix = []
    clipped_recon_loss_matrix = []
    clip_freq_matrix = []
    
    for t in timesteps:
        # Each entry is a list of samples, where each sample has [num_noises] values
        # Convert to [num_samples, num_noises] array and take mean across samples
        recon_loss_per_sample = np.array(steps_data[t]['recon_loss'])  # Shape: [num_samples, num_noises]
        clipped_recon_per_sample = np.array(steps_data[t]['clipped_recon_loss'])
        clip_freq_per_sample = np.array(steps_data[t]['clip_freq'])
        
        # Mean across samples for each noise: [num_noises]
        recon_loss_matrix.append(recon_loss_per_sample.mean(axis=0))
        clipped_recon_loss_matrix.append(clipped_recon_per_sample.mean(axis=0))
        clip_freq_matrix.append(clip_freq_per_sample.mean(axis=0))
    
    # Transpose to get [num_noises, num_steps]
    recon_loss_matrix = np.array(recon_loss_matrix).T  # Shape: [num_noises, num_steps]
    clipped_recon_loss_matrix = np.array(clipped_recon_loss_matrix).T
    clip_freq_matrix = np.array(clip_freq_matrix).T    # Shape: [num_noises, num_steps]
    
    # Get ddim_idx for x-axis labels
    ddim_indices = [steps_data[t]['ddim_idx'] for t in timesteps]
    
    # === Plot 1: Per-noise recon loss (pre-clamp) across all DDIM steps ===
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Use colormap for different timesteps
    colors = cm.viridis(np.linspace(0, 1, num_steps))
    
    noise_indices = np.arange(num_noises)
    for step_idx, (ddim_idx, t) in enumerate(zip(ddim_indices, timesteps)):
        ax.scatter(noise_indices, recon_loss_matrix[:, step_idx], 
                  alpha=0.6, s=20, color=colors[step_idx], 
                  label=f'DDIM idx={ddim_idx}, t={t}')
    
    ax.set_xlabel('Noise Index', fontsize=12)
    ax.set_ylabel('Recon Loss (MSE, Pre-Clamp)', fontsize=12)
    ax.set_title(f'{title_prefix} - {model_name} (DDIM={ddim_steps}): Recon Loss (Pre-Clamp) Per Noise', 
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8, ncol=2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name.lower()}_per_noise_recon_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # === Plot 2: Per-noise clipped recon loss (post-clamp) across all DDIM steps ===
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    for step_idx, (ddim_idx, t) in enumerate(zip(ddim_indices, timesteps)):
        ax.scatter(noise_indices, clipped_recon_loss_matrix[:, step_idx], 
                  alpha=0.6, s=20, color=colors[step_idx], 
                  label=f'DDIM idx={ddim_idx}, t={t}')
    
    ax.set_xlabel('Noise Index', fontsize=12)
    ax.set_ylabel('Clipped Recon Loss (MSE, Post-Clamp)', fontsize=12)
    ax.set_title(f'{title_prefix} - {model_name} (DDIM={ddim_steps}): Clipped Recon Loss (Post-Clamp) Per Noise', 
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8, ncol=2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name.lower()}_per_noise_clipped_recon_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # === Plot 3: Sum and Max recon loss (pre-clamp) across DDIM steps per noise ===
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Sum across DDIM steps for each noise
    sum_recon_loss = recon_loss_matrix.sum(axis=1)  # Sum along steps dimension
    corr_sum = np.corrcoef(noise_indices, sum_recon_loss)[0, 1]
    axes[0].scatter(noise_indices, sum_recon_loss, alpha=0.6, s=30, color='blue')
    axes[0].set_xlabel('Noise Index', fontsize=12)
    axes[0].set_ylabel('Sum Recon Loss (Pre-Clamp)', fontsize=12)
    axes[0].set_title(f'{model_name} (DDIM={ddim_steps}): Sum Recon Loss Across Steps (r={corr_sum:.3f})', 
                     fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Max across DDIM steps for each noise
    max_recon_loss = recon_loss_matrix.max(axis=1)  # Max along steps dimension
    corr_max = np.corrcoef(noise_indices, max_recon_loss)[0, 1]
    axes[1].scatter(noise_indices, max_recon_loss, alpha=0.6, s=30, color='red')
    axes[1].set_xlabel('Noise Index', fontsize=12)
    axes[1].set_ylabel('Max Recon Loss (Pre-Clamp)', fontsize=12)
    axes[1].set_title(f'{model_name} (DDIM={ddim_steps}): Max Recon Loss Across Steps (r={corr_max:.3f})', 
                     fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name.lower()}_sum_max_recon_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # === Plot 4: Sum and Max clipped recon loss (post-clamp) across DDIM steps per noise ===
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Sum across DDIM steps for each noise
    sum_clipped_recon_loss = clipped_recon_loss_matrix.sum(axis=1)
    corr_sum_clipped = np.corrcoef(noise_indices, sum_clipped_recon_loss)[0, 1]
    axes[0].scatter(noise_indices, sum_clipped_recon_loss, alpha=0.6, s=30, color='darkblue')
    axes[0].set_xlabel('Noise Index', fontsize=12)
    axes[0].set_ylabel('Sum Clipped Recon Loss (Post-Clamp)', fontsize=12)
    axes[0].set_title(f'{model_name} (DDIM={ddim_steps}): Sum Clipped Recon Loss Across Steps (r={corr_sum_clipped:.3f})', 
                     fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Max across DDIM steps for each noise
    max_clipped_recon_loss = clipped_recon_loss_matrix.max(axis=1)
    corr_max_clipped = np.corrcoef(noise_indices, max_clipped_recon_loss)[0, 1]
    axes[1].scatter(noise_indices, max_clipped_recon_loss, alpha=0.6, s=30, color='darkred')
    axes[1].set_xlabel('Noise Index', fontsize=12)
    axes[1].set_ylabel('Max Clipped Recon Loss (Post-Clamp)', fontsize=12)
    axes[1].set_title(f'{model_name} (DDIM={ddim_steps}): Max Clipped Recon Loss Across Steps (r={corr_max_clipped:.3f})', 
                     fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name.lower()}_sum_max_clipped_recon_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # === Plot 5 & 6: Clip frequency (if requested) ===
    if plot_clip_freq:
        # Per-noise clip freq
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        for step_idx, (ddim_idx, t) in enumerate(zip(ddim_indices, timesteps)):
            ax.scatter(noise_indices, clip_freq_matrix[:, step_idx], 
                      alpha=0.6, s=20, color=colors[step_idx], 
                      label=f'DDIM idx={ddim_idx}, t={t}')
        
        ax.set_xlabel('Noise Index', fontsize=12)
        ax.set_ylabel('Clip Frequency', fontsize=12)
        ax.set_title(f'{title_prefix} - {model_name} (DDIM={ddim_steps}): Clip Freq Per Noise', 
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8, ncol=2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{model_name.lower()}_per_noise_clip_freq.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Sum and Max clip freq across DDIM steps per noise
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        sum_clip_freq = clip_freq_matrix.sum(axis=1)
        corr_sum_clip = np.corrcoef(noise_indices, sum_clip_freq)[0, 1]
        axes[0].scatter(noise_indices, sum_clip_freq, alpha=0.6, s=30, color='green')
        axes[0].set_xlabel('Noise Index', fontsize=12)
        axes[0].set_ylabel('Sum Clip Frequency', fontsize=12)
        axes[0].set_title(f'{model_name} (DDIM={ddim_steps}): Sum Clip Freq Across Steps (r={corr_sum_clip:.3f})', 
                         fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        max_clip_freq = clip_freq_matrix.max(axis=1)
        corr_max_clip = np.corrcoef(noise_indices, max_clip_freq)[0, 1]
        axes[1].scatter(noise_indices, max_clip_freq, alpha=0.6, s=30, color='purple')
        axes[1].set_xlabel('Noise Index', fontsize=12)
        axes[1].set_ylabel('Max Clip Frequency', fontsize=12)
        axes[1].set_title(f'{model_name} (DDIM={ddim_steps}): Max Clip Freq Across Steps (r={corr_max_clip:.3f})', 
                         fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{model_name.lower()}_sum_max_clip_freq.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"All-steps plots saved for {model_name} model to {save_dir}")
