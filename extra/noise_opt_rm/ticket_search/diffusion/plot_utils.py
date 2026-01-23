import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Import all-steps plotting
from noise_opt_rm.ticket_search.diffusion.plot_all_steps import plot_all_steps_analysis


def plot_losses(losses, save_path=None):
    """Plot the evaluated losses."""
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(losses)), losses, alpha=0.6)
    plt.xlabel('Noise Sample Index')
    plt.ylabel('Loss Value')
    plt.title('Diffusion Loss Distribution')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.close()


def plot_dotprods(dotprods, save_path=None):
    """Plot the evaluated dot products."""
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(dotprods)), dotprods, alpha=0.6, color='green')
    plt.xlabel('Noise Sample Index')
    plt.ylabel('Dot Product Value')
    plt.title('Dot Product Distribution')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.close()


def plot_noise_magnitudes(magnitudes, save_path=None):
    """Plot the L2 norms (magnitudes) of noise samples."""
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(magnitudes)), magnitudes, alpha=0.6, color='purple')
    plt.xlabel('Noise Sample Index')
    plt.ylabel('Noise Magnitude (L2 Norm)')
    plt.title('Noise Magnitude Distribution')
    plt.grid(True, alpha=0.3)
    
    # Add mean line
    mean_mag = np.mean(magnitudes)
    plt.axhline(y=mean_mag, color='r', linestyle='--', linewidth=2, alpha=0.7, label=f'Mean: {mean_mag:.4f}')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.close()


def plot_correlation_results(results_success, save_dir, exp_name='', task_name=''):
    """Create plots for MSE and cosine similarity correlation analysis."""
    import os
    from scipy.stats import spearmanr
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract success data
    success_keys = sorted(results_success.keys())
    num_noises = len(results_success[success_keys[0]]['mse_losses'])
    success_mse = np.array([results_success[k]['mse_losses'] for k in success_keys])
    success_cosine = np.array([results_success[k]['cosine_similarities'] for k in success_keys])
    
    stats = ['mean', 'max', 'min']
    mse_success = {stat: getattr(success_mse, stat)(axis=0) for stat in stats}
    cosine_success = {stat: getattr(success_cosine, stat)(axis=0) for stat in stats}
    noise_indices = np.arange(num_noises)
    
    # Build main title
    title_prefix = f"{task_name.upper()} - {exp_name}" if task_name and exp_name else (task_name.upper() if task_name else exp_name)
    
    # Plot: Success (MSE + Cosine)
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    if title_prefix:
        fig.suptitle(f"{title_prefix} - Success Samples", fontsize=14, fontweight='bold', y=0.995)
    for i, stat in enumerate(stats):
        # MSE plot
        axes[i, 0].scatter(noise_indices, mse_success[stat], alpha=0.6, s=30, color='green')
        axes[i, 0].set_ylabel(f'{stat.title()} MSE', fontsize=11)
        # Compute Spearman correlation between noise index and MSE
        corr_mse, _ = spearmanr(noise_indices, mse_success[stat])
        axes[i, 0].set_title(f'{stat.title()} MSE Success (ρ={corr_mse:.3f})', fontsize=12, fontweight='bold')
        axes[i, 0].grid(True, alpha=0.3)
        
        # Cosine plot
        axes[i, 1].scatter(noise_indices, cosine_success[stat], alpha=0.6, s=30, color='blue')
        axes[i, 1].set_ylabel(f'{stat.title()} Cosine Sim', fontsize=11)
        # Compute Spearman correlation between noise index and cosine similarity
        corr_cosine, _ = spearmanr(noise_indices, cosine_success[stat])
        axes[i, 1].set_title(f'{stat.title()} Cosine Similarity Success (ρ={corr_cosine:.3f})', fontsize=12, fontweight='bold')
        axes[i, 1].grid(True, alpha=0.3)
        
        if i == 2:
            axes[i, 0].set_xlabel('Noise Index', fontsize=11)
            axes[i, 1].set_xlabel('Noise Index', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'correlation_success.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_gte_correlation_results(results_success, save_dir, exp_name='', task_name='', low_ddim=8, high_ddim=20):
    """Create plots for GTE (Global Truncation Error): ld_gt, hd_gt, and gte MSE correlation analysis."""
    import os
    from scipy.stats import spearmanr
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract success data for three metrics
    success_keys = sorted(results_success.keys())
    num_noises = len(results_success[success_keys[0]]['ld_gt_mse_losses'])
    
    success_ld_gt = np.array([results_success[k]['ld_gt_mse_losses'] for k in success_keys])
    success_hd_gt = np.array([results_success[k]['hd_gt_mse_losses'] for k in success_keys])
    success_gte = np.array([results_success[k]['gte_losses'] for k in success_keys])
    
    stats = ['mean', 'max', 'min']
    ld_gt_success = {stat: getattr(success_ld_gt, stat)(axis=0) for stat in stats}
    hd_gt_success = {stat: getattr(success_hd_gt, stat)(axis=0) for stat in stats}
    gte_success = {stat: getattr(success_gte, stat)(axis=0) for stat in stats}
    noise_indices = np.arange(num_noises)
    
    # Build main title
    if task_name and exp_name:
        title_prefix = f"{task_name.upper()} - {exp_name}"
    elif task_name:
        title_prefix = f"{task_name.upper()}"
    elif exp_name:
        title_prefix = exp_name
    else:
        title_prefix = "GTE Analysis"
    
    # Plot 1: Mean Success (LD-GT + HD-GT + GTE)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{title_prefix} - Mean Success Samples", fontsize=14, fontweight='bold', y=1.02)
    
    stat = 'mean'
    # LD-GT MSE plot
    axes[0].scatter(noise_indices, ld_gt_success[stat], alpha=0.6, s=30, color='green')
    axes[0].set_ylabel(f'Mean MSE', fontsize=11)
    axes[0].set_xlabel('Noise Index', fontsize=11)
    corr_ld_gt, _ = spearmanr(noise_indices, ld_gt_success[stat])
    axes[0].set_title(f'LD-GT MSE (DDIM={low_ddim}, ρ={corr_ld_gt:.3f})', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # HD-GT MSE plot
    axes[1].scatter(noise_indices, hd_gt_success[stat], alpha=0.6, s=30, color='blue')
    axes[1].set_ylabel(f'Mean MSE', fontsize=11)
    axes[1].set_xlabel('Noise Index', fontsize=11)
    corr_hd_gt, _ = spearmanr(noise_indices, hd_gt_success[stat])
    axes[1].set_title(f'HD-GT MSE (DDIM={high_ddim}, ρ={corr_hd_gt:.3f})', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # GTE plot
    axes[2].scatter(noise_indices, gte_success[stat], alpha=0.6, s=30, color='purple')
    axes[2].set_ylabel(f'Mean GTE', fontsize=11)
    axes[2].set_xlabel('Noise Index', fontsize=11)
    corr_gte, _ = spearmanr(noise_indices, gte_success[stat])
    axes[2].set_title(f'GTE (LD={low_ddim}, HD={high_ddim}, ρ={corr_gte:.3f})', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'gte_correlation_mean_success.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Max and Min Success (LD-GT + HD-GT + GTE)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"{title_prefix} - Max/Min Success Samples", fontsize=14, fontweight='bold', y=0.995)
    
    for i, stat in enumerate(['max', 'min']):
        # LD-GT MSE plot
        axes[i, 0].scatter(noise_indices, ld_gt_success[stat], alpha=0.6, s=30, color='green')
        axes[i, 0].set_ylabel(f'{stat.title()} MSE', fontsize=11)
        corr_ld_gt, _ = spearmanr(noise_indices, ld_gt_success[stat])
        axes[i, 0].set_title(f'{stat.title()} LD-GT MSE (DDIM={low_ddim}, ρ={corr_ld_gt:.3f})', fontsize=12, fontweight='bold')
        axes[i, 0].grid(True, alpha=0.3)
        
        # HD-GT MSE plot
        axes[i, 1].scatter(noise_indices, hd_gt_success[stat], alpha=0.6, s=30, color='blue')
        axes[i, 1].set_ylabel(f'{stat.title()} MSE', fontsize=11)
        corr_hd_gt, _ = spearmanr(noise_indices, hd_gt_success[stat])
        axes[i, 1].set_title(f'{stat.title()} HD-GT MSE (DDIM={high_ddim}, ρ={corr_hd_gt:.3f})', fontsize=12, fontweight='bold')
        axes[i, 1].grid(True, alpha=0.3)
        
        # GTE plot
        axes[i, 2].scatter(noise_indices, gte_success[stat], alpha=0.6, s=30, color='purple')
        axes[i, 2].set_ylabel(f'{stat.title()} GTE', fontsize=11)
        corr_gte, _ = spearmanr(noise_indices, gte_success[stat])
        axes[i, 2].set_title(f'{stat.title()} GTE (LD={low_ddim}, HD={high_ddim}, ρ={corr_gte:.3f})', fontsize=12, fontweight='bold')
        axes[i, 2].grid(True, alpha=0.3)
        
        if i == 1:
            axes[i, 0].set_xlabel('Noise Index', fontsize=11)
            axes[i, 1].set_xlabel('Noise Index', fontsize=11)
            axes[i, 2].set_xlabel('Noise Index', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'gte_correlation_maxmin_success.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_approx_loss_results(results_success, save_dir, exp_name='', task_name='', low_ddim=8, high_ddim=20, 
                            ddim_idx=0, ld_timestep=None, hd_timestep=None, plot_clip_freq=False):
    """Plot approximation loss metrics: LD and HD variants of clip_freq, recon_loss, and clipped_recon_loss."""
    import os
    from scipy.stats import spearmanr
    os.makedirs(save_dir, exist_ok=True)
    
    success_keys = sorted(results_success.keys())
    num_noises = len(results_success[success_keys[0]]['ld_clip_freq'])
    
    ld_clip = np.array([results_success[k]['ld_clip_freq'] for k in success_keys])
    ld_recon = np.array([results_success[k]['ld_recon_loss'] for k in success_keys])
    ld_clipped_recon = np.array([results_success[k]['ld_clipped_recon_loss'] for k in success_keys])
    hd_clip = np.array([results_success[k]['hd_clip_freq'] for k in success_keys])
    hd_recon = np.array([results_success[k]['hd_recon_loss'] for k in success_keys])
    hd_clipped_recon = np.array([results_success[k]['hd_clipped_recon_loss'] for k in success_keys])
    
    stats = ['mean', 'max', 'min']
    ld_clip_stats = {stat: getattr(ld_clip, stat)(axis=0) for stat in stats}
    ld_recon_stats = {stat: getattr(ld_recon, stat)(axis=0) for stat in stats}
    ld_clipped_recon_stats = {stat: getattr(ld_clipped_recon, stat)(axis=0) for stat in stats}
    hd_clip_stats = {stat: getattr(hd_clip, stat)(axis=0) for stat in stats}
    hd_recon_stats = {stat: getattr(hd_recon, stat)(axis=0) for stat in stats}
    hd_clipped_recon_stats = {stat: getattr(hd_clipped_recon, stat)(axis=0) for stat in stats}
    noise_indices = np.arange(num_noises)
    
    title_prefix = f"{task_name.upper()} - {exp_name}" if task_name and exp_name else (task_name.upper() if task_name else exp_name or "Approx Loss")
    
    # Create title with timestep info
    title_ddim_info = f"DDIM idx={ddim_idx}"
    if ld_timestep is not None or hd_timestep is not None:
        if ld_timestep == hd_timestep and ld_timestep is not None:
            title_ddim_info = f"DDIM idx={ddim_idx}, timestep={ld_timestep}"
        else:
            title_ddim_info = f"DDIM idx={ddim_idx}"
            if ld_timestep is not None:
                title_ddim_info += f", LD t={ld_timestep}"
            if hd_timestep is not None:
                title_ddim_info += f", HD t={hd_timestep}"
    
    stat = 'mean'
    
    # Plot recon loss (before clipping)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{title_prefix} - Mean Success Recon Loss (Pre-Clamp) ({title_ddim_info})", fontsize=14, fontweight='bold', y=1.02)
    
    # LD subplot with timestep info
    corr_ld, _ = spearmanr(noise_indices, ld_recon_stats[stat])
    ld_title = f'LD Recon Loss (DDIM={low_ddim}, ρ={corr_ld:.3f})'
    if ld_timestep is not None:
        ld_title += f'\\nt={ld_timestep}'
    axes[0].scatter(noise_indices, ld_recon_stats[stat], alpha=0.6, s=30, color='purple')
    axes[0].set_ylabel('Mean Recon Loss', fontsize=11)
    axes[0].set_xlabel('Noise Index', fontsize=11)
    axes[0].set_title(ld_title, fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # HD subplot with timestep info
    corr_hd, _ = spearmanr(noise_indices, hd_recon_stats[stat])
    hd_title = f'HD Recon Loss (DDIM={high_ddim}, ρ={corr_hd:.3f})'
    if hd_timestep is not None:
        hd_title += f'\\nt={hd_timestep}'
    axes[1].scatter(noise_indices, hd_recon_stats[stat], alpha=0.6, s=30, color='orange')
    axes[1].set_ylabel('Mean Recon Loss', fontsize=11)
    axes[1].set_xlabel('Noise Index', fontsize=11)
    axes[1].set_title(hd_title, fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'approx_loss_recon.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot clipped recon loss (after clipping)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{title_prefix} - Mean Success Clipped Recon Loss (Post-Clamp) ({title_ddim_info})", fontsize=14, fontweight='bold', y=1.02)
    
    # LD subplot with timestep info
    corr_ld_clipped, _ = spearmanr(noise_indices, ld_clipped_recon_stats[stat])
    ld_clipped_title = f'LD Clipped Recon Loss (DDIM={low_ddim}, ρ={corr_ld_clipped:.3f})'
    if ld_timestep is not None:
        ld_clipped_title += f'\\nt={ld_timestep}'
    axes[0].scatter(noise_indices, ld_clipped_recon_stats[stat], alpha=0.6, s=30, color='darkviolet')
    axes[0].set_ylabel('Mean Clipped Recon Loss', fontsize=11)
    axes[0].set_xlabel('Noise Index', fontsize=11)
    axes[0].set_title(ld_clipped_title, fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # HD subplot with timestep info
    corr_hd_clipped, _ = spearmanr(noise_indices, hd_clipped_recon_stats[stat])
    hd_clipped_title = f'HD Clipped Recon Loss (DDIM={high_ddim}, ρ={corr_hd_clipped:.3f})'
    if hd_timestep is not None:
        hd_clipped_title += f'\\nt={hd_timestep}'
    axes[1].scatter(noise_indices, hd_clipped_recon_stats[stat], alpha=0.6, s=30, color='darkorange')
    axes[1].set_ylabel('Mean Clipped Recon Loss', fontsize=11)
    axes[1].set_xlabel('Noise Index', fontsize=11)
    axes[1].set_title(hd_clipped_title, fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'approx_loss_clipped_recon.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot clip freq separately if requested
    if plot_clip_freq:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"{title_prefix} - Mean Success Clip Freq ({title_ddim_info})", fontsize=14, fontweight='bold', y=1.02)
        
        # LD clip freq with timestep
        corr_ld_clip, _ = spearmanr(noise_indices, ld_clip_stats[stat])
        ld_clip_title = f'LD Clip Freq (DDIM={low_ddim}, ρ={corr_ld_clip:.3f})'
        if ld_timestep is not None:
            ld_clip_title += f'\\nt={ld_timestep}'
        axes[0].scatter(noise_indices, ld_clip_stats[stat], alpha=0.6, s=30, color='green')
        axes[0].set_ylabel('Mean Clip Freq', fontsize=11)
        axes[0].set_xlabel('Noise Index', fontsize=11)
        axes[0].set_title(ld_clip_title, fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # HD clip freq with timestep
        corr_hd_clip, _ = spearmanr(noise_indices, hd_clip_stats[stat])
        hd_clip_title = f'HD Clip Freq (DDIM={high_ddim}, ρ={corr_hd_clip:.3f})'
        if hd_timestep is not None:
            hd_clip_title += f'\\nt={hd_timestep}'
        axes[1].scatter(noise_indices, hd_clip_stats[stat], alpha=0.6, s=30, color='blue')
        axes[1].set_ylabel('Mean Clip Freq', fontsize=11)
        axes[1].set_xlabel('Noise Index', fontsize=11)
        axes[1].set_title(hd_clip_title, fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'approx_loss_clip_freq.png'), dpi=300, bbox_inches='tight')
        plt.close()

