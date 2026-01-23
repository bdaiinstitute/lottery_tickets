"""
This file evaluates GTE- global truncation error. 
It does this by evaluating the difference of outputs from high-step (say 8) and low-step (say 2) DDIM rollouts.
GTE can either be evaluate over rollouts collected from the policy (with even higher DDIM steps- say 20),
or on ground-truth action chunks from the training dataset.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

os.environ["MUJOCO_GL"] = "egl"
os.environ["WANDB_MODE"] = "disabled"

BASE_DIR = Path(__file__).resolve().parents[3]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

log = logging.getLogger(__name__)

from dppo.agent.pretrain.train_agent import batch_to_device
from dppo.agent.pretrain.train_diffusion_agent import TrainDiffusionAgent
from noise_opt_rm.ticket_search.diffusion.loss_utils import (load_noise_samples, 
                                                             load_episodes_from_path,
                                                             get_succ_fail_samples, 
                                                             filter_samples_based_on_gripper,
                                                             generate_noise_samples,
                                                             save_gte_correlation_summary_statistics)
from noise_opt_rm.ticket_search.diffusion.plot_utils import plot_noise_magnitudes, plot_correlation_results
 
TASK_CONFIGS = {
    "lift": "cfg/robomimic/diffusion_loss/lift_pre_diffusion_mlp.yaml",
    "can": "cfg/robomimic/diffusion_loss/can_pre_diffusion_mlp.yaml",
    "square": "cfg/robomimic/diffusion_loss/square_pre_diffusion_mlp.yaml",
    "transport": "cfg/robomimic/diffusion_loss/transport_pre_diffusion_mlp.yaml",
}

def p_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task_name", default="can", choices=list(TASK_CONFIGS.keys()))
    p.add_argument("--num_noises", type=int, default=448)
    p.add_argument("--seed", type=int, default=42024)
    p.add_argument("--noise_path", type=str, default=None, help="Path to saved noise samples .npy file")
    p.add_argument("--episodes_dir", type=str, 
                   default="/lam-248-lambdafs/teams/proj-compose/opatil/src/dsrl/logs_res_rm/policy_record/can/base_policy_seeds1_evals100_seed202020_ddim8_20251211_002544", help="Path to directory containing episode_*.npy files")
    p.add_argument("--max_samples", type=int, default=5000, help="Maximum number of samples to evaluate")
    p.add_argument("--low_ddim", type=int, default=8, help="DDIM steps for low-dimensional model")
    p.add_argument("--high_ddim", type=int, default=20, help="DDIM steps for high-dimensional model")
    p.add_argument("--exp_name", type=str, default="", help="Experiment name to append to output folder")
    return p.parse_args()

GRIPPER_DIM_WEIGHT = 1
class GblTruncationError(TrainDiffusionAgent):
    """
    TrainDiffusionAgent was subclassed to get access to the training dataset.
    Note that we do instantiate DiffusionEval over
    DiffusionModel by modifying the config that goes into TrainDiffusionAgent.
    """
    def __init__(self, cfg, low_ddim=8, high_ddim=20):
        super().__init__(cfg)
        self.cfg = cfg
        self.low_ddim = low_ddim
        self.high_ddim = high_ddim
        self.model_low_ddim = self.model
        cfg.model.ddim_steps = high_ddim
        self.model_high_ddim = hydra.utils.instantiate(cfg.model)

    def eval_action_gte_correlation(self, eval_noises, num_noises, episodes_dir, 
                            batch_size=64, max_samples=5000):
        """Evaluate GTE for success samples to analyze correlation."""
        results = {'success': {}}
        
        # Check if episodes_dir is "dataset" - use training dataset instead
        if episodes_dir and episodes_dir.lower() == "dataset":
            print(f"Using training dataset for evaluation samples...")
            dataset = self.dataset_train
            success_samples = []
            num_samples = min(len(dataset), max_samples)
            for idx in range(num_samples):
                batch = dataset[idx]
                # Dataset returns a namedtuple: Batch(actions, conditions)
                action = batch.actions
                obs = batch.conditions
                
                # Create sample_data dict matching expected format
                sample_data = {
                    'actions': action.cpu().numpy() if hasattr(action, 'cpu') else action,  # Shape: (action_horizon, action_dim)
                    'state': obs['state'].cpu().numpy() if isinstance(obs, dict) and 'state' in obs else (obs.cpu().numpy() if hasattr(obs, 'cpu') else obs)
                }
                success_samples.append((idx, 0, sample_data))  # (ep_idx, timestep_idx, sample_data)
            print(f"Loaded {len(success_samples)} samples from training dataset")
        else:
            all_episodes = load_episodes_from_path(episodes_dir)
            # success_samples, fail_samples = filter_samples_based_on_gripper(all_episodes, max_samples=5000, 
            #                                                                 span=3, use_span_filter=True) 
            success_samples, fail_samples = get_succ_fail_samples(all_episodes=all_episodes)
        
        # Evaluate success samples only
        print(f"Evaluating {len(success_samples)} success samples...")
        for count, (ep_idx, timestep_idx, sample_data) in enumerate(success_samples, 1):
            print(f"Evaluating success sample {count}/{len(success_samples)}: episode={ep_idx}, timestep={timestep_idx}")
            ld_gt_mse, hd_gt_mse, gte = self.eval_gte_for_sample(
                num_noises=num_noises,
                noise_list=eval_noises,
                sample_data=sample_data,
                batch_size=batch_size
            )
            sample_key = f"ep{ep_idx}_t{timestep_idx}"
            results['success'][sample_key] = {
                'ld_gt_mse_losses': ld_gt_mse,
                'hd_gt_mse_losses': hd_gt_mse,
                'gte_losses': gte
            }
        return results

    def eval_gte_for_sample(self, num_noises, noise_list, sample_data,
                                            batch_size=64):
        """Evaluate GTE after complete diffusion inference for multiple noise samples on a single sample."""

        assert num_noises % batch_size == 0, f"num_noises {num_noises} must be a multiple of batch_size {batch_size}"
        
        # Ground truth actions from sample (action chunks at this timestep)
        gt_actions = torch.from_numpy(sample_data['actions']).to(self.cfg.device).float()  # Shape: (action_chunks, action_dim)
        obs = torch.from_numpy(sample_data['state']).to(self.cfg.device).float()  # Current state
        
        # Prepare conditions (repeat for batch)
        if len(obs.shape)==1:
            obs = obs.unsqueeze(0)
        batch_conditions = {
            'state': obs.repeat(batch_size, 1)
        }
        
        device = self.cfg.device
        action_shape = (batch_size, gt_actions.shape[0], gt_actions.shape[1])
        
        # Convert noise_list to batched tensors
        def to_batch(batch_noise):
            noise = torch.stack([torch.from_numpy(n) for n in batch_noise]).to(device)
            return noise.reshape(action_shape) if noise.shape != action_shape else noise
        batched_noise_list = [to_batch(noise_list[i:i+batch_size]) for i in range(0, len(noise_list), batch_size)]
        
        ld_gt_mse_losses = []
        hd_gt_mse_losses = []
        gte_losses = []
        self.model_low_ddim.controllable_noise = True
        self.model_high_ddim.controllable_noise = True
        
        # Disable gradient tracking for evaluation
        with torch.no_grad():
            for noise_batch in batched_noise_list:
                # Set controllable noise in conditions for inference
                batch_conditions['noise_action'] = noise_batch                
                
                # Run inference
                ld_sample_output = self.model_low_ddim.forward(cond=batch_conditions, deterministic=True)
                ld_denoised_actions = ld_sample_output.trajectories  # Shape: (batch_size, horizon_steps, action_dim)
                
                hd_sample_output = self.model_high_ddim.forward(cond=batch_conditions, deterministic=True)
                hd_denoised_actions = hd_sample_output.trajectories  # Shape: (batch_size, horizon_steps, action_dim)
                
                # Compute MSE between denoised and ground truth actions
                gt_actions_batch = gt_actions.unsqueeze(0).repeat(batch_size, 1, 1)
                
                # Flatten for loss calculation
                ld_denoised_flat = ld_denoised_actions.reshape(batch_size, -1)
                hd_denoised_flat = hd_denoised_actions.reshape(batch_size, -1)
                gt_flat = gt_actions_batch.reshape(batch_size, -1)
                
                ld_gt_mse = ((ld_denoised_flat - gt_flat) ** 2).mean(dim=1)
                hd_gt_mse = ((hd_denoised_flat - gt_flat) ** 2).mean(dim=1)
                gte = ((hd_denoised_flat - ld_denoised_flat) ** 2).mean(dim=1)

                ld_gt_mse_losses.extend(ld_gt_mse.detach().cpu().tolist())
                hd_gt_mse_losses.extend(hd_gt_mse.detach().cpu().tolist())
                gte_losses.extend(gte.detach().cpu().tolist())
                    
        return ld_gt_mse_losses, hd_gt_mse_losses, gte_losses


def run_gte_correlation_eval(eval_agent, eval_noises, save_dir, 
                             max_samples=5000, episodes_dir=None, noises_were_generated=False,
                             exp_name='', task_name='', low_ddim=8, high_ddim=20):
    """Evaluate GTE correlation between LTS and successful ep samples and LTs"""
    num_noises = len(eval_noises)
    results = eval_agent.eval_action_gte_correlation(
        eval_noises=eval_noises,
        num_noises=num_noises,
        episodes_dir=episodes_dir,
        max_samples=max_samples,
    )
    results_success = results['success']
    
    # Sort noises if they were generated (not loaded) - use gte_losses for sorting
    if noises_were_generated:
        print("\nSorting generated noises by mean GTE across samples...")
        # Calculate mean GTE per noise
        gte_per_noise = [[] for _ in range(num_noises)]
        for sample_dict in results_success.values():
            for noise_idx, mse in enumerate(sample_dict['gte_losses']):
                gte_per_noise[noise_idx].append(mse)
        mean_gte = [np.mean(vals) for vals in gte_per_noise]
        sorted_idx = np.argsort(mean_gte)
        
        # Reorder
        eval_noises = [eval_noises[i] for i in sorted_idx]
        results_success = {k: {
            'ld_gt_mse_losses': [v['ld_gt_mse_losses'][i] for i in sorted_idx],
            'hd_gt_mse_losses': [v['hd_gt_mse_losses'][i] for i in sorted_idx],
            'gte_losses': [v['gte_losses'][i] for i in sorted_idx]
        } for k, v in results_success.items()}
        results = {'success': results_success}
        print(f"Noises sorted by mean GTE.")
    
    # Save results
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "gte_correlation_results.npy"), dict(results))
    
    # Save noises in reloadable format
    noises_array = np.array(eval_noises)
    np.save(os.path.join(save_dir, "noise_samples.npy"), noises_array)
    print(f"Noise samples saved to {os.path.join(save_dir, 'noise_samples.npy')}")
    
    # Calculate and save noise magnitudes
    noise_magnitudes = np.linalg.norm(noises_array.reshape(len(eval_noises), -1), axis=1)
    np.save(os.path.join(save_dir, "noise_magnitudes.npy"), noise_magnitudes)
    plot_noise_magnitudes(noise_magnitudes, os.path.join(save_dir, "noise_magnitudes.png"))
    
    # Prepare JSON-serializable results
    json_results = {
        'success': {k: v for k, v in results_success.items()},
    }
    with open(os.path.join(save_dir, "gte_correlation_results.json"), "w") as f:
        json.dump(json_results, f, indent=2)
    
    # Create correlation plots for all three metrics
    from noise_opt_rm.ticket_search.diffusion.plot_utils import plot_gte_correlation_results
    plot_gte_correlation_results(results_success, save_dir, exp_name=exp_name, task_name=task_name, 
                                low_ddim=low_ddim, high_ddim=high_ddim)
    
    # Save summary statistics
    save_gte_correlation_summary_statistics(results_success, num_noises, save_dir)
    
    print(f"GTE correlation evaluation results saved to {save_dir}")
    return results


OmegaConf.register_new_resolver("eval", eval)
def main():
    args = p_args()
    base_path = str(BASE_DIR)
    config_path = os.path.join(base_path, TASK_CONFIGS[args.task_name])
    config_dir = os.path.dirname(config_path)
    config_name = os.path.basename(config_path).replace('.yaml', '')
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_name)

    # Allow mutation of config (parity with eval script conveniences)
    OmegaConf.set_struct(cfg, False)
    cfg.seed = args.seed
    
    # Override ddim_steps with low_ddim
    if args.low_ddim is not None:
        cfg.model.ddim_steps = args.low_ddim
    
    if not hasattr(cfg, 'device') or cfg.device is None:
        cfg.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'	
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Instantiate EvalDiffusionLoss with the same config
    eval_agent = GblTruncationError(cfg, low_ddim=args.low_ddim, high_ddim=args.high_ddim)
    
    # Load or generate noises
    noises_were_generated = False
    if args.noise_path:
        eval_noises = load_noise_samples(args.noise_path, args.num_noises)
        # Extract parent directories from noise path for organization
        path_parts = Path(args.noise_path).parts
        if 'lottery_ticket_results' in path_parts:
            idx = path_parts.index('lottery_ticket_results')
            noise_source = os.path.join("loaded_noise", *path_parts[idx+1:-1])
        else:
            noise_source = "loaded_noise"
    else:
        eval_noises = generate_noise_samples(eval_agent.dataset_train, eval_agent.cfg, num_noises=args.num_noises*5)
        noise_source = "sampled_noise"
        noises_were_generated = True

    # Inference MSE evaluation
    checkpoint_name = os.path.splitext(os.path.basename(cfg.base_policy_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    inference_folder_name = f"inference_mse_ns{len(eval_noises)}_ms{args.max_samples}_{timestamp}"
    if args.exp_name:
        inference_folder_name = f"{inference_folder_name}_{args.exp_name}"
    inference_save_dir = os.path.join(cfg.logdir, "gte", checkpoint_name, noise_source, inference_folder_name)
    
    _ = run_gte_correlation_eval(
        eval_agent=eval_agent,
        eval_noises=eval_noises,
        save_dir=inference_save_dir,
        max_samples=args.max_samples,
        episodes_dir=args.episodes_dir,
        noises_were_generated=noises_were_generated,
        exp_name=args.exp_name or '',
        task_name=args.task_name,
        low_ddim=args.low_ddim,
        high_ddim=args.high_ddim
    )
    
if __name__ == "__main__":
    main()
