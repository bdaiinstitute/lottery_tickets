"""
This file evaluates action MSE error after diffusion inference.
Action MSE can be evaluated for rollouts collected from the policy,
or on ground-truth action chunks from the training dataset.
Correlation is evaluated between this MSE for an ordered set of tickets and the order itself.
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

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
                                                             save_correlation_summary_statistics)
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
    p.add_argument("--num_noises", type=int, default=4480)
    p.add_argument("--seed", type=int, default=4299924)
    p.add_argument("--noise_path", type=str, default=None, help="Path to saved noise samples .npy file")
    p.add_argument("--episodes_dir", type=str, 
                   default="/lam-248-lambdafs/teams/proj-compose/opatil/src/dsrl/logs_res_rm/policy_record/can/base_policy_seeds1_evals100_seed202020_ddim8_20251211_002544", help="Path to directory containing episode_*.npy files")
    p.add_argument("--max_samples", type=int, default=5000, help="Maximum number of samples to evaluate")
    p.add_argument("--ddim_steps", type=int, default=8, help="DDIM steps to override config value")
    p.add_argument("--exp_name", type=str, default="", help="Experiment name to append to output folder")
    p.add_argument("--k_shortest", type=int, default=None, help="Number of shortest episodes to use from dataset (use None for random sampling)")
    return p.parse_args()

GRIPPER_DIM_WEIGHT = 1
class ActionMSEError(TrainDiffusionAgent):
    """
    TrainDiffusionAgent was subclassed to get access to the training dataset.
    Note that we do instantiate DiffusionEval over
    DiffusionModel by modifying the config that goes into TrainDiffusionAgent.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        
        # Load normalization parameters for action unnormalization
        self.normalize = hasattr(cfg, 'normalization_path') and cfg.normalization_path is not None
        if self.normalize:
            normalization = np.load(cfg.normalization_path)
            self.action_min = torch.from_numpy(normalization['action_min']).to(cfg.device).float()
            self.action_max = torch.from_numpy(normalization['action_max']).to(cfg.device).float()
            print(f"Loaded normalization from {cfg.normalization_path}")
            print(f"Action range: [{self.action_min.cpu().numpy()}] to [{self.action_max.cpu().numpy()}]")
        else:
            print("No normalization will be applied")
    
    def unnormalize_action(self, action):
        """Unnormalize actions from [-1, 1] to original range."""
        if not self.normalize:
            return action
        action = (action + 1) / 2  # [-1, 1] -> [0, 1]
        return action * (self.action_max - self.action_min) + self.action_min


    def eval_action_mse_correlation(self, eval_noises, num_noises, episodes_dir, 
                            batch_size=64, max_samples=10000, k_shortest=None):
        """Evaluate MSE for success samples to analyze correlation."""
        results = {'success': {}}
        
        # Check if episodes_dir is "dataset" - use training dataset instead
        if episodes_dir and episodes_dir.lower() == "dataset":
            print(f"Using training dataset for evaluation samples...")
            dataset = self.dataset_train
            success_samples = []
            num_samples = min(len(dataset), max_samples)
            
            # Filter for shortest k episodes (if k_shortest is specified)
            if k_shortest is not None:
                traj_lengths = [(dataset.traj_lenghts[i], i) for i in range(len(dataset.traj_lenghts))]
                traj_lengths.sort()
                shortest_trajs = [traj_idx for _, traj_idx in traj_lengths[:k_shortest]]
                print(f"Using {k_shortest} shortest episodes: {shortest_trajs}")
                
                # Approach 1: Sliding window to find spans with both gripper open and close
                span = 5  # Span size for sliding window
                for traj_idx in shortest_trajs:
                    # Collect all samples from this trajectory
                    trajectory_samples = []
                    for batch in dataset.get_trajectory(traj_idx):
                        action = batch.actions
                        obs = batch.conditions
                        action_np = action.cpu().numpy() if hasattr(action, 'cpu') else action
                        sample_data = {'actions': action_np, 'state': obs['state'].cpu().numpy() if isinstance(obs, dict) and 'state' in obs else (obs.cpu().numpy() if hasattr(obs, 'cpu') else obs)}
                        trajectory_samples.append((traj_idx, len(trajectory_samples), sample_data))
                    
                    # Sliding window approach within this trajectory: find spans with both open and close
                    for i in range(max(0, len(trajectory_samples) - span + 1)):
                        window = trajectory_samples[i:i+span]
                        has_open = any(np.any(s[2]['actions'][:, -1] < 0.0) for s in window)
                        has_close = any(np.any(s[2]['actions'][:, -1] > 0.0) for s in window)
                        if has_open and has_close:
                            for s in window:
                                if s not in success_samples:
                                    success_samples.append(s)
                
                # # Approach 2: Take all samples from shortest episodes
                # for traj_idx in shortest_trajs:
                #     for batch in dataset.get_trajectory(traj_idx):
                #         action = batch.actions
                #         obs = batch.conditions
                #         action_np = action.cpu().numpy() if hasattr(action, 'cpu') else action
                #         sample_data = {'actions': action_np, 'state': obs['state'].cpu().numpy() if isinstance(obs, dict) and 'state' in obs else (obs.cpu().numpy() if hasattr(obs, 'cpu') else obs)}
                #         success_samples.append((traj_idx, len(success_samples), sample_data))
            else:
                # Shuffle indices for random sampling
                print(f"Using random sampling from entire dataset")
                indices = np.arange(num_samples)
                np.random.shuffle(indices)
                for idx in indices:
                    batch = dataset[idx]
                    # Dataset returns a namedtuple: Batch(actions, conditions)
                    action = batch.actions
                    obs = batch.conditions
                    action_np = action.cpu().numpy() if hasattr(action, 'cpu') else action

                    # # Filter for gripper activation
                    # gripper_values = action_np[:, -1]  # Last dimension is gripper
                    # if not (gripper_values > 0.5).any():  # Skip if no gripper activation
                    #     continue
                    sample_data = {
                        'actions': action_np,  # Shape: (action_horizon, action_dim)
                        'state': obs['state'].cpu().numpy() if isinstance(obs, dict) and 'state' in obs else (obs.cpu().numpy() if hasattr(obs, 'cpu') else obs)
                    }
                    success_samples.append((idx, 0, sample_data))  # (ep_idx, timestep_idx, sample_data)
            print(f"Loaded {len(success_samples)} samples from training dataset")
        else:
            all_episodes = load_episodes_from_path(episodes_dir)
            # success_samples, fail_samples = filter_samples_based_on_gripper(all_episodes, max_samples=10000, 
            #                                                                 span=3, use_span_filter=True) 
            success_samples, fail_samples = get_succ_fail_samples(all_episodes=all_episodes)
        
        # Evaluate success samples only
        print(f"Evaluating {len(success_samples)} success samples...")
        for count, (ep_idx, timestep_idx, sample_data) in enumerate(success_samples, 1):
            print(f"Evaluating success sample {count}/{len(success_samples)}: episode={ep_idx}, timestep={timestep_idx}")
            mse_losses, cosine_sims = self.eval_mse_for_sample(
                num_noises=num_noises,
                noise_list=eval_noises,
                sample_data=sample_data,
                batch_size=batch_size
            )
            sample_key = f"ep{ep_idx}_t{timestep_idx}"
            results['success'][sample_key] = {'mse_losses': mse_losses, 'cosine_similarities': cosine_sims}
        return results

    def eval_mse_for_sample(self, num_noises, noise_list, sample_data,
                                            batch_size=64):
        """Evaluate MSE after complete diffusion inference for multiple noise samples on a single sample."""

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
        
        mse_losses = []
        cosine_similarities = []
        self.model.controllable_noise = True
        
        # Disable gradient tracking for evaluation
        with torch.no_grad():
            for noise_batch in batched_noise_list:
                # Set controllable noise in conditions for inference
                batch_conditions['noise_action'] = noise_batch                
                
                # Run inference
                sample_output = self.model.forward(cond=batch_conditions, deterministic=True)
                denoised_actions = sample_output.trajectories  # Shape: (batch_size, horizon_steps, action_dim)
                
                # Unnormalize actions before computing MSE
                denoised_actions = self.unnormalize_action(denoised_actions)
                
                # Compute MSE between denoised and ground truth actions
                gt_actions_batch = gt_actions.unsqueeze(0).repeat(batch_size, 1, 1)
                
                # Flatten for loss calculation
                denoised_flat = denoised_actions.reshape(batch_size, -1)
                gt_flat = gt_actions_batch.reshape(batch_size, -1)
                
                # Weighted MSE loss
                weights = torch.ones_like(denoised_flat)
                action_dim = gt_actions.shape[1]
                horizon = gt_actions.shape[0]
                # Set weight for gripper dimension across all timesteps
                for t in range(horizon):
                    weights[:, t * action_dim + (action_dim - 1)] = GRIPPER_DIM_WEIGHT          
                mse = ((denoised_flat - gt_flat) ** 2 * weights).mean(dim=1)
                
                # Cosine similarity (per sample)
                denoised_norm = torch.norm(denoised_flat, dim=1)
                gt_norm = torch.norm(gt_flat, dim=1)
                cosine_sim = (denoised_flat * gt_flat).sum(dim=1) / (denoised_norm * gt_norm + 1e-8)
                
                mse_losses.extend(mse.detach().cpu().tolist())
                cosine_similarities.extend(cosine_sim.detach().cpu().tolist())
                    
        return mse_losses, cosine_similarities


def run_mse_correlation_eval(eval_agent, eval_noises, save_dir, 
                             max_samples=10000, episodes_dir=None, noises_were_generated=False,
                             exp_name='', task_name='', k_shortest=None):
    """Evaluate MSE correlation between LTS and successful ep samples and LTs"""
    num_noises = len(eval_noises)
    results = eval_agent.eval_action_mse_correlation(
        eval_noises=eval_noises,
        num_noises=num_noises,
        episodes_dir=episodes_dir,
        max_samples=max_samples,
        k_shortest=k_shortest,
    )
    results_success = results['success']
    
    # Sort and save noises by different indicators (only if generated)
    if noises_were_generated:
        os.makedirs(save_dir, exist_ok=True)
        
        # Collect metrics per noise
        mse_per_noise = [[] for _ in range(num_noises)]
        cosine_per_noise = [[] for _ in range(num_noises)]
        for sample_dict in results_success.values():
            for noise_idx, (mse, cos) in enumerate(zip(sample_dict['mse_losses'], sample_dict['cosine_similarities'])):
                mse_per_noise[noise_idx].append(mse)
                cosine_per_noise[noise_idx].append(cos)
        
        # Calculate statistics and sort
        for metric_name, metric_data in [('mse', mse_per_noise), ('cosine', cosine_per_noise)]:
            for stat in ['mean', 'min', 'max']:
                stat_vals = [getattr(np, stat)(vals) for vals in metric_data]
                sorted_idx = np.argsort(stat_vals)
                if metric_name == 'cosine':
                    sorted_idx = sorted_idx[::-1]
                sorted_noises = np.array([eval_noises[i] for i in sorted_idx])
                np.save(os.path.join(save_dir, f"noises_sorted_{metric_name}_{stat}.npy"), sorted_noises)
                print(f"Saved noises sorted by {stat} {metric_name}")
        
        # Default sort by mean MSE for results
        mean_mse = [np.mean(vals) for vals in mse_per_noise]
        sorted_idx = np.argsort(mean_mse)
        eval_noises = [eval_noises[i] for i in sorted_idx]
        results_success = {k: {'mse_losses': [v['mse_losses'][i] for i in sorted_idx],
                               'cosine_similarities': [v['cosine_similarities'][i] for i in sorted_idx]}
                          for k, v in results_success.items()}
        results = {'success': results_success}
        print(f"Results reordered by mean MSE")
    
    # Save results
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "mse_correlation_results.npy"), dict(results))
    
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
    with open(os.path.join(save_dir, "mse_correlation_results.json"), "w") as f:
        json.dump(json_results, f, indent=2)
    
    # Create correlation plots
    plot_correlation_results(results_success, save_dir, exp_name=exp_name, task_name=task_name)
    
    # Save summary statistics
    save_correlation_summary_statistics(results_success, num_noises, save_dir)
    
    print(f"mse correlation evaluation results saved to {save_dir}")
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
    
    # Override ddim_steps if provided
    if args.ddim_steps is not None:
        cfg.model.ddim_steps = args.ddim_steps
    
    if not hasattr(cfg, 'device') or cfg.device is None:
        cfg.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'	
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Instantiate EvalDiffusionLoss with the same config
    eval_agent = ActionMSEError(cfg)
    
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
        eval_noises = generate_noise_samples(eval_agent.dataset_train, eval_agent.cfg, num_noises=args.num_noises)
        noise_source = "sampled_noise"
        noises_were_generated = True

    # Inference MSE evaluation
    checkpoint_name = os.path.splitext(os.path.basename(cfg.base_policy_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    inference_folder_name = f"inference_mse_ns{len(eval_noises)}_ms{args.max_samples}_ddim{cfg.model.ddim_steps}_{timestamp}"
    if args.exp_name:
        inference_folder_name = f"{inference_folder_name}_{args.exp_name}"
    inference_save_dir = os.path.join(cfg.logdir, "mse", checkpoint_name, noise_source, inference_folder_name)
    
    # Save config to the output directory
    os.makedirs(inference_save_dir, exist_ok=True)
    with open(os.path.join(inference_save_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    
    _ = run_mse_correlation_eval(
        eval_agent=eval_agent,
        eval_noises=eval_noises,
        save_dir=inference_save_dir,
        max_samples=args.max_samples,
        episodes_dir=args.episodes_dir,
        noises_were_generated=noises_were_generated,
        exp_name=args.exp_name or '',
        task_name=args.task_name,
        k_shortest=args.k_shortest
    )
    
if __name__ == "__main__":
    main()
