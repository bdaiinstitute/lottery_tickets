# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from pathlib import Path

import hydra
import imageio
import torch
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf

from lottery_tickets.franka_sim_lt.gym_utils import make_frankasim_env
from lottery_tickets.franka_sim_lt.models_utils import FMPolicyInterface, load_flow_matching_model

def collect_rollouts(policy, cfg):
    print(f"Building environment: {cfg.evaluation.env_name}")
    env = make_frankasim_env(
        cfg.evaluation.env_name,
        env_kwargs=cfg.evaluation.env_kwargs,
    )

    num_episodes = cfg.evaluation.num_episodes  # X

    print(f"Running {num_episodes} episodes with standard Gaussian noise...")
    print(f"Sampling {num_noises} noise vectors for variance eval")

    obs_rollouts = [] # list of episodes, where each episode is a list of observations (T, obs_dim)
    frames_rollouts = [] # list of episodes, where each episode is a list of frames (T, H, W, C)
    rollout_lengths = []
    return_rollouts = [] # list of episodes, where each episode is a list of episode rewards (T,)

    # ------------------------------------------------------------
    # Collect rollouts using standard Gaussian noise
    # ------------------------------------------------------------
    for episode in range(num_episodes):
        print(f"Collecting rollout for episode {episode+1}/{num_episodes}...")
        episode_returns = 0
        policy.reset()
        obs, _ = env.reset()
        done = False
        frames = []
        obs_list = []
        step = 0

        while not done:
            obs_list.append(obs['state']) # for franka_sim, obs['state'] is shape (1,10)
            action = policy(obs, init_x = None)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            frames.append(env.render()) # each frame is shape [128, 256, 3]
            step += 1
            episode_returns += reward.item()

        # Store episode information
        obs_rollouts.append(np.concatenate(obs_list,axis=0))
        rollout_lengths.append(len(obs_list))
        frames_rollouts.append(np.stack(frames,axis=0))
        return_rollouts.append(episode_returns)

        # Save video + print episode stats
        video_file = video_path / f"ep_video_{episode}.mp4"
        imageio.mimsave(video_file, frames, fps=30)
        print(f"[Episode {episode+1}] returns={episode_returns,}, steps={step}, saved video to {video_file}")

    # Concatenate each rollouts into a single array 
    obs_rollouts = np.stack(obs_rollouts,axis=0) # (num_episodes, T, obs_dim)
    frames_rollouts = np.stack(frames_rollouts,axis=0) # (num_episodes, T, H, W, C)
    return_rollouts = np.array(return_rollouts) # (num_episodes,)
    # Save the observations, frames, and returns
    np.save(video_path / "obs_rollouts.npy", obs_rollouts)
    np.save(video_path / "frames_rollouts.npy", frames_rollouts)
    np.save(video_path / "return_rollouts.npy", return_rollouts)
    
    return obs_rollouts, frames_rollouts, return_rollouts

def eval_variance(cfg: DictConfig) -> None:
    """
    Evaluates variance of action chunks across multiple noise initializations
    for fixed rollouts of observations.
    """
    video_path = Path(cfg.evaluation.video_save_path)
    video_path.mkdir(parents=True, exist_ok=True)

    num_noises = cfg.num_noises      # Y

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model from {cfg.evaluation.model_path}")
    fm_model, config = load_flow_matching_model(cfg.evaluation.model_path, device)

    policy = FMPolicyInterface(
        fm_model,
        cfg.evaluation.chunk_size,
        device,
        use_state_history=cfg.evaluation.use_state_history,
        state_history_length=cfg.evaluation.state_history_length,
    )

    # ------------------------------------------------------------
    # 1) Get rollouts (either live, or load from file)
    # ------------------------------------------------------------
    
    # check if load_rollout_dir is provided, if so load the rollouts from file, otherwise collect new rollouts
    if cfg.get("load_rollout_dir", None) is not None:
        rollout_dir = Path(cfg.load_rollout_dir)
        print(f"Loading rollouts from {rollout_dir}")
        obs_rollouts = np.load(rollout_dir / "obs_rollouts.npy")
        frames_rollouts = np.load(rollout_dir / "frames_rollouts.npy")
        return_rollouts = np.load(rollout_dir / "return_rollouts.npy")
    else:
        obs_rollouts, frames_rollouts, return_rollouts = collect_rollouts(policy, cfg)

    # ------------------------------------------------------------
    # 2) Sample Y noise vectors
    # ------------------------------------------------------------
    init_x_size = 4 * cfg.evaluation.chunk_size # action space is 4 in frankasim (x,y,z,gripper)
    noise_bank = torch.randn((num_noises, init_x_size), device=device)
    torch.save(noise_bank, video_path / "noise_bank.pt")

    # ------------------------------------------------------------
    # 3) For each episode + observation, compute variance
    # ------------------------------------------------------------
    all_episode_variances = []

    for ep_idx, obs_list in enumerate(obs_rollouts):
        print(f"Computing variances for episode {ep_idx}/{obs_rollouts.shape[0] - 1}")

        episode_variances = []  # list of (T, action_dim * chunk_size)

        for t, obs in enumerate(obs_list):
            action_chunks = []

            for n in range(num_noises):
                policy.reset()  # important: reset policy state for fair comparison
                with torch.no_grad():
                    action_chunk = policy(obs, init_x=noise_bank[n:n+1])
                action_chunks.append(action_chunk)

            action_chunks = np.stack(action_chunks, axis=0)
            var_t = np.var(action_chunks, axis=0)  # per action-dimension variance
            
            if np.max(var_t).item() >= 1.0:
                breakpoint()    
        
            episode_variances.append(var_t)


        episode_variances = np.stack(episode_variances, axis=0)  # (T, D)
        all_episode_variances.append(episode_variances)

        np.save(plot_path / f"episode_{ep_idx}_variance.npy", episode_variances)

    # ------------------------------------------------------------
    # 4) Plot variance over time for each episode + action dim
    # ------------------------------------------------------------
    plot_path = video_path / "variance_plots"
    plot_path.mkdir(parents=True, exist_ok=True)
    for ep_idx, ep_var in enumerate(all_episode_variances):
        T, D = ep_var.shape

        plt.figure(figsize=(12, 6))
        for d in range(D):
            plt.plot(
                np.arange(T),
                ep_var[:, d],
                alpha=0.7,
                linewidth=1,
                label=f"action_dim_{d}"
            )

        plt.xlabel("Timestep")
        plt.ylabel("Variance across noise vectors")
        plt.title(f"Episode {ep_idx} – Per-action-dimension variance over time")
        plt.legend(ncol=2, fontsize=8)  # tweak ncol based on how many dims you have
        plt.tight_layout()

        fig_path = plot_path / f"episode_{ep_idx}_variance.png"
        plt.savefig(fig_path)
        plt.close()

        print(f"Saved variance plot to {fig_path}")

    env.close()
    print("Variance evaluation complete!")


@hydra.main(version_base=None, config_path="cfgs", config_name="fm")
def main(cfg: DictConfig):
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    eval_variance(cfg)


if __name__ == "__main__":
    main()
