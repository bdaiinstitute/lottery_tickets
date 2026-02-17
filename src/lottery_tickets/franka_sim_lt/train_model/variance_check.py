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
    video_path = Path(cfg.evaluation.video_save_path)

    print(f"Building environment: {cfg.evaluation.env_name}")
    env = make_frankasim_env(
        cfg.evaluation.env_name,
        env_kwargs=cfg.evaluation.env_kwargs,
    )

    num_episodes = cfg.evaluation.num_episodes  # X
    init_x_size = 4 * cfg.evaluation.chunk_size # action space is 4 in frankasim (x,y,z,gripper)


    print(f"Running {num_episodes} episodes with standard Gaussian noise...")
    obs_rollouts = [] # list of episodes, where each episode is a list of observations (T, obs_dim)
    frames_rollouts = [] # list of episodes, where each episode is a list of frames (T, H, W, C)
    rollout_lengths = []
    return_rollouts = [] # list of episodes, where each episode is a list of episode rewards (T,)
    action_rollouts = [] # list of episodes, where each episode is a list of actions (T, action_dim)
    noise_rollouts = [] # list of episodes, where each episode is a list of noise vectors (T, init_x_size) 
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
        action_list = []
        noise_list = []
        step = 0

        while not done:
            obs_list.append(obs['state']) # for franka_sim, obs['state'] is shape (1,10)

            init_x = torch.randn((1, init_x_size), device=cfg.device)
            action = policy(obs, init_x = init_x)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            frames.append(env.render()) # each frame is shape [128, 256, 3]
            action_list.append(action)
            noise_list.append(init_x.cpu().numpy())
            step += 1
            episode_returns += reward.item()

        # Store episode information
        obs_rollouts.append(np.concatenate(obs_list,axis=0))
        action_rollouts.append(np.stack(action_list,axis=0))
        rollout_lengths.append(len(obs_list))
        frames_rollouts.append(np.stack(frames,axis=0))
        return_rollouts.append(episode_returns)
        noise_rollouts.append(np.concatenate(noise_list,axis=0))

        # Save video + print episode stats
        video_file = video_path / f"ep_video_{episode}.mp4"
        imageio.mimsave(video_file, frames, fps=30)
        print(f"[Episode {episode+1}] returns={episode_returns,}, steps={step}, saved video to {video_file}")

    env.close()

    # Concatenate each rollouts into a single array 
    obs_rollouts = np.stack(obs_rollouts,axis=0) # (num_episodes, T, obs_dim)
    action_rollouts = np.stack(action_rollouts,axis=0) # (num_episodes, T, action_dim)
    frames_rollouts = np.stack(frames_rollouts,axis=0) # (num_episodes, T, H, W, C)
    return_rollouts = np.array(return_rollouts) # (num_episodes,)
    noise_rollouts = np.stack(noise_rollouts,axis=0) # (num_episodes, T, init_x_size)

    # Save the observations, frames, and returns
    np.save(video_path / "obs_rollouts.npy", obs_rollouts)
    np.save(video_path / "action_rollouts.npy", action_rollouts)
    np.save(video_path / "frames_rollouts.npy", frames_rollouts)
    np.save(video_path / "return_rollouts.npy", return_rollouts)
    np.save(video_path / "noise_rollouts.npy", noise_rollouts)
    
    return obs_rollouts, action_rollouts, frames_rollouts, return_rollouts, noise_rollouts

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
        action_rollouts = np.load(rollout_dir / "action_rollouts.npy")
        frames_rollouts = np.load(rollout_dir / "frames_rollouts.npy")
        return_rollouts = np.load(rollout_dir / "return_rollouts.npy")
        noise_rollouts = np.load(rollout_dir / "noise_rollouts.npy")
    else:
        obs_rollouts, action_rollouts, frames_rollouts, return_rollouts, noise_rollouts = collect_rollouts(policy, cfg)

    # ------------------------------------------------------------
    # 2) Sample Y noise vectors
    # ------------------------------------------------------------
    print(f"Sampling {num_noises} noise vectors for variance eval")
    init_x_size = 4 * cfg.evaluation.chunk_size # action space is 4 in frankasim (x,y,z,gripper)
    noise_bank = torch.randn((num_noises, init_x_size), device=device)
    torch.save(noise_bank, video_path / "noise_bank.pt")

    # ------------------------------------------------------------
    # 3) For each episode + observation, compute variance
    # ------------------------------------------------------------
    plot_path = video_path / "variance_plots"
    plot_path.mkdir(parents=True, exist_ok=True)

    all_episode_variances = []

    for ep_idx in range(obs_rollouts.shape[0]): # for each episode
        print(f"Computing variances for episode {ep_idx}/{obs_rollouts.shape[0] - 1}")

        episode_variances = []  # list of (T, action_chunk_size)

        for t in range(obs_rollouts[ep_idx].shape[0]): # for each step in episode
            action_chunks = []
            cur_obs = {'state': np.expand_dims(obs_rollouts[ep_idx][t],axis=0)}
            
            if t % 8 == 0: #only check when chunks were actually generated by the new noise. TODO: make this more elegant.     
                # first, for debugging, check that the action chunk from noise_rollouts matches action_rollouts
                policy.reset()
                with torch.no_grad():
                    action_chunk = policy(cur_obs, init_x=torch.from_numpy(noise_rollouts[ep_idx,t:t+1]).to(device))
                    assert np.allclose(action_chunk, action_rollouts[ep_idx,t:t+1], atol=1e-5), f"Action chunk from noise_rollouts does not match action_rollouts at episode {ep_idx}, step {t}"

            # Now compute action chunks for each noise vector, keeping the observation fixed, and compute variance across noise dimension
            for n in range(num_noises): # for each noise vector
                policy.reset()  # important: reset policy state for fair comparison
                with torch.no_grad():
                    action_chunk = policy(cur_obs, init_x=noise_bank[n:n+1]) # (1, action_chunk_size)
                action_chunks.append(action_chunk) 

            # stack action_chunks so shape is (num_noises, action_chunk_size), then compute variance across noise dimension
            action_chunks = np.stack(action_chunks, axis=0)
            var_t = np.var(action_chunks, axis=0)  # per action-dimension variance
            
            if np.max(var_t).item() >= 1.0:
                print("variance is really high...")
        
            episode_variances.append(var_t)


        episode_variances = np.stack(episode_variances, axis=0)  # (T, D)
        all_episode_variances.append(episode_variances)

        np.save(plot_path / f"episode_{ep_idx}_variance.npy", episode_variances)
        print(f"Saved episode {ep_idx} variances to {plot_path / f'episode_{ep_idx}_variance.npy'}")

    # ------------------------------------------------------------
    # 4) Plot variance over time for each episode + action dim
    # ------------------------------------------------------------
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

    print("Variance evaluation complete!")


@hydra.main(version_base=None, config_path="cfgs", config_name="fm")
def main(cfg: DictConfig):
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    eval_variance(cfg)


if __name__ == "__main__":
    main()
