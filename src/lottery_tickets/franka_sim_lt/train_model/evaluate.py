# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from pathlib import Path

import gymnasium as gym
import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from squirl_launcher.common.evaluation_ts import evaluate
from squirl_launcher.experimental.fm_utils import FMPolicyInterface, load_fm_model
from squirl_launcher.utils.launcher_ts import build_env


def evaluate_fm_policy(cfg: DictConfig):
    """Evaluate Flow Matching policy in the environment."""

    # Set device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the trained FM model
    print(f"Loading model from {cfg.evaluation.model_path}")
    fm_model, config = load_fm_model(cfg.evaluation.model_path, device)
    print(f"Model config: {config}")

    # Create policy interface
    policy = FMPolicyInterface(
        fm_model,
        cfg.evaluation.chunk_size,
        device,
        use_state_history=cfg.evaluation.use_state_history,
        state_history_length=cfg.evaluation.state_history_length,
    )

    # Build environment
    print(f"Building environment: {cfg.evaluation.env_name}")
    try:
        # Try to build environment using squirl launcher
        env = build_env(cfg.evaluation.env_name, env_kwargs=cfg.evaluation.env_kwargs)
    except Exception as e:
        print(f"Failed to build environment with squirl launcher: {e}")
        print("Falling back to direct gym environment creation")
        env = gym.make(cfg.evaluation.env_name)

    print(f"Environment action space: {env.action_space}")
    print(f"Environment observation space: {env.observation_space}")

    # Create video save directory
    video_path = Path(cfg.evaluation.video_save_path)
    video_path.mkdir(parents=True, exist_ok=True)

    # Run evaluation
    print(f"Starting evaluation: {cfg.evaluation.num_episodes} episodes")
    eval_stats = evaluate(
        policy_fn=policy,
        env=env,
        num_episodes=cfg.evaluation.num_episodes,
        train_ep_idx=0,  # We're just evaluating, not training
        save_path=video_path,
        max_steps_per_episode=cfg.evaluation.max_steps_per_episode,
        device=device,
        reset_fn=policy.reset,
    )

    # Print results
    print("\n=== Evaluation Results ===")
    print(f"final.episode.r: {eval_stats['final.episode.r']}")
    print(f"\nVideos saved to: {video_path}")

    return eval_stats


@hydra.main(version_base=None, config_path="cfgs", config_name="fm")
def main(cfg: DictConfig):
    """Main evaluation script with Hydra configuration."""

    # Print configuration
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Run evaluation
    evaluate_fm_policy(cfg)

    print("Evaluation complete!")


if __name__ == "__main__":
    main()
