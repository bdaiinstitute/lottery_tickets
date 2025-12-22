# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from pathlib import Path

import hydra
import imageio
import torch
from omegaconf import DictConfig, OmegaConf

from lottery_tickets.franka_sim_lt.gym_utils import make_frankasim_env
from lottery_tickets.franka_sim_lt.models_utils import FMPolicyInterface, load_fm_model
import numpy as np

def evaluate_fm_policy(cfg: DictConfig) -> None:
    """
    Evaluates a Flow Matching policy in the environment.
    
    Args:
        cfg: The policy configuration dictionary.
    """

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
    env = make_frankasim_env(
        cfg.evaluation.env_name, env_kwargs=cfg.evaluation.env_kwargs
    )

    print(f"Environment action space: {env.action_space}")
    print(f"Environment observation space: {env.observation_space}")

    # Create video save directory
    video_path = Path(cfg.evaluation.video_save_path)
    video_path.mkdir(parents=True, exist_ok=True)

    # Check if running original policy, with a new lottery ticket, or a saved lottery ticket
    # check if origianl policy
    if cfg.get("new_noise", False):
        init_x_size = env.action_space.shape[0] * cfg.evaluation.chunk_size
        print("Evaluating new lottery ticket with new noise initialization.")
        init_x = torch.randn((1, init_x_size), device=cfg.device)
    elif cfg.get("noise_path", False):
        init_x = torch.load(cfg.noise_path)
        print(f"Evaluating {cfg.noise_path}.")
    elif cfg.get("original_policy", False):
        init_x = None
        print("Evaluating original policy without lottery ticket.")
    else:
        raise RuntimeError("Need to either test original policy, test a new ticket, or eval an existing ticket")

    # Evaluate policy
    num_episodes = cfg.evaluation.num_episodes
    total_reward_list = []
    for episode in range(num_episodes):
        policy.reset()
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        step = 0
        frames = []

        while not done:
            action = policy(obs, init_x=init_x)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1
            frames.append(env.render())

        total_reward_list.append(total_reward)
        print(
            f"Episode {episode + 1}/{num_episodes} - Steps: {step}, Total Reward: {total_reward:.2f}"
        )
        video_file = video_path / f"ep_video_{episode}.mp4"
        print(f"saved video to : {video_file}")
        imageio.mimsave(video_file, frames, fps=30)

    avg_reward = sum(total_reward_list) / num_episodes
    print(f"Average Reward over {num_episodes} episodes: {avg_reward:.2f}")

    # turn the total_reward_list into a numpy array and save it to file
    total_reward_list_np = np.array(total_reward_list)
    np.save(video_path.parent / "total_reward_list.npy", total_reward_list_np)

    if 'new_noise' in cfg.keys() or 'noise_path' in cfg.keys():
        torch.save(init_x, video_path.parent / "init_x.pt")

    env.close()


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
