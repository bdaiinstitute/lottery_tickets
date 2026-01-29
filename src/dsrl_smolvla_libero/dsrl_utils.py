import os
import json

import hydra
import numpy as np
import torch
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from lerobot.policies.factory import make_policy


def load_libero_policy(cfg):
    """Load SmolVLA policy for libero using lerobot's make_policy."""
    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
        rename_map=getattr(cfg, "rename_map", {}),
    )
    policy.eval()
    print(f">>> Loaded policy from: {cfg.policy.pretrained_path}")
    print(f">>> Policy device: {policy.config.device}")
    return policy


class LoggingCallback(BaseCallback):
    def __init__(
        self,
        action_chunk=50,
        log_freq=1000,
        use_wandb=True,
        record_video=False,
        video_fps=20,
        video_every_eval=10,
        eval_env=None,
        eval_freq=70,
        eval_episodes=2,
        verbose=1,
        rew_offset=0,
        num_train_env=1,
        num_eval_env=1,
        algorithm="dsrl_sac",
        max_steps=-1,
    ):
        super().__init__(verbose)
        self.action_chunk = action_chunk
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.use_wandb = use_wandb
        self.record_video = record_video
        self.video_fps = video_fps
        self.video_every_eval = video_every_eval
        self.eval_env = eval_env
        self.eval_episodes = eval_episodes
        self.eval_freq = eval_freq
        self.log_count = 0
        self.total_reward = 0
        self.rew_offset = rew_offset
        self.total_timesteps = 0
        self.num_train_env = num_train_env
        self.num_eval_env = num_eval_env
        self.episode_success = np.zeros(self.num_train_env)
        self.episode_completed = np.zeros(self.num_train_env)
        self.algorithm = algorithm
        self.max_steps = max_steps

    def _on_step(self):
        for info in self.locals["infos"]:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
        rew = self.locals["rewards"]
        self.total_reward += np.mean(rew)
        self.episode_success[rew > -self.rew_offset] = 1
        self.episode_completed[self.locals["dones"]] = 1
        self.total_timesteps += self.action_chunk * self.model.n_envs
        if self.n_calls % self.log_freq == 0:
            if len(self.episode_rewards) > 0:
                if self.use_wandb:
                    self.log_count += 1
                    wandb.log(
                        {
                            "train/ep_len_mean": np.mean(self.episode_lengths),
                            "train/success_rate": np.sum(self.episode_success)
                            / np.sum(self.episode_completed),
                            "train/ep_rew_mean": np.mean(self.episode_rewards),
                            "train/rew_mean": np.mean(self.total_reward),
                            "train/timesteps": self.total_timesteps,
                            "train/ent_coef": self.locals["self"].logger.name_to_value[
                                "train/ent_coef"
                            ],
                            "train/actor_loss": self.locals[
                                "self"
                            ].logger.name_to_value["train/actor_loss"],
                            "train/critic_loss": self.locals[
                                "self"
                            ].logger.name_to_value["train/critic_loss"],
                            "train/ent_coef_loss": self.locals[
                                "self"
                            ].logger.name_to_value["train/ent_coef_loss"],
                        },
                        step=self.log_count,
                    )
                    if np.sum(self.episode_completed) > 0:
                        wandb.log(
                            {
                                "train/success_rate": np.sum(self.episode_success)
                                / np.sum(self.episode_completed),
                            },
                            step=self.log_count,
                        )
                self.episode_rewards = []
                self.episode_lengths = []
                self.total_reward = 0
                self.episode_success = np.zeros(self.num_train_env)
                self.episode_completed = np.zeros(self.num_train_env)

        if self.n_calls % self.eval_freq == 0:
            self.evaluate(self.locals["self"])
        return True

    def evaluate(self, agent):
        if self.eval_episodes > 0:
            env = self.eval_env
            with torch.no_grad():
                success, rews = [], []
                rew_total, total_ep = 0, 0
                rew_ep = np.zeros(self.num_eval_env)

                should_log_video = (
                    self.use_wandb
                    and self.record_video
                    and (self.log_count % self.video_every_eval == 0)
                )
                video_frames = []
                for i in range(self.eval_episodes):
                    obs = env.reset()
                    if isinstance(obs, dict):
                        n_envs = obs[list(obs.keys())[0]].shape[0]
                    else:
                        n_envs = obs.shape[0]
                    success_i = np.zeros(n_envs)
                    r = []
                    for _ in range(self.max_steps):
                        action, _ = agent.predict(obs, deterministic=True)
                        next_obs, reward, done, info = env.step(action)
                        obs = next_obs

                        if should_log_video and i == 0 and isinstance(obs, dict):
                            for v in obs.values():
                                if (
                                    isinstance(v, np.ndarray)
                                    and v.dtype == np.uint8
                                    and v.ndim == 4
                                ):
                                    video_frames.append(v[0])
                                    break

                        rew_ep += reward
                        rew_total += sum(rew_ep[done])
                        rew_ep[done] = 0
                        total_ep += np.sum(done)
                        success_i[reward > -self.rew_offset] = 1
                        r.append(reward)
                    success.append(success_i.mean())
                    rews.append(np.mean(np.array(r)))
                    print(f"eval episode {i} at timestep {self.total_timesteps}")
                success_rate = np.mean(success)
                if total_ep > 0:
                    avg_rew = rew_total / total_ep
                else:
                    avg_rew = 0
                if self.use_wandb:
                    payload = {
                        "eval/success_rate": success_rate,
                        "eval/reward": avg_rew,
                        "eval/timesteps": self.total_timesteps,
                    }
                    if should_log_video and len(video_frames) > 0:
                        payload["eval/video"] = wandb.Video(
                            np.stack(video_frames, axis=0),
                            fps=self.video_fps,
                            format="gif",
                        )
                    wandb.log(payload, step=self.log_count)

    def set_timesteps(self, timesteps):
        self.total_timesteps = timesteps


def collect_rollouts(model, env, num_steps, base_policy, cfg):
    """Collect initial rollouts using random noise for DSRL training.
    Fixed DSRL bugs here:
    env is wrapped by SmolVLAPolicyEnvWrapper which expects noise as actions.
    The wrapper internally calls base_policy.predict_action_chunk(obs, noise).

    If cfg.noise_shrink is True, noise is sampled only for action_dim and replicated
    across the chunk dimension, reducing the effective noise space.
    """
    obs = env.reset()
    total_episodes = 0
    # Use chunk_size for the full action chunk (not n_action_steps which is for execution)
    chunk_size = base_policy.config.chunk_size
    action_dim = base_policy.config.max_action_dim

    for i in range(num_steps):
        # Generate random noise in the action space expected by SAC
        if cfg.noise_shrink:
            noise = torch.randn(cfg.n_envs, action_dim).to(device=cfg.device)
            noise = noise.clamp(-cfg.train.action_magnitude, cfg.train.action_magnitude)
            noise_flat = noise.cpu().numpy()
        else:
            noise = torch.randn(cfg.n_envs, chunk_size, action_dim).to(
                device=cfg.device
            )
            noise = noise.clamp(-cfg.train.action_magnitude, cfg.train.action_magnitude)
            noise_flat = noise.reshape(cfg.n_envs, -1).cpu().numpy()
        # Scale to SAC's action space (the wrapper will unscale it)
        action_store = model.policy.scale_action(noise_flat)

        # Step the environment (SmolVLAPolicyEnvWrapper handles the rest)
        next_obs, reward, done, info = env.step(action_store)

        # Store in replay buffer
        model.replay_buffer.add(
            obs=obs,
            next_obs=next_obs,
            action=action_store,
            reward=reward,
            done=done,
            infos=info,
        )
        # Count completed episodes
        total_episodes += done.sum()
        obs = next_obs
    model.replay_buffer.final_offline_step()

    # Return total steps and episodes collected
    total_steps = num_steps * cfg.n_envs * chunk_size
    return total_steps, int(total_episodes)


class CheckpointCallbackOnEpisodes(BaseCallback):
    """Callback for saving a model at specific episode milestones (1k, 5k, 10k).
    IMP: this callback considers each action in a chunk as different timestep.
    """

    def __init__(
        self,
        save_path: str,
        action_chunk: int = 4,
        name_prefix: str = "rl_model",
        save_replay_buffer: bool = False,
        save_vecnormalize: bool = False,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_path = save_path
        self.action_chunk = action_chunk
        self.name_prefix = name_prefix
        self.save_replay_buffer = save_replay_buffer
        self.save_vecnormalize = save_vecnormalize
        self.checkpoint_episodes = [0, 1000, 5000, 10000, 20_000]
        self.saved_checkpoints = set()
        self.initial_steps = 0  # same as used by LoggingCallback
        self.initial_episodes = 0

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def set_initial_counters(self, steps: int, episodes: int) -> None:
        """Set initial step and episode counters from pre-training rollouts."""
        self.initial_steps = steps
        self.initial_episodes = episodes
        if self.verbose >= 1:
            print(f"Set initial counters: {steps} steps, {episodes} episodes")

    def _checkpoint_path(
        self,
        episode_num: int,
        timesteps: int,
        checkpoint_type: str = "",
        extension: str = "",
    ) -> str:
        """Helper to get checkpoint path for each type of checkpoint."""
        return os.path.join(
            self.save_path,
            f"{self.name_prefix}_{checkpoint_type}{episode_num}ep_{timesteps}steps.{extension}",
        )

    def _serialize_locals(self, locals_dict: dict) -> dict:
        """Convert locals to a JSON-serializable format."""
        serializable = {}
        for key, value in locals_dict.items():
            if isinstance(value, (int, float, str, bool, type(None))):
                serializable[key] = value
        return serializable

    def _on_step(self) -> bool:
        # Add initial counters to current counts
        current_episode_num = self.model._episode_num + self.initial_episodes
        # IMP: this considers a chunked action as a single timestep
        current_timesteps = (
            self.model.num_timesteps * self.action_chunk
        ) + self.initial_steps

        # Check if we've reached a checkpoint milestone
        for checkpoint_ep in self.checkpoint_episodes:
            if (
                current_episode_num >= checkpoint_ep
                and checkpoint_ep not in self.saved_checkpoints
            ):
                self.saved_checkpoints.add(checkpoint_ep)

                model_path = self._checkpoint_path(
                    checkpoint_ep, current_timesteps, extension="zip"
                )
                self.model.save(model_path)
                if self.verbose >= 1:
                    print(f"Saving model checkpoint to {model_path}")

                # Save locals to JSON
                locals_path = self._checkpoint_path(
                    checkpoint_ep, current_timesteps, extension="json"
                )
                serializable_locals = self._serialize_locals(self.locals)
                serializable_locals["_episode_num"] = current_episode_num
                serializable_locals["_num_timesteps"] = current_timesteps

                with open(locals_path, "w") as f:
                    json.dump(serializable_locals, f, indent=2)
                if self.verbose >= 1:
                    print(f"Saving locals to {locals_path}")

                if (
                    self.save_replay_buffer
                    and hasattr(self.model, "replay_buffer")
                    and self.model.replay_buffer is not None
                ):
                    # If model has a replay buffer, save it too
                    replay_buffer_path = self._checkpoint_path(
                        checkpoint_ep,
                        current_timesteps,
                        "replay_buffer_",
                        extension="pkl",
                    )
                    self.model.save_replay_buffer(replay_buffer_path)  # type: ignore[attr-defined]
                    if self.verbose >= 1:
                        print(
                            f"Saving model replay buffer checkpoint to {replay_buffer_path}"
                        )

                if (
                    self.save_vecnormalize
                    and self.model.get_vec_normalize_env() is not None
                ):
                    # Save the VecNormalize statistics
                    vec_normalize_path = self._checkpoint_path(
                        checkpoint_ep,
                        current_timesteps,
                        "vecnormalize_",
                        extension="pkl",
                    )
                    self.model.get_vec_normalize_env().save(vec_normalize_path)  # type: ignore[union-attr]
                    if self.verbose >= 1:
                        print(f"Saving model VecNormalize to {vec_normalize_path}")

        return True
