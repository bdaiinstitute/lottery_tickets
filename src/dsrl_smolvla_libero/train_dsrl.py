"""
The intent of tis code is to generate the Libero results
on smolvla using SB3 stack. This closely follows DSRL
implementation for robomimic, and differs in quite a few ways from
their Pi0 implementation. For example:
- We don't use LearnedStdTanhNormalPolicy as the policy class
- Don't create a bottleneck in feature encoding passed to the actor
- Finally since robomimic was state-dim, we use small cnn to encode images

Libero wrappers: `AsyncVectorEnv [gym] <- LiberoEnv(p: gym.Env) [lerobot] <- OffScreenRenderEnv(p: ControlEnv) <- BDDLBaseEnv [Libero] <- SingleArmEnv <- ManipulatorEnv <- RobotEnv <- MujocoEnv [Robosuite]`
Reward structure: libero (BDDLBaseEnv) overrides done on success check and sets reward as 1 on success, 0 otherwise.
Done/terminated/truncated: terminated is set based on is_success or done signal from libero. truncated is always set to False
"""

import math
import os
import random
import sys
import warnings
import multiprocessing as mp
from dataclasses import asdict

import numpy as np
import torch
import wandb
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from feature_extractors import LiberoFeatureExtractor

from lerobot.configs import parser

warnings.filterwarnings("ignore")

# Set up multiprocessing for libero/MuJoCo
os.environ["MUJOCO_GL"] = "egl"
if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set

base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_path)
sys.path.append(os.path.join(base_path, "dppo"))

# os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_INIT_TIMEOUT"] = "600"
os.environ["WANDB_START_METHOD"] = "thread"

from env_utils import (
    ActionChunkWrapper,
    SmolVLAPolicyEnvWrapper,
    make_libero_env,
)
from dsrl_smolvla_libero.dsrl_utils import (
    LoggingCallback,
    CheckpointCallbackOnEpisodes,
    collect_rollouts,
    load_libero_policy,
)
from dsrl_smolvla_libero.config_dsrl_libero import DSRLLiberoConfig


@parser.wrap()
def main(cfg: DSRLLiberoConfig):

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Load SmolVLA
    base_policy = load_libero_policy(cfg)
    print(f"Using policy: {cfg.policy.pretrained_path}")

    if cfg.use_wandb:
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.name,
            group=cfg.wandb.group,
            monitor_gym=True,
            save_code=True,
            config=asdict(cfg),
        )

    ep_len = int(cfg.max_episode_steps / cfg.policy.n_action_steps)
    num_env = cfg.n_envs

    def make_env():
        env = make_libero_env(
            task=cfg.env.task,
            task_id=getattr(cfg.env, "task_id", 0),
            render=False,
        )
        env = ActionChunkWrapper(env, cfg, max_episode_steps=cfg.max_episode_steps)
        return env

    env = make_vec_env(make_env, n_envs=num_env, vec_env_cls=SubprocVecEnv)
    if cfg.algorithm == "dsrl_sac":
        env = SmolVLAPolicyEnvWrapper(env, cfg, base_policy)
    env.seed(cfg.seed + 1)
    post_linear_modules = None
    if cfg.train.use_layer_norm:
        post_linear_modules = [torch.nn.LayerNorm]

    net_arch = []
    for _ in range(cfg.train.num_layers):
        net_arch.append(cfg.train.layer_size)
    policy_kwargs = dict(
        features_extractor_class=LiberoFeatureExtractor,
        features_extractor_kwargs=dict(
            features_dim=256,
            state_mlp_layers=[256, 256],
            cnn_features_dim=128,
            activation_fn=torch.nn.ReLU,
        ),
        net_arch=dict(pi=net_arch, qf=net_arch),
        activation_fn=torch.nn.Tanh,
        log_std_init=0.0,
        post_linear_modules=post_linear_modules,
        n_critics=cfg.train.n_critics,
    )
    model = SAC(
        "MultiInputPolicy",  # Changed from MlpPolicy to handle dict observations
        env,
        learning_rate=cfg.train.actor_lr,
        buffer_size=20000000,
        learning_starts=1,
        batch_size=cfg.train.batch_size,
        tau=cfg.train.tau,
        gamma=cfg.train.discount,
        train_freq=cfg.train.train_freq,
        gradient_steps=cfg.train.utd,
        action_noise=None,
        optimize_memory_usage=False,
        ent_coef="auto" if cfg.train.ent_coef == -1 else cfg.train.ent_coef,
        target_update_interval=1,
        target_entropy=(
            "auto" if cfg.train.target_ent == -1 else cfg.train.target_ent
        ),
        use_sde=False,
        sde_sample_freq=-1,
        tensorboard_log=cfg.logdir,
        verbose=2,
        policy_kwargs=policy_kwargs,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=cfg.save_model_interval // num_env,
        save_path=str(cfg.logdir / "checkpoint"),
        name_prefix="ft_policy",
        save_replay_buffer=cfg.save_replay_buffer,
        save_vecnormalize=True,
    )

    checkpoint_callback_on_episodes = CheckpointCallbackOnEpisodes(
        save_path=str(cfg.logdir / "checkpoint"),
        action_chunk=cfg.policy.n_action_steps,
        name_prefix="ft_policy",
        save_replay_buffer=True,
        save_vecnormalize=True,
        verbose=1,
    )

    num_env_eval = cfg.n_eval_envs
    eval_env = make_vec_env(make_env, n_envs=num_env_eval, vec_env_cls=SubprocVecEnv)
    if cfg.algorithm == "dsrl_sac":
        eval_env = SmolVLAPolicyEnvWrapper(eval_env, cfg, base_policy)
    eval_env.seed(cfg.seed + num_env + 1)

    logging_callback = LoggingCallback(
        action_chunk=cfg.policy.n_action_steps,
        eval_episodes=int(cfg.num_evals / num_env_eval),
        log_freq=ep_len,
        use_wandb=cfg.use_wandb,
        eval_env=eval_env,
        eval_freq=cfg.eval_interval,
        num_train_env=num_env,
        num_eval_env=num_env_eval,
        rew_offset=cfg.reward_offset,
        algorithm=cfg.algorithm,
        max_steps=ep_len,
    )
    logging_callback.evaluate(model)
    logging_callback.log_count += 1

    # this is used with respect to each environment and considers the whole chunk as a single step
    if cfg.train.init_rollout_steps > 0:
        # this return total steps considering each action in chunk as different timestep
        init_steps, init_episodes = collect_rollouts(
            model, env, cfg.train.init_rollout_steps, base_policy, cfg
        )
        logging_callback.set_timesteps(init_steps)
        checkpoint_callback_on_episodes.set_initial_counters(init_steps, init_episodes)

    callbacks = [checkpoint_callback, checkpoint_callback_on_episodes, logging_callback]
    model.learn(total_timesteps=20_000_000, callback=callbacks)

    if len(cfg.name) > 0:
        model.save(str(cfg.logdir / "checkpoint" / "final"))

    env.close()
    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
