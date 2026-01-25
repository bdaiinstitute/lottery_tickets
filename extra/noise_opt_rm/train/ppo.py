"""PPO training with discrete noise candidates relocated under train/ ."""
import math
import os
import random
import sys
import warnings

import gym
import hydra
import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

warnings.filterwarnings("ignore")

base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_path)
sys.path.append(os.path.join(base_path, 'dppo'))

# os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_INIT_TIMEOUT"] = "600"
os.environ["WANDB_START_METHOD"] = "thread"

from noise_opt_rm.utils.env_utils import (ActionChunkWrapper, DiffusionPolicyEnvWrapper,
                                     ObservationWrapperGym, ObservationWrapperRobomimic,
                                     make_robomimic_env)
from noise_opt_rm.train.train_utils.ppo_callbacks import PPONoiseLoggingCallback
from noise_opt_rm.train.train_utils.utils import LoggingCallback, build_noise_library, load_base_policy

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)

@hydra.main(config_path=os.path.join(base_path, "cfg/robomimic/ppo"), config_name="can.yaml", version_base=None)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if cfg.use_wandb:
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.name,
            group=cfg.wandb.group,
            monitor_gym=True,
            save_code=True,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    MAX_STEPS = int(cfg.env.max_episode_steps / cfg.act_steps)

    def make_env():
        if cfg.env_name in ['halfcheetah-medium-v2', 'hopper-medium-v2', 'walker2d-medium-v2']:
            env = gym.make(cfg.env_name)
            env = ObservationWrapperGym(env, cfg.normalization_path)
        elif cfg.env_name in ['lift', 'can', 'square', 'transport']:
            env = make_robomimic_env(env=cfg.env_name, normalization_path=cfg.normalization_path, low_dim_keys=cfg.env.wrappers.robomimic_lowdim.low_dim_keys, dppo_path=cfg.dppo_path)
            env = ObservationWrapperRobomimic(env, reward_offset=cfg.env.reward_offset)
        env = ActionChunkWrapper(env, cfg, max_episode_steps=cfg.env.max_episode_steps)
        return env

    base_policy = load_base_policy(cfg)
    noise_library = build_noise_library(cfg)
    if noise_library is not None:
        os.makedirs(cfg.logdir, exist_ok=True)
        np.save(os.path.join(cfg.logdir, "noise_library.npy"), noise_library)
        print(f">>> Saved noise library to {os.path.join(cfg.logdir, 'noise_library.npy')}")

    env = make_vec_env(make_env, n_envs=cfg.env.n_envs, vec_env_cls=SubprocVecEnv)
    env = DiffusionPolicyEnvWrapper(env, cfg, base_policy, noise_library=noise_library)
    env.seed(cfg.seed + 1)

    net_arch = [cfg.train.layer_size for _ in range(cfg.train.num_layers)]
    policy_kwargs = dict(
        net_arch=dict(pi=net_arch, vf=net_arch),
        activation_fn=torch.nn.Tanh,
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=cfg.train.learning_rate,
        n_steps=cfg.train.n_steps,
        batch_size=cfg.train.batch_size,
        n_epochs=cfg.train.n_epochs,
        gamma=cfg.train.gamma,
        gae_lambda=cfg.train.gae_lambda,
        clip_range=cfg.train.clip_range,
        ent_coef=cfg.train.ent_coef,
        vf_coef=cfg.train.vf_coef,
        max_grad_norm=cfg.train.max_grad_norm,
        device="cpu",  # explicit device as in reference ppo.py
        tensorboard_log=cfg.logdir,
        verbose=2,
        policy_kwargs=policy_kwargs,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=cfg.save_model_interval//cfg.env.n_envs,
        save_path=cfg.logdir + '/checkpoint/',
        name_prefix='ft_policy',
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    eval_env = make_vec_env(make_env, n_envs=cfg.env.n_eval_envs, vec_env_cls=SubprocVecEnv)
    eval_env = DiffusionPolicyEnvWrapper(eval_env, cfg, base_policy, noise_library=noise_library)
    eval_env.seed(cfg.seed + cfg.env.n_envs + 1)

    logging_callback = PPONoiseLoggingCallback(
        action_chunk=cfg.act_steps,
        eval_episodes=int(cfg.num_evals / cfg.env.n_eval_envs),
        log_freq=MAX_STEPS,
        use_wandb=cfg.use_wandb,
        eval_env=eval_env,
        eval_freq=cfg.eval_interval,
        num_train_env=cfg.env.n_envs,
        num_eval_env=cfg.env.n_eval_envs,
        rew_offset=cfg.env.reward_offset,
        algorithm='ppo',
        max_steps=MAX_STEPS,
        deterministic_eval=cfg.deterministic_eval,
    )

    logging_callback.evaluate(model, deterministic=False)
    if cfg.deterministic_eval:
        logging_callback.evaluate(model, deterministic=True)
    logging_callback.log_count += 1

    callbacks = [checkpoint_callback, logging_callback]
    total_timesteps = cfg.train.total_timesteps if hasattr(cfg.train, 'total_timesteps') else 20_000_000
    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    if len(cfg.name) > 0:
        model.save(cfg.logdir + "/checkpoint/final")

    env.close()
    if cfg.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()