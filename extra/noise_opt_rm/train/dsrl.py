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
from stable_baselines3 import DSRL, SAC
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
                                     ObservationWrapperRobomimic, make_robomimic_env)
from noise_opt_rm.train.train_utils.utils import (LoggingCallback, CheckpointCallbackOnEpisodes, collect_rollouts,
                                                   load_base_policy, load_offline_data)

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)

@hydra.main(
    config_path=os.path.join(base_path, "cfg/robomimic"), config_name="dsrl_can.yaml", version_base=None
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Log DDIM steps configuration
    print(f"Using DDIM steps: {cfg.model.ddim_steps}")

    if cfg.use_wandb:
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.name,
            group=cfg.wandb.group,
            monitor_gym=True,
            save_code=True,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wandb.config.update({"ddim_steps": cfg.model.ddim_steps})

    MAX_STEPS = int(cfg.env.max_episode_steps / cfg.act_steps)

    num_env = cfg.env.n_envs
    def make_env():
        env = make_robomimic_env(env=cfg.env_name, normalization_path=cfg.normalization_path, low_dim_keys=cfg.env.wrappers.robomimic_lowdim.low_dim_keys, dppo_path=cfg.dppo_path)
        env = ObservationWrapperRobomimic(env, reward_offset=cfg.env.reward_offset)
        env = ActionChunkWrapper(env, cfg, max_episode_steps=cfg.env.max_episode_steps)
        return env

    base_policy = load_base_policy(cfg)
    env = make_vec_env(make_env, n_envs=num_env, vec_env_cls=SubprocVecEnv)
    if cfg.algorithm == 'dsrl_sac':
        env = DiffusionPolicyEnvWrapper(env, cfg, base_policy)
    env.seed(cfg.seed + 1)
    post_linear_modules = None
    if cfg.train.use_layer_norm:
        post_linear_modules = [torch.nn.LayerNorm]

    net_arch = []
    for _ in range(cfg.train.num_layers):
        net_arch.append(cfg.train.layer_size)
    policy_kwargs = dict(
        net_arch=dict(pi=net_arch, qf=net_arch),
        activation_fn=torch.nn.Tanh,
        log_std_init=0.0,
        post_linear_modules=post_linear_modules,
        n_critics=cfg.train.n_critics,
    )
    if cfg.algorithm == 'dsrl_sac':
        model = SAC(
            "MlpPolicy",
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
            target_entropy="auto" if cfg.train.target_ent == -1 else cfg.train.target_ent,
            use_sde=False,
            sde_sample_freq=-1,
            tensorboard_log=cfg.logdir,
            verbose=2,
            policy_kwargs=policy_kwargs,
        )
    elif cfg.algorithm == 'dsrl_na':
        model = DSRL(
            "MlpPolicy",
            env,
            learning_rate=cfg.train.actor_lr,
            buffer_size=10000000,
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
            target_entropy="auto" if cfg.train.target_ent == -1 else cfg.train.target_ent,
            use_sde=False,
            sde_sample_freq=-1,
            tensorboard_log=cfg.logdir,
            verbose=2,
            policy_kwargs=policy_kwargs,
            diffusion_policy=base_policy,
            diffusion_act_dim=(cfg.act_steps, cfg.action_dim),
            noise_critic_grad_steps=cfg.train.noise_critic_grad_steps,
            critic_backup_combine_type=cfg.train.critic_backup_combine_type,
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=cfg.save_model_interval//num_env,
        save_path=cfg.logdir+'/checkpoint/',
        name_prefix='ft_policy',
        save_replay_buffer=cfg.save_replay_buffer,
        save_vecnormalize=True,
    )

    checkpoint_callback_on_episodes = CheckpointCallbackOnEpisodes(
        save_path=cfg.logdir+'/checkpoint/',
        action_chunk=cfg.act_steps,
        name_prefix='ft_policy',
        save_replay_buffer=True,
        save_vecnormalize=True,
        verbose=1,
    )

    num_env_eval = cfg.env.n_eval_envs
    eval_env = make_vec_env(make_env, n_envs=num_env_eval, vec_env_cls=SubprocVecEnv)
    if cfg.algorithm == 'dsrl_sac':
        eval_env = DiffusionPolicyEnvWrapper(eval_env, cfg, base_policy)
    eval_env.seed(cfg.seed + num_env + 1)

    logging_callback = LoggingCallback(
        action_chunk = cfg.act_steps,
        eval_episodes = int(cfg.num_evals / num_env_eval),
        log_freq=MAX_STEPS,
        use_wandb=cfg.use_wandb,
        eval_env=eval_env,
        eval_freq=cfg.eval_interval,
        num_train_env=num_env,
        num_eval_env=num_env_eval,
        rew_offset=cfg.env.reward_offset,
        algorithm=cfg.algorithm,
        max_steps=MAX_STEPS,
        deterministic_eval=cfg.deterministic_eval,
    )

    logging_callback.evaluate(model, deterministic=False)
    if cfg.deterministic_eval:
        logging_callback.evaluate(model, deterministic=True)
    logging_callback.log_count += 1

    # this is used with respect to each environment and considers the whole chunk as a single step 
    if cfg.train.init_rollout_steps > 0:
        # this return total steps considering each action in chunk as different timestep
        init_steps, init_episodes = collect_rollouts(model, env, cfg.train.init_rollout_steps, base_policy, cfg)
        logging_callback.set_timesteps(init_steps)
        checkpoint_callback_on_episodes.set_initial_counters(init_steps, init_episodes)

    callbacks = [checkpoint_callback, checkpoint_callback_on_episodes, logging_callback]
    model.learn(
        total_timesteps=20_000_000,
        callback = callbacks
    )

    if len(cfg.name) > 0:
        model.save(cfg.logdir+"/checkpoint/final")

    env.close()
    if cfg.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()