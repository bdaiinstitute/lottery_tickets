import numpy as np
import torch
import wandb
from stable_baselines3.common.callbacks import BaseCallback

class PPONoiseLoggingCallback(BaseCallback):
    """Separate logging callback for PPO so original DSRL callback remains untouched."""
    def __init__(self,
                 action_chunk=4,
                 log_freq=1000,
                 use_wandb=True,
                 eval_env=None,
                 eval_freq=70,
                 eval_episodes=2,
                 verbose=1,
                 rew_offset=0,
                 num_train_env=1,
                 num_eval_env=1,
                 algorithm='ppo',
                 max_steps=-1,
                 deterministic_eval=False):
        super().__init__(verbose)
        self.action_chunk = action_chunk
        self.log_freq = log_freq
        self.use_wandb = use_wandb
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.log_count = 0
        self.rew_offset = rew_offset
        self.total_timesteps = 0
        self.num_train_env = num_train_env
        self.num_eval_env = num_eval_env
        self.algorithm = algorithm
        self.max_steps = max_steps
        self.deterministic_eval = deterministic_eval
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_success = np.zeros(self.num_train_env)
        self.episode_completed = np.zeros(self.num_train_env)
        self.total_reward = 0

    def _on_step(self):
        for info in self.locals['infos']:
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                self.episode_lengths.append(info['episode']['l'])
        rew = self.locals['rewards']
        self.total_reward += np.mean(rew)
        self.episode_success[rew > -self.rew_offset] = 1
        self.episode_completed[self.locals['dones']] = 1
        self.total_timesteps += self.action_chunk * self.model.n_envs

        if self.n_calls % self.log_freq == 0 and len(self.episode_rewards) > 0:
            if self.use_wandb:
                self.log_count += 1
                payload = {
                    'train/ep_len_mean': np.mean(self.episode_lengths),
                    'train/success_rate': np.sum(self.episode_success) / max(np.sum(self.episode_completed), 1),
                    'train/ep_rew_mean': np.mean(self.episode_rewards),
                    'train/rew_mean': np.mean(self.total_reward),
                    'train/timesteps': self.total_timesteps,
                }
                logger_values = getattr(self.locals['self'].logger, 'name_to_value', {})
                for k in ['train/value_loss','train/approx_kl','train/clip_fraction','train/entropy_loss','train/explained_variance']:
                    if k in logger_values:
                        payload[k] = logger_values[k]
                wandb.log(payload, step=self.log_count)
            self.episode_rewards = []
            self.episode_lengths = []
            self.total_reward = 0
            self.episode_success = np.zeros(self.num_train_env)
            self.episode_completed = np.zeros(self.num_train_env)

        if self.n_calls % self.eval_freq == 0:
            self.evaluate(self.locals['self'], deterministic=False)
            if self.deterministic_eval:
                self.evaluate(self.locals['self'], deterministic=True)
        return True

    def evaluate(self, agent, deterministic=False):
        if self.eval_episodes <= 0:
            return
        env = self.eval_env
        success, rews = [], []
        rew_total, total_ep = 0, 0
        rew_ep = np.zeros(self.num_eval_env)
        with torch.no_grad():
            for i in range(self.eval_episodes):
                obs = env.reset()
                success_i = np.zeros(obs.shape[0])
                r = []
                for _ in range(self.max_steps):
                    action, _ = agent.predict(obs, deterministic=deterministic)
                    next_obs, reward, done, info = env.step(action)
                    obs = next_obs
                    rew_ep += reward
                    rew_total += sum(rew_ep[done])
                    rew_ep[done] = 0
                    total_ep += np.sum(done)
                    success_i[reward > -self.rew_offset] = 1
                    r.append(reward)
                success.append(success_i.mean())
                rews.append(np.mean(np.array(r)))
            success_rate = np.mean(success)
            avg_rew = rew_total / total_ep if total_ep > 0 else 0
            if self.use_wandb:
                name = 'eval'
                if deterministic:
                    wandb.log({f'{name}/success_rate_deterministic': success_rate,
                               f'{name}/reward_deterministic': avg_rew}, step=self.log_count)
                else:
                    wandb.log({f'{name}/success_rate': success_rate,
                               f'{name}/reward': avg_rew,
                               f'{name}/timesteps': self.total_timesteps}, step=self.log_count)

    def set_timesteps(self, timesteps):
        self.total_timesteps = timesteps