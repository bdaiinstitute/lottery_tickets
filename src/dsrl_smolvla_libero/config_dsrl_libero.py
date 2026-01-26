"""Configuration dataclasses for DSRL training on Libero."""

import datetime as dt
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import EnvConfig

logger = getLogger(__name__)


@dataclass
class TrainConfig:
    """Training configuration for DSRL with libero."""

    tau: float = 0.005
    actor_lr: float = 3e-4
    batch_size: int = 16
    train_freq: int = 1
    utd: int = 20
    use_layer_norm: bool = True
    layer_size: int = 128
    num_layers: int = 3
    discount: float = 0.999
    ent_coef: float = -1.0
    target_ent: float = 0.0
    init_rollout_steps: int = 1501
    action_magnitude: float = 1
    n_critics: int = 10


@dataclass
class WandbConfig:
    """Wandb logging configuration."""

    project: str = "lt_libero_baselines"
    run: str = ""
    group: str = ""


@dataclass
class DSRLLiberoConfig:
    """Main configuration for DSRL training on Libero."""

    # Required lerobot configs
    env: EnvConfig
    policy: PreTrainedConfig | None = None
    rename_map: dict[str, str] = field(default_factory=dict)

    # Training settings
    train: TrainConfig = field(default_factory=TrainConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    # General settings
    name: str = "libero_dsrl"
    logdir: Path | None = None
    algorithm: str = "dsrl_sac"
    seed: int = 1
    device: str = "cuda:0"
    use_wandb: bool = True
    load_offline_data: bool = False
    eval_interval: int = 3000
    num_evals: int = 200
    save_model_interval: int = 50000
    save_replay_buffer: bool = False
    n_envs: int = 4
    n_eval_envs: int = 25
    reward_offset: float = 1.0
    max_episode_steps: int = 280

    def __post_init__(self) -> None:
        """Load policy config from pretrained path if provided."""
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(
                policy_path, cli_overrides=cli_overrides
            )
            self.policy.pretrained_path = Path(policy_path)
        else:
            logger.warning(
                "No pretrained path was provided, policy will be built from scratch (random weights)."
            )

        # Set default logdir if not provided
        if not self.logdir:
            now = dt.datetime.now()
            task_name = getattr(self.env, "task", "libero")
            log_dir = f"{task_name}_{self.algorithm}_{now:%Y-%m-%d}_{now:%H-%M-%S}_{self.seed}"
            self.logdir = Path("./logs_libero_dsrl") / log_dir

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]
