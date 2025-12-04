# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import torch
import torch.nn.functional as F
from torch import nn

def dict_to_device(dict: dict[str, torch.Tensor], device: str | torch.device):
    return {k: v.to(device) for k, v in dict.items()}


class DiffusionBackboneSimple(nn.Module):
    def __init__(
        self,
        x_dim: int = 2,
        state_dim: int = 0,
        hidden_dim: int = 128,
        num_layers: int = 4,
    ):
        """Initialize a simple MLP backbone for diffusion models."""
        super().__init__()
        # MLP on concatenated [x, state, time_emb]
        layers = []
        in_dim = x_dim + state_dim + 1
        for _ in range(num_layers - 1):
            layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
            ]
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, x_dim))  # final E
        self.mlp = nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor, state: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the MLP with concatenated inputs."""
        h = torch.cat([x, state, t.unsqueeze(-1)], dim=-1)
        return self.mlp(h)


class FM(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        sample_shape: tuple[int, ...],
        state_shape: tuple[int, ...],
        n_inference_steps: int,
        use_midpoint: float = True,
        use_bridge: bool = False,
        bridge_alpha: float = 0.1,
    ) -> None:
        """Initialize the Flow Matching model with specified backbone and parameters."""
        super().__init__()
        self.backbone = backbone
        self.sample_shape = sample_shape
        self.state_shape = state_shape
        self.n_inference_steps = n_inference_steps
        self.use_midpoint = use_midpoint
        self.use_bridge = use_bridge
        self.bridge_alpha = bridge_alpha

    def compute_loss(self, x_1: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Compute the flow matching loss between predicted and target flow."""
        x_0 = torch.randn_like(x_1)
        t = torch.rand(x_1.shape[0], device=x_1.device)
        t_ext = t.unsqueeze(-1)
        x_t = (1 - t_ext) * x_0 + t_ext * x_1
        dx_t = x_1 - x_0

        pred = self.backbone(x_t, state, t)
        return F.mse_loss(pred, dx_t)

    def step_dist(
        self,
        x_t: torch.Tensor,
        state: torch.Tensor,
        t_start: torch.Tensor,
        t_end: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute the drift and diffusion terms for one integration step."""
        delta_t = t_end - t_start
        delta_t_ext = delta_t.unsqueeze(-1)

        if self.use_midpoint:
            t_mid = t_start + 0.5 * delta_t
            drift = self.backbone(
                x_t + self.backbone(x_t, state, t_start) * 0.5 * delta_t_ext,
                state,
                t_mid,
            )
        else:
            drift = self.backbone(x_t, state, t_start)

        if self.use_bridge:
            std = (
                self.bridge_alpha
                * self.get_bridge_sigma(t_start).unsqueeze(-1)
                * torch.sqrt(torch.clamp(delta_t_ext, min=1e-12))
            )

        return delta_t_ext * drift, std if self.use_bridge else None

    def step(
        self,
        x_t: torch.Tensor,
        state: torch.Tensor,
        t_start: torch.Tensor,
        t_end: torch.Tensor,
    ) -> torch.Tensor:
        """One integration step using the midpoint method."""
        drift, std = self.step_dist(x_t=x_t, state=state, t_start=t_start, t_end=t_end)

        noise = 0.0
        if self.use_bridge:
            noise = torch.randn_like(x_t) * std

        return x_t + drift + noise

    def sample_action(
        self,
        state: torch.Tensor,
        batch_size: int | None = None,
        device: torch.device | str | None = None,
        init_x: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Sample actions from the flow model given state observations."""
        if device is None:
            device = next(self.parameters()).device

        # Infer batch_size from state if not provided
        if batch_size is None:
            batch_size = state.shape[0]

        # Ensure state is on the correct device
        state = state.to(device)
        assert state.shape == (
            batch_size,
            *self.state_shape,
        ), f"Expected state shape {(batch_size, *self.state_shape)}, got {state.shape}"

        if init_x is not None:
            assert init_x.shape == (batch_size, *self.sample_shape)
            x = init_x.to(device)
        else:
            x = torch.randn(batch_size, *self.sample_shape, device=device)
        trajectory = [x.clone()]
        t_steps = self.get_timesteps().to(x.device)

        for i in range(self.n_inference_steps):
            t_start = torch.full((x.shape[0],), t_steps[i], device=x.device)
            t_end = torch.full((x.shape[0],), t_steps[i + 1], device=x.device)
            x = self.step(x, state, t_start, t_end)
            trajectory.append(x.clone())

        action = trajectory[-1]
        trajectory = torch.stack(trajectory, dim=0)
        init_noise = trajectory[0]
        trajectory = trajectory[1:]  # Remove the initial noise.

        metadata = {
            "init_noise": init_noise,
            "sample": trajectory,
        }
        metadata = dict_to_device(metadata, "cpu")

        return action, metadata

    @staticmethod
    def append_init_noise_to_traces(
        metadata: dict[str, torch.Tensor], traces_key: str
    ) -> torch.Tensor:
        """Append initial noise to the trajectory traces for visualization."""
        init_noise = metadata["init_noise"]  # shape: [B, D]
        traces = metadata[traces_key]  # shape: [T, B, D]
        return torch.cat([init_noise.unsqueeze(0), traces], dim=0)  # shape: [T+1, B, D]

    def get_trainable_parameters(self) -> list[torch.nn.Parameter]:
        """Get trainable parameters from the backbone network."""
        return list(self.backbone.parameters())

    def get_timesteps(self) -> torch.Tensor:
        """Generate linearly spaced timesteps for the integration process."""
        return torch.linspace(0, 1.0, self.n_inference_steps + 1)

    @staticmethod
    def get_bridge_sigma(tt: torch.Tensor) -> torch.Tensor:
        """Compute the bridge variance schedule as a function of time."""
        return torch.sqrt(
            torch.clamp(tt, 1e-6, 1 - 1e-6) * (1 - torch.clamp(tt, 1e-6, 1 - 1e-6))
        )

    def set_bridge_alpha(self, alpha: float) -> None:
        """Set the bridge diffusion coefficient for stochastic sampling."""
        assert self.use_bridge, "Bridge is not enabled."
        self.bridge_alpha = alpha
