# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import torch
import torch.nn.functional as F
from torch import nn

def dict_to_device(tensor_dict: dict[str, torch.Tensor], device: str | torch.device) -> dict[str, torch.Tensor]:
    """
    Utility function that moves all entries in a tensor dictionary to a specific device and returns a new tensor.
    
    Args:
        tensor_dict: The dictionary mapping string keys to torch tensor values.
        device: The target device to move the tensors to.

    Returns:
        A new dictionary with all tensors moved to the specified device.
    """
    return {k: v.to(device) for k, v in tensor_dict.items()}


class DiffusionBackboneSimple(nn.Module):
    def __init__(
        self,
        x_dim: int = 2,
        state_dim: int = 0,
        hidden_dim: int = 128,
        num_layers: int = 4,
    ):
        """
        Initialize a simple MLP backbone for diffusion models.
        
        Args:
            x_dim: The dimension of the noise vector.
            state_dim: The dimension of the state vector.
            hidden_dim: The hidden layer dimension.
            num_layers: The total number of layers in the MLP.
        """
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
        """
        Forward pass through the MLP with concatenated inputs.
        
        Args:
            x: The noise tensor.
            state: The state tensor.
            t: The diffusion time step.

        Returns:
            The output tensor from the MLP.
        """
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
        """
        Initialize the Flow Matching model with specified backbone and parameters.
        
        Args:
            backbone: The backbone network.
            sample_shape: The shape of the noise sample tensor.
            state_shape: The shape of the state tensor.
            n_inference_steps: The number of flow steps to use at inference time.
            use_midpoint: Use midpoint method for integration steps.
            use_bridge: Use stochastic bridge sampling.
            bridge_alpha: The bridge diffusion coefficient (if use_bridge is True).
        """
        super().__init__()
        self.backbone = backbone
        self.sample_shape = sample_shape
        self.state_shape = state_shape
        self.n_inference_steps = n_inference_steps
        self.use_midpoint = use_midpoint
        self.use_bridge = use_bridge
        self.bridge_alpha = bridge_alpha

    def compute_loss(self, x_1: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Compute the flow matching loss between predicted and target flow.
        
        Args:
            x_1: The target sample tensor at time t=1 (denoised object)
            state: The state tensor, used for conditioning.

        Returns:
            The computed MSE loss tensor on predicted vs target flow.
        """
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
        """Compute the drift and diffusion terms for one integration step.
        
        Args:
            x_t: The current sample tensor at time t_start.
            state: The state tensor, used for conditioning.
            t_start: The start time tensor for the integration step.
            t_end: The end time tensor for the integration step.

        Returns:
            A tuple of (drift tensor, diffusion std tensor or None).
        """
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
        """One integration step using the midpoint method.
        
        Args:
            x_t: The current sample tensor at time t_start.
            state: The state tensor, used for conditioning.
            t_start: The start time tensor for the integration step.
            t_end: The end time tensor for the integration step.

        Returns:
            The updated sample tensor at time t_end.
        """
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
        """Sample actions from the flow model given state observations.
        
        Args:
            state: The state tensor for conditioning.
            batch_size: The batch size for sampling. If None, inferred from state.
            device: The torch device to perform sampling on. If None, uses model device.
            init_x: Optional initial noise tensor to start the sampling from.
        """
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
        """Append initial noise to the trajectory traces for visualization.
        
        Args:
            metadata: The metadata dictionary containing 'init_noise' and traces.
            traces_key: The key in metadata for the trajectory traces.

        Returns:
            The concatenated tensor of shape [T+1, B, D] with initial noise prepended.
        """
        init_noise = metadata["init_noise"]  # shape: [B, D]
        traces = metadata[traces_key]  # shape: [T, B, D]
        return torch.cat([init_noise.unsqueeze(0), traces], dim=0)  # shape: [T+1, B, D]

    def get_trainable_parameters(self) -> list[torch.nn.Parameter]:
        """Get trainable parameters from the backbone network.
        
        Returns:
            A list of trainable parameters.
        """
        return list(self.backbone.parameters())

    def get_timesteps(self) -> torch.Tensor:
        """Generate linearly spaced timesteps for the integration process.
        
        Returns:
            A tensor of shape [n_inference_steps + 1] with linearly spaced timesteps.
        """
        return torch.linspace(0, 1.0, self.n_inference_steps + 1)

    @staticmethod
    def get_bridge_sigma(tt: torch.Tensor) -> torch.Tensor:
        """Compute the bridge variance schedule as a function of time.
        
        Args:
            tt: The time tensor.

        Returns:
            The bridge standard deviation tensor.
        """
        return torch.sqrt(
            torch.clamp(tt, 1e-6, 1 - 1e-6) * (1 - torch.clamp(tt, 1e-6, 1 - 1e-6))
        )

    def set_bridge_alpha(self, alpha: float) -> None:
        """Set the bridge diffusion coefficient for stochastic sampling.
        
        Args:
            alpha: The new bridge diffusion coefficient.
        """
        assert self.use_bridge, "Bridge is not enabled."
        self.bridge_alpha = alpha
