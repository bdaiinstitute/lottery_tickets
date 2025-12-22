# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import torch
from lottery_tickets.franka_sim_lt.models import FlowMatching
from hydra.utils import instantiate
from omegaconf import OmegaConf
from typing import Any
import numpy as np

class FMPolicyInterface:
    """Policy interface that wraps Flow Matching model to work with gym environments."""

    def __init__(
        self,
        fm_model: FlowMatching,
        chunk_size: int,
        device: str | torch.device = "cpu",
        use_state_history: bool = False,
        state_history_length: int = 4,
    ):
        self.fm_model = fm_model
        self.chunk_size = chunk_size
        self.device = device
        self.use_state_history = use_state_history
        self.state_history_length = state_history_length
        self.action_buffer = []  # Buffer to store future actions from chunks
        self.step_in_chunk = 0
        self.state_history_buffer = []  # Buffer to store state history
        self.episode_step = 0  # Track steps to detect episode resets

    def __call__(
        self, obs: dict[str, Any], init_x: torch.Tensor | None = None
    ) -> np.ndarray:
        """
        Convert gym observation to action using Flow Matching model.
        Implements action chunking by buffering future actions.
        """
        # Extract current state from observation
        current_state = obs["state"]
        # Remove extra dimension if present (e.g., from batched environments)
        if current_state.ndim == 2 and current_state.shape[0] == 1:
            current_state = current_state.squeeze(0)

        # Update state history buffer
        self.state_history_buffer.append(current_state.flatten())
        # Keep only the required history length
        if len(self.state_history_buffer) > self.state_history_length:
            self.state_history_buffer.pop(0)

        # If we have buffered actions, use them first
        if len(self.action_buffer) > 0:
            action = self.action_buffer.pop(0)
            return action

        # Create state input (current state + history if enabled)
        if self.use_state_history:
            # Create padded history if we don't have enough states yet
            padded_history = []
            for i in range(self.state_history_length):
                if i < len(self.state_history_buffer):
                    # Use actual historical state (most recent first)
                    hist_idx = -(i + 1)  # -1 = current, -2 = previous, etc.
                    padded_history.append(self.state_history_buffer[hist_idx])
                else:
                    # Pad with the earliest available state
                    padded_history.append(self.state_history_buffer[0])

            # Concatenate states: [current, prev_1, prev_2, ...]
            state = np.concatenate(padded_history)
        else:
            state = current_state.flatten()

        # Generate new action chunk
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Sample action chunk from flow matching model
            action_chunk, _ = self.fm_model.sample_action(state_tensor, init_x=init_x)
            action_chunk = action_chunk.squeeze(0).cpu().numpy()

            # Reshape action chunk back to (chunk_size, action_dim)
            single_action_dim = action_chunk.shape[0] // self.chunk_size
            action_chunk = action_chunk.reshape(self.chunk_size, single_action_dim)

            # Use first action immediately, buffer the rest
            current_action = action_chunk[0]
            for i in range(1, self.chunk_size):
                self.action_buffer.append(action_chunk[i])

        return current_action

    def reset(self):
        """Reset the policy state (clear buffers)."""
        self.action_buffer.clear()
        self.state_history_buffer.clear()


def load_fm_model(
    model_path: str, device: str | torch.device = "cpu"
) -> tuple[FlowMatching, dict]:
    """Load Flow Matching model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint["config"]

    # Convert config back to OmegaConf for instantiate compatibility
    cfg = OmegaConf.create(config)

    # Recreate model architecture using hydra instantiate
    backbone = instantiate(
        cfg.model.backbone,
        x_dim=config["action_dim"],
        state_dim=config["state_dim"],
    )

    fm_model = instantiate(
        cfg.model.fm,
        backbone=backbone,
        sample_shape=(config["action_dim"],),
        state_shape=(config["state_dim"],),
    ).to(device)

    fm_model.load_state_dict(checkpoint["model_state_dict"])
    fm_model.eval()

    return fm_model, config
