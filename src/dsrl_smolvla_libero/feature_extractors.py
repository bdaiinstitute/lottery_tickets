"""Custom feature extractors for processing Libero dictionary observations in Stable-Baselines3."""

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict


class LiberoFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor for Libero dictionary observations with images and state.
    Works with flattened observation spaces where:
    - Image keys don't have nested prefixes (e.g., "image" instead of "observation.images.image")
    - State is a single flattened vector with key "state"

    Normalization:
    - Images are normalized from [0, 255] to [0, 1] in forward pass
    - State is passed through as-is (robot state values are already in reasonable ranges)
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 256,
        state_mlp_layers: list = [256, 256],
        cnn_features_dim: int = 128,
    ):
        super().__init__(observation_space, features_dim=features_dim)

        # Collect state keys (look for "state" or nested state keys)
        self.state_keys = [
            k
            for k in observation_space.spaces.keys()
            if k == "state"
            or "observation.state" in k
            or "observation.environment" in k
        ]

        # Build state MLP (no normalization - state values are already reasonable)
        if self.state_keys:
            state_dim = sum(
                int(np.prod(observation_space.spaces[k].shape)) for k in self.state_keys
            )
            state_layers = []
            input_dim = state_dim
            for hidden_dim in state_mlp_layers:
                state_layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU()])
                input_dim = hidden_dim
            self.state_mlp = nn.Sequential(*state_layers)
            self.state_output_dim = input_dim
        else:
            self.state_mlp = None
            self.state_output_dim = 0

        # Build image encoders - look for flat image keys or nested image keys
        self.image_keys = [
            k
            for k in observation_space.spaces.keys()
            if "observation.images" in k
            or (
                k not in self.state_keys and len(observation_space.spaces[k].shape) == 3
            )
        ]
        self.image_encoders = nn.ModuleDict()
        for key in self.image_keys:
            img_space = observation_space.spaces[key]
            n_channels = img_space.shape[0]
            encoder = self._create_cnn(n_channels, cnn_features_dim)
            # Replace dots with underscores for valid ModuleDict key
            self.image_encoders[key.replace(".", "_")] = encoder

        # Final projection
        total_dim = self.state_output_dim + len(self.image_keys) * cnn_features_dim
        self.final_projection = nn.Linear(total_dim, features_dim)

    def _create_cnn(self, n_input_channels: int, output_dim: int) -> nn.Module:
        """CNN for 64x64 images matching Jax reference:
        features=(32,32,32,32), strides=(2,1,1,1), 3x3 kernels, VALID padding.

        Note: Images are normalized from [0, 255] to [0, 1] in forward() before being passed here.

        Output size calculation for 64x64 input with VALID (padding=0):
        - After Conv2d(stride=2, kernel=3): (64-3)/2+1 = 31
        - After Conv2d(stride=1, kernel=3): 31-3+1 = 29
        - After Conv2d(stride=1, kernel=3): 29-3+1 = 27
        - After Conv2d(stride=1, kernel=3): 27-3+1 = 25
        Final: 32 * 25 * 25 = 20000

        Then bottleneck: Dense(output_dim) -> LayerNorm -> Tanh
        """
        return nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            # Bottleneck: Dense -> LayerNorm -> Tanh
            nn.Linear(32 * 25 * 25, output_dim),  # 20000 for 64x64 input
            nn.LayerNorm(output_dim),
            nn.Tanh(),
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        features = []

        # Process state - normalize to reasonable range
        if self.state_mlp is not None and self.state_keys:
            state_tensors = [
                (
                    observations[k].flatten(start_dim=1)
                    if observations[k].dim() > 2
                    else observations[k]
                )
                for k in self.state_keys
                if k in observations
            ]
            if state_tensors:
                state = torch.cat(state_tensors, dim=1).float()
                features.append(self.state_mlp(state))

        # Process images - normalize from [0, 255] to [0, 1]
        for img_key in self.image_keys:
            if img_key in observations:
                img = observations[img_key]
                if img.dim() == 3:
                    img = img.unsqueeze(0)
                # Normalize images from uint8 [0, 255] to float [0, 1]
                img = img.float() / 255.0
                encoder_key = img_key.replace(".", "_")
                features.append(self.image_encoders[encoder_key](img))

        combined = torch.cat(features, dim=1)
        return self.final_projection(combined)


class LiberoPixelOnlyExtractor(BaseFeaturesExtractor):
    """
    Feature extractor for Libero that only uses image observations (no state).
    Works with flattened observation spaces where image keys are flat (e.g., "image").

    Normalization:
    - Images are normalized from [0, 255] to [0, 1] in forward pass
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 256,
        cnn_features_dim: int = 128,
    ):
        super().__init__(observation_space, features_dim=features_dim)

        # Collect state keys to exclude them
        state_keys = [
            k
            for k in observation_space.spaces.keys()
            if k == "state"
            or "observation.state" in k
            or "observation.environment" in k
        ]

        # Build image encoders - look for flat image keys or nested image keys (exclude state)
        self.image_keys = [
            k
            for k in observation_space.spaces.keys()
            if "observation.images" in k
            or (k not in state_keys and len(observation_space.spaces[k].shape) == 3)
        ]
        self.image_encoders = nn.ModuleDict()
        for key in self.image_keys:
            img_space = observation_space.spaces[key]
            n_channels = img_space.shape[0]
            encoder = self._create_cnn(n_channels, cnn_features_dim)
            self.image_encoders[key.replace(".", "_")] = encoder
            break

        # Final projection
        total_dim = cnn_features_dim
        self.final_projection = nn.Linear(total_dim, features_dim)

    def _create_cnn(self, n_input_channels: int, output_dim: int) -> nn.Module:
        """CNN for 64x64 images with architecture: features=(32,32,32,32), strides=(2,1,1,1), 3x3 kernels.

        Note: Images are normalized from [0, 255] to [0, 1] in forward() before being passed here.

        Output size for 64x64 input: 32 * 25 * 25 = 20000
        """
        return nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 25 * 25, output_dim),  # 20000 for 64x64 input
            nn.LayerNorm(output_dim),
            nn.Tanh(),
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        features = []
        for img_key in self.image_keys:
            if img_key in observations:
                img = observations[img_key]
                if img.dim() == 3:
                    img = img.unsqueeze(0)
                # Normalize images from uint8 [0, 255] to float [0, 1]
                img = img.float() / 255.0
                encoder_key = img_key.replace(".", "_")
                features.append(self.image_encoders[encoder_key](img))
                break

        combined = torch.cat(features, dim=1)
        return self.final_projection(combined)


class StateOnlyExtractor(BaseFeaturesExtractor):
    """Feature extractor using only robot state (no images)."""

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim=features_dim)
        state_dim = int(np.prod(observation_space["state"].shape))
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.mlp(observations["state"].float())
