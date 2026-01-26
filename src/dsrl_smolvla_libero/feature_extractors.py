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
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 256,
        state_mlp_layers: list = [256, 256],
        cnn_features_dim: int = 128,
    ):
        super().__init__(observation_space, features_dim=features_dim)

        # Collect state keys
        self.state_keys = [
            k
            for k in observation_space.spaces.keys()
            if "observation.state" in k or "observation.environment" in k
        ]

        # Build state MLP
        if self.state_keys:
            state_dim = sum(
                np.prod(observation_space.spaces[k].shape) for k in self.state_keys
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

        # Build image encoders
        self.image_keys = [
            k for k in observation_space.spaces.keys() if "observation.images" in k
        ]
        self.image_encoders = nn.ModuleDict()
        for key in self.image_keys:
            img_space = observation_space.spaces[key]
            n_channels = img_space.shape[0]
            encoder = self._create_cnn(n_channels, cnn_features_dim)
            self.image_encoders[key.replace(".", "_")] = encoder

        # Final projection
        total_dim = self.state_output_dim + len(self.image_keys) * cnn_features_dim
        self.final_projection = nn.Linear(total_dim, features_dim)

    def _create_cnn(self, n_input_channels: int, output_dim: int) -> nn.Module:
        """CNN with architecture: features=(32,32,32,32), strides=(2,1,1,1), 3x3 kernels."""
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
            nn.Linear(32 * 51 * 51, output_dim),
            nn.ReLU(),
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        features = []

        # Process state
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
                state = torch.cat(state_tensors, dim=1)
                features.append(self.state_mlp(state))

        # Process images
        for img_key in self.image_keys:
            if img_key in observations:
                img = observations[img_key]
                if img.dim() == 3:
                    img = img.unsqueeze(0)
                encoder_key = img_key.replace(".", "_")
                features.append(self.image_encoders[encoder_key](img))

        combined = torch.cat(features, dim=1)
        return self.final_projection(combined)


class LiberoPixelOnlyExtractor(BaseFeaturesExtractor):
    """
    Feature extractor for Libero that only uses image observations (no state).
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 256,
        cnn_features_dim: int = 128,
    ):
        super().__init__(observation_space, features_dim=features_dim)

        # Build image encoders
        self.image_keys = [
            k for k in observation_space.spaces.keys() if "observation.images" in k
        ]
        self.image_encoders = nn.ModuleDict()
        for key in self.image_keys:
            img_space = observation_space.spaces[key]
            n_channels = img_space.shape[0]
            encoder = self._create_cnn(n_channels, cnn_features_dim)
            self.image_encoders[key.replace(".", "_")] = encoder

        # Final projection
        total_dim = len(self.image_keys) * cnn_features_dim
        self.final_projection = nn.Linear(total_dim, features_dim)

    def _create_cnn(self, n_input_channels: int, output_dim: int) -> nn.Module:
        """CNN with architecture: features=(32,32,32,32), strides=(2,1,1,1), 3x3 kernels."""
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
            nn.Linear(32 * 51 * 51, output_dim),
            nn.ReLU(),
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        features = []
        for img_key in self.image_keys:
            if img_key in observations:
                img = observations[img_key]
                if img.dim() == 3:
                    img = img.unsqueeze(0)
                encoder_key = img_key.replace(".", "_")
                features.append(self.image_encoders[encoder_key](img))

        combined = torch.cat(features, dim=1)
        return self.final_projection(combined)
