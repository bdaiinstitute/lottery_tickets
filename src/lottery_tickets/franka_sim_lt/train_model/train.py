# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import pickle
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset


def save_checkpoint(
    cfg: DictConfig,
    fm_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    state_dim: int,
    action_dim: int,
    save_path: Path,
    filename: str,
) -> None:
    """Save model checkpoint with configuration.

    Args:
        cfg: Hydra configuration
        fm_model: The flow matching model
        optimizer: The optimizer
        epoch: Current epoch
        loss: Current loss
        state_dim: State dimension
        action_dim: Action dimension
        save_path: Directory to save checkpoint
        filename: Checkpoint filename
    """
    # Add dynamic dimensions to config for saving
    config_with_dims = OmegaConf.create(cfg)
    OmegaConf.set_struct(config_with_dims, False)  # Allow adding new keys
    config_with_dims.state_dim = state_dim
    config_with_dims.action_dim = action_dim

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": fm_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "config": OmegaConf.to_container(config_with_dims, resolve=True),
    }

    torch.save(checkpoint, save_path / filename)
    print(f"saved checkpoint to: {save_path / filename}")


class ActionChunkDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        chunk_size: int = 8,
        use_state_history: bool = False,
        state_history_length: int = 4,
    ):
        """Dataset for state-action chunk pairs.

        Args:
            data_path: Path to the pickle file containing episodes
            chunk_size: Number of future actions to predict
            use_state_history: Whether to include state history
            state_history_length: Number of past states to include
        """
        self.chunk_size = chunk_size
        self.use_state_history = use_state_history
        self.state_history_length = state_history_length

        with open(data_path, "rb") as f:
            data = pickle.load(f)

        # Assume data is a list of episodes
        assert isinstance(data, list), "Data should be a list of episodes"
        assert all(isinstance(ep, list) for ep in data), (
            "Each episode should be a list of transitions"
        )

        self.episodes = data
        self.num_episodes = len(self.episodes)

        # Create index mapping for variable-length episodes
        # Each entry is (episode_idx, step_idx) for a valid transition
        self.indices = []
        for episode_idx, episode in enumerate(self.episodes):
            for step_idx in range(len(episode)):
                self.indices.append((episode_idx, step_idx))

    def __len__(self):
        """Get length of self.indices."""
        return len(self.indices)

    def __getitem__(self, idx : int) -> dict:
        """
        Get state and action chunk for the given index.

        Args:
            idx: Index of the data point
        
        Returns:
            dict containing "state" and "action_chunk"
        """
        episode_idx, step_idx = self.indices[idx]

        episode = self.episodes[episode_idx]
        current_transition = episode[step_idx]

        # Extract current state
        current_state = current_transition["observations"]["state"].flatten()

        # Create state input (current state + history if enabled)
        if self.use_state_history:
            state_history = []
            for i in range(self.state_history_length):
                # Look back i+1 steps (0 = current, 1 = previous, etc.)
                hist_step_idx = step_idx - i
                if hist_step_idx >= 0:
                    # Use actual historical state
                    hist_state = episode[hist_step_idx]["observations"][
                        "state"
                    ].flatten()
                else:
                    # Pad with the first state if we don't have enough history
                    hist_state = episode[0]["observations"]["state"].flatten()
                state_history.append(hist_state)

            # Concatenate states: [current, prev_1, prev_2, ...]
            state = np.concatenate(state_history)
        else:
            state = current_state

        # Create action chunk by looking at future transitions
        action_chunk = []
        episode_length = len(episode)
        for i in range(self.chunk_size):
            future_step = step_idx + i

            if future_step < episode_length:
                # Use future action
                future_action = episode[future_step]["actions"]
            else:
                # If at end of episode, pad with the last action
                future_action = episode[-1]["actions"]

            action_chunk.append(future_action)

        action_chunk = np.array(action_chunk)

        return {
            "state": torch.FloatTensor(state),
            "action_chunk": torch.FloatTensor(
                action_chunk.flatten()
            ),  # Flatten action chunk
        }


def train_flow_matching_policy(cfg: DictConfig) -> None:
    """
    Train Flow Matching policy on state-action data.
    
    Args:
        cfg: The policy configuration dictionary.
    """

    # Set device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataset and dataloader
    dataset = ActionChunkDataset(
        data_path=cfg.dataset.data_path,
        chunk_size=cfg.dataset.chunk_size,
        use_state_history=cfg.dataset.use_state_history,
        state_history_length=cfg.dataset.state_history_length,
    )
    dataloader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True)
    episode_lengths = [len(episode) for episode in dataset.episodes]
    print(f"Dataset loaded with {dataset.num_episodes} episodes")
    print(
        f"Episode lengths: min={min(episode_lengths)}, max={max(episode_lengths)}, avg={np.mean(episode_lengths):.1f}"
    )
    print(f"Total dataset size: {len(dataset)}")

    # Get data dimensions from first sample
    sample = dataset[0]
    state_dim = sample["state"].shape[0]
    action_dim = sample["action_chunk"].shape[0]

    print(f"State dimension: {state_dim}")
    print(f"Action chunk dimension: {action_dim}")

    # Create Flow Matching model using hydra instantiate
    backbone = instantiate(
        cfg.model.backbone,
        x_dim=action_dim,
        state_dim=state_dim,
    )

    fm_model = instantiate(
        cfg.model.fm,
        backbone=backbone,
        sample_shape=(action_dim,),
        state_shape=(state_dim,),
    ).to(device)

    # Setup optimizer
    optimizer = torch.optim.Adam(
        fm_model.get_trainable_parameters(), lr=cfg.training.learning_rate
    )

    # Create save directory
    save_path = Path(cfg.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Training loop
    fm_model.train()
    for epoch in range(cfg.training.num_epochs):
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            states = batch["state"].to(device)
            action_chunks = batch["action_chunk"].to(device)

            loss = fm_model.compute_loss(action_chunks, states)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}: Average Loss = {avg_loss:.6f}")

        # Save checkpoint at interval
        if (epoch + 1) % cfg.training.checkpoint_interval == 0:
            save_checkpoint(
                cfg=cfg,
                fm_model=fm_model,
                optimizer=optimizer,
                epoch=epoch + 1,
                loss=avg_loss,
                state_dim=state_dim,
                action_dim=action_dim,
                save_path=save_path,
                filename=f"fm_policy_epoch_{epoch + 1}.pt",
            )
            print(f"Saved checkpoint at epoch {epoch + 1}")

    # Save final model
    save_checkpoint(
        cfg=cfg,
        fm_model=fm_model,
        optimizer=optimizer,
        epoch=cfg.training.num_epochs,
        loss=avg_loss,
        state_dim=state_dim,
        action_dim=action_dim,
        save_path=save_path,
        filename="fm_policy_final.pt",
    )
    print("Training completed! Final model saved.")

    # Save config for reproducibility
    OmegaConf.save(cfg, save_path / "config.yaml")

    return fm_model


@hydra.main(version_base=None, config_path="cfgs", config_name="fm")
def main(cfg: DictConfig):
    """Main training script with Hydra configuration.
    
    Args:
        cfg: The configuration object.
    """

    # Print configuration
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Set random seed if specified
    if hasattr(cfg, "seed"):
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

    print("Starting training...")
    train_flow_matching_policy(cfg)


if __name__ == "__main__":
    main()
