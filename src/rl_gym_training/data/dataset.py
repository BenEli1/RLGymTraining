"""PyTorch datasets for sequence modeling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from rl_gym_training.data.data_loader import N_ACTIONS
from rl_gym_training.shared.config import DataConfig


class WorkoutSequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Sequences of state plus one-hot action mapped to next state."""

    def __init__(self, frame: pd.DataFrame, config: DataConfig) -> None:
        self.state_columns = config.state_columns
        self.sequence_length = config.sequence_length
        self.samples: list[tuple[np.ndarray, np.ndarray]] = []
        input_size = len(config.state_columns) + N_ACTIONS
        for _, group in frame.sort_values(config.day_column).groupby(config.trainee_column):
            states = group[config.state_columns].to_numpy(dtype=np.float32)
            actions = group[config.action_column].to_numpy(dtype=np.int64)
            if len(group) <= config.sequence_length:
                continue
            for start in range(len(group) - config.sequence_length):
                end = start + config.sequence_length
                sequence = np.zeros((config.sequence_length, input_size), dtype=np.float32)
                sequence[:, : len(config.state_columns)] = states[start:end]
                sequence[np.arange(config.sequence_length), len(config.state_columns) + actions[start:end]] = 1.0
                target = states[end]
                self.samples.append((sequence, target))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sequence, target = self.samples[index]
        return torch.from_numpy(sequence), torch.from_numpy(target)
