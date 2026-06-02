"""Preprocessing for chronological workout sequences."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rl_gym_training.shared.config import DataConfig


@dataclass
class StandardScaler:
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "StandardScaler":
        self.mean_ = values.mean(axis=0)
        self.scale_ = values.std(axis=0)
        self.scale_[self.scale_ == 0.0] = 1.0
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler must be fitted before transform.")
        return (values - self.mean_) / self.scale_


@dataclass(frozen=True)
class DataSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def chronological_split(frame: pd.DataFrame, config: DataConfig) -> DataSplits:
    """Split each trainee chronologically and concatenate the partitions."""
    train_parts: list[pd.DataFrame] = []
    validation_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for _, group in frame.sort_values(config.day_column).groupby(config.trainee_column):
        n = len(group)
        train_end = max(int(n * config.train_ratio), config.sequence_length + 1)
        validation_end = max(int(n * (config.train_ratio + config.validation_ratio)), train_end + 1)
        train_parts.append(group.iloc[:train_end])
        validation_parts.append(group.iloc[train_end:validation_end])
        test_parts.append(group.iloc[validation_end:])
    return DataSplits(
        train=pd.concat(train_parts).reset_index(drop=True),
        validation=pd.concat(validation_parts).reset_index(drop=True),
        test=pd.concat(test_parts).reset_index(drop=True),
    )


def fit_scaler(frame: pd.DataFrame, state_columns: list[str]) -> StandardScaler:
    return StandardScaler().fit(frame[state_columns].to_numpy(dtype=np.float32))


def apply_scaler(frame: pd.DataFrame, state_columns: list[str], scaler: StandardScaler) -> pd.DataFrame:
    scaled = frame.copy()
    scaled[state_columns] = scaler.transform(frame[state_columns].to_numpy(dtype=np.float32))
    return scaled
