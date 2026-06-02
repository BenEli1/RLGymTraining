"""World-model training service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from rl_gym_training.data.data_loader import N_ACTIONS
from rl_gym_training.data.dataset import WorkoutSequenceDataset
from rl_gym_training.models.lstm_world_model import LSTMWorldModel
from rl_gym_training.shared.config import AppConfig


@dataclass(frozen=True)
class WorldModelTrainingResult:
    train_losses: list[float]
    validation_loss: float
    checkpoint_path: Path


def train_world_model(
    train_dataset: WorkoutSequenceDataset,
    validation_dataset: WorkoutSequenceDataset,
    config: AppConfig,
    checkpoint_path: Path,
) -> WorldModelTrainingResult:
    model = LSTMWorldModel(
        input_size=len(config.data.state_columns) + N_ACTIONS,
        state_size=len(config.data.state_columns),
        hidden_size=config.world_model.hidden_size,
        num_layers=config.world_model.num_layers,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.world_model.learning_rate)
    loss_fn = nn.MSELoss()
    loader = DataLoader(train_dataset, batch_size=config.world_model.batch_size, shuffle=True)
    train_losses: list[float] = []
    for _ in range(config.world_model.epochs):
        epoch_losses: list[float] = []
        for sequence, target in loader:
            prediction = model(sequence)
            loss = loss_fn(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        train_losses.append(float(sum(epoch_losses) / max(len(epoch_losses), 1)))
    validation_loss = evaluate_world_model_loss(model, validation_dataset)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(checkpoint_path)
    return WorldModelTrainingResult(train_losses, validation_loss, checkpoint_path)


def evaluate_world_model_loss(model: LSTMWorldModel, dataset: WorkoutSequenceDataset) -> float:
    if len(dataset) == 0:
        return float("nan")
    loss_fn = nn.MSELoss()
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    losses: list[float] = []
    with torch.no_grad():
        for sequence, target in loader:
            losses.append(float(loss_fn(model(sequence), target)))
    return float(sum(losses) / len(losses))
