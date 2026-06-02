"""Typed configuration loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rl_gym_training.shared.paths import resolve_path


@dataclass(frozen=True)
class DataConfig:
    raw_path: Path
    processed_dir: Path
    use_synthetic_fallback: bool
    synthetic_trainees: int
    synthetic_days: int
    sequence_length: int
    train_ratio: float
    validation_ratio: float
    state_columns: list[str]
    action_column: str
    trainee_column: str
    day_column: str


@dataclass(frozen=True)
class WorldModelConfig:
    hidden_size: int
    num_layers: int
    learning_rate: float
    epochs: int
    batch_size: int


@dataclass(frozen=True)
class RLConfig:
    episode_length: int
    gamma: float
    policy_learning_rate: float
    critic_learning_rate: float
    entropy_coefficient: float
    reinforce_episodes: int
    a2c_episodes: int
    normalize_returns: bool


@dataclass(frozen=True)
class RewardConfig:
    progress_weight: float
    readiness_weight: float
    fatigue_penalty: float
    overload_penalty: float
    invalid_action_penalty: float
    consistency_bonus: float


@dataclass(frozen=True)
class AppConfig:
    seed: int
    data: DataConfig
    world_model: WorldModelConfig
    rl: RLConfig
    reward: RewardConfig


def load_config(path: str | Path = "config/setup.yaml") -> AppConfig:
    config_path = resolve_path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw: dict[str, Any] = yaml.safe_load(stream)
    data = raw["data"]
    return AppConfig(
        seed=int(raw["seed"]),
        data=DataConfig(
            raw_path=resolve_path(data["raw_path"]),
            processed_dir=resolve_path(data["processed_dir"]),
            use_synthetic_fallback=bool(data["use_synthetic_fallback"]),
            synthetic_trainees=int(data["synthetic_trainees"]),
            synthetic_days=int(data["synthetic_days"]),
            sequence_length=int(data["sequence_length"]),
            train_ratio=float(data["train_ratio"]),
            validation_ratio=float(data["validation_ratio"]),
            state_columns=list(data["state_columns"]),
            action_column=str(data["action_column"]),
            trainee_column=str(data["trainee_column"]),
            day_column=str(data["day_column"]),
        ),
        world_model=WorldModelConfig(**raw["world_model"]),
        rl=RLConfig(**raw["rl"]),
        reward=RewardConfig(**raw["reward"]),
    )
