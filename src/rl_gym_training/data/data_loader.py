"""Data loading and synthetic fallback generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rl_gym_training.shared.config import DataConfig

ACTION_REST = 0
ACTION_CARDIO = 1
ACTION_STRENGTH = 2
ACTION_MIXED = 3
ACTION_NAMES = {
    ACTION_REST: "rest",
    ACTION_CARDIO: "cardio",
    ACTION_STRENGTH: "strength",
    ACTION_MIXED: "mixed",
}
N_ACTIONS = len(ACTION_NAMES)


def load_workout_data(config: DataConfig, seed: int) -> pd.DataFrame:
    """Load a CSV dataset or create a clearly marked synthetic fallback."""
    if config.raw_path.exists():
        return _load_csv(config.raw_path, config)
    if not config.use_synthetic_fallback:
        msg = f"Dataset not found at {config.raw_path}. Enable synthetic fallback or provide CSV."
        raise FileNotFoundError(msg)
    return generate_synthetic_workout_data(
        trainees=config.synthetic_trainees,
        days=config.synthetic_days,
        seed=seed,
    )


def _load_csv(path: Path, config: DataConfig) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        config.trainee_column,
        config.day_column,
        config.action_column,
        *config.state_columns,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    frame = frame.sort_values([config.trainee_column, config.day_column]).reset_index(drop=True)
    numeric_columns = [*config.state_columns, config.action_column, config.day_column]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    frame[config.state_columns] = frame.groupby(config.trainee_column, group_keys=False)[
        config.state_columns
    ].apply(lambda group: group.ffill().bfill())
    frame = frame.dropna(subset=required)
    frame[config.action_column] = frame[config.action_column].astype(int)
    return frame


def generate_synthetic_workout_data(trainees: int, days: int, seed: int) -> pd.DataFrame:
    """Generate small educational data for tests and demos, not a real medical dataset."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []
    for trainee_id in range(trainees):
        readiness = rng.uniform(0.45, 0.75)
        fatigue = rng.uniform(0.1, 0.35)
        strength = rng.uniform(0.35, 0.65)
        endurance = rng.uniform(0.35, 0.65)
        soreness = rng.uniform(0.05, 0.25)
        previous_action = ACTION_REST
        for day in range(days):
            if fatigue > 0.7 or soreness > 0.65:
                action = ACTION_REST
            else:
                action = int(rng.choice([ACTION_CARDIO, ACTION_STRENGTH, ACTION_MIXED, ACTION_REST]))
            load = {ACTION_REST: 0.0, ACTION_CARDIO: 0.45, ACTION_STRENGTH: 0.55, ACTION_MIXED: 0.65}[action]
            rows.append(
                {
                    "trainee_id": f"T{trainee_id:03d}",
                    "day": day,
                    "readiness": readiness,
                    "fatigue": fatigue,
                    "strength": strength,
                    "endurance": endurance,
                    "soreness": soreness,
                    "action": action,
                    "previous_action": previous_action,
                    "synthetic": 1,
                }
            )
            strength += 0.025 * (action in [ACTION_STRENGTH, ACTION_MIXED]) - 0.01 * fatigue
            endurance += 0.025 * (action in [ACTION_CARDIO, ACTION_MIXED]) - 0.008 * fatigue
            fatigue = 0.55 * fatigue + load * 0.35 + rng.normal(0.0, 0.03)
            soreness = 0.55 * soreness + load * 0.30 + rng.normal(0.0, 0.025)
            readiness = 0.75 - 0.45 * fatigue - 0.25 * soreness + rng.normal(0.0, 0.03)
            readiness = float(np.clip(readiness, 0.0, 1.0))
            fatigue = float(np.clip(fatigue, 0.0, 1.0))
            strength = float(np.clip(strength, 0.0, 1.0))
            endurance = float(np.clip(endurance, 0.0, 1.0))
            soreness = float(np.clip(soreness, 0.0, 1.0))
            previous_action = action
    return pd.DataFrame(rows)
