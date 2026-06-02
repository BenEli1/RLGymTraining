from pathlib import Path

import numpy as np
import pandas as pd

from rl_gym_training.data.data_loader import (
    N_ACTIONS,
    generate_synthetic_workout_data,
    load_workout_data,
)
from rl_gym_training.data.dataset import WorkoutSequenceDataset
from rl_gym_training.data.preprocessing import apply_scaler, chronological_split, fit_scaler
from rl_gym_training.shared.config import load_config
from rl_gym_training.shared.seed import set_seed


def test_config_loads_repo_relative_paths():
    config = load_config()
    assert config.data.sequence_length > 0
    assert isinstance(config.data.raw_path, Path)


def test_seed_reproducibility_for_synthetic_data():
    set_seed(7)
    first = generate_synthetic_workout_data(trainees=2, days=8, seed=7)
    set_seed(7)
    second = generate_synthetic_workout_data(trainees=2, days=8, seed=7)
    pd.testing.assert_frame_equal(first, second)


def test_loader_uses_synthetic_fallback_when_csv_missing(tmp_path):
    config = load_config()
    data_config = config.data.__class__(
        **{**config.data.__dict__, "raw_path": tmp_path / "missing.csv", "synthetic_trainees": 2}
    )
    frame = load_workout_data(data_config, seed=1)
    assert not frame.empty
    assert int(frame["synthetic"].iloc[0]) == 1


def test_split_scaler_and_dataset_shapes():
    config = load_config()
    frame = generate_synthetic_workout_data(trainees=3, days=16, seed=4)
    splits = chronological_split(frame, config.data)
    assert splits.train["day"].max() < frame["day"].max()
    scaler = fit_scaler(splits.train, config.data.state_columns)
    scaled_train = apply_scaler(splits.train, config.data.state_columns, scaler)
    dataset = WorkoutSequenceDataset(scaled_train, config.data)
    sequence, target = dataset[0]
    assert sequence.shape == (
        config.data.sequence_length,
        len(config.data.state_columns) + N_ACTIONS,
    )
    assert target.shape == (len(config.data.state_columns),)
    assert np.isfinite(sequence.numpy()).all()
