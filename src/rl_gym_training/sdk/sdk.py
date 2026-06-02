"""Facade used by CLI, tests, and future UI layers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from rl_gym_training.data.data_loader import N_ACTIONS, load_workout_data
from rl_gym_training.data.dataset import WorkoutSequenceDataset
from rl_gym_training.data.preprocessing import apply_scaler, chronological_split, fit_scaler
from rl_gym_training.models.actor_critic import ActorCritic
from rl_gym_training.models.lstm_world_model import LSTMWorldModel
from rl_gym_training.models.policy_network import PolicyNetwork
from rl_gym_training.rl.environment import WorkoutWorldModelEnv
from rl_gym_training.rl.reward import RewardFunction
from rl_gym_training.services.evaluation_service import evaluate_policy, evaluate_random_policy
from rl_gym_training.services.train_a2c_service import run_a2c_training
from rl_gym_training.services.train_lstm_service import train_world_model
from rl_gym_training.services.train_reinforce_service import run_reinforce_training
from rl_gym_training.shared.config import AppConfig, load_config
from rl_gym_training.shared.paths import resolve_path
from rl_gym_training.shared.seed import set_seed


@dataclass(frozen=True)
class PreparedData:
    train: WorkoutSequenceDataset
    validation: WorkoutSequenceDataset
    test: WorkoutSequenceDataset
    initial_states: np.ndarray


class RLGymTrainingSDK:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()
        set_seed(self.config.seed)

    def prepare_data(self) -> PreparedData:
        frame = load_workout_data(self.config.data, self.config.seed)
        splits = chronological_split(frame, self.config.data)
        scaler = fit_scaler(splits.train, self.config.data.state_columns)
        train_frame = apply_scaler(splits.train, self.config.data.state_columns, scaler)
        validation_frame = apply_scaler(splits.validation, self.config.data.state_columns, scaler)
        test_frame = apply_scaler(splits.test, self.config.data.state_columns, scaler)
        self.config.data.processed_dir.mkdir(parents=True, exist_ok=True)
        train_frame.to_csv(self.config.data.processed_dir / "train.csv", index=False)
        validation_frame.to_csv(self.config.data.processed_dir / "validation.csv", index=False)
        test_frame.to_csv(self.config.data.processed_dir / "test.csv", index=False)
        return PreparedData(
            train=WorkoutSequenceDataset(train_frame, self.config.data),
            validation=WorkoutSequenceDataset(validation_frame, self.config.data),
            test=WorkoutSequenceDataset(test_frame, self.config.data),
            initial_states=train_frame[self.config.data.state_columns].to_numpy(dtype=np.float32),
        )

    def train_world_model(self, checkpoint_path: str | Path = "results/lstm_world_model.pt"):
        prepared = self.prepare_data()
        return train_world_model(
            prepared.train, prepared.validation, self.config, resolve_path(checkpoint_path)
        )

    def make_environment(self, checkpoint_path: str | Path = "results/lstm_world_model.pt"):
        prepared = self.prepare_data()
        checkpoint = resolve_path(checkpoint_path)
        if checkpoint.exists():
            world_model = LSTMWorldModel.load(checkpoint)
        else:
            result = train_world_model(prepared.train, prepared.validation, self.config, checkpoint)
            world_model = LSTMWorldModel.load(result.checkpoint_path)
        return WorkoutWorldModelEnv(
            world_model=world_model,
            reward_function=RewardFunction(self.config.reward),
            initial_states=prepared.initial_states,
            sequence_length=self.config.data.sequence_length,
            episode_length=self.config.rl.episode_length,
            seed=self.config.seed,
        )

    def train_reinforce(self) -> dict[str, object]:
        env = self.make_environment()
        policy = PolicyNetwork(len(self.config.data.state_columns), N_ACTIONS)
        metrics = run_reinforce_training(policy, env, self.config)
        evaluation = evaluate_policy(policy, env, episodes=2)
        output = {"training": asdict(metrics), "evaluation": asdict(evaluation)}
        self._write_json("results/reinforce_metrics.json", output)
        return output

    def train_a2c(self) -> dict[str, object]:
        env = self.make_environment()
        model = ActorCritic(len(self.config.data.state_columns), N_ACTIONS)
        metrics = run_a2c_training(model, env, self.config)
        evaluation = evaluate_policy(model.actor, env, episodes=2)
        output = {"training": asdict(metrics), "evaluation": asdict(evaluation)}
        self._write_json("results/a2c_metrics.json", output)
        return output

    def evaluate_random(self) -> dict[str, object]:
        env = self.make_environment()
        output = asdict(evaluate_random_policy(env, episodes=2))
        self._write_json("results/random_policy_metrics.json", output)
        return output

    @staticmethod
    def _write_json(path: str | Path, payload: dict[str, object]) -> None:
        target = resolve_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
