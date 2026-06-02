"""Learned-world workout environment."""

from __future__ import annotations

from collections import deque

import numpy as np
import torch

from rl_gym_training.data.data_loader import N_ACTIONS
from rl_gym_training.models.lstm_world_model import LSTMWorldModel
from rl_gym_training.rl.action_masking import valid_action_mask
from rl_gym_training.rl.reward import RewardFunction


class WorkoutWorldModelEnv:
    def __init__(
        self,
        world_model: LSTMWorldModel,
        reward_function: RewardFunction,
        initial_states: np.ndarray,
        sequence_length: int,
        episode_length: int,
        seed: int,
    ) -> None:
        self.world_model = world_model
        self.reward_function = reward_function
        self.initial_states = initial_states.astype(np.float32)
        self.sequence_length = sequence_length
        self.episode_length = episode_length
        self.rng = np.random.default_rng(seed)
        self.history: deque[np.ndarray] = deque(maxlen=sequence_length)
        self.previous_action: int | None = None
        self.steps = 0
        self.state = self.initial_states[0]

    def reset(self) -> np.ndarray:
        self.steps = 0
        self.previous_action = None
        self.state = self.initial_states[int(self.rng.integers(0, len(self.initial_states)))].copy()
        self.history.clear()
        for _ in range(self.sequence_length):
            self.history.append(np.concatenate([self.state, _one_hot(0)]).astype(np.float32))
        return self.state.copy()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, float | bool]]:
        mask = valid_action_mask(self.state, self.previous_action)
        valid = bool(mask[action])
        sequence = torch.tensor(np.stack(self.history), dtype=torch.float32).unsqueeze(0)
        sequence[0, -1, len(self.state) :] = torch.tensor(_one_hot(action), dtype=torch.float32)
        with torch.no_grad():
            next_state = self.world_model(sequence).squeeze(0).cpu().numpy().astype(np.float32)
        next_state = np.clip(next_state, -3.0, 3.0)
        reward = self.reward_function(self.state, next_state, action, valid, self.previous_action)
        self.previous_action = action
        self.state = next_state
        self.history.append(np.concatenate([self.state, _one_hot(action)]).astype(np.float32))
        self.steps += 1
        done = self.steps >= self.episode_length
        return self.state.copy(), reward, done, {"valid_action": valid}


def _one_hot(action: int) -> np.ndarray:
    values = np.zeros(N_ACTIONS, dtype=np.float32)
    values[action] = 1.0
    return values
