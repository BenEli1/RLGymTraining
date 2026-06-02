"""Configurable reward for safe workout progress."""

from __future__ import annotations

import numpy as np

from rl_gym_training.data.data_loader import ACTION_REST
from rl_gym_training.shared.config import RewardConfig


class RewardFunction:
    """Reward balances progress, readiness, and safety."""

    def __init__(self, config: RewardConfig) -> None:
        self.config = config

    def __call__(
        self,
        state: np.ndarray,
        next_state: np.ndarray,
        action: int,
        valid_action: bool = True,
        previous_action: int | None = None,
    ) -> float:
        readiness_delta = next_state[0] - state[0]
        fatigue_delta = next_state[1] - state[1]
        strength_delta = next_state[2] - state[2]
        endurance_delta = next_state[3] - state[3]
        soreness = next_state[4]
        progress = strength_delta + endurance_delta
        reward = self.config.progress_weight * progress
        reward += self.config.readiness_weight * readiness_delta
        reward -= self.config.fatigue_penalty * max(0.0, fatigue_delta)
        reward -= self.config.overload_penalty * max(0.0, soreness - 0.7)
        if previous_action is not None and action == previous_action and action != ACTION_REST:
            reward += self.config.consistency_bonus
        if not valid_action:
            reward -= self.config.invalid_action_penalty
        return float(np.nan_to_num(reward))
