"""Safety action masking."""

from __future__ import annotations

import numpy as np
import torch

from rl_gym_training.data.data_loader import (
    ACTION_CARDIO,
    ACTION_MIXED,
    ACTION_REST,
    ACTION_STRENGTH,
    N_ACTIONS,
)


def valid_action_mask(
    state: np.ndarray | torch.Tensor, previous_action: int | None = None
) -> np.ndarray:
    values = state.detach().cpu().numpy() if isinstance(state, torch.Tensor) else state
    readiness, fatigue, _, _, soreness = values[:5]
    mask = np.ones(N_ACTIONS, dtype=bool)
    if fatigue > 0.75 or soreness > 0.7 or readiness < 0.25:
        mask[[ACTION_CARDIO, ACTION_STRENGTH, ACTION_MIXED]] = False
    if previous_action in {ACTION_STRENGTH, ACTION_MIXED} and fatigue > 0.55:
        mask[[ACTION_STRENGTH, ACTION_MIXED]] = False
    mask[ACTION_REST] = True
    return mask
