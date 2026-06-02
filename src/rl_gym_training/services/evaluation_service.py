"""Evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from rl_gym_training.models.policy_network import PolicyNetwork
from rl_gym_training.rl.action_masking import valid_action_mask


@dataclass(frozen=True)
class PolicyEvaluation:
    average_return: float
    safety_violations: int
    action_distribution: dict[int, int]


def evaluate_policy(policy: PolicyNetwork, env, episodes: int) -> PolicyEvaluation:
    returns: list[float] = []
    safety_violations = 0
    action_counts: dict[int, int] = {}
    for _ in range(episodes):
        state = env.reset()
        done = False
        episode_return = 0.0
        while not done:
            mask_array = valid_action_mask(state, env.previous_action)
            with torch.no_grad():
                probs = policy(
                    torch.tensor(state, dtype=torch.float32).unsqueeze(0),
                    torch.tensor(mask_array).unsqueeze(0),
                )
            action = int(torch.argmax(probs, dim=-1).item())
            state, reward, done, info = env.step(action)
            episode_return += float(reward)
            safety_violations += int(not bool(info["valid_action"]))
            action_counts[action] = action_counts.get(action, 0) + 1
        returns.append(episode_return)
    return PolicyEvaluation(
        average_return=float(np.mean(returns)),
        safety_violations=safety_violations,
        action_distribution=action_counts,
    )


def evaluate_random_policy(env, episodes: int) -> PolicyEvaluation:
    rng = np.random.default_rng(123)
    returns: list[float] = []
    safety_violations = 0
    action_counts: dict[int, int] = {}
    for _ in range(episodes):
        state = env.reset()
        done = False
        total = 0.0
        while not done:
            valid_actions = np.flatnonzero(valid_action_mask(state, env.previous_action))
            action = int(rng.choice(valid_actions))
            state, reward, done, info = env.step(action)
            total += float(reward)
            safety_violations += int(not bool(info["valid_action"]))
            action_counts[action] = action_counts.get(action, 0) + 1
        returns.append(total)
    return PolicyEvaluation(float(np.mean(returns)), safety_violations, action_counts)
