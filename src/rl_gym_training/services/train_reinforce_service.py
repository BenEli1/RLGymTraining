"""REINFORCE service wrapper."""

from __future__ import annotations

from rl_gym_training.models.policy_network import PolicyNetwork
from rl_gym_training.rl.reinforce import ReinforceMetrics, train_reinforce
from rl_gym_training.shared.config import AppConfig


def run_reinforce_training(policy: PolicyNetwork, env, config: AppConfig) -> ReinforceMetrics:
    return train_reinforce(
        policy=policy,
        env=env,
        episodes=config.rl.reinforce_episodes,
        gamma=config.rl.gamma,
        learning_rate=config.rl.policy_learning_rate,
        normalize_returns=config.rl.normalize_returns,
    )
