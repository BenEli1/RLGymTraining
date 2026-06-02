"""A2C service wrapper."""

from __future__ import annotations

from rl_gym_training.models.actor_critic import ActorCritic
from rl_gym_training.rl.a2c import A2CMetrics, train_a2c
from rl_gym_training.shared.config import AppConfig


def run_a2c_training(model: ActorCritic, env, config: AppConfig) -> A2CMetrics:
    return train_a2c(
        model=model,
        env=env,
        episodes=config.rl.a2c_episodes,
        gamma=config.rl.gamma,
        actor_learning_rate=config.rl.policy_learning_rate,
        critic_learning_rate=config.rl.critic_learning_rate,
        entropy_coefficient=config.rl.entropy_coefficient,
    )
