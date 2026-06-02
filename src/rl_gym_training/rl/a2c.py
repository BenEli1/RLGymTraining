"""Advantage Actor-Critic training utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical

from rl_gym_training.models.actor_critic import ActorCritic
from rl_gym_training.rl.action_masking import valid_action_mask


@dataclass(frozen=True)
class A2CMetrics:
    episode_returns: list[float]
    actor_losses: list[float]
    critic_losses: list[float]


def td_target(
    reward: torch.Tensor, next_value: torch.Tensor, done: torch.Tensor, gamma: float
) -> torch.Tensor:
    return reward + gamma * next_value * (1.0 - done)


def advantage(target: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return target - value


def a2c_losses(
    log_prob: torch.Tensor,
    value: torch.Tensor,
    target: torch.Tensor,
    entropy: torch.Tensor,
    entropy_coefficient: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    adv = advantage(target, value)
    actor_loss = -(log_prob * adv.detach()) - entropy_coefficient * entropy
    critic_loss = nn.functional.mse_loss(value, target.detach())
    return actor_loss.mean(), critic_loss


def train_a2c(
    model: ActorCritic,
    env,
    episodes: int,
    gamma: float,
    actor_learning_rate: float,
    critic_learning_rate: float,
    entropy_coefficient: float,
) -> A2CMetrics:
    actor_optimizer = torch.optim.Adam(model.actor.parameters(), lr=actor_learning_rate)
    critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=critic_learning_rate)
    actor_losses: list[float] = []
    critic_losses: list[float] = []
    episode_returns: list[float] = []
    for _ in range(episodes):
        state = env.reset()
        total_return = 0.0
        done = False
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            mask = torch.tensor(valid_action_mask(state, env.previous_action)).unsqueeze(0)
            probs = model.actor(state_tensor, mask)
            dist = Categorical(probs=probs)
            action = dist.sample()
            value = model.critic(state_tensor)
            next_state, reward, done, _ = env.step(int(action.item()))
            next_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                next_value = model.critic(next_tensor)
                target = td_target(
                    torch.tensor([reward], dtype=torch.float32),
                    next_value,
                    torch.tensor([float(done)], dtype=torch.float32),
                    gamma,
                )
            actor_loss, critic_loss = a2c_losses(
                dist.log_prob(action),
                value,
                target,
                dist.entropy(),
                entropy_coefficient,
            )
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            (actor_loss + critic_loss).backward()
            actor_optimizer.step()
            critic_optimizer.step()
            actor_losses.append(float(actor_loss.detach()))
            critic_losses.append(float(critic_loss.detach()))
            total_return += float(reward)
            state = next_state
        episode_returns.append(total_return)
    return A2CMetrics(episode_returns, actor_losses, critic_losses)
