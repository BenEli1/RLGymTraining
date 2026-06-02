"""REINFORCE policy-gradient utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.distributions import Categorical

from rl_gym_training.models.policy_network import PolicyNetwork
from rl_gym_training.rl.action_masking import valid_action_mask


@dataclass(frozen=True)
class ReinforceMetrics:
    episode_returns: list[float]
    losses: list[float]


def discounted_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    returns: list[float] = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + gamma * running
        returns.append(running)
    return torch.tensor(list(reversed(returns)), dtype=torch.float32)


def reinforce_loss(log_probs: list[torch.Tensor], returns: torch.Tensor) -> torch.Tensor:
    stacked = torch.stack(log_probs)
    return -(stacked * returns).sum()


def train_reinforce(
    policy: PolicyNetwork,
    env,
    episodes: int,
    gamma: float,
    learning_rate: float,
    normalize_returns: bool,
) -> ReinforceMetrics:
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    episode_returns: list[float] = []
    losses: list[float] = []
    for _ in range(episodes):
        state = env.reset()
        log_probs: list[torch.Tensor] = []
        rewards: list[float] = []
        done = False
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            mask = torch.tensor(valid_action_mask(state, env.previous_action)).unsqueeze(0)
            probs = policy(state_tensor, mask)
            dist = Categorical(probs=probs)
            action = dist.sample()
            next_state, reward, done, _ = env.step(int(action.item()))
            log_probs.append(dist.log_prob(action).squeeze(0))
            rewards.append(float(reward))
            state = next_state
        returns = discounted_returns(rewards, gamma)
        if normalize_returns and len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        loss = reinforce_loss(log_probs, returns)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        episode_returns.append(float(sum(rewards)))
        losses.append(float(loss.detach()))
    return ReinforceMetrics(episode_returns=episode_returns, losses=losses)
