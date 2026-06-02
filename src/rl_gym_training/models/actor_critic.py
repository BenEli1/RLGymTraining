"""Actor-Critic network components."""

from __future__ import annotations

import torch
from torch import nn

from rl_gym_training.models.policy_network import PolicyNetwork


class CriticNetwork(nn.Module):
    def __init__(self, state_size: int, hidden_size: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


class ActorCritic(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 32) -> None:
        super().__init__()
        self.actor = PolicyNetwork(state_size, action_size, hidden_size)
        self.critic = CriticNetwork(state_size, hidden_size)
