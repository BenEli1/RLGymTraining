"""Stochastic policy network."""

from __future__ import annotations

import torch
from torch import nn


class PolicyNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, state: torch.Tensor, action_mask: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.net(state)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask.bool(), -1e9)
        return torch.softmax(logits, dim=-1)
