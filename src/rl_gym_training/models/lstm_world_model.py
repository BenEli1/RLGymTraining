"""LSTM world model for trainee state dynamics."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class LSTMWorldModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        state_size: int,
        hidden_size: int = 32,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, state_size)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(sequence)
        return self.head(output[:, -1, :])

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "state_dict": self.state_dict(),
                "input_size": self.input_size,
                "state_size": self.state_size,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> LSTMWorldModel:
        payload = torch.load(path, map_location="cpu")
        model = cls(
            input_size=payload["input_size"],
            state_size=payload["state_size"],
            hidden_size=payload["hidden_size"],
            num_layers=payload["num_layers"],
        )
        model.load_state_dict(payload["state_dict"])
        return model
