from __future__ import annotations

import torch
from torch import nn


class StudentWorldModel(nn.Module):
    def __init__(
        self,
        obs_dim: int = 4,
        act_dim: int = 1,
        hidden_dim: int = 256,
        num_layers: int = 3,
        use_gru: bool = True,
        delta_limit: float = 3.0,
    ):
        super().__init__()
        self.use_gru = bool(use_gru)
        self.delta_limit = float(delta_limit)

        self.input_layer = nn.Linear(obs_dim + act_dim, hidden_dim)

        self.mid_layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.mid_layer2 = nn.Linear(hidden_dim, hidden_dim)

        self.head = nn.Linear(hidden_dim, obs_dim)
        self.act = nn.SiLU()

    def initial_hidden(self, batch_size: int, device: torch.device):
        return None

    def forward(self, obs_norm: torch.Tensor, act_norm: torch.Tensor, hidden=None):
        if self.training:
            obs_norm = obs_norm + torch.randn_like(obs_norm) * 0.015

        x = self.act(self.input_layer(torch.cat([obs_norm, act_norm], dim=-1)))

        residual = x
        x = self.act(self.mid_layer1(x))
        x = self.act(self.mid_layer2(x) + residual)

        raw_delta = self.head(x)
        delta = self.delta_limit * torch.tanh(raw_delta / self.delta_limit)
        return delta, hidden