"""MLP policy networks for FutsalMOT-RL."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPPolicy(nn.Module):
    """Simple MLP policy for continuous action space.

    Architecture: obs → FC(128) → ReLU → FC(128) → ReLU → FC(act_dim) → Tanh
    Outputs actions in [-1, 1].
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: list[int] | None = None,
        act_dim: int = 2,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [128, 128]

        layers: list[nn.Module] = []
        in_size = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            in_size = h
        layers.append(nn.Linear(in_size, act_dim))

        self.net = nn.Sequential(*layers)
        self._obs_dim = obs_dim
        self._act_dim = act_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass, returns actions in [-1, 1]."""
        return torch.tanh(self.net(obs))

    def get_action(
        self, obs: torch.Tensor | np.ndarray, deterministic: bool = True
    ) -> np.ndarray:
        """Get action for inference.

        Args:
            obs: Observation tensor or array.
            deterministic: If True, return mean action (no noise).

        Returns:
            Action array of shape (act_dim,).
        """
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        self.eval()
        with torch.no_grad():
            action = self.forward(obs)

        return action.squeeze(0).cpu().numpy()


class MLPActorCritic(nn.Module):
    """Actor-Critic network for PPO.

    Actor: MLPPolicy (shared or separate)
    Critic: Separate MLP for value function.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: list[int] | None = None,
        act_dim: int = 2,
        shared_backbone: bool = False,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [128, 128]

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.shared_backbone = shared_backbone

        if shared_backbone:
            backbone_layers: list[nn.Module] = []
            in_size = obs_dim
            for h in hidden_sizes:
                backbone_layers.append(nn.Linear(in_size, h))
                backbone_layers.append(nn.ReLU())
                in_size = h
            self.backbone = nn.Sequential(*backbone_layers)

            self.actor_head = nn.Linear(in_size, act_dim)
            self.critic_head = nn.Linear(in_size, 1)
        else:
            self.backbone = None
            self.actor = MLPPolicy(obs_dim, hidden_sizes, act_dim)
            critic_layers: list[nn.Module] = []
            in_size = obs_dim
            for h in hidden_sizes:
                critic_layers.append(nn.Linear(in_size, h))
                critic_layers.append(nn.ReLU())
                in_size = h
            critic_layers.append(nn.Linear(in_size, 1))
            self.critic = nn.Sequential(*critic_layers)

        # Fixed log_std for Gaussian policy (PPO)
        self.log_std = nn.Parameter(torch.zeros(act_dim) - 0.5)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (action, value)."""
        if self.shared_backbone and self.backbone is not None:
            features = self.backbone(obs)
            action = torch.tanh(self.actor_head(features))
            value = self.critic_head(features)
        else:
            action = self.actor(obs)
            value = self.critic(obs)
        return action, value

    def get_action_and_value(
        self, obs: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log_prob, entropy, and value for PPO.

        Args:
            obs: Observation tensor.
            action: Optional action to evaluate. If None, sample from policy.

        Returns:
            (action, log_prob, entropy, value)
        """
        mean_action, value = self.forward(obs)

        # For continuous actions, use a Gaussian policy with learnable log_std
        std = self.log_std.exp().expand_as(mean_action)

        dist = torch.distributions.Normal(mean_action, std)

        if action is None:
            # Sample raw Gaussian, then tanh-squash
            raw_action = dist.rsample()          # u ~ N(mean, std)
            action = torch.tanh(raw_action)       # a = tanh(u)
        else:
            # Action is already tanh-squashed — invert to get raw sample u
            raw_action = torch.clamp(action, -0.999, 0.999)
            raw_action = 0.5 * torch.log((1.0 + raw_action) / (1.0 - raw_action))

        # Compute log-probability with tanh correction:
        #   log pi(a|s) = log N(u | mean, std) - sum log(1 - tanh(u)^2)
        log_prob = dist.log_prob(raw_action)
        log_prob -= torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)

        entropy = dist.entropy().sum(dim=-1)

        return action, log_prob, entropy, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get value estimate for an observation."""
        if self.shared_backbone and self.backbone is not None:
            features = self.backbone(obs)
            return self.critic_head(features)
        return self.critic(obs)

    def get_action(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Get action for inference (compatible with non-PPO usage)."""
        if isinstance(obs, np.ndarray):
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
        else:
            obs_tensor = obs.float().unsqueeze(0)

        self.eval()
        with torch.no_grad():
            if deterministic:
                mean_action, _ = self.forward(obs_tensor)
                return mean_action.squeeze(0).cpu().numpy()
            else:
                action, _, _, _ = self.get_action_and_value(obs_tensor)
                return action.squeeze(0).cpu().numpy()
