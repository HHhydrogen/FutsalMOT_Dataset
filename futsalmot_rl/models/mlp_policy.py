"""MLP policy networks for FutsalMOT-RL."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _get_device(module: nn.Module) -> torch.device:
    """Get the device of a module's first parameter."""
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _to_tensor(obs: torch.Tensor | np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert numpy array or tensor to tensor on the given device."""
    if isinstance(obs, np.ndarray):
        return torch.as_tensor(obs, dtype=torch.float32, device=device)
    return obs.to(device)


class MLPPolicy(nn.Module):
    """Simple MLP policy. forward() returns RAW unbounded mean — NO tanh."""

    def __init__(self, obs_dim: int, hidden_sizes: list[int] | None = None, act_dim: int = 2):
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
        self.hidden_sizes = tuple(int(v) for v in (hidden_sizes or [128, 128]))

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def act_dim(self) -> int:
        return self._act_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return RAW unbounded mean — NO tanh."""
        return self.net(obs)

    def deterministic_action_tensor(self, obs: torch.Tensor) -> torch.Tensor:
        """Return tanh-squashed action tensor, same as get_action(..., deterministic=True)."""
        return torch.tanh(self.forward(obs))

    def get_action(self, obs: torch.Tensor | np.ndarray, deterministic: bool = True) -> np.ndarray:
        device = _get_device(self)
        obs_t = _to_tensor(obs, device)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        self.eval()
        with torch.no_grad():
            raw_mean = self.forward(obs_t)
            if deterministic:
                action = torch.tanh(raw_mean)
            else:
                std = raw_mean.new_ones(raw_mean.shape[-1]) * 0.5
                dist = torch.distributions.Normal(raw_mean, std.expand_as(raw_mean))
                action = torch.tanh(dist.rsample())
        return action.squeeze(0).cpu().numpy()


class MLPActorCritic(nn.Module):
    """Actor-Critic network for PPO.

    Supports both shared and non-shared backbone.
    """

    def __init__(self, obs_dim: int, hidden_sizes: list[int] | None = None, act_dim: int = 2, shared_backbone: bool = False):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [128, 128]
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.shared_backbone = bool(shared_backbone)
        self.hidden_sizes = tuple(int(v) for v in (hidden_sizes or [128, 128]))

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

        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    # ── Internal: unified raw_mean + value path ─────────────────

    def _raw_mean_and_value(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (raw_mean, value) regardless of shared/non-shared backbone."""
        if self.shared_backbone and self.backbone is not None:
            features = self.backbone(obs)
            raw_mean = self.actor_head(features)
            value = self.critic_head(features)
        else:
            raw_mean = self.actor(obs)
            value = self.critic(obs)
        return raw_mean, value

    # ── Public methods ──────────────────────────────────────────

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Deterministic forward: (action=tanh(raw_mean), value)."""
        raw_mean, value = self._raw_mean_and_value(obs)
        return torch.tanh(raw_mean), value

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stochastic forward for PPO.

        Returns (action, log_prob, entropy, value).
        """
        raw_mean, value = self._raw_mean_and_value(obs)
        std = self.log_std.exp().expand_as(raw_mean)
        dist = torch.distributions.Normal(raw_mean, std)

        if action is None:
            raw_sample = dist.rsample()
            action = torch.tanh(raw_sample)
        else:
            raw_sample = torch.clamp(action, -0.999, 0.999)
            raw_sample = 0.5 * torch.log((1.0 + raw_sample) / (1.0 - raw_sample))

        log_prob = dist.log_prob(raw_sample)
        log_prob -= torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """State-value estimate."""
        _, value = self._raw_mean_and_value(obs)
        return value

    def get_action(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Inference action in [-1, 1]."""
        device = _get_device(self)
        obs_t = _to_tensor(obs, device)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)

        self.eval()
        with torch.no_grad():
            if deterministic:
                action, _ = self.forward(obs_t)
            else:
                action, _, _, _ = self.get_action_and_value(obs_t)
        return action.squeeze(0).cpu().numpy()
