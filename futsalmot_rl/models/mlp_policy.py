"""MLP policy networks for FutsalMOT-RL.

Architecture notes (correctness-critical):
  - MLPPolicy.forward() returns RAW (unbounded) mean — NO tanh.
  - Tanh-squashing is applied ONCE, in the policy distribution layer.
  - MLPActorCritic.forward() applies tanh for the deterministic action.
  - get_action_and_value() constructs Normal(mean, std), rsample() → tanh().
  - Log-prob includes the tanh Jacobian correction:
      log pi(a|s) = log N(u|mu,sigma) - sum_i log(1 - a_i^2)
    where u = atanh(a) is the pre-tanh sample.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class MLPPolicy(nn.Module):
    """Simple MLP policy for continuous action space.

    forward() returns RAW unbounded mean — does NOT apply tanh.
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
        # Final linear layer — NO tanh (tanh happens in get_action_and_value)
        layers.append(nn.Linear(in_size, act_dim))

        self.net = nn.Sequential(*layers)
        self._obs_dim = obs_dim
        self._act_dim = act_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass, returns RAW unbounded mean."""
        return self.net(obs)

    def get_action(self, obs: torch.Tensor | np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Get action for inference. Applies tanh-squash.

        Args:
            obs: Observation tensor or array.
            deterministic: If True, return tanh(mean) (no sampling noise).

        Returns:
            Action array of shape (act_dim,) in [-1, 1].
        """
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        self.eval()
        with torch.no_grad():
            raw_mean = self.forward(obs)
            if deterministic:
                action = torch.tanh(raw_mean)
            else:
                std = raw_mean.new_ones(raw_mean.shape[-1]) * 0.5
                dist = torch.distributions.Normal(raw_mean, std.expand_as(raw_mean))
                action = torch.tanh(dist.rsample())
        return action.squeeze(0).cpu().numpy()


class MLPActorCritic(nn.Module):
    """Actor-Critic network for PPO.

    Actor: MLPPolicy that outputs RAW unbounded mean.
    Critic: Separate MLP for value function.

    Forward returns action=tanh(raw_mean) for deterministic evaluation.
    get_action_and_value handles the full stochastic policy:
      Normal(raw_mean, std) → rsample() → tanh() → action
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
            self.actor_head = nn.Linear(in_size, act_dim)  # raw mean, NO tanh
            self.critic_head = nn.Linear(in_size, 1)
        else:
            self.backbone = None
            self.actor = MLPPolicy(obs_dim, hidden_sizes, act_dim)  # raw mean
            critic_layers: list[nn.Module] = []
            in_size = obs_dim
            for h in hidden_sizes:
                critic_layers.append(nn.Linear(in_size, h))
                critic_layers.append(nn.ReLU())
                in_size = h
            critic_layers.append(nn.Linear(in_size, 1))
            self.critic = nn.Sequential(*critic_layers)

        # Learnable log_std for Gaussian policy
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (action, value).

        Action is tanh(raw_mean) — deterministic mode.
        Value is the scalar state-value estimate.
        """
        if self.shared_backbone and self.backbone is not None:
            features = self.backbone(obs)
            raw_mean = self.actor_head(features)
            value = self.critic_head(features)
        else:
            raw_mean = self.actor(obs)  # MLPPolicy returns raw mean
            value = self.critic(obs)
        action = torch.tanh(raw_mean)
        return action, value

    def get_action_and_value(
        self, obs: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log_prob, entropy, and value for PPO.

        The policy is a tanh-squashed Gaussian:
          u ~ N(raw_mean, std)    # raw (unbounded) space
          a = tanh(u)              # single tanh squashing

        Args:
            obs: Observation tensor.
            action: Optional action to evaluate. If None, sample from policy.

        Returns:
            (action, log_prob, entropy, value)
        """
        raw_mean, value = self.actor(obs), self.critic(obs)  # raw mean, no tanh
        std = self.log_std.exp().expand_as(raw_mean)

        dist = torch.distributions.Normal(raw_mean, std)

        if action is None:
            # Sample raw Gaussian → tanh-squash
            raw_sample = dist.rsample()  # u ~ N(raw_mean, std)
            action = torch.tanh(raw_sample)  # a = tanh(u) — single squash
        else:
            # Action is tanh-squashed — invert to get raw sample for log-prob
            raw_sample = torch.clamp(action, -0.999, 0.999)
            raw_sample = 0.5 * torch.log((1.0 + raw_sample) / (1.0 - raw_sample))

        # Log-probability with tanh Jacobian correction:
        #   log pi(a|s) = log N(u|raw_mean,std) - sum log(1 - tanh(u)^2)
        log_prob = dist.log_prob(raw_sample)
        log_prob -= torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)

        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy, value

    def get_action(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Get action for inference.

        Args:
            obs: Observation array.
            deterministic: If True, return tanh(raw_mean).

        Returns:
            Action array in [-1, 1].
        """
        if isinstance(obs, np.ndarray):
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
        else:
            obs_tensor = obs.float().unsqueeze(0)

        self.eval()
        with torch.no_grad():
            if deterministic:
                action, _ = self.forward(obs_tensor)
            else:
                action, _, _, _ = self.get_action_and_value(obs_tensor)
        return action.squeeze(0).cpu().numpy()
