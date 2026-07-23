"""Policy model persistence utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from futsalmot_rl.core.rl_io import write_json_atomic
from futsalmot_rl.models.mlp_policy import MLPActorCritic, MLPPolicy


def save_policy(
    policy: torch.nn.Module,
    path: str | Path,
    config: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> None:
    """Save a policy model and optional metadata.

    The saved file is a dict with 'model_state_dict', 'config', and 'metrics'.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    save_dict: dict[str, Any] = {
        "model_state_dict": policy.state_dict(),
    }
    if config is not None:
        save_dict["config"] = config
    if metrics is not None:
        save_dict["metrics"] = metrics

    # Save model type info for loading
    if isinstance(policy, MLPActorCritic):
        save_dict["model_type"] = "MLPActorCritic"
        save_dict["obs_dim"] = policy.obs_dim
        save_dict["act_dim"] = policy.act_dim
    elif isinstance(policy, MLPPolicy):
        save_dict["model_type"] = "MLPPolicy"
        save_dict["obs_dim"] = policy._obs_dim
        save_dict["act_dim"] = policy._act_dim

    torch.save(save_dict, str(path))


def load_policy(
    path: str | Path,
    device: torch.device | None = None,
) -> tuple[torch.nn.Module, dict[str, Any] | None, dict[str, Any] | None]:
    """Load a policy model and its metadata.

    Returns (policy, config, metrics).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dict = torch.load(str(path), map_location=device, weights_only=True)
    model_type = save_dict.get("model_type", "MLPPolicy")
    obs_dim = save_dict.get("obs_dim", 39)
    act_dim = save_dict.get("act_dim", 2)

    if model_type == "MLPActorCritic":
        policy = MLPActorCritic(obs_dim, act_dim=act_dim)
    else:
        policy = MLPPolicy(obs_dim, act_dim=act_dim)

    policy.load_state_dict(save_dict["model_state_dict"])
    policy.to(device)
    policy.eval()

    config = save_dict.get("config")
    metrics = save_dict.get("metrics")
    return policy, config, metrics


def save_checkpoint(
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str | Path,
) -> None:
    """Save a training checkpoint including optimizer state."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        str(path),
    )


def load_checkpoint(
    path: str | Path,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | None = None,
) -> tuple[int, float]:
    """Load a training checkpoint.

    Returns (epoch, loss).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(str(path), map_location=device, weights_only=True)
    policy.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint.get("epoch", 0), checkpoint.get("loss", float("inf"))
