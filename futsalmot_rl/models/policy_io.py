"""Policy model persistence with architecture-aware save/load."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from futsalmot_rl.models.mlp_policy import MLPActorCritic, MLPPolicy


def save_policy(
    policy: torch.nn.Module,
    path: str | Path,
    config: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> None:
    """Save a policy model, architecture, and optional metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    save_dict: dict[str, Any] = {
        "checkpoint_version": 2,
        "model_state_dict": policy.state_dict(),
    }
    if config is not None:
        save_dict["config"] = config
    if metrics is not None:
        save_dict["metrics"] = metrics

    # Architecture metadata
    obs_dim = getattr(policy, "obs_dim", getattr(policy, "_obs_dim", None))
    act_dim = getattr(policy, "act_dim", getattr(policy, "_act_dim", None))
    hidden_sizes = getattr(policy, "hidden_sizes", None)

    if isinstance(policy, MLPActorCritic):
        save_dict["model_type"] = "MLPActorCritic"
        save_dict["architecture"] = {
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "hidden_sizes": list(hidden_sizes) if hidden_sizes else [128, 128],
            "shared_backbone": bool(policy.shared_backbone),
        }
    elif isinstance(policy, MLPPolicy):
        save_dict["model_type"] = "MLPPolicy"
        save_dict["architecture"] = {
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "hidden_sizes": list(hidden_sizes) if hidden_sizes else [128, 128],
        }

    torch.save(save_dict, str(path))


def _infer_architecture_from_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Infer model architecture by inspecting state dict keys."""
    arch: dict[str, Any] = {}
    has_backbone = any(k.startswith("backbone.") for k in state_dict)
    has_actor = any(k.startswith("actor.") for k in state_dict)
    has_critic = any(k.startswith("critic.") for k in state_dict)

    if has_actor or has_critic:
        arch["model_type"] = "MLPActorCritic"
    else:
        arch["model_type"] = "MLPPolicy"

    arch["shared_backbone"] = has_backbone

    # Infer hidden sizes from net.0.weight or actor.net.0.weight
    prefix = "actor.net.0.weight" if has_actor else "net.0.weight"
    weight = state_dict.get(prefix)
    if weight is not None:
        arch["obs_dim"] = weight.shape[1]
        arch["hidden_sizes"] = [weight.shape[0], 128]  # best guess
        # Look for deeper layers
        layer2_key = prefix.replace("0.weight", "2.weight")
        w2 = state_dict.get(layer2_key)
        if w2 is not None:
            arch["hidden_sizes"][1] = w2.shape[0]
    else:
        arch["obs_dim"] = 38
        arch["hidden_sizes"] = [128, 128]

    # act_dim from last layer
    for k in state_dict:
        if k.endswith(".weight") and "backbone" not in k and "critic" not in k and "head" not in k:
            parts = k.split(".")
            if len(parts) >= 2 and parts[-2].isdigit():
                layer_idx = int(parts[-2])
                if layer_idx >= 4:  # deep enough to be output
                    arch["act_dim"] = state_dict[k].shape[0]
                    break
    if "act_dim" not in arch:
        arch["act_dim"] = 2

    return arch


def load_policy(
    path: str | Path,
    device: torch.device | None = None,
) -> tuple[torch.nn.Module, dict[str, Any] | None, dict[str, Any] | None]:
    """Load a policy model, restoring architecture from checkpoint.

    Returns:
        (policy, config, metrics)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dict = torch.load(str(path), map_location=device, weights_only=True)
    state_dict = save_dict["model_state_dict"]
    architecture = save_dict.get("architecture")
    config = save_dict.get("config")
    metrics = save_dict.get("metrics")

    if architecture is not None:
        obs_dim = int(architecture["obs_dim"])
        act_dim = int(architecture["act_dim"])
        hidden_sizes = [int(v) for v in architecture.get("hidden_sizes", [128, 128])]
        shared_backbone = bool(architecture.get("shared_backbone", False))
        model_type = save_dict.get("model_type", architecture.get("model_type", "MLPPolicy"))

        if model_type == "MLPActorCritic":
            policy: torch.nn.Module = MLPActorCritic(
                obs_dim, hidden_sizes=hidden_sizes, act_dim=act_dim,
                shared_backbone=shared_backbone,
            )
        else:
            policy = MLPPolicy(obs_dim, hidden_sizes=hidden_sizes, act_dim=act_dim)
    else:
        # Legacy checkpoint — infer from state dict and config
        inferred = _infer_architecture_from_state_dict(state_dict)
        obs_dim = int(save_dict.get("obs_dim", inferred.get("obs_dim", 38)))
        act_dim = int(save_dict.get("act_dim", inferred.get("act_dim", 2)))
        hidden_sizes = (
            config.get("hidden_sizes", [128, 128])
            if config and isinstance(config, dict)
            else inferred.get("hidden_sizes", [128, 128])
        )

        if inferred.get("model_type") == "MLPActorCritic":
            policy = MLPActorCritic(
                obs_dim, hidden_sizes=hidden_sizes, act_dim=act_dim,
                shared_backbone=inferred.get("shared_backbone", False),
            )
        else:
            policy = MLPPolicy(obs_dim, hidden_sizes=hidden_sizes, act_dim=act_dim)

    policy.load_state_dict(state_dict)
    policy.to(device)
    policy.eval()

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

    Returns:
        (epoch, loss)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(str(path), map_location=device, weights_only=True)
    policy.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint.get("epoch", 0), checkpoint.get("loss", float("inf"))
