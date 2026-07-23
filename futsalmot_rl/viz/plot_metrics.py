"""Metrics plotting utilities for FutsalMOT-RL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


def plot_loss_curve(
    log_path: str | Path,
    output_path: str | Path,
    title: str = "Training Loss",
) -> None:
    """Plot training/validation loss curve from a JSONL log file.

    Expects each line: {"epoch": int, "train_loss": float, "val_loss": float}
    """
    epochs: list[int] = []
    train_losses: list[float] = []
    val_losses: list[float] = []

    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                epochs.append(entry.get("epoch", len(epochs) + 1))
                train_losses.append(entry.get("train_loss", 0.0))
                val_losses.append(entry.get("val_loss", 0.0))

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(epochs, train_losses, label="Train", color="#2196F3", linewidth=2)
    if any(v != 0.0 for v in val_losses):
        ax.plot(epochs, val_losses, label="Validation", color="#FF6F00", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_reward_curve(
    log_path: str | Path,
    output_path: str | Path,
    title: str = "PPO Training Reward",
) -> None:
    """Plot reward curve from a PPO JSONL log file.

    Expects each line: {"global_step": int, "mean_episode_reward": float, ...}
    """
    steps: list[int] = []
    rewards: list[float] = []

    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                steps.append(entry.get("global_step", 0))
                rewards.append(entry.get("mean_episode_reward", 0.0))

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(steps, rewards, label="Mean Episode Reward", color="#4CAF50", linewidth=2)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_bar(
    metrics: dict[str, dict[str, Any]],
    output_path: str | Path,
    metric_keys: list[str] | None = None,
    title: str = "Policy Comparison",
) -> None:
    """Plot a grouped bar chart comparing multiple policies.

    Args:
        metrics: {policy_name: {metric_key: value}}
        output_path: Output PNG path.
        metric_keys: List of metric keys to plot. If None, use all numeric ones.
        title: Chart title.
    """
    if not metrics:
        return

    # Determine metric keys
    if metric_keys is None:
        # Use all numeric metrics from the first policy
        first = next(iter(metrics.values()))
        metric_keys = [k for k, v in first.items() if isinstance(v, (int, float))]
        # Limit to meaningful ones
        metric_keys = [k for k in metric_keys if not k.startswith("_")]

    if not metric_keys:
        return

    policy_names = list(metrics.keys())
    n_metrics = len(metric_keys)
    n_policies = len(policy_names)

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    colors = ["#2196F3", "#FF6F00", "#4CAF50"]

    for i, metric_key in enumerate(metric_keys):
        ax = axes[i]
        values = []
        for pname in policy_names:
            v = metrics[pname].get(metric_key, 0)
            values.append(v if isinstance(v, (int, float)) else 0)

        bars = ax.bar(policy_names, values, color=colors[:n_policies])
        ax.set_title(metric_key, fontsize=10)
        ax.set_ylabel("Value")
        ax.tick_params(axis="x", rotation=45)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.1f}" if isinstance(val, float) else str(val),
                ha="center", va="bottom", fontsize=8,
            )

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
