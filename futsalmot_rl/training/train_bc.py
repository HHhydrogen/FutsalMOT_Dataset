"""Behavior Cloning (BC) training for FutsalMOT-RL."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from futsalmot_rl.core.rl_io import write_json_atomic
from futsalmot_rl.core.rl_paths import (
    DEMOS_DIR,
    MODELS_DIR,
    TRAIN_LOGS_DIR,
    VIDEOS_DIR,
    ensure_dirs,
)
from futsalmot_rl.data.demo_dataset import DemoDataset
from futsalmot_rl.features.obs_builder import get_obs_dim
from futsalmot_rl.models.mlp_policy import MLPPolicy
from futsalmot_rl.models.policy_io import save_policy


def train_bc(
    demo_index_path: str | Path = DEMOS_DIR / "demo_index.json",
    model_out: str | Path = MODELS_DIR / "defender_follow_bc_v1.pt",
    config: dict[str, Any] | None = None,
    device: str = "auto",
    video_callback=None,
) -> dict[str, Any]:
    """Run behavior cloning training.

    Args:
        demo_index_path: Path to demo_index.json.
        model_out: Path for the saved model.
        config: Training configuration dict (or None for defaults).
        device: 'auto', 'cpu', or 'cuda'.
        video_callback: Optional callable(policy, epoch, output_dir) for video.

    Returns:
        Summary dict with training metrics.
    """
    # ── Device ──────────────────────────────────────────────────
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"Using device: {device}")

    # ── Config ──────────────────────────────────────────────────
    cfg: dict[str, Any] = {
        "epochs": 100,
        "batch_size": 512,
        "learning_rate": 0.0003,
        "weight_decay": 0.00001,
        "train_split": 0.8,
        "val_split": 0.1,
        "eval_interval_epochs": 5,
        "hidden_sizes": [128, 128],
    }
    if config is not None:
        cfg.update(config)

    obs_dim = get_obs_dim()

    # ── Data ────────────────────────────────────────────────────
    print("Loading training data...")
    train_dataset = DemoDataset(
        demo_index_path,
        split="train",
        train_ratio=cfg["train_split"],
        val_ratio=cfg["val_split"],
    )
    val_dataset = DemoDataset(
        demo_index_path,
        split="val",
        train_ratio=cfg["train_split"],
        val_ratio=cfg["val_split"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples:   {len(val_dataset)}")
    print(f"  Obs dim:       {obs_dim}")

    # ── Model ────────────────────────────────────────────────────
    policy = MLPPolicy(obs_dim, hidden_sizes=cfg["hidden_sizes"], act_dim=2)
    policy.to(device)
    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    loss_fn = nn.MSELoss()

    # ── Training loop ───────────────────────────────────────────
    log_dir = TRAIN_LOGS_DIR / "bc"
    ensure_dirs()

    log_path = log_dir / "train_log.jsonl"
    best_val_loss = float("inf")
    best_epoch = 0
    summary: dict[str, Any] = {
        "config": cfg,
        "epochs": [],
        "best_val_loss": None,
        "best_epoch": None,
        "total_train_time_s": 0.0,
    }

    train_start = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        # Train
        policy.train()
        train_loss = 0.0
        n_train_batches = 0

        for obs_batch, act_batch in train_loader:
            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)

            pred = policy(obs_batch)
            loss = loss_fn(pred, act_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_train_batches += 1

        avg_train_loss = train_loss / max(1, n_train_batches)

        # Validate
        policy.eval()
        val_loss = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for obs_batch, act_batch in val_loader:
                obs_batch = obs_batch.to(device)
                act_batch = act_batch.to(device)
                pred = policy(obs_batch)
                loss = loss_fn(pred, act_batch)
                val_loss += loss.item()
                n_val_batches += 1

        avg_val_loss = val_loss / max(1, n_val_batches)

        # Log
        log_entry = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_model_path = Path(model_out).with_name(
                Path(model_out).stem + "_best" + Path(model_out).suffix
            )
            save_policy(policy, best_model_path, config=cfg, metrics={
                "epoch": epoch,
                "val_loss": avg_val_loss,
                "train_loss": avg_train_loss,
                "best_val_loss": best_val_loss,
            })

        # Print progress
        if epoch % 5 == 0 or epoch == 1 or epoch == cfg["epochs"]:
            print("  Epoch {}/{}: train_loss={:.6f} val_loss={:.6f} {}".format(
                epoch, cfg["epochs"], avg_train_loss, avg_val_loss,
                "(best)" if epoch == best_epoch else "",
            ))

        # Video callback
        if (
            video_callback is not None
            and epoch % cfg["eval_interval_epochs"] == 0
        ):
            try:
                video_callback(policy, epoch, VIDEOS_DIR / "bc")
            except Exception as exc:
                print(f"    [WARNING] Video callback failed: {exc}")

        summary["epochs"].append(log_entry)

    # ── Finalize ────────────────────────────────────────────────
    total_time = time.time() - train_start
    summary["best_val_loss"] = best_val_loss
    summary["best_epoch"] = best_epoch
    summary["total_train_time_s"] = total_time

    # Save final model
    save_policy(policy, model_out, config=cfg, metrics=summary)
    print(f"\nTraining complete ({total_time:.1f}s)")
    print(f"Best val loss: {best_val_loss:.6f} (epoch {best_epoch})")
    print(f"Final model: {model_out}")

    # Generate loss curve
    try:
        _plot_loss_curve(log_path, log_dir / "loss_curve.png")
    except Exception as exc:
        print(f"  [WARNING] Loss curve plot failed: {exc}")

    # Save summary
    summary_path = log_dir / "bc_summary.json"
    write_json_atomic(summary_path, summary)
    print(f"Summary: {summary_path}")

    return summary


def _plot_loss_curve(log_path: Path, output_path: Path) -> None:
    """Plot training and validation loss curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs: list[int] = []
    train_losses: list[float] = []
    val_losses: list[float] = []

    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                epochs.append(entry["epoch"])
                train_losses.append(entry["train_loss"])
                val_losses.append(entry["val_loss"])

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(epochs, train_losses, label="Train Loss", color="#2196F3")
    ax.plot(epochs, val_losses, label="Val Loss", color="#FF6F00")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Behavior Cloning Loss Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
