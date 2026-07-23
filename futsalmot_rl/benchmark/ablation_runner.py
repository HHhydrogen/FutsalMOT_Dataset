"""PPO from scratch ablation runner — trains PPO without BC initialization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from futsalmot_rl.core.rl_io import write_json_atomic
from futsalmot_rl.core.rl_paths import MODELS_DIR, ensure_dirs
from futsalmot_rl.envs.defender_follow_env import FutsalDefenderFollowEnv
from futsalmot_rl.training.train_ppo import PPOTrainer


def run_ablation_ppo_scratch(
    source_path: str | Path,
    output_dir: str | Path,
    total_timesteps: int = 500000,
    eval_interval: int = 25000,
    device: str = "auto",
) -> dict[str, Any]:
    """Train PPO from scratch (no BC init) with the same settings as PPO v2.

    Args:
        source_path: Path to source A3.3 config.
        output_dir: Output directory for ablation results.
        total_timesteps: Total training steps.
        eval_interval: Evaluation interval.
        device: Device for training.

    Returns:
        Summary dict.
    """
    output_dir = Path(output_dir)
    ensure_dirs()

    print("=" * 60)
    print("PPO from Scratch — Ablation Experiment")
    print(f"Source: {source_path}")
    print(f"Total timesteps: {total_timesteps}")
    print("=" * 60)

    # Same reward config as PPO v2
    reward_cfg = {
        "out_of_bounds_penalty": -10.0,
        "collision_penalty": -5.0,
        "boundary_proximity_weight": -0.02,
        "boundary_proximity_margin_cm": 300.0,
        "goal_side_bonus": 0.5,
        "goal_side_penalty": -0.5,
        "acceleration_penalty": -0.002,
    }

    train_env = FutsalDefenderFollowEnv(
        source_episode_path=str(source_path),
        reward_config=reward_cfg,
    )
    eval_env = FutsalDefenderFollowEnv(
        source_episode_path=str(source_path),
        reward_config=reward_cfg,
    )

    trainer = PPOTrainer(
        env=train_env,
        eval_env=eval_env,
        config={
            "total_timesteps": total_timesteps,
            "learning_rate": 0.0001,
            "n_steps": 2048,
            "batch_size": 64,
            "reward_config": reward_cfg,
        },
        device=device,
    )

    # No BC initialization — train from scratch

    log_dir = output_dir / "train_logs"
    model_dir = output_dir / "models"

    summary = trainer.train(
        total_timesteps=total_timesteps,
        log_dir=log_dir,
        model_dir=model_dir,
        eval_interval=eval_interval,
    )

    # Copy reward curve to ablation dir
    src_curve = log_dir / "reward_curve.png"
    if src_curve.is_file():
        import shutil
        shutil.copy2(str(src_curve), str(output_dir / "ppo_scratch_reward_curve.png"))

    # Evaluate
    best_model_path = model_dir / "defender_follow_ppo_v1_best.pt"
    if best_model_path.is_file():
        print("\nEvaluating best model...")
        eval_metrics = _evaluate_scratch_model(best_model_path, source_path, device)
        summary["eval_metrics"] = eval_metrics
        eval_report_path = output_dir / "ppo_scratch_eval_report.json"
        write_json_atomic(eval_report_path, eval_metrics)

    # Compare with BC-init
    bc_model = MODELS_DIR / "defender_follow_ppo_v1_best.pt"
    if bc_model.is_file() and best_model_path.is_file():
        _write_ablation_report(
            best_model_path,
            bc_model,
            source_path,
            output_dir,
            device,
        )

    train_env.close()
    eval_env.close()

    return summary


def _evaluate_scratch_model(
    model_path: Path,
    source_path: str | Path,
    device: str = "auto",
) -> dict[str, Any]:
    """Evaluate the scratch model on a single episode."""
    from futsalmot_rl.models.policy_io import load_policy

    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    policy, _, _ = load_policy(str(model_path), device=device)

    env = FutsalDefenderFollowEnv(
        source_episode_path=str(source_path),
        reward_config={
            "out_of_bounds_penalty": -10.0,
            "collision_penalty": -5.0,
            "boundary_proximity_weight": -0.02,
            "boundary_proximity_margin_cm": 300.0,
            "goal_side_bonus": 0.5,
            "goal_side_penalty": -0.5,
            "acceleration_penalty": -0.002,
        },
    )

    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    oob = 0
    coll = 0
    step = 0

    while not done:
        action = policy.get_action(obs, deterministic=True)
        obs_next, reward, term, trunc, info = env.step(action)
        total_reward += float(reward)
        if info.get("out_of_bounds"):
            oob += 1
        if info.get("collision"):
            coll += 1
        done = term or trunc
        obs = obs_next
        step += 1

    env.close()

    return {
        "total_reward": total_reward,
        "out_of_bounds": oob,
        "collisions": coll,
        "episode_length": step,
    }


def _write_ablation_report(
    scratch_model_path: Path,
    bc_init_model_path: Path,
    source_path: str | Path,
    output_dir: Path,
    device: str = "auto",
) -> None:
    """Write ablation comparison report as Markdown."""
    from futsalmot_rl.benchmark.metrics import compute_policy_metrics
    from futsalmot_rl.data.a33_reader import get_player_positions_2d, load_a33_config
    from futsalmot_rl.models.policy_io import load_policy

    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # Load source rule data
    cfg = load_a33_config(source_path)
    all_pos = get_player_positions_2d(cfg)
    target_pos = np.array(all_pos.get("Player_01", []), dtype=np.float32)
    other_pos = {}
    for pid, pos in all_pos.items():
        if pid != "Player_05":
            other_pos[pid] = np.array(pos, dtype=np.float32)

    results: dict[str, dict[str, Any]] = {}

    # Rule
    rule_pos = np.array(all_pos.get("Player_05", []), dtype=np.float32)
    results["rule"] = compute_policy_metrics(rule_pos, target_pos, all_player_positions=other_pos)

    # Scratch
    scratch_policy, _, _ = load_policy(str(scratch_model_path), device=device)
    env = FutsalDefenderFollowEnv(source_episode_path=str(source_path))
    scratch_pos_list = []
    obs, info = env.reset()
    init_pos = info.get("all_positions", {}).get("Player_05", (0, 0))
    scratch_pos_list.append((float(init_pos[0]), float(init_pos[1])))
    done = False
    while not done:
        action = scratch_policy.get_action(obs, deterministic=True)
        obs_next, _, term, trunc, info = env.step(action)
        pos = info.get("all_positions", {}).get("Player_05", (0, 0))
        scratch_pos_list.append((float(pos[0]), float(pos[1])))
        done = term or trunc
        obs = obs_next
    env.close()
    scratch_pos = np.array(scratch_pos_list, dtype=np.float32)
    results["ppo_from_scratch"] = compute_policy_metrics(
        scratch_pos, target_pos, all_player_positions=other_pos,
    )

    # BC-init
    bc_policy, _, _ = load_policy(str(bc_init_model_path), device=device)
    env = FutsalDefenderFollowEnv(source_episode_path=str(source_path))
    bc_pos_list = []
    obs, info = env.reset()
    init_pos = info.get("all_positions", {}).get("Player_05", (0, 0))
    bc_pos_list.append((float(init_pos[0]), float(init_pos[1])))
    done = False
    while not done:
        action = bc_policy.get_action(obs, deterministic=True)
        obs_next, _, term, trunc, info = env.step(action)
        pos = info.get("all_positions", {}).get("Player_05", (0, 0))
        bc_pos_list.append((float(pos[0]), float(pos[1])))
        done = term or trunc
        obs = obs_next
    env.close()
    bc_init_pos = np.array(bc_pos_list, dtype=np.float32)
    results["ppo_bc_init"] = compute_policy_metrics(
        bc_init_pos, target_pos, all_player_positions=other_pos,
    )

    # Generate Markdown
    lines = [
        "# PPO from Scratch — Ablation Report\n",
        "Comparison of PPO trained from scratch vs. PPO with BC initialization.\n",
        "## Metrics\n",
        "| Metric | Rule | PPO from Scratch | PPO + BC Init |",
        "|--------|------|-----------------|---------------|",
    ]
    key_metrics = [
        ("mean_marking_distance_cm", "Mean Marking Distance (cm)"),
        ("std_marking_distance_cm", "Std Marking Distance (cm)"),
        ("goal_side_success_rate", "Goal-side Success Rate"),
        ("out_of_bounds_count", "Out of Bounds"),
        ("collision_count", "Collisions"),
        ("max_speed_cm_s", "Max Speed (cm/s)"),
    ]
    for key, label in key_metrics:
        vals = []
        for name in ["rule", "ppo_from_scratch", "ppo_bc_init"]:
            if name in results:
                vals.append(f"{results[name].get(key, 0):.2f}")
            else:
                vals.append("N/A")
        lines.append("| {} | {} |".format(label, " | ".join(vals)))

    lines.append("")
    lines.append("## Conclusion\n")

    scratch_oob = results.get("ppo_from_scratch", {}).get("out_of_bounds_count", 0)
    bc_init_oob = results.get("ppo_bc_init", {}).get("out_of_bounds_count", 0)
    scratch_coll = results.get("ppo_from_scratch", {}).get("collision_count", 0)
    bc_init_coll = results.get("ppo_bc_init", {}).get("collision_count", 0)
    scratch_mark = results.get("ppo_from_scratch", {}).get("mean_marking_distance_cm", 0)
    bc_init_mark = results.get("ppo_bc_init", {}).get("mean_marking_distance_cm", 0)

    lines.append(f"- PPO from scratch OOB: {scratch_oob} vs BC-init OOB: {bc_init_oob}")
    lines.append(f"- PPO from scratch collisions: {scratch_coll} vs BC-init collisions: {bc_init_coll}")
    lines.append(f"- PPO from scratch marking: {scratch_mark:.1f}cm vs BC-init marking: {bc_init_mark:.1f}cm")

    if bc_init_oob <= scratch_oob and bc_init_coll <= scratch_coll:
        lines.append("- **BC initialization improves training stability.**")
    else:
        lines.append("- BC init and scratch perform similarly.")

    report_path = output_dir / "ablation_bc_init_report.md"
    write_json_atomic(report_path.with_suffix(".json"), results)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nAblation report: {report_path}")
