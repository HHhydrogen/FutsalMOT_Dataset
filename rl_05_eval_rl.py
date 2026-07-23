#!/usr/bin/env python3
"""
rl_05_eval_rl.py — Evaluate a trained PPO policy, record final videos.

Usage:
    D:/Anaconda/envs/yolov11/python.exe rl_05_eval_rl.py
    D:/Anaconda/envs/yolov11/python.exe rl_05_eval_rl.py --model path/to/ppo_best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_CODE_ROOT = Path(__file__).resolve().parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from futsalmot_rl.core.rl_io import write_json_atomic
from futsalmot_rl.core.rl_paths import (
    MODELS_DIR,
    REPORTS_DIR,
    RUNS_DIR,
    VIDEOS_DIR,
    ensure_dirs,
)
from futsalmot_rl.envs.defender_follow_env import FutsalDefenderFollowEnv
from futsalmot_rl.models.policy_io import load_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained PPO policy.")
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to PPO model (default: defender_follow_ppo_v1_best.pt)",
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="Source A3.3 config path (default: production_run)",
    )
    parser.add_argument(
        "--n-episodes", type=int, default=5,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device (auto/cpu/cuda)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dirs()

    # Find model
    model_path = (
        Path(args.model)
        if args.model
        else MODELS_DIR / "defender_follow_ppo_v1_best.pt"
    )
    if not model_path.is_file():
        print("[ERROR] PPO model not found: {}".format(model_path))
        print("  Run rl_04_train_ppo.py first.")
        return 1

    # Find source
    source_path = args.source
    if source_path is None:
        candidates = [
            RUNS_DIR / "production_run" / "episode_random_0001_t1_a33.json",
        ]
        for c in candidates:
            if c.is_file():
                source_path = str(c)
                break

    if source_path is None or not Path(source_path).is_file():
        print("[ERROR] No source A3.3 file found.")
        return 1

    device_str = args.device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print("=" * 60)
    print("FutsalMOT-RL PPO Evaluation")
    print("Model:    {}".format(model_path))
    print("Source:   {}".format(source_path))
    print("Device:   {}".format(device))
    print("Episodes: {}".format(args.n_episodes))
    print("=" * 60)

    # Load model
    policy, _, _ = load_policy(model_path, device=device)
    policy.eval()
    print("Model loaded.")

    # Create environment with optimized reward config
    reward_cfg = {
        "out_of_bounds_penalty": -10.0,
        "collision_penalty": -5.0,
        "boundary_proximity_weight": -0.02,
        "boundary_proximity_margin_cm": 300.0,
        "goal_side_bonus": 0.5,
        "goal_side_penalty": -0.5,
        "acceleration_penalty": -0.002,
    }
    env = FutsalDefenderFollowEnv(source_episode_path=source_path, reward_config=reward_cfg)

    # Run evaluation
    all_rewards: list[float] = []
    all_lengths: list[int] = []
    out_of_bounds_total = 0
    collision_total = 0
    marking_distances: list[float] = []

    for ep in range(1, args.n_episodes + 1):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        ep_len = 0

        while not done:
            action = policy.get_action(obs, deterministic=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            ep_len += 1
            done = terminated or truncated
            obs = next_obs

            if info.get("out_of_bounds", False):
                out_of_bounds_total += 1
            if info.get("collision", False):
                collision_total += 1
            dist = info.get("distance_to_target")
            if dist is not None:
                marking_distances.append(float(dist))

        all_rewards.append(ep_reward)
        all_lengths.append(ep_len)
        print("  Episode {}: reward={:.3f} length={}".format(ep, ep_reward, ep_len))

    # Record final video
    print("\nRecording final video...")
    try:
        from futsalmot_rl.viz.video_recorder import record_episode_video

        def policy_fn(obs, deterministic=True):
            return policy.get_action(obs, deterministic)

        seq_id = getattr(env, "seq_id", "eval")
        video_path = VIDEOS_DIR / "final" / "final_rl_{}_episode_001.mp4".format(seq_id)

        result = record_episode_video(
            env, policy_fn, video_path, fps=15,
            title="RL Final - {}".format(seq_id),
        )
        if result:
            print("  Final video: {}".format(result))

        # Second video
        env2 = FutsalDefenderFollowEnv(source_episode_path=source_path, reward_config=reward_cfg)
        video_path2 = VIDEOS_DIR / "final" / "final_rl_{}_episode_002.mp4".format(seq_id)
        result2 = record_episode_video(
            env2, policy_fn, video_path2, fps=15,
            title="RL Final - {} ep2".format(seq_id),
        )
        if result2:
            print("  Final video: {}".format(result2))
        env2.close()

    except Exception as exc:
        print("  [WARNING] Final video generation failed: {}".format(exc))

    env.close()

    # Compute metrics
    metrics = {
        "episode_reward_mean": float(np.mean(all_rewards)),
        "episode_reward_std": float(np.std(all_rewards)),
        "episode_length_mean": float(np.mean(all_lengths)),
        "out_of_bounds_total": out_of_bounds_total,
        "collision_total": collision_total,
        "mean_marking_distance_cm": float(np.mean(marking_distances)) if marking_distances else None,
        "std_marking_distance_cm": float(np.std(marking_distances)) if marking_distances else None,
        "n_episodes": args.n_episodes,
    }

    print("\n--- RL Evaluation Results ---")
    for key, value in metrics.items():
        if value is not None:
            print("  {}: {:.3f}".format(key, value) if isinstance(value, float) else "  {}: {}".format(key, value))

    report_path = REPORTS_DIR / "rl_eval_report.json"
    write_json_atomic(report_path, metrics)
    print("\nReport: {}".format(report_path))

    # Check acceptance criteria
    issues = []
    if metrics["out_of_bounds_total"] > 0:
        issues.append("out_of_bounds_count = {}".format(metrics["out_of_bounds_total"]))
    if metrics["collision_total"] > 0:
        issues.append("collision_count = {}".format(metrics["collision_total"]))

    if issues:
        print("\n[INFO] Evaluation notes: {}".format("; ".join(issues)))
    else:
        print("\n[PASS] No out-of-bounds or collision events.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
