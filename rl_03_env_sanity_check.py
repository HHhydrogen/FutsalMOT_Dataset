#!/usr/bin/env python3
"""
rl_03_env_sanity_check.py — Verify the RL environment works correctly.

Tests three policies: zero, random, and rule_replay.
Expected: rule_replay_reward > zero_reward > random_reward.

Usage:
    D:/Anaconda/envs/yolov11/python.exe rl_03_env_sanity_check.py
    D:/Anaconda/envs/yolov11/python.exe rl_03_env_sanity_check.py --source path/to/a33.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_CODE_ROOT = Path(__file__).resolve().parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from futsalmot_rl.core.rl_io import write_json_atomic
from futsalmot_rl.core.rl_paths import (
    REPORTS_DIR,
    RUNS_DIR,
    VIDEOS_DIR,
    ensure_dirs,
)
from futsalmot_rl.data.a33_reader import get_player_positions_2d, load_a33_config
from futsalmot_rl.envs.defender_follow_env import FutsalDefenderFollowEnv
from futsalmot_rl.features.action_builder import (
    PLAYER_05_MAX_SPEED_CM_S,
    extract_action_from_trajectory,
)


def zero_policy(obs, deterministic=True):
    """Always output zero velocity."""
    return np.array([0.0, 0.0], dtype=np.float32)


def random_policy(obs, deterministic=True):
    """Output random actions uniformly in [-1, 1]."""
    return np.random.uniform(-1.0, 1.0, size=2).astype(np.float32)


def make_rule_replay_policy(env, positions):
    """Create a policy that replays the rule trajectory actions.

    Since the env controls Player_05, we need to extract actions from
    the rule trajectory that match what the env was doing.
    """
    def rule_replay(obs, deterministic=True):
        frame = min(env.current_frame, len(positions) - 2)
        return extract_action_from_trajectory(
            positions, frame, fps=env.fps, max_speed=PLAYER_05_MAX_SPEED_CM_S
        )
    return rule_replay


def run_eval(env, policy, policy_name: str, n_episodes: int = 3) -> dict:
    """Run evaluation episodes with a given policy."""
    rewards: list[float] = []
    episode_lengths: list[int] = []
    out_of_bounds_count = 0
    collision_count = 0

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        ep_len = 0

        while not done:
            action = policy(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            ep_len += 1
            done = terminated or truncated
            obs = next_obs

            if info.get("out_of_bounds", False):
                out_of_bounds_count += 1
            if info.get("collision", False):
                collision_count += 1

        rewards.append(ep_reward)
        episode_lengths.append(ep_len)

    return {
        "policy": policy_name,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "out_of_bounds_total": out_of_bounds_count,
        "collision_total": collision_count,
        "n_episodes": n_episodes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity check the FutsalDefenderFollowEnv.")
    parser.add_argument(
        "--source", type=str, default=None,
        help="Path to source A3.3 JSON (default: first available in runs/)",
    )
    parser.add_argument(
        "--n-episodes", type=int, default=3,
        help="Number of episodes per policy",
    )
    parser.add_argument(
        "--record-video", action="store_true", default=True,
        help="Record videos for the sanity check runs",
    )
    return parser.parse_args()


def record_policy_video(env, policy, output_path: Path, title: str = "") -> None:
    """Record a single episode video."""
    try:
        from futsalmot_rl.viz.video_recorder import record_episode_video

        result = record_episode_video(
            env, policy, output_path, fps=15,
            title=title,
        )
        if result:
            print("    Video saved: {}".format(result))
    except Exception as exc:
        print("    [WARNING] Video recording failed: {}".format(exc))


def main() -> int:
    args = parse_args()
    ensure_dirs()

    # Find source A3.3 file
    source_path = None
    if args.source:
        source_path = Path(args.source)
    else:
        candidates = [
            RUNS_DIR / "production_run" / "episode_random_0001_t1_a33.json",
        ]
        for c in candidates:
            if c.is_file():
                source_path = c
                break

    if source_path is None or not source_path.is_file():
        print("[ERROR] No source A3.3 file found. Run the main pipeline first.")
        return 1

    print("=" * 60)
    print("FutsalMOT-RL Environment Sanity Check")
    print("Source: {}".format(source_path))
    print("=" * 60)

    # Create environment
    env = FutsalDefenderFollowEnv(source_episode_path=source_path)

    # Get rule positions for the agent (to replay)
    cfg = load_a33_config(source_path)
    all_positions = get_player_positions_2d(cfg)
    agent_rule_positions = all_positions.get("Player_05", [])

    # Policies
    policies = [
        ("zero", zero_policy, None),
        ("random", random_policy, None),
        ("rule_replay", make_rule_replay_policy(env, agent_rule_positions), None),
    ]

    results: list[dict] = []
    for name, policy_fn, _ in policies:
        print("\nTesting {} policy...".format(name))
        result = run_eval(env, policy_fn, name, args.n_episodes)
        results.append(result)
        print("  Mean reward: {:.3f} ± {:.3f}".format(
            result["mean_reward"], result["std_reward"]
        ))
        print("  Out of bounds: {}  Collisions: {}".format(
            result["out_of_bounds_total"], result["collision_total"]
        ))

        # Record video
        if args.record_video:
            video_path = VIDEOS_DIR / "rl_eval" / "sanity_{}_{}.mp4".format(
                name, Path(source_path).stem
            )
            env_copy = FutsalDefenderFollowEnv(source_episode_path=source_path)
            if name == "rule_replay":
                pos_copy = get_player_positions_2d(load_a33_config(source_path)).get("Player_05", [])
                rule_fn = make_rule_replay_policy(env_copy, pos_copy)
                record_policy_video(env_copy, rule_fn, video_path, title="Sanity: {}".format(name))
            else:
                record_policy_video(env_copy, policy_fn, video_path, title="Sanity: {}".format(name))
            env_copy.close()

    env.close()

    # Check expected ordering
    rule_reward = results[2]["mean_reward"]
    zero_reward = results[0]["mean_reward"]
    random_reward = results[1]["mean_reward"]

    ordering_ok = rule_reward > zero_reward > random_reward
    print("\n--- Sanity Check Result ---")
    print("  Rule replay:  {:.3f}".format(rule_reward))
    print("  Zero policy:  {:.3f}".format(zero_reward))
    print("  Random:       {:.3f}".format(random_reward))
    print("  Expected ordering: rule_replay > zero > random")
    print("  Result: {}".format("PASS" if ordering_ok else "FAIL"))

    # Save report
    report = {
        "schema_version": "RL_ENV_SANITY_V1",
        "source_config": str(source_path.resolve()),
        "n_episodes": args.n_episodes,
        "ordering_ok": ordering_ok,
        "results": results,
    }
    report_path = REPORTS_DIR / "env_sanity_report.json"
    write_json_atomic(report_path, report)
    print("\nReport: {}".format(report_path))

    if ordering_ok:
        print("\n[DONE] Environment sanity check PASSED.")
        return 0
    else:
        print("\n[FAIL] Environment sanity check FAILED.")
        print("  The reward ordering is unexpected. Review the reward function.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
