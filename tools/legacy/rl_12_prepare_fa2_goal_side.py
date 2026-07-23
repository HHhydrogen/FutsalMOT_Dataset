#!/usr/bin/env python3
"""
rl_12_prepare_fa2_goal_side.py — Prepare FA-2 Goal-side Defense task.

This script:
1. Defines FA-2 metrics
2. Defines FA-2 reward v3
3. Tests FA-2 metrics on existing rule/BC/PPO policies
4. Generates FA-2 README

Usage:
    D:/Anaconda/envs/yolov11/python.exe rl_12_prepare_fa2_goal_side.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_CODE_ROOT = Path(__file__).resolve().parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from futsalmot_rl.academy.fa2_goal_side_defense import (
    FA2_REWARD_CONFIG,
    FA2_TRAIN_CONFIG,
    compute_shot_lane_block_score,
)
from futsalmot_rl.core.rl_io import write_json_atomic, write_text_atomic
from futsalmot_rl.core.rl_paths import (
    ABLATIONS_DIR,
    MODELS_DIR,
    RUNS_DIR,
    ensure_dirs,
)
from futsalmot_rl.data.a33_reader import get_player_positions_2d, load_a33_config
from futsalmot_rl.envs.defender_follow_env import FutsalDefenderFollowEnv
from futsalmot_rl.models.policy_io import load_policy


def compute_fa2_metrics(positions, target_positions):
    """Compute FA-2 specific metrics for a trajectory."""
    T = min(len(positions), len(target_positions))
    own_goal = (-1950.0, 0.0)

    goal_side_count = 0
    behind_count = 0
    total_lane_score = 0.0
    goal_line_distances = []

    for i in range(T):
        px, py = positions[i]
        tx, ty = target_positions[i]
        gx, gy = own_goal

        # Goal-side check
        dist_to_goal = np.hypot(px - gx, py - gy)
        dist_target_to_goal = np.hypot(tx - gx, ty - gy)
        vec_to_target = (tx - px, ty - py)
        vec_to_goal = (gx - px, gy - py)
        dot = vec_to_target[0] * vec_to_goal[0] + vec_to_target[1] * vec_to_goal[1]

        if dot > 0 and dist_to_goal < dist_target_to_goal:
            goal_side_count += 1
        else:
            behind_count += 1

        # Lane block
        lane_score = compute_shot_lane_block_score(
            (float(px), float(py)), (float(tx), float(ty)), own_goal
        )
        total_lane_score += lane_score

        # Goal line offset
        goal_line_distances.append(abs(px - gx))

    return {
        "goal_side_success_rate": goal_side_count / max(1, T),
        "time_behind_attacker_ratio": behind_count / max(1, T),
        "mean_goal_line_offset_cm": float(np.mean(goal_line_distances)),
        "mean_shot_lane_block_score": total_lane_score / max(1, T),
    }


def main() -> int:
    ensure_dirs()
    output_dir = Path("Saved/FutsalMOT_RL/academy_fa2")
    output_dir.mkdir(parents=True, exist_ok=True)

    source = RUNS_DIR / "production_run" / "episode_random_0001_t1_a33.json"

    print("=" * 60)
    print("FA-2 Goal-side Defense — Preparation")
    print("=" * 60)

    # Load source data
    cfg = load_a33_config(source)
    all_pos = get_player_positions_2d(cfg)
    target_pos = np.array(all_pos.get("Player_01", []), dtype=np.float32)
    rule_pos = np.array(all_pos.get("Player_05", []), dtype=np.float32)

    # Compute FA-2 metrics for rule
    rule_fa2 = compute_fa2_metrics(rule_pos, target_pos)
    print("\n--- Rule Baseline FA-2 Metrics ---")
    for k, v in rule_fa2.items():
        print("  {}: {:.3f}".format(k, v))

    # Compute for BC
    bc_model = MODELS_DIR / "defender_follow_bc_v1_best.pt"
    if bc_model.is_file():
        bc_policy, _, _ = load_policy(str(bc_model))
        env = FutsalDefenderFollowEnv(source_episode_path=str(source))
        bc_pos = []
        obs, info = env.reset()
        init_pos = info.get("all_positions", {}).get("Player_05", (0, 0))
        bc_pos.append((float(init_pos[0]), float(init_pos[1])))
        done = False
        while not done:
            action = bc_policy.get_action(obs, deterministic=True)
            obs_next, _, term, trunc, info = env.step(action)
            pos = info.get("all_positions", {}).get("Player_05", (0, 0))
            bc_pos.append((float(pos[0]), float(pos[1])))
            done = term or trunc
            obs = obs_next
        env.close()
        bc_pos_arr = np.array(bc_pos, dtype=np.float32)
        bc_fa2 = compute_fa2_metrics(bc_pos_arr, target_pos)
        print("\n--- BC FA-2 Metrics ---")
        for k, v in bc_fa2.items():
            print("  {}: {:.3f}".format(k, v))
    else:
        bc_fa2 = {}

    # Compute for PPO v2
    ppo_model = MODELS_DIR / "defender_follow_ppo_v1_best.pt"
    if ppo_model.is_file():
        ppo_policy, _, _ = load_policy(str(ppo_model))
        env = FutsalDefenderFollowEnv(source_episode_path=str(source))
        ppo_pos = []
        obs, info = env.reset()
        init_pos = info.get("all_positions", {}).get("Player_05", (0, 0))
        ppo_pos.append((float(init_pos[0]), float(init_pos[1])))
        done = False
        while not done:
            action = ppo_policy.get_action(obs, deterministic=True)
            obs_next, _, term, trunc, info = env.step(action)
            pos = info.get("all_positions", {}).get("Player_05", (0, 0))
            ppo_pos.append((float(pos[0]), float(pos[1])))
            done = term or trunc
            obs = obs_next
        env.close()
        ppo_pos_arr = np.array(ppo_pos, dtype=np.float32)
        ppo_fa2 = compute_fa2_metrics(ppo_pos_arr, target_pos)
        print("\n--- PPO v2 FA-2 Metrics ---")
        for k, v in ppo_fa2.items():
            print("  {}: {:.3f}".format(k, v))
    else:
        ppo_fa2 = {}

    # Write FA-2 report
    report = {
        "task": "FA-2 Goal-side Defense",
        "description": "Player_05 maintains position between Player_01 and own goal",
        "reward_config": FA2_REWARD_CONFIG,
        "training_config": FA2_TRAIN_CONFIG,
        "baseline_metrics": {
            "rule": rule_fa2,
            "bc": bc_fa2,
            "ppo_v2": ppo_fa2,
        },
    }
    report_path = output_dir / "fa2_preparation_report.json"
    write_json_atomic(report_path, report)
    print("\nReport: {}".format(report_path))

    # Write FA-2 README
    readme = """# FA-2: Goal-side Defense

## Task Description
Player_05 (defender) must maintain position between Player_01 (attacker)
and the own goal (-X direction), forming a proper defensive block.

## Current Baseline Metrics

| Metric | Rule | BC | PPO v2 |
|--------|------|----|--------|
| Goal-side Success Rate | {rule_gs:.3f} | {bc_gs:.3f} | {ppo_gs:.3f} |
| Behind Attacker Ratio | {rule_behind:.3f} | {bc_behind:.3f} | {ppo_behind:.3f} |
| Shot Lane Block Score | {rule_lane:.3f} | {bc_lane:.3f} | {ppo_lane:.3f} |

## Reward Config
```
goal_side_bonus = 1.0
goal_side_penalty = -1.0
marking_point_weight = -0.004
distance_band_weight = -0.003
ideal_mark_distance_cm = 150.0
```
(All other rewards same as FA-1 v2)

## Training Config
- BC initialization from FA-1 model
- Total timesteps: 500000
- Learning rate: 0.0001
- Eval interval: 25000

## Acceptance Criteria
- goal_side_success_rate >= 0.30
- out_of_bounds == 0
- collisions <= 5
- trajectory validation errors == 0

## Next Steps
1. Train FA-2 with reward v3
2. Evaluate FA-2 metrics
3. Export A3.3 trajectory
4. UE closed-loop verification
""".format(
        rule_gs=rule_fa2.get("goal_side_success_rate", 0),
        bc_gs=bc_fa2.get("goal_side_success_rate", 0),
        ppo_gs=ppo_fa2.get("goal_side_success_rate", 0),
        rule_behind=rule_fa2.get("time_behind_attacker_ratio", 0),
        bc_behind=bc_fa2.get("time_behind_attacker_ratio", 0),
        ppo_behind=ppo_fa2.get("time_behind_attacker_ratio", 0),
        rule_lane=rule_fa2.get("mean_shot_lane_block_score", 0),
        bc_lane=bc_fa2.get("mean_shot_lane_block_score", 0),
        ppo_lane=ppo_fa2.get("mean_shot_lane_block_score", 0),
    )
    readme_path = output_dir / "FA2_README.md"
    write_text_atomic(readme_path, readme)
    print("FA-2 README: {}".format(readme_path))

    print("\n[DONE] FA-2 preparation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
