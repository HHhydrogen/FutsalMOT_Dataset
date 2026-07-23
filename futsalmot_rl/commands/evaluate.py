"""`futsalmot-rl evaluate` — evaluate BC or RL policy."""

from __future__ import annotations

import argparse
from pathlib import Path

from futsalmot_rl.core.local_config import get_repo_root


def _resolve_model_dir(project_root: str) -> Path:
    if project_root:
        return Path(project_root) / "Saved" / "FutsalMOT_RL" / "models"
    return get_repo_root().parent.parent.parent / "Saved" / "FutsalMOT_RL" / "models"


def _resolve_source() -> str:
    runs_dir = get_repo_root() / "configs" / "runs"
    candidates = [
        runs_dir / "production_run" / "episode_random_0001_t1_a33.json",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    return ""


def register_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("evaluate", help="Evaluate BC or RL policy")
    subs = p.add_subparsers(dest="eval_cmd")
    bc = subs.add_parser("bc", help="Evaluate BC policy")
    bc.add_argument("--model", type=str, default=None)
    bc.add_argument("--max-episodes", type=int, default=3)
    rl = subs.add_parser("rl", help="Evaluate RL (PPO) policy")
    rl.add_argument("--model", type=str, default=None)
    rl.add_argument("--n-episodes", type=int, default=3)
    sanity = subs.add_parser("sanitize-env", help="Environment sanity check")
    sanity.add_argument("--n-episodes", type=int, default=2)


def run(args: argparse.Namespace, project_root: str) -> int:
    import numpy as np
    from futsalmot_rl.models.policy_io import load_policy
    from futsalmot_rl.envs.defender_follow_env import FutsalDefenderFollowEnv

    source = _resolve_source()
    if not source:
        print("No source A3.3 found. Generate trajectories first.", file=__import__("sys").stderr)
        return 1

    model_dir = _resolve_model_dir(project_root)

    if args.eval_cmd == "bc":
        model = args.model or str(model_dir / "defender_follow_bc_v1_best.pt")
        if not Path(model).is_file():
            print(f"Model not found: {model}")
            return 1
        policy, _, _ = load_policy(model)
        env = FutsalDefenderFollowEnv(source_episode_path=source)
        obs, _ = env.reset()
        done = False
        rewards = []
        while not done:
            action = policy.get_action(obs, deterministic=True)
            obs_next, r, term, trunc, _ = env.step(action)
            rewards.append(float(r))
            done = term or trunc
            obs = obs_next
        env.close()
        print(f"BC eval: mean_reward={float(np.mean(rewards)):.3f}")
        return 0

    if args.eval_cmd == "rl":
        model = args.model or str(model_dir / "defender_follow_ppo_v1_best.pt")
        if not Path(model).is_file():
            print(f"Model not found: {model}")
            return 1
        policy, _, _ = load_policy(model)
        env = FutsalDefenderFollowEnv(source_episode_path=source)
        all_rewards = []
        for _ in range(args.n_episodes):
            obs, _ = env.reset()
            done = False
            ep_r = 0.0
            while not done:
                action = policy.get_action(obs, deterministic=True)
                obs_next, r, term, trunc, _ = env.step(action)
                ep_r += float(r)
                done = term or trunc
                obs = obs_next
            all_rewards.append(ep_r)
        env.close()
        mean_r = float(np.mean(all_rewards))
        std_r = float(np.std(all_rewards))
        print(f"RL eval: mean_reward={mean_r:.3f} std={std_r:.3f}")
        return 0

    if args.eval_cmd == "sanitize-env":
        from futsalmot_rl.features.action_builder import PLAYER_05_MAX_SPEED_CM_S, extract_action_from_trajectory
        from futsalmot_rl.data.a33_reader import get_player_positions_2d, load_a33_config

        cfg = load_a33_config(source)
        all_pos = get_player_positions_2d(cfg)
        agent_positions = all_pos.get("Player_05", [])
        env = FutsalDefenderFollowEnv(source_episode_path=source)
        obs, _ = env.reset()
        done = False
        rule_reward = 0.0
        while not done:
            frame = min(env.current_frame, len(agent_positions) - 2)
            action = extract_action_from_trajectory(agent_positions, frame, fps=30, max_speed=PLAYER_05_MAX_SPEED_CM_S)
            obs_next, r, term, trunc, _ = env.step(action)
            rule_reward += float(r)
            done = term or trunc
            obs = obs_next
        env.close()
        print(f"Rule replay reward: {rule_reward:.3f}")
        return 0

    print(f"Unknown evaluate command: {args.eval_cmd}")
    return 1
