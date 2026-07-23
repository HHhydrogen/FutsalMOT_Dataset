"""Policy rollout utilities for FutsalMOT-RL."""

from __future__ import annotations

from typing import Any

import numpy as np

from futsalmot_rl.envs.defender_follow_env import FutsalDefenderFollowEnv


def rollout_episode(
    env: FutsalDefenderFollowEnv,
    policy: Any,
    deterministic: bool = True,
    collect_all_info: bool = False,
) -> dict[str, Any]:
    """Rollout a policy in the environment for one episode.

    Args:
        env: The FutsalDefenderFollowEnv instance.
        policy: A policy callable taking (obs, deterministic) → action.
        deterministic: Whether to use deterministic actions.
        collect_all_info: If True, store full per-frame positions.

    Returns:
        Dict with rollout data, optionally including full trajectories.
    """
    obs, info = env.reset()
    done = False

    # Storage — capture initial position from frame 0
    agent_id = env.agent_id
    init_pos = info.get("all_positions", {}).get(agent_id, (0.0, 0.0))
    agent_positions: list[tuple[float, float]] = [
        (float(init_pos[0]), float(init_pos[1]))
    ]
    agent_velocities: list[tuple[float, float]] = [(0.0, 0.0)]
    rewards: list[float] = []
    actions_list: list[np.ndarray] = []
    infos: list[dict] = []

    step = 0
    total_reward = 0.0

    while not done:
        action = policy(obs, deterministic=deterministic)
        next_obs, reward, terminated, truncated, info = env.step(action)

        agent_positions.append(
            (float(info.get("all_positions", {}).get(agent_id, (0, 0))[0]),
             float(info.get("all_positions", {}).get(agent_id, (0, 0))[1]))
        )
        agent_velocities.append(info.get("agent_velocity", (0.0, 0.0)))
        rewards.append(float(reward))
        actions_list.append(action)
        if collect_all_info:
            infos.append(info)

        total_reward += float(reward)
        step += 1
        done = terminated or truncated
        obs = next_obs

    result: dict[str, Any] = {
        "agent_positions": np.array(agent_positions, dtype=np.float32),
        "agent_velocities": np.array(agent_velocities, dtype=np.float32),
        "rewards": np.array(rewards, dtype=np.float32),
        "actions": np.array(actions_list, dtype=np.float32),
        "total_reward": total_reward,
        "n_frames": step,
        "seq_id": getattr(env, "seq_id", "unknown"),
        "agent_id": env.agent_id,
    }

    if collect_all_info:
        result["infos"] = infos

    return result
