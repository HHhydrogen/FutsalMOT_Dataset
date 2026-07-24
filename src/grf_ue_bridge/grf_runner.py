"""Run a Google Research Football environment and collect raw observations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass
class StepSnapshot:
    """Captured state from one environment step."""

    step: int
    observation: Dict[str, Any]
    reward: Any
    done: bool
    info: Dict[str, Any]


@dataclass
class EpisodeResult:
    """Complete episode data collected from GRF."""

    scenario: str
    seed: int
    num_steps: int
    steps_left_at_start: int
    score: Tuple[int, int]
    snapshots: List[StepSnapshot] = field(default_factory=list)


_BUILTIN_AI_ACTION_INDEX = 19  # action_builtin_ai in action_set='v2'


def create_env(scenario: str, seed: int, render: bool = False, **kwargs):
    """Create a GRF environment with built-in AI only.

    Uses action_set='v2' and controls 1 left player with builtin_ai
    actions, so all players use built-in AI while we get observations.

    Args:
        scenario: GRF scenario name (e.g. '5_vs_5').
        seed: Random seed.
        render: Whether to render.
        **kwargs: Additional args passed to create_environment.

    Returns:
        A GRF environment instance.
    """
    from gfootball.env import create_environment

    # Remove user-supplied control-player kwargs (we force builtin_ai mode)
    force_kwargs = {
        "number_of_left_players_agent_controls": 1,
        "number_of_right_players_agent_controls": 0,
        "other_config_options": {"action_set": "v2"},
    }
    for k in force_kwargs:
        kwargs.pop(k, None)

    env = create_environment(
        env_name=scenario,
        representation="raw",
        render=render,
        write_video=False,
        write_full_episode_dumps=False,
        write_goal_dumps=False,
        **force_kwargs,
        **kwargs,
    )
    return env


def run_episode(
    scenario: str = "5_vs_5",
    seed: int = 42,
    num_steps: int = 300,
    render: bool = False,
    builtin_ai_action_index: int = _BUILTIN_AI_ACTION_INDEX,
    **kwargs,
) -> EpisodeResult:
    """Run one episode in a GRF environment and record all steps.

    All players are controlled by built-in AI. One left player is nominally
    controlled by us but receives the builtin_ai action, so behaviour is
    identical to full built-in AI gameplay.

    Args:
        scenario: GRF scenario name (e.g. '5_vs_5').
        seed: Random seed.
        render: Whether to render.
        builtin_ai_action_index: Action index for built-in AI (19 for v2).
        **kwargs: Additional args passed to create_environment.

    Returns:
        EpisodeResult containing all captured snapshots.
    """
    env = create_env(scenario, seed, render, **kwargs)
    obs = env.reset()
    snapshots = []

    score = [0, 0]
    steps_left_at_start = 0

    for step in range(num_steps):
        # Use builtin_ai action for all controlled players
        actions = [builtin_ai_action_index] * len(obs)

        obs, rew, done, info = env.step(actions)
        ob = obs[0]  # observation for the controlled player

        if step == 0:
            steps_left_at_start = int(ob["steps_left"])
            score = list(ob["score"])

        snapshot = StepSnapshot(
            step=step,
            observation=_extract_observation(ob),
            reward=rew,
            done=done,
            info=info,
        )
        snapshots.append(snapshot)

        if done:
            obs = env.reset()

    final_obs = env.unwrapped.observation()[0]
    final_score = list(final_obs["score"]) if "score" in final_obs else score

    env.close()

    return EpisodeResult(
        scenario=scenario,
        seed=seed,
        num_steps=num_steps,
        steps_left_at_start=steps_left_at_start,
        score=tuple(final_score),
        snapshots=snapshots,
    )


def _extract_observation(raw_obs: dict) -> Dict[str, Any]:
    """Extract and convert relevant fields from raw observation.

    Converts numpy arrays to lists for JSON serialization.
    """
    result: Dict[str, Any] = {}
    for key in [
        "ball",
        "ball_direction",
        "ball_owned_player",
        "ball_owned_team",
        "left_team",
        "left_team_direction",
        "left_team_roles",
        "left_team_active",
        "right_team",
        "right_team_direction",
        "right_team_roles",
        "right_team_active",
        "score",
        "game_mode",
        "steps_left",
        "sticky_actions",
    ]:
        val = raw_obs[key]
        if isinstance(val, np.ndarray):
            result[key] = val.tolist()
        elif isinstance(val, (np.integer,)):
            result[key] = int(val)
        elif isinstance(val, (np.floating,)):
            result[key] = float(val)
        else:
            result[key] = val
    return result
