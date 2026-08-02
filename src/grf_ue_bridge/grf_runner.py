"""运行 Google Research Football 环境并采集原始观测。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass
class StepSnapshot:
    """一个环境步中捕获的状态。"""

    step: int
    observation: Dict[str, Any]
    reward: Any
    done: bool
    info: Dict[str, Any]


@dataclass
class EpisodeResult:
    """从 GRF 采集到的完整 episode 数据。"""

    scenario: str
    seed: int
    num_steps: int
    steps_left_at_start: int
    score: Tuple[int, int]
    snapshots: List[StepSnapshot] = field(default_factory=list)


_BUILTIN_AI_ACTION_INDEX = 19  # action_set='v2' 中的 action_builtin_ai


def create_env(scenario: str, seed: int, render: bool = False, **kwargs):
    """创建一个仅使用内置 AI 的 GRF 环境。

    使用 action_set='v2'，并只控制 1 名左队球员（动作恒为 builtin_ai），
    这样所有球员都使用内置 AI，而我们仍能拿到完整观测。

    Args:
        scenario: GRF 场景名（例如 '5_vs_5'）。
        seed: 随机种子。
        render: 是否渲染。
        **kwargs: 传给 create_environment 的额外参数。

    Returns:
        一个 GRF 环境实例。
    """
    from gfootball.env import create_environment

    # 移除用户传入的控制球员相关参数（我们强制使用 builtin_ai 模式）
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
    """在 GRF 环境中运行一个 episode 并记录所有步。

    所有球员都由内置 AI 控制。我们名义上控制 1 名左队球员，但给它发送
    builtin_ai 动作，因此行为与完全内置 AI 对局一致。

    Args:
        scenario: GRF 场景名（例如 '5_vs_5'）。
        seed: 随机种子。
        render: 是否渲染。
        builtin_ai_action_index: 内置 AI 的动作索引（v2 为 19）。
        **kwargs: 传给 create_environment 的额外参数。

    Returns:
        包含所有捕获快照的 EpisodeResult。
    """
    env = create_env(scenario, seed, render, **kwargs)
    obs = env.reset()
    snapshots = []

    score = [0, 0]
    steps_left_at_start = 0

    for step in range(num_steps):
        # 对所有受控球员使用 builtin_ai 动作
        actions = [builtin_ai_action_index] * len(obs)

        obs, rew, done, info = env.step(actions)
        ob = obs[0]  # 受控球员的观测

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
    """从原始观测中提取并转换相关字段。

    将 numpy 数组转换为列表以便 JSON 序列化。
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
