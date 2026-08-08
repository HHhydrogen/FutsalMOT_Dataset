"""运行 Google Research Football 环境并采集原始观测。"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .seeds import EpisodeSeeds, derive_episode_seeds


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
    seeds: Optional[EpisodeSeeds] = field(
        default=None,
        metadata={"doc": "实际使用的种子集合（含传入 GRF 的 game_engine_random_seed）"},
    )


_BUILTIN_AI_ACTION_INDEX = 19  # action_set='v2' 中的 action_builtin_ai


def _apply_scenario_overrides(
    env,
    game_duration: Optional[int] = None,
    left_team_difficulty: Optional[float] = None,
    right_team_difficulty: Optional[float] = None,
) -> None:
    """后置覆盖 GRF 场景配置（回合长度 game_duration / 双方 AI 难度）。

    这些字段由场景模块在 build_scenario 中设定；`env.reset()` 会重建场景，
    因此每次 reset 后都必须重新调用本函数。None 表示不覆盖（用场景默认）。

    Args:
        env: create_environment 返回的环境（未包到最内层）。
        game_duration: 单个回合的引擎帧数。
        left_team_difficulty / right_team_difficulty: 左/右队 AI 难度（0~1）。
    """
    cfg = env.unwrapped._env._env.config
    if game_duration is not None:
        cfg.game_duration = int(game_duration)
    if left_team_difficulty is not None:
        cfg.left_team_difficulty = float(left_team_difficulty)
    if right_team_difficulty is not None:
        cfg.right_team_difficulty = float(right_team_difficulty)


def create_env(
    scenario: str,
    seed: int,
    render: bool = False,
    game_duration: Optional[int] = None,
    left_team_difficulty: Optional[float] = None,
    right_team_difficulty: Optional[float] = None,
    **kwargs,
):
    """创建一个仅使用内置 AI 的 GRF 环境。

    使用 action_set='v2'，并只控制 1 名左队球员（动作恒为 builtin_ai），
    这样所有球员都使用内置 AI，而我们仍能拿到完整观测。

    随机种子体系：从 root seed 派生命名空间子 seed，并：
      - `game_engine_random_seed` 经 other_config_options 真正传入 GRF 引擎
        （未设置时 gfootball 用 random.randint 兜底，非可复现）；
      - Python 标准库 random 与 numpy 分别用派生 seed 设置（全局副作用，
        属预期——episode 生成通常在专用进程内完成）。

    Args:
        scenario: GRF 场景名（例如 '5_vs_5'）。
        seed: 随机根 seed。
        render: 是否渲染。
        game_duration / left_team_difficulty / right_team_difficulty:
            场景覆盖（回合长度 / AI 难度），None = 用场景默认。
        **kwargs: 传给 create_environment 的额外参数；其中
            other_config_options 会被合并（本函数补充 action_set 与
            game_engine_random_seed，不覆盖调用者已提供的同名键之外的选项）。

    Returns:
        一个 GRF 环境实例。
    """
    from gfootball.env import create_environment

    root_seed = int(seed)
    seeds = derive_episode_seeds(root_seed)

    # 合并调用者的 other_config_options，但强制覆盖我们依赖的键
    user_options = dict(kwargs.pop("other_config_options", None) or {})
    other_config_options = {
        **user_options,
        "action_set": "v2",
        "game_engine_random_seed": seeds.grf_game_engine_seed,
    }

    # 移除用户传入的控制球员相关参数（我们强制使用 builtin_ai 模式）
    force_kwargs = {
        "number_of_left_players_agent_controls": 1,
        "number_of_right_players_agent_controls": 0,
        "other_config_options": other_config_options,
    }
    for k in force_kwargs:
        kwargs.pop(k, None)

    # 明确设置 Python / NumPy 随机种子（全局副作用，见 docstring）
    random.seed(seeds.python_seed)
    np.random.seed(seeds.numpy_seed)

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
    _apply_scenario_overrides(
        env, game_duration, left_team_difficulty, right_team_difficulty
    )
    return env


def run_episode(
    scenario: str = "5_vs_5",
    seed: int = 42,
    num_steps: int = 300,
    render: bool = False,
    game_duration: Optional[int] = None,
    left_team_difficulty: Optional[float] = None,
    right_team_difficulty: Optional[float] = None,
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
        game_duration / left_team_difficulty / right_team_difficulty:
            场景覆盖（回合长度 / AI 难度），None = 用场景默认；
            每次 env.reset() 后重新应用（reset 会重建场景配置）。
        builtin_ai_action_index: 内置 AI 的动作索引（v2 为 19）。
        **kwargs: 传给 create_environment 的额外参数。

    Returns:
        包含所有捕获快照的 EpisodeResult（含实际使用的种子集合 seeds）。
    """
    root_seed = int(seed)
    seeds = derive_episode_seeds(root_seed)
    env = create_env(
        scenario,
        root_seed,
        render,
        game_duration=game_duration,
        left_team_difficulty=left_team_difficulty,
        right_team_difficulty=right_team_difficulty,
        **kwargs,
    )
    obs = env.reset()
    _apply_scenario_overrides(
        env, game_duration, left_team_difficulty, right_team_difficulty
    )
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
            _apply_scenario_overrides(
                env, game_duration, left_team_difficulty, right_team_difficulty
            )

    final_obs = env.unwrapped.observation()[0]
    final_score = list(final_obs["score"]) if "score" in final_obs else score

    env.close()

    return EpisodeResult(
        scenario=scenario,
        seed=root_seed,
        num_steps=num_steps,
        steps_left_at_start=steps_left_at_start,
        score=tuple(final_score),
        snapshots=snapshots,
        seeds=seeds,
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
