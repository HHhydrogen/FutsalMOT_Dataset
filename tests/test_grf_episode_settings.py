"""真实 GRF 回合长度 / AI 难度覆盖集成测试（`-m grf_integration` 才运行）。

验证 grf_runner 对 C++ 场景配置的后置覆盖确实生效：

  回合长度（game_duration）→ steps_left 反映更长回合（避免采集步数耗尽回合→重置瞬移）；
  AI 难度（left/right_team_difficulty）→ 引擎场景配置被覆盖。

默认套件不运行（pyproject addopts: -m 'not grf_integration'）。
本地运行：
    uv run pytest -m grf_integration -q
"""

from __future__ import annotations

import pytest

from grf_ue_bridge.grf_runner import create_env, run_episode

pytestmark = [pytest.mark.grf_integration]


def _has_gfootball() -> bool:
    try:
        import gfootball  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


pytestmark = [pytest.mark.grf_integration,
              pytest.mark.skipif(not _has_gfootball(), reason="gfootball 未安装")]


class TestEpisodeLengthOverride:
    def test_default_uses_scenario_duration(self):
        r = run_episode(scenario="5_vs_5", seed=42, num_steps=3)
        assert r.steps_left_at_start > 0
        # 5_vs_5 场景默认 game_duration=3000，steps_left 从 ~3000 开始
        assert 2500 <= r.steps_left_at_start <= 4000

    def test_game_duration_extends_episode(self):
        r = run_episode(scenario="5_vs_5", seed=42, num_steps=3, game_duration=30000)
        assert r.steps_left_at_start >= 30000
        # 采集过程中持续递减且远未耗尽
        assert r.snapshots[-1].observation["steps_left"] > 25000


class TestDifficultyOverride:
    def test_difficulty_applies_to_engine(self):
        env = create_env(
            "5_vs_5", 42,
            left_team_difficulty=0.7, right_team_difficulty=0.3,
        )
        try:
            cfg = env.unwrapped._env._env.config
            assert abs(cfg.left_team_difficulty - 0.7) < 1e-3
            assert abs(cfg.right_team_difficulty - 0.3) < 1e-3
        finally:
            env.close()

    def test_none_leaves_scenario_default(self):
        env = create_env("5_vs_5", 42)
        try:
            cfg = env.unwrapped._env._env.config
            assert cfg.game_duration == 3000
            # float32 精度：用近似比较
            assert pytest.approx(cfg.left_team_difficulty, abs=1e-3) == 0.05
            assert pytest.approx(cfg.right_team_difficulty, abs=1e-3) == 0.05
        finally:
            env.close()
