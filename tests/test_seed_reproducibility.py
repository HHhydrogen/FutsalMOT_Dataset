"""真实 GRF seed 复现性集成测试（`-m grf_integration` 才运行）。

每个用例在独立 Python 子进程中运行短 episode（steps=4），证明：

  6.2 同 seed、独立进程复现 —— frames.jsonl SHA-256 完全一致；
  6.3 不同 seed 轨迹不同；
  6.4 运行顺序不依赖 —— 1001 → 1002 → 1001，首尾一致。

默认套件不运行（pyproject addopts: -m 'not grf_integration'）。
本地运行：
    uv run pytest -m grf_integration -q
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.grf_integration


def _has_gfootball() -> bool:
    try:
        import gfootball  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


pytestmark = [pytest.mark.grf_integration, pytest.mark.skipif(not _has_gfootball(), reason="gfootball 未安装")]

# 子进程导出代码：独立进程、独立解释器状态，验证跨进程初始化差异。
_EXPORT_CODE = r"""
import hashlib, json, sys
from pathlib import Path
from grf_ue_bridge.config import ExportConfig
from grf_ue_bridge.exporter import export_episode
from grf_ue_bridge.grf_runner import run_episode

out, seed, steps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
cfg = ExportConfig(scenario="5_vs_5", seed=seed, num_steps=steps)
result = run_episode(
    scenario=cfg.scenario, seed=cfg.seed, num_steps=cfg.num_steps, render=False
)
export_episode(cfg, result, Path(out))
h = hashlib.sha256(Path(out, "frames.jsonl").read_bytes()).hexdigest()
meta = json.load(open(Path(out, "meta.json"), encoding="utf-8"))
frames = [json.loads(l) for l in open(Path(out, "frames.jsonl"), encoding="utf-8")]
json.dump(
    {
        "hash": h,
        "meta": meta,
        "steps": len(frames),
        "score": frames[-1]["score"],
        "ball_first": frames[0]["ball"]["position_m"],
        "player_first": frames[0]["players"][0]["position_m"],
    },
    open(Path(out, "result.json"), "w", encoding="utf-8"),
)
"""

_STEPS = 4


def _run_export(seed: int) -> dict:
    """在独立 Python 子进程中导出短 episode，返回结果 dict。"""
    with tempfile.TemporaryDirectory(prefix="grfue_seed_") as td:
        proc = subprocess.run(
            [sys.executable, "-c", _EXPORT_CODE, td, str(seed), str(_STEPS)],
            capture_output=True,
            text=True,
            cwd=".",
        )
        if proc.returncode != 0:
            raise AssertionError(f"seed 子进程失败 rc={proc.returncode}: {proc.stderr[-2000:]}")
        return json.load(open(Path(td) / "result.json", encoding="utf-8"))


class TestSeedReproducibility:
    def test_same_seed_two_processes_identical(self):
        a = _run_export(1001)
        b = _run_export(1001)
        assert a["hash"] == b["hash"]  # 同 seed、独立进程 → frames.jsonl 完全一致
        assert a["steps"] == b["steps"] == _STEPS
        assert a["score"] == b["score"]
        assert a["ball_first"] == b["ball_first"]
        assert a["player_first"] == b["player_first"]
        # meta.randomness 一致
        assert a["meta"]["randomness"] == b["meta"]["randomness"]
        assert a["meta"]["randomness"]["root_seed"] == 1001
        assert a["meta"]["source"]["seed"] == 1001

    def test_different_seeds_differ(self):
        h1 = _run_export(1001)["hash"]
        h2 = _run_export(1002)["hash"]
        h3 = _run_export(1003)["hash"]
        assert h1 != h2
        assert h1 != h3
        assert h2 != h3

    def test_seed_order_independent(self):
        h1 = _run_export(1001)["hash"]
        h2 = _run_export(1002)["hash"]
        h3 = _run_export(1001)["hash"]  # 再次运行 1001
        assert h1 == h3  # 结果不依赖进程内之前运行的其他 episode
        assert h2 != h1
