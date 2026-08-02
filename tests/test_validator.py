"""episode 验证器的测试。"""

import json
import tempfile
from pathlib import Path

from grf_ue_bridge.validator import validate_episode


def _write_episode(meta: dict, frames: list[dict]) -> Path:
    """写入一个临时 episode 目录并返回其路径。"""
    tmp = Path(tempfile.mkdtemp())
    with open(tmp / "meta.json", "w") as f:
        json.dump(meta, f)
    with open(tmp / "frames.jsonl", "w") as f:
        for frame in frames:
            f.write(json.dumps(frame) + "\n")
    return tmp


def _make_frame(step: int, **overrides) -> dict:
    """创建一个合法的帧，可用 overrides 覆盖字段。"""
    frame = {
        "step": step,
        "time_seconds": step * 0.1,
        "score": [0, 0],
        "ball": {"position_m": [0.0, 0.0, 0.11], "source_grf_position": [0.0, 0.0, 0.11]},
        "players": [],
    }
    # 5 名左队 + 5 名右队球员
    for i in range(5):
        frame["players"].append({"id": f"L{i}", "position_m": [float(-10 + i), 0.0, 0.0]})
    for i in range(5):
        frame["players"].append({"id": f"R{i}", "position_m": [float(10 - i), 0.0, 0.0]})
    frame.update(overrides)
    return frame


def _default_meta(num_steps: int = 300) -> dict:
    return {
        "schema": "grf_ue_episode",
        "version": 1,
        "timing": {
            "source_step_seconds": 0.1,
            "playback_fps": 30,
            "num_steps": num_steps,
        },
        "field": {"length_m": 40.0, "width_m": 20.0},
    }


class TestValidator:
    def test_valid_episode(self):
        meta = _default_meta(5)
        frames = [_make_frame(i) for i in range(5)]
        path = _write_episode(meta, frames)
        assert validate_episode(path) == 0

    def test_missing_meta(self):
        tmp = Path(tempfile.mkdtemp())
        assert validate_episode(tmp) == 1

    def test_wrong_schema(self):
        meta = _default_meta(5)
        meta["schema"] = "wrong"
        frames = [_make_frame(i) for i in range(5)]
        path = _write_episode(meta, frames)
        assert validate_episode(path) == 1

    def test_wrong_version(self):
        meta = _default_meta(5)
        meta["version"] = 2
        frames = [_make_frame(i) for i in range(5)]
        path = _write_episode(meta, frames)
        assert validate_episode(path) == 1

    def test_wrong_frame_count(self):
        meta = _default_meta(5)
        frames = [_make_frame(i) for i in range(3)]  # 3 帧而不是 5 帧
        path = _write_episode(meta, frames)
        assert validate_episode(path) == 1

    def test_wrong_player_count(self):
        meta = _default_meta(2)
        frames = [_make_frame(i, players=_make_frame(i)["players"][:5]) for i in range(2)]
        path = _write_episode(meta, frames)
        assert validate_episode(path) == 1

    def test_invalid_id(self):
        meta = _default_meta(2)
        players = _make_frame(0)["players"]
        players[0] = {"id": "X0", "position_m": [0.0, 0.0, 0.0]}
        frames = [_make_frame(i, players=players[:10]) for i in range(2)]
        path = _write_episode(meta, frames)
        assert validate_episode(path) == 1

    def test_duplicate_id(self):
        meta = _default_meta(2)
        players = _make_frame(0)["players"]
        players[1] = {"id": "L0", "position_m": [0.0, 0.0, 0.0]}
        frames = [_make_frame(i, players=players[:10]) for i in range(2)]
        path = _write_episode(meta, frames)
        assert validate_episode(path) == 1

    def test_non_finite_position(self):
        meta = _default_meta(2)
        players = _make_frame(0)["players"]
        players[0] = {"id": "L0", "position_m": [float("nan"), 0.0, 0.0]}
        frames = [_make_frame(i, players=players[:10]) for i in range(2)]
        path = _write_episode(meta, frames)
        assert validate_episode(path) == 1

    def test_player_z_not_zero(self):
        meta = _default_meta(2)
        players = _make_frame(0)["players"]
        players[0] = {"id": "L0", "position_m": [0.0, 0.0, 0.5]}
        frames = [_make_frame(i, players=players[:10]) for i in range(2)]
        path = _write_episode(meta, frames)
        assert validate_episode(path) == 1

    def test_out_of_bounds(self):
        meta = _default_meta(2)
        players = _make_frame(0)["players"]
        players[0] = {"id": "L0", "position_m": [100.0, 0.0, 0.0]}
        frames = [_make_frame(i, players=players[:10]) for i in range(2)]
        path = _write_episode(meta, frames)
        assert validate_episode(path) == 1

    def test_wrong_step_number(self):
        meta = _default_meta(3)
        frames = [_make_frame(i + 1) for i in range(3)]  # step 为 1,2,3 而不是 0,1,2
        path = _write_episode(meta, frames)
        assert validate_episode(path) == 1
