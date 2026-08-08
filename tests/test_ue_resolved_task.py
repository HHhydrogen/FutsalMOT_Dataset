"""UE 端 resolved task 契约测试（不导入 unreal，只验证 JSON 契约）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from grf_ue_bridge.config import resolver


def _make_resolved(tmp_path: Path, pin_repo_root: Path):
    repo = pin_repo_root
    ds = tmp_path / "ds"
    (repo / "ue" / "actor_mapping.example.json").parent.mkdir(parents=True, exist_ok=True)
    (repo / "ue" / "actor_mapping.example.json").write_text('{}', encoding="utf-8")
    task = {
        "schema": "futsalmot_dataset_task", "version": 2,
        "task_id": "ue_t1", "episode_name": "episode_ue_t1",
        "dataset_root": str(ds), "ue_project_root": str(repo),
        "export": {"scenario": "5_vs_5", "seed": 42, "num_steps": 300,
                   "playback_fps": 30},
        "ue": {"actor_mapping": "ue/actor_mapping.example.json",
               "sequences": [{"name": "LS_Cam_01", "camera_actor": "CineCam_01"}],
               "annotation_export": {"cameras": ["CineCam_01"], "image_width": 1920,
                                     "image_height": 1080}},
        "postprocess": {}, "audit": {},
    }
    (repo / "task.json").write_text(json.dumps(task), encoding="utf-8")
    return resolver.resolve_task(repo / "task.json"), repo


class TestResolvedTaskContract:
    def test_contains_ue_required_fields(self, tmp_path, pin_repo_root):
        rt, repo = _make_resolved(tmp_path, pin_repo_root)
        d = rt.model_dump(by_alias=True)
        assert d["schema"] == "futsalmot_resolved_task"
        assert d["version"] == 1
        # run_task.py 需要读取的字段
        assert d["trajectory_output"]
        assert d["dataset_root"]
        assert d["actor_mapping"]
        assert "ue_profile" in d and d["ue_profile"]["sequences"]
        assert d["ue_profile"]["annotation_export"]["cameras"] == ["CineCam_01"]

    def test_load_resolved_rejects_bad_schema(self, tmp_path, pin_repo_root):
        rt, repo = _make_resolved(tmp_path, pin_repo_root)
        path = resolver.save_resolved_task(rt, repo)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["schema"] = "wrong"
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match="schema"):
            resolver.load_resolved_task(path)

    def test_load_resolved_rejects_bad_version(self, tmp_path, pin_repo_root):
        rt, repo = _make_resolved(tmp_path, pin_repo_root)
        path = resolver.save_resolved_task(rt, repo)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["version"] = 99
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match="version"):
            resolver.load_resolved_task(path)

    def test_sanitized_no_absolute(self, tmp_path, pin_repo_root):
        rt, repo = _make_resolved(tmp_path, pin_repo_root)
        sane = resolver.sanitize_resolved_task(rt)
        blob = json.dumps(sane)
        assert "\\\\" not in blob  # 无反斜杠
        assert not any(f"{chr(65+i)}:/" in blob for i in range(6))  # 无盘符 A:-F:

    def test_runtime_path_inside_gitignored(self, tmp_path, pin_repo_root):
        rt, repo = _make_resolved(tmp_path, pin_repo_root)
        p = resolver.save_resolved_task(rt, repo)
        assert ".futsalmot" in p.parts
