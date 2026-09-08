"""UE 端 resolved task 契约测试（不导入 unreal，只验证 JSON 契约）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from grf_ue_bridge.config import resolver
from ue import run_task


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


def _camera_task_fields():
    profiles = {}
    mapping = {}
    for index, camera_id in enumerate(("C1", "C2", "C3", "C4", "C5"), 1):
        profiles[camera_id] = {
            "type": "static_surveillance",
            "coverage": "full_field",
            "resolution": [1920, 1080],
            "lens": {"focal_length_mm": 10.0 if camera_id == "C5" else 15.0},
            "position_m": [float(index), -8.0, 8.0],
            "rotation_deg": [0.0, 45.0, 0.0],
            "height_m": 8.0,
            "distortion": None,
        }
        mapping[camera_id] = {
            "actor": f"CineCam_{index:02d}",
            "sequence": f"LS_Cam_{index:02d}",
        }
    return {
        "simulation": {
            "camera": {
                "distribution": {
                    "system": "dual",
                    "anchors": ["C1", "C2", "C3", "C4", "C5"],
                    "static_surveillance_ratio": [0.70, 0.80],
                    "dynamic_broadcast_ratio": [0.20, 0.30],
                },
                "profiles": profiles,
            }
        },
        "ue": {"camera_mapping": mapping},
    }


def _partial_camera_task_fields():
    fields = _camera_task_fields()
    fields["simulation"]["camera"]["profiles"]["P01"] = {
        "type": "static_surveillance",
        "coverage": "partial_field",
        "region": "left_half",
    }
    fields["ue"]["camera_mapping"]["P01"] = {
        "actor": "CineCam_P01",
        "sequence": "LS_Cam_P01",
    }
    return fields


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

    def test_serializes_simulation_and_canonical_camera_mapping(
        self, tmp_path, pin_repo_root
    ):
        rt, repo = _make_resolved(tmp_path, pin_repo_root)
        task = json.loads((repo / "task.json").read_text(encoding="utf-8"))
        task.update(_camera_task_fields())
        (repo / "task.json").write_text(json.dumps(task), encoding="utf-8")
        rt = resolver.resolve_task(repo / "task.json")

        data = rt.model_dump(by_alias=True)

        assert data["simulation"]["camera"]["distribution"]["anchors"] == [
            "C1", "C2", "C3", "C4", "C5"
        ]
        assert data["ue_profile"]["camera_mapping"]["C5"] == {
            "actor": "CineCam_05",
            "sequence": "LS_Cam_05",
        }

    def test_serializes_trimmed_canonical_camera_mapping(
        self, tmp_path, pin_repo_root
    ):
        rt, repo = _make_resolved(tmp_path, pin_repo_root)
        task = json.loads((repo / "task.json").read_text(encoding="utf-8"))
        task.update(_camera_task_fields())
        task["ue"]["camera_mapping"]["C5"] = {
            "actor": "  CineCam_Main  ",
            "sequence": " LS_Cam_Main ",
        }
        (repo / "task.json").write_text(json.dumps(task), encoding="utf-8")

        resolved = resolver.resolve_task(repo / "task.json")

        assert resolved.ue_profile["camera_mapping"]["C5"] == {
            "actor": "CineCam_Main",
            "sequence": "LS_Cam_Main",
        }

    def test_serializes_p01_camera_mapping_and_profile(self, tmp_path, pin_repo_root):
        _make_resolved(tmp_path, pin_repo_root)
        task = json.loads((pin_repo_root / "task.json").read_text(encoding="utf-8"))
        task.update(_partial_camera_task_fields())
        task["simulation"]["camera"]["distribution"]["episode_profile"] = (
            "anchor_plus_one_partial"
        )
        (pin_repo_root / "task.json").write_text(json.dumps(task), encoding="utf-8")

        resolved = resolver.resolve_task(pin_repo_root / "task.json")
        data = resolved.model_dump(by_alias=True)

        assert data["simulation"]["camera"]["distribution"]["episode_profile"] == (
            "anchor_plus_one_partial"
        )
        assert data["ue_profile"]["camera_mapping"]["P01"] == {
            "actor": "CineCam_P01",
            "sequence": "LS_Cam_P01",
        }
        assert data["simulation"]["camera"]["profiles"]["P01"]["lens"]["focal_length_mm"] == 12.0

    def test_runtime_contract_selects_p01_after_anchors_and_preserves_override(
        self, tmp_path, pin_repo_root
    ):
        _make_resolved(tmp_path, pin_repo_root)
        task = json.loads((pin_repo_root / "task.json").read_text(encoding="utf-8"))
        task.update(_partial_camera_task_fields())
        task["simulation"]["camera"]["distribution"]["episode_profile"] = (
            "anchor_plus_one_partial"
        )
        (pin_repo_root / "task.json").write_text(json.dumps(task), encoding="utf-8")

        resolved = resolver.resolve_task(pin_repo_root / "task.json").model_dump(by_alias=True)

        selected = run_task._resolve_runtime_camera_selection(
            resolved,
            legacy_sequences=[],
            legacy_cameras=[],
        )
        assert [entry["camera_id"] for entry in selected["entries"]] == [
            "C1", "C2", "C3", "C4", "C5", "P01"
        ]
        assert selected["sequences"][-1] == {
            "name": "LS_Cam_P01",
            "camera_actor": "CineCam_P01",
            "camera_id": "P01",
        }

        override = run_task._resolve_runtime_camera_selection(
            resolved,
            legacy_sequences=[],
            legacy_cameras=[],
            camera_ids=["P01"],
        )
        assert [entry["camera_id"] for entry in override["entries"]] == ["P01"]

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
