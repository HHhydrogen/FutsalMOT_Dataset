"""路径解析 / 可移植性 / resolved task 测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from grf_ue_bridge.config import resolver
from grf_ue_bridge.config.paths import (
    PLACEHOLDER_DATASET_ROOT,
    PLACEHOLDER_REPO_ROOT,
    PLACEHOLDER_UE_PROJECT_ROOT,
    resolve_task_relative,
)


def _make_task_dir(tmp_path: Path, *, dataset_root: Path = None) -> Path:
    """构造含内联单 config 的目录（机器路径直接写在 task 内）。"""
    base = tmp_path / "repo"
    base.mkdir(parents=True, exist_ok=True)
    ds = Path(dataset_root) if dataset_root is not None else (tmp_path / "ds")
    ue = tmp_path / "ue"
    task = {
        "schema": "futsalmot_dataset_task", "version": 2,
        "task_id": "res_t1", "episode_name": "episode_res_t1",
        "dataset_root": str(ds), "ue_project_root": str(ue),
        "export": {"scenario": "5_vs_5", "seed": 42, "num_steps": 300,
                   "playback_fps": 30},
        "ue": {"actor_mapping": "ue/actor_mapping.example.json",
               "sequences": [{"name": "LS_Cam_01", "camera_actor": "CineCam_01"}],
               "annotation_export": {"cameras": ["CineCam_01"], "image_width": 1920,
                                     "image_height": 1080}},
        "postprocess": {"workers": 4, "validation_level": "full"},
        "audit": {"expected_cameras": 1, "expected_frames_per_camera": 300},
    }
    (base / "task.json").write_text(json.dumps(task), encoding="utf-8")
    return base / "task.json"


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


def _p01_mapping_fields():
    fields = _partial_camera_task_fields()
    fields["ue"]["camera_mapping"]["P01"] = {
        "actor": "CineCam_P01",
        "sequence": "LS_Cam_P01",
    }
    return fields


class TestResolvePaths:
    def test_resolve_task_absolute_paths(self, tmp_path, pin_repo_root):
        tf = _make_task_dir(tmp_path)
        rt = resolver.resolve_task(tf)
        assert rt.task_id == "res_t1"
        assert Path(rt.trajectory_output).is_absolute()
        assert Path(rt.dataset_episode_dir).is_absolute()
        assert Path(rt.dataset_episode_dir).parent == Path(rt.dataset_root)
        assert Path(rt.dataset_episode_dir).name == "episode_res_t1"
        assert Path(rt.actor_mapping).is_absolute()

    def test_export_seed_applied(self, tmp_path, pin_repo_root):
        tf = _make_task_dir(tmp_path)
        task = json.loads(tf.read_text(encoding="utf-8"))
        task["export"]["seed"] = 123
        tf.write_text(json.dumps(task), encoding="utf-8")
        rt = resolver.resolve_task(tf)
        assert rt.export_profile["seed"] == 123

    def test_all_outputs_under_dataset_root(self, tmp_path, pin_repo_root):
        """轨迹与数据集都落 <dataset_root>/<episode_name>/ 自包含。"""
        tf = _make_task_dir(tmp_path, dataset_root=tmp_path / "custom_ds")
        rt = resolver.resolve_task(tf)
        assert Path(rt.dataset_root) == (tmp_path / "custom_ds").resolve()
        assert Path(rt.trajectory_output) == (
            tmp_path / "custom_ds" / "episode_res_t1").resolve()
        assert Path(rt.dataset_episode_dir) == (
            tmp_path / "custom_ds" / "episode_res_t1").resolve()

    def test_wrong_camera_expectation(self, tmp_path, pin_repo_root):
        tf = _make_task_dir(tmp_path)
        task = json.loads(tf.read_text(encoding="utf-8"))
        task["audit"]["expected_cameras"] = 4  # ue 只有 1 相机
        tf.write_text(json.dumps(task), encoding="utf-8")
        problems = resolver.validate_task(tf)
        assert any("相机数" in p for p in problems)

    def test_missing_dataset_root_fails(self, tmp_path, pin_repo_root):
        tf = _make_task_dir(tmp_path)
        task = json.loads(tf.read_text(encoding="utf-8"))
        del task["dataset_root"]
        tf.write_text(json.dumps(task), encoding="utf-8")
        problems = resolver.validate_task(tf)
        assert problems  # 解析失败（dataset_root 缺失）

    def test_resolved_task_propagates_simulation_and_mapping(self, tmp_path, pin_repo_root):
        tf = _make_task_dir(tmp_path)
        task = json.loads(tf.read_text(encoding="utf-8"))
        task.update(_camera_task_fields())
        tf.write_text(json.dumps(task), encoding="utf-8")

        resolved = resolver.resolve_task(tf)

        assert resolved.simulation["camera"]["distribution"]["anchors"] == [
            "C1", "C2", "C3", "C4", "C5"
        ]
        assert resolved.ue_profile["camera_mapping"]["C5"] == {
            "actor": "CineCam_05",
            "sequence": "LS_Cam_05",
        }
        assert resolved.simulation["camera"]["profiles"]["C1"]["resolution"] == [1920, 1080]
        assert resolved.simulation["camera"]["profiles"]["C1"]["lens"]["focal_length_mm"] == 15.0
        assert resolved.simulation["camera"]["profiles"]["C5"]["lens"]["focal_length_mm"] == 10.0

    def test_resolved_task_carries_partial_p01_profile(self, tmp_path, pin_repo_root):
        tf = _make_task_dir(tmp_path)
        task = json.loads(tf.read_text(encoding="utf-8"))
        task.update(_partial_camera_task_fields())
        tf.write_text(json.dumps(task), encoding="utf-8")

        resolved = resolver.resolve_task(tf)

        p01 = resolved.simulation["camera"]["profiles"]["P01"]
        assert p01["coverage"] == "partial_field"
        assert p01["region"] == "left_half"
        assert p01["position_m"] == [-10.0, -25.0, 16.0]
        assert p01["lens"]["focal_length_mm"] == 12.0

    def test_resolved_task_carries_p01_mapping(self, tmp_path, pin_repo_root):
        tf = _make_task_dir(tmp_path)
        task = json.loads(tf.read_text(encoding="utf-8"))
        task.update(_p01_mapping_fields())
        tf.write_text(json.dumps(task), encoding="utf-8")

        resolved = resolver.resolve_task(tf)

        assert resolved.ue_profile["camera_mapping"]["P01"] == {
            "actor": "CineCam_P01",
            "sequence": "LS_Cam_P01",
        }

    def test_resolved_task_propagates_partial_episode_profile(self, tmp_path, pin_repo_root):
        tf = _make_task_dir(tmp_path)
        task = json.loads(tf.read_text(encoding="utf-8"))
        task.update(_partial_camera_task_fields())
        task["simulation"]["camera"]["distribution"]["episode_profile"] = (
            "anchor_plus_one_partial"
        )
        tf.write_text(json.dumps(task), encoding="utf-8")

        resolved = resolver.resolve_task(tf)

        assert resolved.simulation["camera"]["distribution"]["episode_profile"] == (
            "anchor_plus_one_partial"
        )
        assert "P01" in resolved.simulation["camera"]["profiles"]

    def test_resolved_task_trims_camera_mapping_strings(self, tmp_path, pin_repo_root):
        tf = _make_task_dir(tmp_path)
        task = json.loads(tf.read_text(encoding="utf-8"))
        task.update(_camera_task_fields())
        task["ue"]["camera_mapping"]["C5"] = {
            "actor": "  CineCam_Main  ",
            "sequence": " LS_Cam_Main ",
        }
        tf.write_text(json.dumps(task), encoding="utf-8")

        resolved = resolver.resolve_task(tf)

        assert resolved.ue_profile["camera_mapping"] == {
            "C1": {"actor": "CineCam_01", "sequence": "LS_Cam_01"},
            "C2": {"actor": "CineCam_02", "sequence": "LS_Cam_02"},
            "C3": {"actor": "CineCam_03", "sequence": "LS_Cam_03"},
            "C4": {"actor": "CineCam_04", "sequence": "LS_Cam_04"},
            "C5": {"actor": "CineCam_Main", "sequence": "LS_Cam_Main"},
        }

    def test_legacy_task_keeps_empty_simulation(self, tmp_path, pin_repo_root):
        resolved = resolver.resolve_task(_make_task_dir(tmp_path))
        assert resolved.simulation == {}

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            (lambda fields: fields["ue"]["camera_mapping"].pop("C5"), "C5"),
            (lambda fields: fields["ue"]["camera_mapping"].update(
                {"C6": fields["ue"]["camera_mapping"].pop("C5")}
            ), "C6"),
            (lambda fields: fields["simulation"]["camera"]["profiles"].update(
                {"C6": fields["simulation"]["camera"]["profiles"].pop("C5")}
            ), "C6"),
        ],
    )
    def test_rejects_invalid_canonical_ids(self, tmp_path, pin_repo_root, change, message):
        tf = _make_task_dir(tmp_path)
        task = json.loads(tf.read_text(encoding="utf-8"))
        fields = _camera_task_fields()
        task.update(fields)
        change(task)
        tf.write_text(json.dumps(task), encoding="utf-8")

        with pytest.raises(ValueError, match=message):
            resolver.resolve_task(tf)

    @pytest.mark.parametrize("field", ["actor", "sequence"])
    def test_rejects_missing_mapping_field_with_canonical_id(
        self, tmp_path, pin_repo_root, field
    ):
        tf = _make_task_dir(tmp_path)
        task = json.loads(tf.read_text(encoding="utf-8"))
        task.update(_camera_task_fields())
        del task["ue"]["camera_mapping"]["C3"][field]
        tf.write_text(json.dumps(task), encoding="utf-8")

        with pytest.raises(ValueError, match=f"C3.*{field}"):
            resolver.resolve_task(tf)

    def test_rejects_duplicate_ue_actor_with_canonical_id(self, tmp_path, pin_repo_root):
        tf = _make_task_dir(tmp_path)
        task = json.loads(tf.read_text(encoding="utf-8"))
        task.update(_camera_task_fields())
        task["ue"]["camera_mapping"]["C5"]["actor"] = "CineCam_01"
        tf.write_text(json.dumps(task), encoding="utf-8")

        with pytest.raises(ValueError, match="C5.*CineCam_01"):
            resolver.resolve_task(tf)

    def test_validate_task_reports_camera_contract_failure(self, tmp_path, pin_repo_root):
        tf = _make_task_dir(tmp_path)
        task = json.loads(tf.read_text(encoding="utf-8"))
        task.update(_camera_task_fields())
        task["ue"]["camera_mapping"]["C5"]["actor"] = "CineCam_01"
        tf.write_text(json.dumps(task), encoding="utf-8")

        problems = resolver.validate_task(tf)

        assert any("C5" in problem and "重复" in problem for problem in problems)


class TestResolvedTaskFile:
    def test_save_load_roundtrip(self, tmp_path, pin_repo_root):
        tf = _make_task_dir(tmp_path)
        rt = resolver.resolve_task(tf)
        runtime = resolver.save_resolved_task(rt, Path(rt.repo_root))
        assert runtime.is_file()
        loaded = resolver.load_resolved_task(runtime)
        assert loaded.task_id == "res_t1"
        assert loaded.trajectory_output == rt.trajectory_output

    def test_load_rejects_bad_schema(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"schema": "nope", "version": 1}), encoding="utf-8")
        with pytest.raises(ValueError, match="schema"):
            resolver.load_resolved_task(bad)

    def test_runtime_dir_ignored(self, repo_root):
        gitignore = (repo_root / ".gitignore").read_text(encoding="utf-8")
        assert ".futsalmot/" in gitignore


class TestSanitizedProvenance:
    def test_no_absolute_paths(self, tmp_path, pin_repo_root):
        tf = _make_task_dir(tmp_path)
        rt = resolver.resolve_task(tf)
        sane = resolver.sanitize_resolved_task(rt)
        blob = json.dumps(sane)
        assert "\\" not in blob  # 无反斜杠
        for c in "ABCDEFG":
            assert f"{c}:/" not in blob, f"provenance 含盘符 {c}:/"
        assert PLACEHOLDER_DATASET_ROOT in sane["trajectory_output"]
        assert PLACEHOLDER_DATASET_ROOT in sane["dataset_episode_dir"]
        assert PLACEHOLDER_UE_PROJECT_ROOT in sane["ue_project_root"]
        assert PLACEHOLDER_REPO_ROOT in sane["repo_root"]

    def test_sanitize_posix(self, tmp_path):
        from grf_ue_bridge.config.paths import sanitize_path
        s = sanitize_path(
            "G:/DS/episode_x/img1", tmp_path / "r", tmp_path / "ue", tmp_path / "ds"
        )
        assert "\\" not in s


class TestTaskRelative:
    def test_windows_relative(self, tmp_path):
        p = resolve_task_relative("outputs/ep_x", tmp_path)
        assert p == (tmp_path / "outputs" / "ep_x").resolve()

    def test_absolute_passthrough(self, tmp_path):
        p = resolve_task_relative(str((tmp_path / "abs" / "x").resolve()), tmp_path)
        assert p == (tmp_path / "abs" / "x").resolve()

    def test_escape_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            resolve_task_relative("../..", tmp_path)
