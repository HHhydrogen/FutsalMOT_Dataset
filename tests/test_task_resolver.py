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
