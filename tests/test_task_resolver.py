"""路径解析 / 安全 / 可移植性 / resolved task 测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from grf_ue_bridge.config import loader, resolver
from grf_ue_bridge.config.paths import (
    ENV_DATASET_ROOT,
    ENV_UE_PROJECT_ROOT,
    PLACEHOLDER_DATASET_ROOT,
    PLACEHOLDER_REPO_ROOT,
    PLACEHOLDER_UE_PROJECT_ROOT,
    resolve_task_relative,
)


def _make_task_dir(tmp_path: Path) -> Path:
    """构造含 task + export/ue profile 的目录（置于 repo 内，供 sanitize 匹配）。"""
    base = tmp_path / "repo"
    base.mkdir(parents=True, exist_ok=True)
    (base / "export.json").write_text(json.dumps({
        "scenario": "5_vs_5", "seed": 42, "num_steps": 300,
        "playback_fps": 30,
    }), encoding="utf-8")
    (base / "ue.json").write_text(json.dumps({
        "schema": "futsalmot_ue_profile", "version": 1,
        "actor_mapping": "ue/actor_mapping.example.json",
        "sequences": [{"name": "LS_Cam_01", "camera_actor": "CineCam_01"}],
        "annotation_export": {"cameras": ["CineCam_01"], "image_width": 1920,
                              "image_height": 1080},
    }), encoding="utf-8")
    task = {
        "schema": "futsalmot_dataset_task", "version": 1,
        "task_id": "res_t1", "episode_name": "episode_res_t1",
        "export_profile": "export.json", "ue_profile": "ue.json",
        "seed": None,
        "paths": {"trajectory_output": "outputs/episode_res_t1",
                  "dataset_output": "episode_res_t1"},
        "postprocess": {"workers": 4, "validation_level": "full"},
        "audit": {"expected_cameras": 1, "expected_frames_per_camera": 300},
    }
    (base / "task.json").write_text(json.dumps(task), encoding="utf-8")
    return base / "task.json"


def _env(repo: Path, ds: Path) -> dict:
    return {ENV_UE_PROJECT_ROOT: str(repo), ENV_DATASET_ROOT: str(ds)}


class TestResolvePaths:
    def test_resolve_task_absolute_paths(self, tmp_path):
        tf = _make_task_dir(tmp_path)
        env = _env(tmp_path / "repo", tmp_path / "ds")
        rt = resolver.resolve_task(tf, env=env)
        assert rt.task_id == "res_t1"
        assert Path(rt.trajectory_output).is_absolute()
        assert Path(rt.dataset_episode_dir).is_absolute()
        assert Path(rt.dataset_episode_dir).parent == Path(rt.dataset_root)
        assert Path(rt.dataset_episode_dir).name == "episode_res_t1"
        assert Path(rt.actor_mapping).is_absolute()

    def test_seed_override_applied(self, tmp_path):
        tf = _make_task_dir(tmp_path)
        env = _env(tmp_path / "repo", tmp_path / "ds")
        # 覆盖 task 的 seed → export profile seed 变化
        task = json.loads(tf.read_text(encoding="utf-8"))
        task["seed"] = 123
        tf.write_text(json.dumps(task), encoding="utf-8")
        rt = resolver.resolve_task(tf, env=env)
        assert rt.export_profile["seed"] == 123

    def test_escape_rejected(self, tmp_path):
        tf = _make_task_dir(tmp_path)
        task = json.loads(tf.read_text(encoding="utf-8"))
        task["paths"]["trajectory_output"] = "../../escape"
        tf.write_text(json.dumps(task), encoding="utf-8")
        with pytest.raises(ValueError, match="逃逸"):
            resolver.resolve_task(tf, env=_env(tmp_path / "repo", tmp_path / "ds"))

    def test_absolute_path_rejected_by_default(self, tmp_path):
        tf = _make_task_dir(tmp_path)
        task = json.loads(tf.read_text(encoding="utf-8"))
        task["paths"]["trajectory_output"] = "C:/abs/path"
        tf.write_text(json.dumps(task), encoding="utf-8")
        with pytest.raises(ValueError, match="绝对路径"):
            resolver.resolve_task(tf, env=_env(tmp_path / "repo", tmp_path / "ds"))

    def test_absolute_path_allowed_with_flag(self, tmp_path):
        tf = _make_task_dir(tmp_path)
        task = json.loads(tf.read_text(encoding="utf-8"))
        task["paths"]["trajectory_output"] = str((tmp_path / "abs_out").resolve())
        tf.write_text(json.dumps(task), encoding="utf-8")
        rt = resolver.resolve_task(
            tf, env=_env(tmp_path / "repo", tmp_path / "ds"),
            allow_absolute_paths=True,
        )
        assert Path(rt.trajectory_output) == (tmp_path / "abs_out").resolve()

    def test_validate_task_episode_name_consistency(self, tmp_path):
        tf = _make_task_dir(tmp_path)
        task = json.loads(tf.read_text(encoding="utf-8"))
        task["paths"]["dataset_output"] = "other_name"
        tf.write_text(json.dumps(task), encoding="utf-8")
        problems = resolver.validate_task(tf, env=_env(tmp_path / "repo", tmp_path / "ds"))
        assert any("episode_name" in p for p in problems)

    def test_wrong_camera_expectation(self, tmp_path):
        tf = _make_task_dir(tmp_path)
        task = json.loads(tf.read_text(encoding="utf-8"))
        task["audit"]["expected_cameras"] = 4  # profile 只有 1 相机
        tf.write_text(json.dumps(task), encoding="utf-8")
        problems = resolver.validate_task(tf, env=_env(tmp_path / "repo", tmp_path / "ds"))
        assert any("相机数" in p for p in problems)


class TestResolvedTaskFile:
    def test_save_load_roundtrip(self, tmp_path):
        tf = _make_task_dir(tmp_path)
        env = _env(tmp_path / "repo", tmp_path / "ds")
        rt = resolver.resolve_task(tf, env=env)
        runtime = resolver.save_resolved_task(rt, tmp_path / "repo")
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
    def test_no_absolute_paths(self, tmp_path):
        tf = _make_task_dir(tmp_path)
        env = _env(tmp_path / "repo", tmp_path / "ds")
        rt = resolver.resolve_task(tf, env=env)
        sane = resolver.sanitize_resolved_task(rt)
        blob = json.dumps(sane)
        assert "\\" not in blob  # 无反斜杠
        for c in "ABCDEFG":
            assert f"{c}:/" not in blob, f"provenance 含盘符 {c}:/"
        assert PLACEHOLDER_REPO_ROOT in sane["trajectory_output"]
        assert PLACEHOLDER_DATASET_ROOT in sane["dataset_episode_dir"]
        assert PLACEHOLDER_UE_PROJECT_ROOT in sane["ue_project_root"]

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

    def test_escape_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            resolve_task_relative("../..", tmp_path)
