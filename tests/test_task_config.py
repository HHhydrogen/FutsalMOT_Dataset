"""本地配置与 task schema 测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from grf_ue_bridge.config import loader
from grf_ue_bridge.config.models import (
    AuditTaskConfig,
    DatasetTaskConfig,
    LocalConfig,
    PostprocessTaskConfig,
)
from grf_ue_bridge.config.paths import (
    ENV_DATASET_ROOT,
    ENV_UE_PROJECT_ROOT,
    LOCAL_CONFIG_FILENAME,
    LOCAL_CONFIG_SCHEMA,
)


def _write_task(path: Path, **over):
    data = {
        "schema": "futsalmot_dataset_task",
        "version": 1,
        "task_id": "t1",
        "episode_name": "episode_t1",
        "export_profile": "export.json",
        "ue_profile": "ue.json",
        "seed": None,
        "paths": {
            "trajectory_output": "outputs/episode_t1",
            "dataset_output": "episode_t1",
        },
        "postprocess": {
            "include_ball": True,
            "workers": 4,
            "chunk_size": 50,
            "png_compress_level": 1,
            "formats": ["json", "mot", "yolo-det", "yolo-seg"],
            "clean_stale": True,
            "validation_level": "full",
        },
        "audit": {"expected_cameras": 4, "expected_frames_per_camera": 300},
    }
    data.update(over)
    (path / "task.json").write_text(json.dumps(data), encoding="utf-8")
    return path / "task.json"


class TestLocalConfig:
    def test_example_file_parses(self, repo_root):
        p = repo_root / ".futsalmot.local.example.json"
        assert p.is_file()
        data = json.loads(p.read_text(encoding="utf-8"))
        assert data["schema"] == LOCAL_CONFIG_SCHEMA
        assert data["version"] == 1

    def test_real_local_ignored(self, repo_root):
        gitignore = (repo_root / ".gitignore").read_text(encoding="utf-8")
        assert ".futsalmot.local.json" in gitignore
        assert "configs/tasks/" in gitignore or "tasks/local/" in gitignore

    def test_load_local_config_file(self, tmp_path, monkeypatch):
        cfg = tmp_path / "local.json"
        cfg.write_text(json.dumps({
            "schema": LOCAL_CONFIG_SCHEMA, "version": 1,
            "ue_project_root": "D:/UE", "dataset_root": "G:/DS",
        }), encoding="utf-8")
        monkeypatch.setenv("FUTSALMOT_LOCAL_CONFIG", str(cfg))
        local = loader.load_local_config()
        assert local is not None
        assert local.ue_project_root == "D:/UE"

    def test_env_overrides_local(self, tmp_path, monkeypatch):
        cfg = tmp_path / "local.json"
        cfg.write_text(json.dumps({
            "schema": LOCAL_CONFIG_SCHEMA, "version": 1,
            "ue_project_root": "D:/UE_LOCAL", "dataset_root": "G:/DS_LOCAL",
        }), encoding="utf-8")
        monkeypatch.setenv("FUTSALMOT_LOCAL_CONFIG", str(cfg))
        monkeypatch.setenv(ENV_UE_PROJECT_ROOT, "D:/UE_ENV")
        monkeypatch.setenv(ENV_DATASET_ROOT, "G:/DS_ENV")
        paths = loader.resolve_local_paths()
        assert paths["ue_project_root"].as_posix() == "D:/UE_ENV"
        assert paths["dataset_root"].as_posix() == "G:/DS_ENV"

    def test_missing_required_fails(self, tmp_path, monkeypatch):
        # 隔离真实 .futsalmot.local.json：指向不存在的 local 配置文件
        monkeypatch.setenv("FUTSALMOT_LOCAL_CONFIG", str(tmp_path / "nope.json"))
        monkeypatch.delenv(ENV_UE_PROJECT_ROOT, raising=False)
        monkeypatch.delenv(ENV_DATASET_ROOT, raising=False)
        with pytest.raises(ValueError, match="dataset_root"):
            loader.resolve_local_paths()

    def test_repo_root_default(self, repo_root):
        paths = loader.resolve_local_paths({
            ENV_UE_PROJECT_ROOT: "D:/UE",
            ENV_DATASET_ROOT: "G:/DS",
        })
        assert paths["repo_root"].resolve() == repo_root.resolve()


class TestTaskSchema:
    def test_valid_task(self, tmp_path):
        tf = _write_task(tmp_path)
        t = loader.load_task_config(tf)
        assert t.task_id == "t1"
        assert t.episode_name == "episode_t1"
        assert t.postprocess.validation_level == "full"

    def test_bad_schema_rejected(self, tmp_path):
        tf = _write_task(tmp_path, schema="other")
        with pytest.raises(ValueError):
            loader.load_task_config(tf)

    def test_bad_task_id_rejected(self, tmp_path):
        tf = _write_task(tmp_path, task_id="bad id!")
        with pytest.raises(Exception):
            loader.load_task_config(tf)

    def test_bad_episode_name_rejected(self, tmp_path):
        tf = _write_task(tmp_path, episode_name="has/slash")
        with pytest.raises(Exception):
            loader.load_task_config(tf)

    def test_bad_workers_rejected(self, tmp_path):
        tf = _write_task(tmp_path, postprocess={"workers": 999})
        with pytest.raises(Exception):
            loader.load_task_config(tf)

    def test_unsupported_format_rejected(self, tmp_path):
        tf = _write_task(tmp_path, postprocess={"formats": ["json", "coco"]})
        with pytest.raises(ValueError, match="格式"):
            loader.load_task_config(tf)  # load 时即调用 validate_formats

    def test_missing_profile_file(self, tmp_path):
        tf = _write_task(tmp_path)
        from grf_ue_bridge.config import resolver
        problems = resolver.validate_task(tf, env={
            ENV_UE_PROJECT_ROOT: "D:/UE", ENV_DATASET_ROOT: "G:/DS",
        })
        assert any("export profile" in p for p in problems)

    def test_defaults(self):
        pp = PostprocessTaskConfig()
        assert pp.workers == 4
        assert pp.validation_level == "full"
        assert "yolo-seg" in pp.formats
        au = AuditTaskConfig()
        assert au.expected_cameras == 4
        assert au.expected_frames_per_camera == 300
