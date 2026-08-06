"""grf-ue task CLI 测试（用 CliRunner，避免真实 UE/GRF）。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from grf_ue_bridge.cli import app
from grf_ue_bridge.config.paths import (
    ENV_DATASET_ROOT,
    ENV_REPO_ROOT,
    ENV_UE_PROJECT_ROOT,
)

runner = CliRunner()


def _valid_png() -> bytes:
    import io
    buf = io.BytesIO()
    Image.new("L", (8, 8), 3).save(buf, format="PNG")
    return buf.getvalue()


def _make_task_dir(tmp_path: Path, cam_count: int = 1, frames: int = 1) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "export.json").write_text(json.dumps({
        "scenario": "5_vs_5", "seed": 42, "num_steps": frames, "playback_fps": 30,
    }), encoding="utf-8")
    cameras = [f"CineCam_0{i}" for i in range(1, cam_count + 1)]
    (repo / "ue.json").write_text(json.dumps({
        "schema": "futsalmot_ue_profile", "version": 1,
        "actor_mapping": "ue/actor_mapping.example.json",
        "sequences": [{"name": f"LS_{c}", "camera_actor": c} for c in cameras],
        "annotation_export": {"cameras": cameras, "image_width": 64, "image_height": 64},
    }), encoding="utf-8")
    task = {
        "schema": "futsalmot_dataset_task", "version": 1,
        "task_id": "cli_t1", "episode_name": "episode_cli_t1",
        "export_profile": "export.json", "ue_profile": "ue.json", "seed": None,
        "paths": {"trajectory_output": "episode_cli_t1",
                  "dataset_output": "episode_cli_t1"},
        "postprocess": {"workers": 2, "validation_level": "full"},
        "audit": {"expected_cameras": cam_count, "expected_frames_per_camera": frames},
    }
    (repo / "task.json").write_text(json.dumps(task), encoding="utf-8")
    return repo / "task.json"


def _make_minimal_dataset(ds_root: Path, ep: str, cam_count: int = 1) -> Path:
    """构造能通过 audit（validation none）的最小数据集目录。"""
    png = _valid_png()
    ep_dir = ds_root / ep
    for i in range(1, cam_count + 1):
        cam = ep_dir / f"CineCam_0{i}"
        (cam / "img1").mkdir(parents=True)
        (cam / "mask").mkdir(parents=True)
        (cam / "render").mkdir(parents=True)
        (cam / "render_mask").mkdir(parents=True)
        (cam / "labels" / "det").mkdir(parents=True)
        (cam / "labels" / "seg").mkdir(parents=True)
        (cam / "gt").mkdir(parents=True)
        (cam / "camera.json").write_text(json.dumps({
            "image_width": 64, "image_height": 64,
            "intrinsics": {"fx": 60.0, "fy": 60.0, "cx": 32.0, "cy": 32.0,
                           "width": 64, "height": 64},
            "extrinsics": {"world_location_m": [0.0, 0.0, 1.0],
                           "forward": [1.0, 0.0, 0.0], "right": [0.0, 1.0, 0.0],
                           "up": [0.0, 0.0, 1.0]},
        }), encoding="utf-8")
        (cam / "seqinfo.ini").write_text("[Sequence]\nfps=30\n", encoding="utf-8")
        (cam / "annotations.jsonl").write_text(
            json.dumps({"frame_index": 1, "source_step": 0, "time_seconds": 0.0,
                        "episode_id": ep, "objects": []}) + "\n", encoding="utf-8")
        (cam / "img1" / "000001.png").write_bytes(png)
        (cam / "mask" / "000001.png").write_bytes(png)
        (cam / "render" / "000000.png").write_bytes(png)
        (cam / "render_mask" / "000000.exr").write_bytes(b"E")
        (cam / "labels" / "det" / "000001.txt").write_text("x\n", encoding="utf-8")
        (cam / "labels" / "seg" / "000001.txt").write_text("x\n", encoding="utf-8")
        (cam / "gt" / "gt.txt").write_text("1,1,1,1,1,1,1,1,1,1\n", encoding="utf-8")
    return ep_dir


def _set_env(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    ds = tmp_path / "ds"
    repo.mkdir(exist_ok=True)
    monkeypatch.setenv(ENV_UE_PROJECT_ROOT, str(repo))
    monkeypatch.setenv(ENV_DATASET_ROOT, str(ds))
    monkeypatch.setenv(ENV_REPO_ROOT, str(repo))
    return repo, ds


class TestTaskValidateCLI:
    def test_validate_pass(self, tmp_path, monkeypatch):
        _set_env(monkeypatch, tmp_path)
        tf = _make_task_dir(tmp_path)
        r = runner.invoke(app, ["task", "validate", str(tf)])
        assert r.exit_code == 0, r.output
        assert "PASS" in r.output

    def test_validate_fail_bad_profile(self, tmp_path, monkeypatch):
        _set_env(monkeypatch, tmp_path)
        tf = _make_task_dir(tmp_path)
        task = json.loads(tf.read_text(encoding="utf-8"))
        task["export_profile"] = "missing.json"
        tf.write_text(json.dumps(task), encoding="utf-8")
        r = runner.invoke(app, ["task", "validate", str(tf)])
        assert r.exit_code == 1
        assert "FAIL" in r.output


class TestTaskResolveCLI:
    def test_resolve_writes_runtime(self, tmp_path, monkeypatch):
        repo, _ = _set_env(monkeypatch, tmp_path)
        tf = _make_task_dir(tmp_path)
        r = runner.invoke(app, ["task", "resolve", str(tf)])
        assert r.exit_code == 0, r.output
        assert "Task ID: cli_t1" in r.output
        runtime = repo / ".futsalmot" / "runtime" / "cli_t1" / "resolved-task.json"
        assert runtime.is_file()

    def test_ue_command_prints_run_task(self, tmp_path, monkeypatch):
        repo, _ = _set_env(monkeypatch, tmp_path)
        tf = _make_task_dir(tmp_path)
        r = runner.invoke(app, ["task", "ue-command", str(tf)])
        assert r.exit_code == 0, r.output
        assert "run_task.py" in r.output
        assert "--resolved-task" in r.output


class TestTaskStatusAudit:
    def test_status_readonly(self, tmp_path, monkeypatch):
        repo, ds = _set_env(monkeypatch, tmp_path)
        _make_minimal_dataset(ds, "episode_cli_t1")
        tf = _make_task_dir(tmp_path)
        r = runner.invoke(app, ["task", "status", str(tf)])
        assert r.exit_code == 0, r.output
        assert "episode_cli_t1" in r.output

    def test_audit_passes_minimal(self, tmp_path, monkeypatch):
        repo, ds = _set_env(monkeypatch, tmp_path)
        _make_minimal_dataset(ds, "episode_cli_t1")
        tf = _make_task_dir(tmp_path)
        r = runner.invoke(app, ["task", "audit", str(tf), "--validation-level", "none"])
        assert r.exit_code == 0, r.output

    def test_postprocess_skip_all_noop(self, tmp_path, monkeypatch):
        repo, ds = _set_env(monkeypatch, tmp_path)
        tf = _make_task_dir(tmp_path)
        r = runner.invoke(app, ["task", "postprocess", str(tf),
                                "--skip-cryptomatte", "--skip-annotate", "--skip-validate"])
        assert r.exit_code == 0, r.output


class TestActiveTask:
    def test_active_cycle(self, tmp_path, monkeypatch, repo_root):
        repo, ds = _set_env(monkeypatch, tmp_path)
        tf = _make_task_dir(tmp_path)
        # 激活写真实仓库 .futsalmot（gitignore），测试后清理
        from grf_ue_bridge.config import resolver
        act_path = resolver.save_active_task(tf, repo_root)
        try:
            r = runner.invoke(app, ["task", "status"])  # 无参数 → active
            assert r.exit_code == 0, r.output
            assert "Active task:" in r.output
            # 显式 task 优先于 active
            r2 = runner.invoke(app, ["task", "status", str(tf)])
            assert r2.exit_code == 0
        finally:
            resolver.clear_active_task(repo_root)
        # deactivate 后无 active → 报错
        runner.invoke(app, ["task", "deactivate"])
        r3 = runner.invoke(app, ["task", "status"])
        assert r3.exit_code == 2

    def test_activate_deactivate_cli(self, tmp_path, monkeypatch, repo_root):
        repo, ds = _set_env(monkeypatch, tmp_path)
        tf = _make_task_dir(tmp_path)
        from grf_ue_bridge.config import resolver
        try:
            r = runner.invoke(app, ["task", "activate", str(tf)])
            assert r.exit_code == 0
            assert resolver.load_active_task(repo_root) is not None
            r2 = runner.invoke(app, ["task", "deactivate"])
            assert r2.exit_code == 0
            assert resolver.load_active_task(repo_root) is None
        finally:
            resolver.clear_active_task(repo_root)
