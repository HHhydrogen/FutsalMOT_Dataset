"""grf-ue task CLI 测试（用 CliRunner，避免真实 UE/GRF）。"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

from PIL import Image
import numpy as np
import pytest
from typer.testing import CliRunner
import inspect

from grf_ue_bridge.cli import app

runner = CliRunner()


def _valid_png() -> bytes:
    import io
    buf = io.BytesIO()
    Image.new("L", (8, 8), 3).save(buf, format="PNG")
    return buf.getvalue()


def _make_task_dir(tmp_path: Path, cam_count: int = 1, frames: int = 1) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    ds = tmp_path / "ds"
    cameras = [f"CineCam_0{i}" for i in range(1, cam_count + 1)]
    task = {
        "schema": "futsalmot_dataset_task", "version": 2,
        "task_id": "cli_t1", "episode_name": "episode_cli_t1",
        "dataset_root": str(ds), "ue_project_root": str(repo),
        "export": {"scenario": "5_vs_5", "seed": 42, "num_steps": frames,
                   "playback_fps": 30},
        "ue": {"actor_mapping": "ue/actor_mapping.example.json",
               "sequences": [{"name": f"LS_{c}", "camera_actor": c} for c in cameras],
               "annotation_export": {"cameras": cameras, "image_width": 64,
                                     "image_height": 64}},
        "postprocess": {"workers": 2, "validation_level": "full"},
        "audit": {"expected_cameras": cam_count, "expected_frames_per_camera": frames},
    }
    (repo / "task.json").write_text(json.dumps(task), encoding="utf-8")
    return repo / "task.json"


def _write_local_config(tmp_path: Path, name: str = "local.json") -> Path:
    dataset_root = tmp_path / f"{name}-dataset"
    dataset_root.mkdir(parents=True)
    ue_root = tmp_path / f"{name}-ue"
    ue_root.mkdir()
    (ue_root / "project.uproject").write_text("{}", encoding="utf-8")
    path = tmp_path / name
    path.write_text(json.dumps({
        "dataset_root": str(dataset_root),
        "ue_project_root": str(ue_root),
    }), encoding="utf-8")
    return path


def _make_v3_task(tmp_path: Path) -> Path:
    path = tmp_path / "task-v3.json"
    path.write_text(json.dumps({
        "schema": "futsalmot_task", "version": 3, "episode_id": "ep_cli_v3",
        "simulation": {"scenario": "5_vs_5", "seed": 17, "steps": 3},
        "cameras": {"C01": "Camera_A", "C02": "Camera_B"},
        "output": {
            "fps": 30, "resolution": [1280, 720],
            "annotations": ["mot", "pose", "mots"],
            "classes": ["player", "ball"],
        },
    }), encoding="utf-8")
    return path


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


class TestTaskValidateCLI:
    @pytest.mark.parametrize("function_name", [
        "task_export", "task_postprocess", "task_audit", "task_cleanup",
        "task_activate", "task_status", "task_ue_command", "task_manifest",
        "task_motion_quality",
    ])
    def test_resolving_task_commands_accept_local_config(self, function_name):
        from grf_ue_bridge import cli
        assert "local_config" in inspect.signature(getattr(cli, function_name)).parameters

    def test_runtime_resolution_for_export_forwards_local_config(self, tmp_path, monkeypatch):
        from grf_ue_bridge import cli
        expected = object()
        seen = {}
        monkeypatch.setattr(cli, "_resolve_task_or_active", lambda task: task)
        monkeypatch.setattr(cli._resolver, "resolve_task", lambda task, local_config=None: seen.update({"task": task, "local": local_config}) or expected)
        task = tmp_path / "task.json"
        local = tmp_path / "local.json"
        assert cli._resolve_runtime(task, local) == (task, expected)
        assert seen == {"task": task, "local": local}

    def test_validate_pass(self, tmp_path, pin_repo_root):
        tf = _make_task_dir(tmp_path)
        r = runner.invoke(app, ["task", "validate", str(tf)])
        assert r.exit_code == 0, r.output
        assert "PASS" in r.output

    def test_validate_fail_missing_dataset_root(self, tmp_path, pin_repo_root):
        tf = _make_task_dir(tmp_path)
        task = json.loads(tf.read_text(encoding="utf-8"))
        del task["dataset_root"]
        tf.write_text(json.dumps(task), encoding="utf-8")
        r = runner.invoke(app, ["task", "validate", str(tf)])
        assert r.exit_code == 1
        assert "FAIL" in r.output

    def test_v3_validate_accepts_explicit_local_config(self, tmp_path, pin_repo_root):
        task = _make_v3_task(tmp_path)
        local = _write_local_config(tmp_path)
        r = runner.invoke(app, ["task", "validate", str(task), "--local-config", str(local)])
        assert r.exit_code == 0, r.output
        assert "PASS" in r.output

    def test_v3_validate_uses_environment_local_config(self, tmp_path, pin_repo_root, monkeypatch):
        task = _make_v3_task(tmp_path)
        local = _write_local_config(tmp_path)
        monkeypatch.setenv("FUTSALMOT_LOCAL_CONFIG", str(local))
        r = runner.invoke(app, ["task", "validate", str(task)])
        assert r.exit_code == 0, r.output

    def test_v3_explicit_local_config_wins_over_environment(self, tmp_path, pin_repo_root, monkeypatch):
        task = _make_v3_task(tmp_path)
        explicit = _write_local_config(tmp_path, "explicit.json")
        environment = _write_local_config(tmp_path, "environment.json")
        monkeypatch.setenv("FUTSALMOT_LOCAL_CONFIG", str(environment))
        r = runner.invoke(app, ["task", "validate", str(task), "--local-config", str(explicit)])
        assert r.exit_code == 0, r.output
        resolved = runner.invoke(app, ["task", "resolve", str(task), "--local-config", str(explicit)])
        assert resolved.exit_code == 0, resolved.output
        runtime = json.loads((pin_repo_root / ".futsalmot" / "runtime" / "ep_cli_v3" / "resolved-task.json").read_text(encoding="utf-8"))
        assert runtime["dataset_root"] == str((tmp_path / "explicit.json-dataset").resolve())
        assert runtime["ue_project_root"] == str((tmp_path / "explicit.json-ue").resolve())
        assert str(tmp_path / "environment.json") not in json.dumps(runtime)

    def test_v3_validate_missing_local_config_fails(self, tmp_path, pin_repo_root, monkeypatch):
        task = _make_v3_task(tmp_path)
        monkeypatch.delenv("FUTSALMOT_LOCAL_CONFIG", raising=False)
        r = runner.invoke(app, ["task", "validate", str(task)])
        assert r.exit_code == 1
        assert "local config" in r.output


class TestTaskResolveCLI:
    def test_resolve_writes_runtime(self, tmp_path, pin_repo_root):
        tf = _make_task_dir(tmp_path)
        r = runner.invoke(app, ["task", "resolve", str(tf)])
        assert r.exit_code == 0, r.output
        assert "Task ID: cli_t1" in r.output
        runtime = pin_repo_root / ".futsalmot" / "runtime" / "cli_t1" / "resolved-task.json"
        assert runtime.is_file()

    def test_ue_command_prints_run_task(self, tmp_path, pin_repo_root):
        tf = _make_task_dir(tmp_path)
        r = runner.invoke(app, ["task", "ue-command", str(tf)])
        assert r.exit_code == 0, r.output
        assert "run_task.py" in r.output
        assert "--resolved-task" in r.output

    def test_v3_resolve_prints_summary_and_uses_local_config(self, tmp_path, pin_repo_root):
        task = _make_v3_task(tmp_path)
        local = _write_local_config(tmp_path)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r = runner.invoke(app, ["task", "resolve", str(task), "--local-config", str(local)])
        assert r.exit_code == 0, r.output
        assert not any(item.category is DeprecationWarning for item in caught)
        for expected in (
            "Episode: ep_cli_v3", "Seed/steps: 17/3", "FPS: 30",
            "Resolution: 1280x720", "C01 -> Camera_A", "C02 -> Camera_B",
            "Expected frames: 9", "Annotations: mot, pose, mots",
            "Classes: player, ball", "Public sequence names:",
            "FutsalMOT_ep_cli_v3_C01", "FutsalMOT_ep_cli_v3_C02",
        ):
            assert expected in r.output
        runtime = pin_repo_root / ".futsalmot" / "runtime" / "ep_cli_v3" / "resolved-task.json"
        assert runtime.is_file()

    def test_v3_resolve_missing_local_config_does_not_create_runtime(self, tmp_path, pin_repo_root, monkeypatch):
        task = _make_v3_task(tmp_path)
        monkeypatch.delenv("FUTSALMOT_LOCAL_CONFIG", raising=False)
        r = runner.invoke(app, ["task", "resolve", str(task)])
        assert r.exit_code != 0
        assert not (pin_repo_root / ".futsalmot" / "runtime" / "ep_cli_v3" / "resolved-task.json").exists()

    def test_v2_resolve_warns_but_still_works_without_local_config(self, tmp_path, pin_repo_root):
        task = _make_task_dir(tmp_path)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r = runner.invoke(app, ["task", "resolve", str(task)])
        assert r.exit_code == 0, r.output
        assert "deprecated" in r.output.lower()
        assert any(item.category is DeprecationWarning for item in caught)
        for expected in (
            "Trajectory output:", "Dataset output:", "Export: scenario=5_vs_5 steps=1 seed=42",
            "UE: 1 cameras", "Postprocess formats:",
            "Episode: episode_cli_t1", "Seed/steps: 42/1", "FPS: 30",
            "Resolution: 64x64", "CineCam_01", "Expected frames: 1",
            "Annotations:", "Classes: player, ball", "Public sequence names: LS_CineCam_01",
        ):
            assert expected in r.output


class TestTaskStatusAudit:
    def test_status_readonly(self, tmp_path, pin_repo_root):
        tf = _make_task_dir(tmp_path)
        _make_minimal_dataset(tmp_path / "ds", "episode_cli_t1")
        r = runner.invoke(app, ["task", "status", str(tf)])
        assert r.exit_code == 0, r.output
        assert "episode_cli_t1" in r.output

    def test_audit_passes_minimal(self, tmp_path, pin_repo_root):
        tf = _make_task_dir(tmp_path)
        _make_minimal_dataset(tmp_path / "ds", "episode_cli_t1")
        r = runner.invoke(app, ["task", "audit", str(tf), "--validation-level", "none"])
        assert r.exit_code == 0, r.output

    def test_postprocess_skip_all_noop(self, tmp_path, pin_repo_root):
        tf = _make_task_dir(tmp_path)
        r = runner.invoke(app, ["task", "postprocess", str(tf),
                                "--skip-cryptomatte", "--skip-annotate", "--skip-validate"])
        assert r.exit_code == 0, r.output

    def test_postprocess_defaults_to_public_writer(self, tmp_path, pin_repo_root, monkeypatch):
        from grf_ue_bridge.config.models import ResolvedTask
        from grf_ue_bridge.workflows.task_postprocess import run_postprocess
        dataset = tmp_path / "public_episode"
        cam = dataset / "Cam_01"
        (cam / "img1").mkdir(parents=True)
        (cam / "gt").mkdir()
        Image.new("RGB", (4, 4), "black").save(cam / "img1" / "000001.png")
        (cam / "camera.json").write_text(json.dumps({
            "camera_id": "Cam_01", "image_width": 4, "image_height": 4,
            "intrinsics": {"width": 4, "height": 4, "fx": 2, "fy": 2, "cx": 2, "cy": 2},
            "extrinsics": {"world_location_m": [0, 0, 0], "forward": [1, 0, 0], "right": [0, 1, 0], "up": [0, 0, 1]},
        }), encoding="utf-8")
        (cam / "annotations.jsonl").write_text(json.dumps({"episode_id": "ep", "frame_index": 1, "objects": [{"entity_id": "L0"}]}) + "\n", encoding="utf-8")
        (cam / "pose_keypoints.jsonl").write_text(json.dumps({"kind": "frame", "frame_index": 1, "objects": [{"entity_id": "L0", "keypoints_world": [[4, 0, 1]] * 17}]}) + "\n", encoding="utf-8")
        mapping = {f"L{i}": f"Player_L{i}" for i in range(5)} | {f"R{i}": f"Player_R{i}" for i in range(5)} | {"BALL": "Ball_01"}
        mapping_path = pin_repo_root / "ue"
        mapping_path.mkdir(exist_ok=True)
        (mapping_path / "actor_mapping.example.json").write_text(json.dumps(mapping), encoding="utf-8")
        monkeypatch.setattr("grf_ue_bridge.public_episode._load_mask_for_frame", lambda *_: np.pad(np.ones((1, 1), dtype="uint8"), ((0, 3), (0, 3))))
        resolved = ResolvedTask(task_id="t", episode_name="ep", source_task_file="task.json", repo_root=str(pin_repo_root), ue_project_root=str(pin_repo_root), dataset_root=str(tmp_path), trajectory_output=str(tmp_path), dataset_episode_dir=str(dataset), actor_mapping=str(mapping_path / "actor_mapping.example.json"), ue_profile={"sequences": [{"name": "LS_Cam_01", "camera_actor": "Cam_01"}]}, postprocess={"public_output": True})
        assert run_postprocess(resolved, skip_validate=False) == 0
        public_cam = dataset / "FutsalMOT_ep_C01"
        assert (public_cam / "img1" / "000001.jpg").exists()
        assert (public_cam / "gt" / "gt.txt").exists()
        assert (public_cam / "gt" / "gt_pose.json").exists()
        assert (public_cam / "gt" / "gt_mots.txt").exists()
        assert (dataset / "episode_manifest.json").exists()
        assert not (cam / "mask").exists()
        assert not (cam / "labels").exists()
        assert not (dataset / "debug").exists()

    def test_v3_postprocess_forwards_selected_output_to_public_writer(self, tmp_path, monkeypatch):
        from grf_ue_bridge.config.models import ResolvedTask
        from grf_ue_bridge.workflows import task_postprocess

        dataset = tmp_path / "public_episode"
        dataset.mkdir()
        mapping_path = tmp_path / "mapping.json"
        mapping_path.write_text("{}", encoding="utf-8")
        resolved = ResolvedTask(
            task_id="t", episode_name="ep", source_task_file="task.json",
            repo_root=str(tmp_path), ue_project_root=str(tmp_path), dataset_root=str(tmp_path),
            trajectory_output=str(tmp_path), dataset_episode_dir=str(dataset),
            actor_mapping=str(mapping_path), ue_profile={"sequences": []},
            postprocess={"public_output": True},
            config_v3={"annotations": ["mot"], "classes": ["player"]},
        )
        seen = {}

        def fake_writer(*args, **kwargs):
            seen.update(kwargs)
            return {"sequences": []}

        monkeypatch.setattr("grf_ue_bridge.public_episode.write_public_episode", fake_writer)
        assert task_postprocess.run_postprocess(resolved, skip_validate=True) == 0
        assert seen["annotations"] == ["mot"]
        assert seen["classes"] == ["player"]

    def test_postprocess_public_output_false_keeps_legacy_skip_behavior(self, tmp_path):
        from grf_ue_bridge.config.models import ResolvedTask
        from grf_ue_bridge.workflows.task_postprocess import run_postprocess
        resolved = ResolvedTask(task_id="t", episode_name="ep", source_task_file="task.json", repo_root=str(tmp_path), ue_project_root=str(tmp_path), dataset_root=str(tmp_path), trajectory_output=str(tmp_path), dataset_episode_dir=str(tmp_path / "ep"), postprocess={"public_output": False})
        assert run_postprocess(resolved, skip_cryptomatte=True, skip_annotate=True, skip_validate=True) == 0
        assert not (tmp_path / "ep" / "episode_manifest.json").exists()


class TestActiveTask:
    def test_active_cycle(self, tmp_path, pin_repo_root):
        tf = _make_task_dir(tmp_path)
        # 激活写 pinned 仓库 .futsalmot（gitignore），测试后清理
        from grf_ue_bridge.config import resolver
        act_path = resolver.save_active_task(tf, pin_repo_root)
        try:
            r = runner.invoke(app, ["task", "status"])  # 无参数 → active
            assert r.exit_code == 0, r.output
            assert "Active task:" in r.output
            # 显式 task 优先于 active
            r2 = runner.invoke(app, ["task", "status", str(tf)])
            assert r2.exit_code == 0
        finally:
            resolver.clear_active_task(pin_repo_root)
        # deactivate 后无 active → 报错
        runner.invoke(app, ["task", "deactivate"])
        r3 = runner.invoke(app, ["task", "status"])
        assert r3.exit_code == 2

    def test_activate_deactivate_cli(self, tmp_path, pin_repo_root):
        tf = _make_task_dir(tmp_path)
        from grf_ue_bridge.config import resolver
        try:
            r = runner.invoke(app, ["task", "activate", str(tf)])
            assert r.exit_code == 0
            assert resolver.load_active_task(pin_repo_root) is not None
            r2 = runner.invoke(app, ["task", "deactivate"])
            assert r2.exit_code == 0
            assert resolver.load_active_task(pin_repo_root) is None
        finally:
            resolver.clear_active_task(pin_repo_root)
