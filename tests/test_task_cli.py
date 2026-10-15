"""grf-ue task CLI 测试（用 CliRunner，避免真实 UE/GRF）。"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest
from typer.testing import CliRunner

from grf_ue_bridge.cli import app
from grf_ue_bridge.appearance import APPEARANCE_POLICY, build_appearance_manifest
from grf_ue_bridge.config.loader import load_task_config
from grf_ue_bridge.seeds import derive_episode_seeds

runner = CliRunner()


def _valid_png() -> bytes:
    import io
    buf = io.BytesIO()
    Image.new("L", (8, 8), 3).save(buf, format="PNG")
    return buf.getvalue()


def _appearance_catalog() -> dict:
    return {
        "mode": "seeded",
        "policy": APPEARANCE_POLICY,
        "catalog_id": "mpfb_players_003_009_v1",
        "models": [
            {
                "player_id": player_id,
                "mesh": f"/Game/Players/Player_{player_id:03d}",
                "anim_class": "/Game/Core/ABP_RuntimeRetarget",
                "skeleton": "/Game/Core/SKEL_MPFB_Base",
            }
            for player_id in range(3, 10)
        ],
        "skins": [
            {
                "skin_id": f"skin-{index:02d}",
                "texture": f"/Game/Textures/T_Skin_{index:02d}",
            }
            for index in range(1, 10)
        ],
        "kits": [
            {
                "set_id": f"{index:02d}",
                "jersey_material": f"/Game/Materials/M_Jersey_Set_{index:02d}",
                "shorts_material": f"/Game/Materials/M_Shorts_Set_{index:02d}",
            }
            for index in range(1, 6)
        ],
        "body_texture_parameter": "SkinTexture",
    }


def _appearance_meta(episode_id: str) -> dict:
    return {
        "schema": "grf_ue_episode",
        "version": 1,
        "episode_id": episode_id,
        "source": {"seed": 42},
        "randomness": derive_episode_seeds(42).model_dump(),
        "timing": {
            "source_step_seconds": 0.1,
            "playback_fps": 10,
            "num_steps": 1,
        },
        "field": {"length_m": 40.0, "width_m": 20.0},
        "entities": [
            {
                "id": entity_id,
                "team": "left" if entity_id.startswith("L") else "right",
                "is_goalkeeper": entity_id in {"L2", "R1"},
            }
            for entity_id in (
                "L0", "L1", "L2", "L3", "L4",
                "R0", "R1", "R2", "R3", "R4",
            )
        ] + [{"id": "BALL", "team": None, "is_goalkeeper": False}],
    }


def _appearance_frame() -> dict:
    return {
        "step": 0,
        "time_seconds": 0.0,
        "score": [0, 0],
        "ball": {"position_m": [0.0, 0.0, 0.11]},
        "players": [
            {"id": entity_id, "position_m": [0.0, 0.0, 0.0]}
            for entity_id in (
                "L0", "L1", "L2", "L3", "L4",
                "R0", "R1", "R2", "R3", "R4",
            )
        ],
        "ball_owned_team": -1,
        "ball_owned_player": -1,
        "game_mode": 0,
    }


def _make_task_dir(
    tmp_path: Path,
    cam_count: int = 1,
    frames: int = 1,
    *,
    appearance: bool = False,
) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "FutsalMOT.uproject").write_text("{}", encoding="utf-8")
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
    if appearance:
        task["ue"]["appearance"] = _appearance_catalog()
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
        (cam / "gt" / "gt.txt").write_text("1,1,1,1,1,1,1,1,1\n", encoding="utf-8")
    return ep_dir


class TestTaskValidateCLI:
    def test_camera_anchor_fixture_loads_with_five_canonical_ids(self, repo_root):
        fixture = repo_root / "configs" / "camera_anchor_smoke_5cam.json"
        raw = json.loads(fixture.read_text(encoding="utf-8"))
        task = load_task_config(fixture)

        assert {"C1", "C2", "C3", "C4", "C5"} <= set(task.ue.camera_mapping)
        assert "P01" in task.ue.camera_mapping
        assert task.simulation.camera.distribution.anchors == [
            "C1", "C2", "C3", "C4", "C5"
        ]
        assert task.simulation.camera.distribution.episode_profile == "anchor_only"
        assert {"C1", "C2", "C3", "C4", "C5"} <= set(task.simulation.camera.profiles)
        assert "P01" in task.simulation.camera.profiles
        expected_profile_fields = {
            "type", "coverage", "resolution", "lens", "position_m",
            "rotation_deg", "height_m",
        }
        assert {"C1", "C2", "C3", "C4", "C5", "P01"} == set(
            raw["simulation"]["camera"]["profiles"]
        )
        for profile in raw["simulation"]["camera"]["profiles"].values():
            assert expected_profile_fields <= set(profile)
        for camera_id, profile in task.simulation.camera.profiles.items():
            assert profile.type == "static_surveillance"
            assert profile.coverage == ("partial_field" if camera_id == "P01" else "full_field")
            assert len(profile.resolution) == 2
            assert profile.lens.focal_length_mm or profile.lens.horizontal_fov_deg
            assert len(profile.position_m) == 3
            assert len(profile.rotation_deg) == 3
            assert profile.height_m > 0
        assert set(task.ue.camera_mapping) == set(task.simulation.camera.profiles)
        assert {"C1", "C2", "C3", "C4", "C5", "P01"} == set(
            raw["ue"]["camera_mapping"]
        )
        assert all(
            {"actor", "sequence"} <= set(mapping)
            for mapping in raw["ue"]["camera_mapping"].values()
        )
        for mapping in task.ue.camera_mapping.values():
            assert mapping.actor
            assert mapping.sequence

        assert raw["ue"]["camera_mapping"]["C5"] == {
            "actor": "CineCam_Main",
            "sequence": "LS_Cam_Main",
        }
        assert raw["ue"]["camera_mapping"]["P01"] == {
            "actor": "CineCam_P01",
            "sequence": "LS_Cam_P01",
        }
        assert raw["simulation"]["camera"]["profiles"]["C5"] == {
            "type": "static_surveillance",
            "coverage": "full_field",
            "resolution": [1920, 1080],
            "lens": {
                "focal_length_mm": 10.0,
                "horizontal_fov_deg": 99.82194519042969,
            },
            "position_m": [0.0, 22.0, 13.0],
            "rotation_deg": [-35.0, -90.0, 0.0],
            "height_m": 13.0,
            "distortion": None,
        }

        distribution = raw["simulation"]["camera"]["distribution"]
        assert "partial" not in distribution
        assert "dynamic_broadcast_ratio" not in distribution
        assert "broadcast" not in distribution
        assert "auto_create" not in raw["ue"]

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
        state_path = pin_repo_root / ".futsalmot" / "runtime" / "cli_t1" / "pipeline_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        assert state["steps"]["ue_sequence"]["status"] == "COMPLETED"
        assert state["steps"]["ue_sequence"]["completion_type"] == "command_generation"
        assert state["steps"]["render"]["status"] == "PENDING"


class TestTaskStatusAudit:
    def test_status_readonly(self, tmp_path, pin_repo_root):
        tf = _make_task_dir(tmp_path)
        _make_minimal_dataset(tmp_path / "ds", "episode_cli_t1")
        r = runner.invoke(app, ["task", "status", str(tf)])
        assert r.exit_code == 0, r.output
        assert "episode_cli_t1" in r.output
        assert "Pipeline State:" in r.output
        assert "export:" in r.output
        assert "Run Manifest:" in r.output

    def test_status_displays_audit_metrics(self, tmp_path, pin_repo_root):
        tf = _make_task_dir(tmp_path)
        episode = _make_minimal_dataset(tmp_path / "ds", "episode_cli_t1")
        audit = episode / "audit"
        audit.mkdir()
        (audit / "soak_audit_report.json").write_text(json.dumps({
            "passed": True,
            "exit_code": 0,
            "errors": [],
            "warnings": [],
            "checks": {},
            "metrics": {"dataset": {"image_count": 1}},
        }), encoding="utf-8")

        r = runner.invoke(app, ["task", "status", str(tf)])

        assert r.exit_code == 0, r.output
        assert "Audit Metrics:" in r.output
        assert "image_count" in r.output
        assert "Schema version:" in r.output
        assert "Current step:" in r.output

    def test_audit_passes_minimal(self, tmp_path, pin_repo_root):
        tf = _make_task_dir(tmp_path)
        _make_minimal_dataset(tmp_path / "ds", "episode_cli_t1")
        r = runner.invoke(app, ["task", "audit", str(tf), "--validation-level", "none"])
        assert r.exit_code == 0, r.output
        state_path = pin_repo_root / ".futsalmot" / "runtime" / "cli_t1" / "pipeline_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        assert state["steps"]["audit"]["status"] == "COMPLETED"
        run_manifest = pin_repo_root / ".futsalmot" / "runtime" / "cli_t1" / "run_manifest.json"
        manifest = json.loads(run_manifest.read_text(encoding="utf-8"))
        assert manifest["validation"]["passed"] is True
        assert manifest["validation"]["audit_report"].endswith("soak_audit_report.json")
        assert not (tmp_path / "ds" / "episode_cli_t1" / "dataset_manifest.json").exists()

    def test_cleanup_manifest_keeps_pre_delete_artifact_counts(self, tmp_path, pin_repo_root):
        tf = _make_task_dir(tmp_path)
        episode = _make_minimal_dataset(tmp_path / "ds", "episode_cli_t1")
        audit = episode / "audit"
        audit.mkdir()
        (audit / "soak_audit_report.json").write_text(json.dumps({
            "passed": True,
            "exit_code": 0,
            "errors": [],
            "warnings": [],
            "checks": {},
        }), encoding="utf-8")

        r = runner.invoke(app, ["task", "cleanup", str(tf), "--apply"])

        assert r.exit_code == 0, r.output
        manifest_path = pin_repo_root / ".futsalmot" / "runtime" / "cli_t1" / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["artifacts"]["images"]["count"] == 1
        assert manifest["runtime"]["finished_at"]

    def test_postprocess_skip_all_noop(self, tmp_path, pin_repo_root):
        tf = _make_task_dir(tmp_path)
        r = runner.invoke(app, ["task", "postprocess", str(tf),
                                "--skip-cryptomatte", "--skip-annotate", "--skip-validate"])
        assert r.exit_code == 0, r.output
        state_path = pin_repo_root / ".futsalmot" / "runtime" / "cli_t1" / "pipeline_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        assert state["steps"]["postprocess"]["status"] == "COMPLETED"

    def test_resume_dry_run_does_not_execute_workflow(self, tmp_path, pin_repo_root, monkeypatch):
        tf = _make_task_dir(tmp_path)
        from grf_ue_bridge import pipeline_state

        path = pipeline_state.state_path(pin_repo_root, "cli_t1")
        state = pipeline_state.create_state("cli_t1")
        pipeline_state.update_step(state, "export", pipeline_state.StepStatus.COMPLETED)
        pipeline_state.update_step(state, "render", pipeline_state.StepStatus.FAILED,
                                   error_message="render_summary missing")
        pipeline_state.save_state(state, path)
        called = []
        monkeypatch.setattr("grf_ue_bridge.workflows.task_export.run_export",
                            lambda *args, **kwargs: called.append("export") or 0)

        r = runner.invoke(app, ["task", "resume", str(tf), "--dry-run"])

        assert r.exit_code == 0, r.output
        assert "export: skip" in r.output
        assert "render: retry" in r.output
        assert called == []

    def test_resume_dry_run_does_not_create_state(self, tmp_path, pin_repo_root):
        tf = _make_task_dir(tmp_path)
        state_path = pin_repo_root / ".futsalmot" / "runtime" / "cli_t1" / "pipeline_state.json"

        r = runner.invoke(app, ["task", "resume", str(tf), "--dry-run"])

        assert r.exit_code == 0, r.output
        assert not state_path.exists()

    @pytest.mark.parametrize(
        "invalid_artifact",
        [
            "missing_manifest", "bad_schema", "changed_frames", "changed_meta",
            "bad_meta", "catalog", "entity_binding", "fingerprint",
            "invalid_hashed_frames",
        ],
    )
    def test_resume_retries_completed_appearance_export_with_invalid_manifest_inputs(
        self, tmp_path, pin_repo_root, invalid_artifact
    ):
        from grf_ue_bridge import pipeline_state

        tf = _make_task_dir(tmp_path, appearance=True)
        from grf_ue_bridge.config import resolver

        resolved = resolver.resolve_task(tf)
        episode = Path(resolved.trajectory_output)
        episode.mkdir(parents=True)
        meta = _appearance_meta(episode.name)
        meta_path = episode / "meta.json"
        frames_path = episode / "frames.jsonl"
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        frames_path.write_text(json.dumps(_appearance_frame()) + "\n", encoding="utf-8")
        manifest = build_appearance_manifest(
            meta, frames_path, resolved.ue_profile["appearance"]
        )
        provenance = episode / "provenance"
        provenance.mkdir()
        manifest_path = provenance / "appearance.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        if invalid_artifact == "missing_manifest":
            manifest_path.unlink()
        elif invalid_artifact == "bad_schema":
            manifest["schema"] = "wrong_schema"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        elif invalid_artifact == "changed_frames":
            frames_path.write_bytes(b'{"step":0,"changed":true}\n')
        elif invalid_artifact == "changed_meta":
            meta["randomness"]["ue_visual_seed"] += 1
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
        elif invalid_artifact == "bad_meta":
            meta_path.write_text("not json", encoding="utf-8")
        elif invalid_artifact == "catalog":
            task_data = json.loads(tf.read_text(encoding="utf-8"))
            task_data["ue"]["appearance"]["models"][0]["mesh"] = "/Game/Players/Changed"
            tf.write_text(json.dumps(task_data), encoding="utf-8")
        elif invalid_artifact == "entity_binding":
            meta["entities"][2]["is_goalkeeper"] = False
            meta["entities"][3]["is_goalkeeper"] = True
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
        elif invalid_artifact == "fingerprint":
            manifest["fingerprint"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        elif invalid_artifact == "invalid_hashed_frames":
            frames_path.write_text("not json\n", encoding="utf-8")
            manifest = build_appearance_manifest(
                meta, frames_path, resolved.ue_profile["appearance"]
            )
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        state_path = pipeline_state.state_path(pin_repo_root, "cli_t1")
        state = pipeline_state.create_state("cli_t1")
        pipeline_state.update_step(state, "export", pipeline_state.StepStatus.COMPLETED)
        pipeline_state.save_state(state, state_path)

        result = runner.invoke(app, [
            "task", "resume", str(tf), "--dry-run",
            "--dataset-root", str(tmp_path / "ds"),
            "--ue-project-root", str(tf.parent),
        ])

        assert result.exit_code == 0, result.output
        assert "export: retry" in result.output

    def test_resume_accepts_valid_appearance_manifest(self, tmp_path, pin_repo_root):
        from grf_ue_bridge import pipeline_state
        from grf_ue_bridge.config import resolver

        tf = _make_task_dir(tmp_path, appearance=True)
        resolved = resolver.resolve_task(tf)
        episode = Path(resolved.trajectory_output)
        episode.mkdir(parents=True)
        meta = _appearance_meta(episode.name)
        meta_path = episode / "meta.json"
        frames_path = episode / "frames.jsonl"
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        frames_path.write_text(json.dumps(_appearance_frame()) + "\n", encoding="utf-8")
        manifest = build_appearance_manifest(
            meta, frames_path, resolved.ue_profile["appearance"]
        )
        provenance = episode / "provenance"
        provenance.mkdir()
        (provenance / "appearance.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

        state_path = pipeline_state.state_path(pin_repo_root, "cli_t1")
        state = pipeline_state.create_state("cli_t1")
        pipeline_state.update_step(state, "export", pipeline_state.StepStatus.COMPLETED)
        pipeline_state.save_state(state, state_path)

        result = runner.invoke(app, [
            "task", "resume", str(tf), "--dry-run",
            "--dataset-root", str(tmp_path / "ds"),
            "--ue-project-root", str(tf.parent),
        ])

        assert result.exit_code == 0, result.output
        assert "export: skip" in result.output

    def test_running_state_blocks_new_postprocess_without_force(self, tmp_path, pin_repo_root):
        tf = _make_task_dir(tmp_path)
        from grf_ue_bridge import pipeline_state

        path = pipeline_state.state_path(pin_repo_root, "cli_t1")
        state = pipeline_state.create_state("cli_t1")
        pipeline_state.update_step(state, "postprocess", pipeline_state.StepStatus.RUNNING)
        pipeline_state.save_state(state, path)

        r = runner.invoke(app, ["task", "postprocess", str(tf), "--skip-cryptomatte",
                                "--skip-annotate", "--skip-validate"])

        assert r.exit_code != 0
        assert "Existing running pipeline state detected" in r.output


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
