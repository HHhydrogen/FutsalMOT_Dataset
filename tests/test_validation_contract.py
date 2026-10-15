import hashlib
import json
import shutil
from pathlib import Path
from typing import Optional

import pytest

from grf_ue_bridge import annotation_validator as annotation_validator_module
from grf_ue_bridge.annotation_validator import validate_annotation_result
from grf_ue_bridge.appearance import build_appearance_manifest
from grf_ue_bridge.dataset_regression import collect_dataset_regression_errors
from grf_ue_bridge.task_requirements import TaskRequirements, resolve_task_requirements
from grf_ue_bridge.validation_result import (
    CheckStatus,
    ValidationResult,
    validation_result_from_report,
)
from grf_ue_bridge.workflows.task_audit import main as audit_main
from grf_ue_bridge.workflows.artifact_cleanup import (
    _validation_gate,
    apply_cleanup,
    plan_cleanup,
)


def _annotation_object():
    return {
        "entity_id": "L0",
        "track_id": 1,
        "class": "player",
        "team": "left",
        "role": None,
        "is_goalkeeper": False,
        "world_position": [0.0, 0.0, 0.0],
        "in_frame": True,
        "truncated": False,
        "visibility": None,
        "raw_bbox_xywh": [1.0, 2.0, 10.0, 20.0],
        "raw_bbox_xyxy": [1.0, 2.0, 11.0, 22.0],
        "bbox_xywh": [1.0, 2.0, 10.0, 20.0],
        "bbox_xyxy": [1.0, 2.0, 11.0, 22.0],
    }


def make_annotation_fixture(tmp_path: Path, mot: bool) -> Path:
    """构造不依赖 mask/RGB 的最小真实 camera 标注目录。"""
    root = tmp_path / "annotations"
    camera = root / "Camera_01"
    (camera / "gt").mkdir(parents=True)
    (camera / "camera.json").write_text(
        json.dumps(
            {
                "camera_id": "Camera_01",
                "image_width": 64,
                "image_height": 64,
                "intrinsics": {
                    "width": 64,
                    "height": 64,
                    "fx": 50.0,
                    "fy": 50.0,
                    "cx": 32.0,
                    "cy": 32.0,
                },
                "extrinsics": {
                    "world_location_m": [0.0, 0.0, 0.0],
                    "forward": [1.0, 0.0, 0.0],
                    "right": [0.0, 1.0, 0.0],
                    "up": [0.0, 0.0, 1.0],
                },
            }
        ),
        encoding="utf-8",
    )
    (camera / "annotations.jsonl").write_text(
        json.dumps(
            {
                "episode_id": "episode_contract",
                "camera_id": "Camera_01",
                "frame_index": 1,
                "source_step": 0,
                "time_seconds": 0.0,
                "objects": [_annotation_object()],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (camera / "seqinfo.ini").write_text("[Sequence]\nname=Camera_01\n", encoding="utf-8")
    if mot:
        (camera / "gt" / "gt.txt").write_text(
            "1,1,1,2,10,20,1,1,1.00\n", encoding="utf-8"
        )
    return root


def _write_audit_camera_fixture(tmp_path: Path) -> Path:
    """构造不含可选产物的最小真实 audit camera 目录。"""
    root = tmp_path / "audit_episode"
    camera = root / "Camera_01"
    (camera / "img1").mkdir(parents=True)
    (camera / "img1" / "000001.png").write_bytes(b"png")
    (camera / "camera.json").write_text(
        json.dumps(
            {
                "camera_id": "Camera_01",
                "image_width": 64,
                "image_height": 64,
                "intrinsics": {
                    "width": 64,
                    "height": 64,
                    "fx": 50.0,
                    "fy": 50.0,
                    "cx": 32.0,
                    "cy": 32.0,
                },
                "extrinsics": {
                    "world_location_m": [0.0, 0.0, 1.0],
                    "forward": [1.0, 0.0, 0.0],
                    "right": [0.0, 1.0, 0.0],
                    "up": [0.0, 0.0, 1.0],
                },
            }
        ),
        encoding="utf-8",
    )
    (camera / "annotations.jsonl").write_text(
        json.dumps(
            {
                "episode_id": "audit_episode",
                "camera_id": "Camera_01",
                "frame_index": 1,
                "source_step": 0,
                "time_seconds": 0.0,
                "objects": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def _appearance_catalog() -> dict:
    """构造审计所需的最小完整外观 catalog。"""
    return {
        "mode": "seeded",
        "policy": "futsalmot_appearance_v1",
        "catalog_id": "audit-appearance-v1",
        "models": [
            {
                "player_id": player_id,
                "mesh": f"/Game/Appearance/Mesh_{player_id}",
                "anim_class": "/Game/Appearance/Anim",
                "skeleton": "/Game/Appearance/Skeleton",
            }
            for player_id in range(3, 10)
        ],
        "skins": [
            {
                "skin_id": f"skin-{index}",
                "texture": f"/Game/Appearance/Skin_{index}",
            }
            for index in range(1, 10)
        ],
        "kits": [
            {
                "set_id": f"{index:02d}",
                "jersey_material": f"/Game/Appearance/Jersey_{index}",
                "shorts_material": f"/Game/Appearance/Shorts_{index}",
            }
            for index in range(1, 6)
        ],
        "body_texture_parameter": "SkinTexture",
    }


def _write_appearance_fixture(
    root: Path,
    *,
    report_overrides: Optional[dict] = None,
    change_frames_after_manifest: bool = False,
    render_required: bool = False,
) -> dict:
    """为 appearance audit 构造 resolved profile、manifest 和 application report。"""
    meta = {
        "schema": "grf_ue_episode",
        "version": 1,
        "episode_id": root.name,
        "source": {"seed": 42},
        "randomness": {
            "policy": "futsalmot_seed_v1",
            "root_seed": 42,
            "ue_visual_seed": 1001,
        },
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
        ],
    }
    frames_path = root / "frames.jsonl"
    frames_path.write_bytes(b'{"step":0}\n')
    (root / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    manifest = build_appearance_manifest(meta, frames_path, _appearance_catalog())
    if change_frames_after_manifest:
        frames_path.write_bytes(b'{"step":1}\n')
    provenance = root / "provenance"
    provenance.mkdir(exist_ok=True)
    (provenance / "ue-profile.json").write_text(
        json.dumps({"appearance": _appearance_catalog()}), encoding="utf-8"
    )
    (provenance / "appearance.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    report = {
        "schema": "futsalmot_appearance_application",
        "version": 1,
        "manifest_fingerprint": manifest["fingerprint"],
        "catalog_fingerprint": manifest["catalog_fingerprint"],
        "trajectory_sha256": hashlib.sha256(
            b'{"step":0}\n'
        ).hexdigest(),
        "state": "restored",
        "errors": [],
        "warnings": [],
        "actors": [
            {
                "entity_id": entity_id,
                "apply_status": "applied",
                "restore_status": "restored",
                "assets": {
                    field: next(
                        item[field]
                        for item in manifest["entities"]
                        if item["entity_id"] == entity_id
                    )
                    for field in (
                        "mesh", "anim_class", "skeleton", "skin_texture",
                        "jersey_material", "shorts_material",
                    )
                },
                "slots": {
                    "body_texture_parameter": manifest["body_texture_parameter"],
                    "Body": next(
                        item["skin_texture"]
                        for item in manifest["entities"]
                        if item["entity_id"] == entity_id
                    ),
                    "Jersey": next(
                        item["jersey_material"]
                        for item in manifest["entities"]
                        if item["entity_id"] == entity_id
                    ),
                    "Shorts": next(
                        item["shorts_material"]
                        for item in manifest["entities"]
                        if item["entity_id"] == entity_id
                    ),
                },
            }
            for entity_id in (
                "L0", "L1", "L2", "L3", "L4",
                "R0", "R1", "R2", "R3", "R4",
            )
        ],
        "mrq_jobs": ([{"camera_id": "Camera_01", "status": "success"}]
                     if render_required else []),
        "mrq_pending": False,
        **({"render_result":
            {
                "status": "success",
                "jobs": [{"camera_id": "Camera_01", "status": "success"}],
                "cameras": {"Camera_01": {"ok": True}},
            }
        } if render_required else {}),
    }
    if report_overrides:
        report.update(report_overrides)
    (provenance / "appearance_application.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    if render_required:
        (root / "render_summary.json").write_text(
            json.dumps({"status": "success", "cameras": {"Camera_01": {"ok": True}}}),
            encoding="utf-8",
        )
    return {"meta": meta, "manifest": manifest, "report": report}


def run_audit_fixture(
    tmp_path: Path,
    *,
    require_render: bool,
    require_mask: bool,
    require_mot: bool,
    require_pose: bool,
    validation_level: str,
    require_yolo_det: bool = False,
    require_yolo_seg: bool = False,
) -> dict:
    root = _write_audit_camera_fixture(tmp_path)
    return run_audit(
        root,
        require_render=require_render,
        require_mask=require_mask,
        require_mot=require_mot,
        require_pose=require_pose,
        validation_level=validation_level,
        require_yolo_det=require_yolo_det,
        require_yolo_seg=require_yolo_seg,
    )


def run_audit(
    root: Path,
    *,
    require_render: bool,
    require_mask: bool,
    require_mot: bool,
    require_pose: bool,
    validation_level: str,
    require_yolo_det: bool = False,
    require_yolo_seg: bool = False,
    episode: Optional[Path] = None,
    appearance_catalog=None,
) -> dict:
    args = [
        "--input",
        str(root),
        "--expected-cameras",
        "1",
        "--expected-frames-per-camera",
        "1",
        "--validation-level",
        validation_level,
        *( ["--episode", str(episode)] if episode is not None else [] ),
        "--render-required",
        str(require_render).lower(),
        "--mask-enabled",
        str(require_mask).lower(),
        "--mot-required",
        str(require_mot).lower(),
        "--yolo-det-required",
        str(require_yolo_det).lower(),
        "--yolo-seg-required",
        str(require_yolo_seg).lower(),
        "--pose-required",
        str(require_pose).lower(),
    ]
    rc = audit_main(
        args,
        appearance_catalog=appearance_catalog,
        episode_dir=episode or root,
    )
    assert rc in (0, 1)
    return json.loads(
        (root / "audit" / "soak_audit_report.json").read_text(encoding="utf-8")
    )


def test_audit_legacy_task_skips_appearance_even_with_residual_files(tmp_path):
    root = _write_audit_camera_fixture(tmp_path)
    provenance = root / "provenance"
    provenance.mkdir()
    (provenance / "appearance.json").write_text("{}", encoding="utf-8")
    (provenance / "appearance_application.json").write_text("{}", encoding="utf-8")
    (provenance / "ue-profile.json").write_text(
        json.dumps({"appearance": _appearance_catalog()}), encoding="utf-8"
    )

    report = run_audit(
        root,
        require_render=False,
        require_mask=False,
        require_mot=False,
        require_pose=False,
        validation_level="none",
        appearance_catalog=None,
    )

    assert report["passed"] is True
    assert report["checks"]["appearance"]["status"] == "skipped"


def test_audit_standalone_ignores_residual_enabled_profile_without_task_config(tmp_path):
    root = _write_audit_camera_fixture(tmp_path)
    provenance = root / "provenance"
    provenance.mkdir()
    (provenance / "ue-profile.json").write_text(
        json.dumps({"appearance": _appearance_catalog()}), encoding="utf-8"
    )
    (provenance / "appearance.json").write_text("{}", encoding="utf-8")
    (provenance / "appearance_application.json").write_text("{}", encoding="utf-8")

    report = run_audit(
        root,
        require_render=False,
        require_mask=False,
        require_mot=False,
        require_pose=False,
        validation_level="none",
    )

    assert report["passed"] is True
    assert report["checks"]["appearance"]["status"] == "skipped"


@pytest.mark.parametrize("appearance_enabled", [False, True])
def test_task_audit_cli_passes_current_resolved_appearance_config(
    monkeypatch, tmp_path, appearance_enabled
):
    from types import SimpleNamespace

    from grf_ue_bridge import cli as cli_module

    root = tmp_path / "episode"
    root.mkdir()
    task_file = tmp_path / "task.json"
    catalog = _appearance_catalog() if appearance_enabled else None
    resolved = SimpleNamespace(
        audit={"expected_cameras": 1, "expected_frames_per_camera": 1},
        artifact_policy={"profile": "full"},
        dataset_episode_dir=str(root),
        trajectory_output=str(root),
        ue_profile={"appearance": catalog} if appearance_enabled else {},
        repo_root=str(tmp_path),
        task_id="audit-cli-test",
    )
    captured = {}

    monkeypatch.setattr(cli_module, "_resolve_runtime", lambda *args: (task_file, resolved))
    monkeypatch.setattr(cli_module, "_run_manifest", lambda *args: ({}, tmp_path / "run.json"))
    monkeypatch.setattr(cli_module, "_save_run_manifest_best_effort", lambda *args: None)
    monkeypatch.setattr(cli_module, "_start_pipeline_step", lambda *args, **kwargs: (object(), tmp_path / "state.json"))
    monkeypatch.setattr(cli_module, "_finish_pipeline_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli_module, "_sync_render_state_from_audit", lambda *args: None)
    monkeypatch.setattr(cli_module, "update_validation", lambda *args: None)
    monkeypatch.setattr(cli_module, "_finish_run_manifest", lambda *args: None)

    def fake_audit_main(argv, **kwargs):
        captured["argv"] = argv
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(
        "grf_ue_bridge.workflows.task_audit.main", fake_audit_main
    )

    cli_module.task_audit(
        task=task_file,
        validation_level="none",
        dataset_root=None,
        ue_project_root=None,
        local_config=None,
        force=False,
    )

    assert captured["appearance_catalog"] == catalog
    assert captured["episode_dir"] == root


def test_audit_enabled_appearance_accepts_matching_restored_report(tmp_path):
    root = _write_audit_camera_fixture(tmp_path)
    _write_appearance_fixture(root, render_required=True)
    (root / "provenance" / "ue-profile.json").unlink()

    report = run_audit(
        root,
        require_render=True,
        require_mask=False,
        require_mot=False,
        require_pose=False,
        validation_level="none",
        episode=root,
        appearance_catalog=_appearance_catalog(),
    )

    assert report["passed"] is True
    assert report["checks"]["appearance"]["status"] == "passed"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda actor: actor.update(apply_status="failed"), "apply_status"),
        (lambda actor: actor["assets"].update(mesh="/Game/Other/Mesh"), "mesh"),
        (lambda actor: actor["slots"].update(Jersey="/Game/Other/Jersey"), "Jersey"),
    ],
)
def test_audit_appearance_requires_applied_manifest_matching_actor_and_slots(
    tmp_path, mutation, message
):
    root = _write_audit_camera_fixture(tmp_path)
    _write_appearance_fixture(root)
    report_path = root / "provenance" / "appearance_application.json"
    application = json.loads(report_path.read_text(encoding="utf-8"))
    mutation(application["actors"][0])
    report_path.write_text(json.dumps(application), encoding="utf-8")

    report = run_audit(
        root,
        require_render=False,
        require_mask=False,
        require_mot=False,
        require_pose=False,
        validation_level="none",
        episode=root,
        appearance_catalog=_appearance_catalog(),
    )

    assert report["passed"] is False
    assert report["checks"]["appearance"]["status"] == "failed"
    assert message in " ".join(report["errors"])


@pytest.mark.parametrize("invariant", [
    "actor_count",
    "actor_entity_ids",
    "actor_restore_status",
    "mrq_pending",
    "mrq_job_status",
])
def test_audit_enabled_appearance_validates_complete_restored_structure(
    tmp_path, invariant
):
    root = _write_audit_camera_fixture(tmp_path)
    _write_appearance_fixture(root)
    report_path = root / "provenance" / "appearance_application.json"
    application = json.loads(report_path.read_text(encoding="utf-8"))
    if invariant == "actor_count":
        application["actors"] = application["actors"][:-1]
    elif invariant == "actor_entity_ids":
        application["actors"][0]["entity_id"] = "UNKNOWN"
    elif invariant == "actor_restore_status":
        application["actors"][0]["restore_status"] = "restoring"
    elif invariant == "mrq_pending":
        application["mrq_pending"] = True
    else:
        application["mrq_jobs"] = [{"camera_id": "Camera_01", "status": "rendering"}]
    report_path.write_text(json.dumps(application), encoding="utf-8")

    report = run_audit(
        root,
        require_render=False,
        require_mask=False,
        require_mot=False,
        require_pose=False,
        validation_level="none",
        episode=root,
        appearance_catalog=_appearance_catalog(),
    )

    assert report["passed"] is False
    assert report["checks"]["appearance"]["status"] == "failed"
    assert any(
        marker in " ".join(report["errors"])
        for marker in ("actors", "entity", "restore_status", "mrq_pending", "terminal")
    )


def test_audit_enabled_appearance_rejects_application_summary_status_mismatch(tmp_path):
    root = _write_audit_camera_fixture(tmp_path)
    _write_appearance_fixture(
        root,
        render_required=True,
        report_overrides={
            "mrq_jobs": [{"camera_id": "Camera_01", "status": "failed"}],
            "render_result": {
                "status": "failed",
                "jobs": [{"camera_id": "Camera_01", "status": "failed"}],
                "cameras": {"Camera_01": {"ok": False}},
            },
        },
    )

    report = run_audit(
        root,
        require_render=True,
        require_mask=False,
        require_mot=False,
        require_pose=False,
        validation_level="none",
        episode=root,
        appearance_catalog=_appearance_catalog(),
    )

    assert report["passed"] is False
    assert report["checks"]["appearance"]["status"] == "failed"
    assert any(
        "render_result.status" in error and "render_summary.status" in error
        for error in report["errors"]
    )


@pytest.mark.parametrize(
    ("application_status", "summary_status", "matches"),
    [
        ("success", "success", True),
        ("skipped", "failed", True),
        ("no_render", "failed", True),
        ("success", "no_render", False),
        ("failed", "success", False),
    ],
)
def test_audit_appearance_render_status_uses_terminal_normalization(
    tmp_path, application_status, summary_status, matches
):
    root = _write_audit_camera_fixture(tmp_path)
    _write_appearance_fixture(root, render_required=True)
    report_path = root / "provenance" / "appearance_application.json"
    application = json.loads(report_path.read_text(encoding="utf-8"))
    application["render_result"]["status"] = application_status
    report_path.write_text(json.dumps(application), encoding="utf-8")
    summary_path = root / "render_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["status"] = summary_status
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    report = run_audit(
        root,
        require_render=True,
        require_mask=False,
        require_mot=False,
        require_pose=False,
        validation_level="none",
        episode=root,
        appearance_catalog=_appearance_catalog(),
    )

    mismatch_errors = [
        error for error in report["errors"]
        if "render_result.status" in error and "render_summary.status" in error
    ]
    assert bool(mismatch_errors) is (not matches)


def _recursive_json():
    return "[" * 1000 + "]" * 1000


@pytest.mark.parametrize("broken_file", [
    "manifest",
    "manifest_validation",
    "application",
    "meta",
    "render_summary",
])
def test_audit_recursive_json_writes_canonical_failed_report(tmp_path, broken_file):
    root = _write_audit_camera_fixture(tmp_path)
    _write_appearance_fixture(root, render_required=broken_file == "render_summary")
    if broken_file in {"manifest", "manifest_validation"}:
        path = root / "provenance" / "appearance.json"
    elif broken_file == "application":
        path = root / "provenance" / "appearance_application.json"
    elif broken_file == "meta":
        path = root / "meta.json"
    else:
        path = root / "render_summary.json"
    if broken_file == "manifest_validation":
        manifest = json.loads(path.read_text(encoding="utf-8"))
        deeply_nested = []
        for _ in range(850):
            deeply_nested = [deeply_nested]
        manifest["extra_recursive"] = deeply_nested
        path.write_text(json.dumps(manifest), encoding="utf-8")
    else:
        path.write_text(_recursive_json(), encoding="utf-8")

    report = run_audit(
        root,
        require_render=broken_file == "render_summary",
        require_mask=False,
        require_mot=False,
        require_pose=False,
        validation_level="none",
        episode=root,
        appearance_catalog=_appearance_catalog(),
    )

    assert (root / "audit" / "soak_audit_report.json").is_file()
    assert report["passed"] is False
    assert report["exit_code"] == 1
    assert report["errors"]


def test_audit_recursive_camera_json_writes_canonical_failed_report(tmp_path):
    root = _write_audit_camera_fixture(tmp_path)
    (root / "Camera_01" / "camera.json").write_text(
        _recursive_json(), encoding="utf-8"
    )

    report = run_audit(
        root,
        require_render=False,
        require_mask=False,
        require_mot=False,
        require_pose=False,
        validation_level="none",
    )

    assert report["passed"] is False
    assert report["exit_code"] == 1
    assert report["checks"]["calibration"]["status"] == "failed"
    assert any("camera.json" in error for error in report["errors"])


def test_audit_recursive_annotations_jsonl_writes_canonical_failed_report(tmp_path):
    root = _write_audit_camera_fixture(tmp_path)
    (root / "Camera_01" / "annotations.jsonl").write_text(
        _recursive_json() + "\n", encoding="utf-8"
    )

    report = run_audit(
        root,
        require_render=False,
        require_mask=False,
        require_mot=False,
        require_pose=False,
        validation_level="none",
    )

    assert report["passed"] is False
    assert report["exit_code"] == 1
    assert report["checks"]["camera_core"]["status"] == "failed"
    assert any("annotations.jsonl" in error for error in report["errors"])


def test_audit_recursive_pose_session_writes_canonical_failed_report(tmp_path):
    root = _write_audit_camera_fixture(tmp_path)
    (root / "pose_session.json").write_text(
        _recursive_json(), encoding="utf-8"
    )

    report = run_audit(
        root,
        require_render=False,
        require_mask=False,
        require_mot=False,
        require_pose=True,
        validation_level="none",
    )

    assert report["passed"] is False
    assert report["exit_code"] == 1
    assert report["checks"]["runtime_pose"]["status"] == "failed"
    assert any("pose_session.json" in error for error in report["errors"])


@pytest.mark.parametrize(
    "broken_file", ["application", "render_summary", "pose_session"]
)
def test_cleanup_recursive_json_fails_closed_and_preserves_transient(
    tmp_path, broken_file
):
    root = tmp_path / "episode"
    transient = root / "Camera_01" / "render" / "frame.png"
    transient.parent.mkdir(parents=True)
    transient.write_bytes(b"render")
    _write_successful_audit(root)
    if broken_file == "application":
        path = root / "provenance" / "appearance_application.json"
        path.parent.mkdir(parents=True)
    elif broken_file == "render_summary":
        path = root / "render_summary.json"
    else:
        path = root / "pose_session.json"
    path.write_text(_recursive_json(), encoding="utf-8")

    result = apply_cleanup(
        root,
        ["Camera_01"],
        resolved=_cleanup_resolved(tmp_path, pose_enabled=broken_file == "pose_session"),
    )

    assert result["ok"] is False
    assert result["deleted_files"] == 0
    assert transient.is_file()
    assert any("appearance" in problem or "render_summary" in problem
               for problem in result["gate_problems"])


def test_audit_enabled_appearance_requires_manifest_even_when_report_exists(tmp_path):
    root = _write_audit_camera_fixture(tmp_path)
    _write_appearance_fixture(root)
    (root / "provenance" / "appearance.json").unlink()

    report = run_audit(
        root,
        require_render=False,
        require_mask=False,
        require_mot=False,
        require_pose=False,
        validation_level="none",
        episode=root,
        appearance_catalog=_appearance_catalog(),
    )

    assert report["passed"] is False
    assert report["checks"]["appearance"]["status"] == "failed"
    assert "appearance.json" in " ".join(report["errors"])


def test_audit_enabled_appearance_rejects_stale_trajectory_report(tmp_path):
    root = _write_audit_camera_fixture(tmp_path)
    _write_appearance_fixture(root, change_frames_after_manifest=True)

    report = run_audit(
        root,
        require_render=False,
        require_mask=False,
        require_mot=False,
        require_pose=False,
        validation_level="none",
        episode=root,
        appearance_catalog=_appearance_catalog(),
    )

    assert report["passed"] is False
    assert report["checks"]["appearance"]["status"] == "failed"
    assert "trajectory_sha256" in " ".join(report["errors"])


def test_audit_enabled_appearance_rejects_failed_application_report(tmp_path):
    root = _write_audit_camera_fixture(tmp_path)
    _write_appearance_fixture(root, report_overrides={"state": "failed"})

    report = run_audit(
        root,
        require_render=False,
        require_mask=False,
        require_mot=False,
        require_pose=False,
        validation_level="none",
        episode=root,
        appearance_catalog=_appearance_catalog(),
    )

    assert report["passed"] is False
    assert report["checks"]["appearance"]["status"] == "failed"
    assert "restored" in " ".join(report["errors"])


@pytest.mark.parametrize("state", ["applied", "rendering", "restore_failed"])
def test_audit_enabled_appearance_rejects_unrestored_report(tmp_path, state):
    root = _write_audit_camera_fixture(tmp_path)
    _write_appearance_fixture(root, report_overrides={"state": state})

    report = run_audit(
        root,
        require_render=False,
        require_mask=False,
        require_mot=False,
        require_pose=False,
        validation_level="none",
        episode=root,
        appearance_catalog=_appearance_catalog(),
    )

    assert report["passed"] is False
    assert report["checks"]["appearance"]["status"] == "failed"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema", "wrong_schema", "schema"),
        ("version", 2, "version"),
        ("version", True, "version"),
        ("manifest_fingerprint", "0" * 64, "manifest fingerprint"),
        ("trajectory_sha256", "0" * 64, "trajectory_sha256"),
    ],
)
def test_audit_enabled_appearance_rejects_application_contract_mismatch(
    tmp_path, field, value, message
):
    root = _write_audit_camera_fixture(tmp_path)
    _write_appearance_fixture(root, report_overrides={field: value})

    report = run_audit(
        root,
        require_render=False,
        require_mask=False,
        require_mot=False,
        require_pose=False,
        validation_level="none",
        episode=root,
        appearance_catalog=_appearance_catalog(),
    )

    assert report["passed"] is False
    assert report["checks"]["appearance"]["status"] == "failed"
    assert message in " ".join(report["errors"])


def test_audit_enabled_appearance_rejects_application_catalog_fingerprint_mismatch(
    tmp_path,
):
    root = _write_audit_camera_fixture(tmp_path)
    _write_appearance_fixture(
        root,
        report_overrides={"catalog_fingerprint": "0" * 64},
        render_required=True,
    )

    report = run_audit(
        root,
        require_render=True,
        require_mask=False,
        require_mot=False,
        require_pose=False,
        validation_level="none",
        episode=root,
        appearance_catalog=_appearance_catalog(),
    )

    assert report["passed"] is False
    assert report["checks"]["appearance"]["status"] == "failed"
    assert "catalog fingerprint" in " ".join(report["errors"])


def test_audit_enabled_appearance_requires_every_render_camera_in_application_report(
    tmp_path,
):
    root = _write_audit_camera_fixture(tmp_path)
    _write_appearance_fixture(
        root,
        report_overrides={"mrq_jobs": [], "render_result": {
            "status": "success",
            "jobs": [],
            "cameras": {},
        }},
        render_required=True,
    )

    report = run_audit(
        root,
        require_render=True,
        require_mask=False,
        require_mot=False,
        require_pose=False,
        validation_level="none",
        episode=root,
        appearance_catalog=_appearance_catalog(),
    )

    assert report["passed"] is False
    assert report["checks"]["appearance"]["status"] == "failed"
    assert "required camera" in " ".join(report["errors"])


def test_audit_enabled_appearance_requires_every_render_summary_camera(tmp_path):
    root = _write_audit_camera_fixture(tmp_path)
    _write_appearance_fixture(root, render_required=True)
    (root / "render_summary.json").write_text(
        json.dumps({"status": "success", "cameras": {}}), encoding="utf-8"
    )

    report = run_audit(
        root,
        require_render=True,
        require_mask=False,
        require_mot=False,
        require_pose=False,
        validation_level="none",
        episode=root,
        appearance_catalog=_appearance_catalog(),
    )

    assert report["passed"] is False
    assert report["checks"]["appearance"]["status"] == "failed"
    assert "render_summary 缺少 required camera" in " ".join(report["errors"])


def test_audit_enabled_appearance_malformed_render_job_status_fails_diagnostically(
    tmp_path,
):
    root = _write_audit_camera_fixture(tmp_path)
    malformed_jobs = [{"camera_id": "Camera_01", "status": []}]
    _write_appearance_fixture(
        root,
        report_overrides={
            "mrq_jobs": malformed_jobs,
            "render_result": {
                "status": "success",
                "jobs": malformed_jobs,
                "cameras": {"Camera_01": {"ok": True}},
            },
        },
        render_required=True,
    )

    report = run_audit(
        root,
        require_render=True,
        require_mask=False,
        require_mot=False,
        require_pose=False,
        validation_level="none",
        episode=root,
        appearance_catalog=_appearance_catalog(),
    )

    assert report["passed"] is False
    assert report["checks"]["appearance"]["status"] == "failed"
    assert any("必须为终态" in error for error in report["errors"])


def test_audit_enabled_appearance_requires_terminal_render_success(tmp_path):
    root = _write_audit_camera_fixture(tmp_path)
    _write_appearance_fixture(
        root,
        render_required=True,
        report_overrides={"render_result": {"status": "rendering"}},
    )

    report = run_audit(
        root,
        require_render=True,
        require_mask=False,
        require_mot=False,
        require_pose=False,
        validation_level="none",
        episode=root,
        appearance_catalog=_appearance_catalog(),
    )

    assert report["passed"] is False
    assert report["checks"]["appearance"]["status"] == "failed"
    assert "终态 render_result" in " ".join(report["errors"])


@pytest.mark.parametrize("status", [["success"], {"status": "success"}])
def test_audit_invalid_render_status_writes_failed_report(tmp_path, status):
    root = _write_audit_camera_fixture(tmp_path)
    _write_appearance_fixture(
        root,
        render_required=True,
        report_overrides={"render_result": {"status": status}},
    )

    report = run_audit(
        root,
        require_render=True,
        require_mask=False,
        require_mot=False,
        require_pose=False,
        validation_level="none",
        episode=root,
        appearance_catalog=_appearance_catalog(),
    )

    assert (root / "audit" / "soak_audit_report.json").is_file()
    assert report["passed"] is False
    assert report["exit_code"] == 1
    assert report["checks"]["appearance"]["status"] == "failed"
    assert "终态 render_result" in " ".join(report["errors"])


def test_audit_overflowing_seed_writes_failed_report_without_crashing(tmp_path):
    root = _write_audit_camera_fixture(tmp_path)
    _write_appearance_fixture(root)
    meta_path = root / "meta.json"
    meta_text = meta_path.read_text(encoding="utf-8")
    meta_path.write_text(
        meta_text.replace('"ue_visual_seed": 1001', '"ue_visual_seed": 1e999'),
        encoding="utf-8",
    )

    report = run_audit(
        root,
        require_render=False,
        require_mask=False,
        require_mot=False,
        require_pose=False,
        validation_level="none",
        episode=root,
        appearance_catalog=_appearance_catalog(),
    )

    assert (root / "audit" / "soak_audit_report.json").is_file()
    assert report["passed"] is False
    assert report["exit_code"] == 1
    assert report["checks"]["appearance"]["status"] == "failed"
    assert "seed" in " ".join(report["errors"])


def test_audit_boolean_manifest_version_writes_failed_report(tmp_path):
    root = _write_audit_camera_fixture(tmp_path)
    artifacts = _write_appearance_fixture(root)
    manifest = artifacts["manifest"]
    manifest["version"] = True
    fingerprint_payload = {
        key: value for key, value in manifest.items() if key != "fingerprint"
    }
    manifest["fingerprint"] = hashlib.sha256(
        json.dumps(
            fingerprint_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    manifest_path = root / "provenance" / "appearance.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    application_path = root / "provenance" / "appearance_application.json"
    application = json.loads(application_path.read_text(encoding="utf-8"))
    application["manifest_fingerprint"] = manifest["fingerprint"]
    application_path.write_text(json.dumps(application), encoding="utf-8")

    report = run_audit(
        root,
        require_render=False,
        require_mask=False,
        require_mot=False,
        require_pose=False,
        validation_level="none",
        episode=root,
        appearance_catalog=_appearance_catalog(),
    )

    assert (root / "audit" / "soak_audit_report.json").is_file()
    assert report["passed"] is False
    assert report["exit_code"] == 1
    assert report["checks"]["appearance"]["status"] == "failed"
    assert "version" in " ".join(report["errors"])


def test_audit_enabled_appearance_accepts_terminal_failed_render_result(tmp_path):
    root = _write_audit_camera_fixture(tmp_path)
    _write_appearance_fixture(
        root,
        render_required=True,
        report_overrides={"mrq_jobs": [{"camera_id": "Camera_01", "status": "failed"}],
                          "render_result": {
                              "status": "failed",
                              "jobs": [{"camera_id": "Camera_01", "status": "failed"}],
                              "cameras": {"Camera_01": {"ok": False}},
                          }},
    )
    (root / "render_summary.json").write_text(
        json.dumps({"status": "failed", "cameras": {"Camera_01": {"ok": False}}}),
        encoding="utf-8",
    )

    report = run_audit(
        root,
        require_render=True,
        require_mask=False,
        require_mot=False,
        require_pose=False,
        validation_level="none",
        episode=root,
        appearance_catalog=_appearance_catalog(),
    )

    assert report["passed"] is False
    assert report["checks"]["appearance"]["status"] == "passed"
    assert report["checks"]["render"]["status"] == "failed"


def test_audit_enabled_appearance_rejects_non_object_manifest(tmp_path):
    root = _write_audit_camera_fixture(tmp_path)
    _write_appearance_fixture(root)
    (root / "provenance" / "appearance.json").write_text("[]", encoding="utf-8")

    report = run_audit(
        root,
        require_render=False,
        require_mask=False,
        require_mot=False,
        require_pose=False,
        validation_level="none",
        episode=root,
        appearance_catalog=_appearance_catalog(),
    )

    assert report["passed"] is False
    assert report["checks"]["appearance"]["status"] == "failed"


def test_audit_json_only_missing_mot_and_pose_is_passed_with_skips(tmp_path):
    report = run_audit_fixture(
        tmp_path,
        require_render=False,
        require_mask=False,
        require_mot=False,
        require_pose=False,
        validation_level="none",
    )

    assert report["passed"] is True
    assert report["checks"]["mot_export"]["status"] == "skipped"
    assert report["checks"]["runtime_pose"]["status"] == "skipped"
    assert report["checks"]["render"]["status"] == "skipped"


def test_audit_required_pose_missing_session_fails(tmp_path):
    report = run_audit_fixture(
        tmp_path,
        require_render=False,
        require_mask=False,
        require_mot=False,
        require_pose=True,
        validation_level="none",
    )

    assert report["passed"] is False
    assert report["checks"]["runtime_pose"]["status"] == "failed"
    assert "pose_session.json" in " ".join(report["errors"])


def test_audit_required_pose_malformed_capture_is_diagnostic_failure(tmp_path):
    root = _write_audit_camera_fixture(tmp_path)
    (root / "pose_session.json").write_text(
        json.dumps({"capture_complete": True}), encoding="utf-8"
    )
    (root / "pose_capture.jsonl").write_text("not-json\n", encoding="utf-8")
    (root / "coco17_3d.jsonl").write_text("{}\n", encoding="utf-8")
    (root / "Camera_01" / "coco17_2d.jsonl").write_text(
        "{}\n", encoding="utf-8"
    )

    report = run_audit(
        root,
        require_render=False,
        require_mask=False,
        require_mot=False,
        require_pose=True,
        validation_level="none",
    )

    assert report["passed"] is False
    assert report["checks"]["runtime_pose"]["status"] == "failed"
    assert "pose_capture.jsonl" in " ".join(report["errors"])


def test_audit_required_mot_invalid_existing_file_fails_without_annotation_validation(tmp_path):
    root = _write_audit_camera_fixture(tmp_path)
    camera = root / "Camera_01"
    (camera / "gt").mkdir()
    (camera / "gt" / "gt.txt").write_text("not,a,valid,mot,row\n", encoding="utf-8")
    (camera / "seqinfo.ini").write_text("[Sequence]\n", encoding="utf-8")

    report = run_audit(
        root,
        require_render=False,
        require_mask=False,
        require_mot=True,
        require_pose=False,
        validation_level="none",
    )

    assert report["passed"] is False
    assert report["checks"]["mot_export"]["status"] == "failed"
    assert "gt.txt" in " ".join(report["errors"])


def test_audit_required_mot_invalid_values_fail_at_validation_none(tmp_path):
    root = _write_audit_camera_fixture(tmp_path)
    camera = root / "Camera_01"
    (camera / "gt").mkdir()
    (camera / "gt" / "gt.txt").write_text(
        "0,0,0,0,0,0,0,0,0\n", encoding="utf-8"
    )
    (camera / "seqinfo.ini").write_text("[Sequence]\n", encoding="utf-8")

    report = run_audit(
        root,
        require_render=False,
        require_mask=False,
        require_mot=True,
        require_pose=False,
        validation_level="none",
    )

    assert report["passed"] is False
    assert report["checks"]["mot_export"]["status"] == "failed"
    assert any(
        marker in " ".join(report["errors"])
        for marker in ("frame", "track_id", "宽/高", "x/y")
    )


def test_audit_optional_mot_missing_seqinfo_is_skipped_not_passed(tmp_path):
    root = _write_audit_camera_fixture(tmp_path)
    camera = root / "Camera_01"
    (camera / "gt").mkdir()
    (camera / "gt" / "gt.txt").write_text(
        "1,1,1,1,1,1,1,1,1\n", encoding="utf-8"
    )

    report = run_audit(
        root,
        require_render=False,
        require_mask=False,
        require_mot=False,
        require_pose=False,
        validation_level="quick",
    )

    assert report["checks"]["mot_export"]["status"] == "skipped"
    assert report["checks"]["mot_export"]["required"] is False
    assert report["validation"]["checks"]["mot_export"]["status"] == "skipped"
    assert report["passed"] is True


def test_audit_warnings_only_passes(tmp_path):
    report = run_audit_fixture(
        tmp_path,
        require_render=False,
        require_mask=False,
        require_mot=False,
        require_pose=False,
        validation_level="none",
    )

    assert report["exit_code"] == 0
    assert report["passed"] is True


def test_audit_required_render_missing_summary_fails(tmp_path):
    report = run_audit_fixture(
        tmp_path,
        require_render=True,
        require_mask=False,
        require_mot=False,
        require_pose=False,
        validation_level="none",
    )

    assert report["passed"] is False
    assert report["checks"]["render"]["status"] == "failed"
    assert report["render_summary"]["ok"] is False
    assert "render_summary.json" in " ".join(report["errors"])


def test_audit_optional_bad_render_summary_is_not_required_failure(tmp_path):
    root = _write_audit_camera_fixture(tmp_path)
    (root / "render_summary.json").write_text(
        json.dumps({"status": "failed", "cameras": {"Camera_01": {"ok": False}}}),
        encoding="utf-8",
    )

    rc = audit_main(
        [
            "--input",
            str(root),
            "--expected-cameras",
            "1",
            "--expected-frames-per-camera",
            "1",
            "--validation-level",
            "none",
            "--render-required",
            "false",
            "--mask-enabled",
            "false",
            "--mot-required",
            "false",
            "--yolo-det-required",
            "false",
            "--yolo-seg-required",
            "false",
            "--pose-required",
            "false",
        ]
    )
    assert rc == 0
    report = json.loads(
        (root / "audit" / "soak_audit_report.json").read_text(encoding="utf-8")
    )
    assert report["passed"] is True
    assert report["checks"]["render"]["status"] == "skipped"
    assert report["render_summary"]["ok"] is False


def test_audit_required_render_non_object_summary_fails_diagnostically(tmp_path):
    root = _write_audit_camera_fixture(tmp_path)
    (root / "render_summary.json").write_text("[]", encoding="utf-8")

    report = run_audit(
        root,
        require_render=True,
        require_mask=False,
        require_mot=False,
        require_pose=False,
        validation_level="none",
    )

    assert report["passed"] is False
    assert report["checks"]["render"]["status"] == "failed"
    assert "JSON 对象" in " ".join(report["errors"])


def test_audit_required_render_success_without_camera_mapping_fails(tmp_path):
    root = _write_audit_camera_fixture(tmp_path)
    (root / "render_summary.json").write_text(
        json.dumps({"status": "success"}), encoding="utf-8"
    )

    report = run_audit(
        root,
        require_render=True,
        require_mask=False,
        require_mot=False,
        require_pose=False,
        validation_level="none",
    )

    assert report["passed"] is False
    assert report["checks"]["render"]["status"] == "failed"
    assert "cameras" in " ".join(report["errors"])


def test_audit_optional_bad_render_camera_entry_is_reported_without_failure(tmp_path):
    root = _write_audit_camera_fixture(tmp_path)
    (root / "render_summary.json").write_text(
        json.dumps({"status": "success", "cameras": {"Camera_01": []}}),
        encoding="utf-8",
    )

    report = run_audit(
        root,
        require_render=False,
        require_mask=False,
        require_mot=False,
        require_pose=False,
        validation_level="none",
    )

    assert report["passed"] is True
    assert report["checks"]["render"]["status"] == "skipped"
    assert report["render_summary"]["ok"] is False
    assert any("render_summary" in warning for warning in report["warnings"])


def test_audit_required_pose_non_object_session_fails_diagnostically(tmp_path):
    root = _write_audit_camera_fixture(tmp_path)
    (root / "pose_session.json").write_text("[]", encoding="utf-8")

    report = run_audit(
        root,
        require_render=False,
        require_mask=False,
        require_mot=False,
        require_pose=True,
        validation_level="none",
    )

    assert report["passed"] is False
    assert report["checks"]["runtime_pose"]["status"] == "failed"
    assert "pose_session.json" in " ".join(report["errors"])
    assert "JSON 对象" in " ".join(report["errors"])


def test_audit_required_pose_malformed_coco17_2d_fails_diagnostically(tmp_path):
    root = _write_audit_camera_fixture(tmp_path)
    (root / "pose_session.json").write_text(
        json.dumps({"capture_complete": True}), encoding="utf-8"
    )
    (root / "pose_capture.jsonl").write_text(
        "\n".join("{}" for _ in range(130)) + "\n", encoding="utf-8"
    )
    (root / "coco17_3d.jsonl").write_text(
        "\n".join("{}" for _ in range(10)) + "\n", encoding="utf-8"
    )
    (root / "Camera_01" / "coco17_2d.jsonl").write_text(
        "not-json\n", encoding="utf-8"
    )

    report = run_audit(
        root,
        require_render=False,
        require_mask=False,
        require_mot=False,
        require_pose=True,
        validation_level="none",
    )

    assert report["passed"] is False
    assert report["checks"]["runtime_pose"]["status"] == "failed"
    assert "coco17_2d.jsonl" in " ".join(report["errors"])


def test_audit_optional_existing_invalid_mot_fails_without_annotation_validation(tmp_path):
    root = _write_audit_camera_fixture(tmp_path)
    camera = root / "Camera_01"
    (camera / "gt").mkdir()
    (camera / "gt" / "gt.txt").write_text("not,a,valid,mot,row\n", encoding="utf-8")
    (camera / "seqinfo.ini").write_text("[Sequence]\n", encoding="utf-8")

    report = run_audit(
        root,
        require_render=False,
        require_mask=False,
        require_mot=False,
        require_pose=False,
        validation_level="none",
    )

    assert report["passed"] is False
    assert report["checks"]["mot_export"]["status"] == "failed"
    assert "gt.txt" in " ".join(report["errors"])


def test_audit_malformed_annotation_objects_fails_without_crashing(tmp_path):
    root = _write_audit_camera_fixture(tmp_path)
    annotation_path = root / "Camera_01" / "annotations.jsonl"
    frame = json.loads(annotation_path.read_text(encoding="utf-8"))
    frame["objects"] = {"unexpected": "object"}
    annotation_path.write_text(json.dumps(frame) + "\n", encoding="utf-8")

    report = run_audit(
        root,
        require_render=False,
        require_mask=False,
        require_mot=False,
        require_pose=False,
        validation_level="none",
    )

    assert report["passed"] is False
    assert any("objects" in error for error in report["errors"])


def test_audit_empty_annotations_fails_without_crashing(tmp_path):
    root = _write_audit_camera_fixture(tmp_path)
    (root / "Camera_01" / "annotations.jsonl").write_text("", encoding="utf-8")

    report = run_audit(
        root,
        require_render=False,
        require_mask=False,
        require_mot=False,
        require_pose=False,
        validation_level="none",
    )

    assert report["passed"] is False
    assert "annotations 帧数" in " ".join(report["errors"])


def test_audit_nested_camera_sections_fail_diagnostically(tmp_path):
    root = _write_audit_camera_fixture(tmp_path)
    camera_json = json.loads(
        (root / "Camera_01" / "camera.json").read_text(encoding="utf-8")
    )
    camera_json["intrinsics"] = ["not", "an", "object"]
    camera_json["extrinsics"] = "not an object"
    (root / "Camera_01" / "camera.json").write_text(
        json.dumps(camera_json), encoding="utf-8"
    )

    report = run_audit(
        root,
        require_render=False,
        require_mask=False,
        require_mot=False,
        require_pose=False,
        validation_level="none",
    )

    assert report["passed"] is False
    assert any("intrinsics" in error for error in report["errors"])
    assert any("extrinsics" in error for error in report["errors"])


@pytest.mark.parametrize("validation_level", ["quick", "full"])
def test_annotation_malformed_object_and_corrupt_mask_are_diagnostic(
    tmp_path, validation_level
):
    root = make_annotation_fixture(tmp_path, mot=False)
    camera = root / "Camera_01"
    annotation_path = camera / "annotations.jsonl"
    frame = json.loads(annotation_path.read_text(encoding="utf-8"))
    frame["objects"].append("not an object")
    annotation_path.write_text(json.dumps(frame) + "\n", encoding="utf-8")
    (camera / "mask").mkdir()
    (camera / "mask" / "000001.png").write_bytes(b"corrupt png")

    result = validate_annotation_result(
        root,
        workers=1,
        validation_level=validation_level,
        require_mot=False,
    )

    assert result.passed is False
    assert any("objects[1]" in error for error in result.errors)
    assert any("000001.png" in error or "mask" in error for error in result.errors)


@pytest.mark.parametrize("validation_level", ["quick", "full"])
def test_annotation_disabled_mask_and_yolo_ignore_malformed_stale_artifacts(
    tmp_path, validation_level
):
    root = make_annotation_fixture(tmp_path, mot=False)
    camera = root / "Camera_01"
    (camera / "mask").mkdir()
    (camera / "mask" / "000001.png").write_bytes(b"corrupt png")
    (camera / "labels" / "det").mkdir(parents=True)
    (camera / "labels" / "det" / "000001.txt").write_text(
        "malformed detection\n", encoding="utf-8"
    )
    (camera / "labels" / "seg").mkdir(parents=True)
    (camera / "labels" / "seg" / "000001.txt").write_text(
        "malformed segmentation\n", encoding="utf-8"
    )

    result = validate_annotation_result(
        root,
        workers=1,
        validation_level=validation_level,
        require_mot=False,
        require_mask=False,
        require_yolo_det=False,
        require_yolo_seg=False,
    )

    assert result.passed is True
    assert result.errors == []


@pytest.mark.parametrize("validation_level", ["quick", "full"])
def test_annotation_disabled_mask_ignores_malformed_mask_config(
    tmp_path, validation_level
):
    root = make_annotation_fixture(tmp_path, mot=False)
    camera = root / "Camera_01"
    (camera / "mask").mkdir()
    (camera / "mask_config.json").write_text("[]", encoding="utf-8")

    result = validate_annotation_result(
        root,
        workers=1,
        validation_level=validation_level,
        require_mot=False,
        require_mask=False,
        require_yolo_det=False,
        require_yolo_seg=False,
    )

    assert result.passed is True
    assert result.errors == []


@pytest.mark.parametrize("validation_level", ["quick", "full"])
def test_annotation_enabled_mask_rejects_non_object_mask_config(
    tmp_path, validation_level
):
    root = make_annotation_fixture(tmp_path, mot=True)
    camera = root / "Camera_01"
    (camera / "mask").mkdir()
    (camera / "mask_config.json").write_text("[]", encoding="utf-8")

    result = validate_annotation_result(
        root,
        workers=1,
        validation_level=validation_level,
        require_mot=True,
        require_mask=True,
    )

    assert result.passed is False
    assert any("mask_config.json" in error for error in result.errors)


def test_audit_malformed_annotation_object_at_validation_none_fails_required_checks(
    tmp_path,
):
    root = _write_audit_camera_fixture(tmp_path)
    annotation_path = root / "Camera_01" / "annotations.jsonl"
    frame = json.loads(annotation_path.read_text(encoding="utf-8"))
    frame["objects"] = ["not an object"]
    annotation_path.write_text(json.dumps(frame) + "\n", encoding="utf-8")

    report = run_audit(
        root,
        require_render=False,
        require_mask=True,
        require_mot=True,
        require_pose=False,
        require_yolo_det=True,
        require_yolo_seg=True,
        validation_level="none",
    )

    assert report["passed"] is False
    assert any("objects[0]" in error for error in report["errors"])
    assert report["checks"]["instance_mask"]["status"] == "failed"
    assert report["checks"]["mot_export"]["status"] == "failed"
    assert report["checks"]["yolo_det"]["status"] == "failed"
    assert report["checks"]["yolo_seg"]["status"] == "failed"


def test_audit_annotation_validator_exception_is_reported(monkeypatch, tmp_path):
    root = _write_audit_camera_fixture(tmp_path)

    def raise_validator(*args, **kwargs):
        raise RuntimeError("synthetic validator failure")

    monkeypatch.setattr(
        annotation_validator_module,
        "validate_annotation_result",
        raise_validator,
    )

    report = run_audit(
        root,
        require_render=False,
        require_mask=False,
        require_mot=False,
        require_pose=False,
        validation_level="quick",
    )

    assert report["passed"] is False
    assert any(
        "validate-annotations" in error and "synthetic validator failure" in error
        for error in report["errors"]
    )
    assert report["checks"]["annotation_validation"]["status"] == "failed"


def test_audit_sync_checks_first_frame_across_cameras(tmp_path):
    root = _write_audit_camera_fixture(tmp_path)
    camera_two = root / "Camera_02"
    shutil.copytree(root / "Camera_01", camera_two)

    camera_json = json.loads((camera_two / "camera.json").read_text(encoding="utf-8"))
    camera_json["extrinsics"]["world_location_m"] = [1.0, 0.0, 1.0]
    (camera_two / "camera.json").write_text(json.dumps(camera_json), encoding="utf-8")
    annotation = json.loads(
        (camera_two / "annotations.jsonl").read_text(encoding="utf-8")
    )
    annotation["time_seconds"] = 1.0
    (camera_two / "annotations.jsonl").write_text(
        json.dumps(annotation) + "\n", encoding="utf-8"
    )

    args = [
        "--input",
        str(root),
        "--expected-cameras",
        "2",
        "--expected-frames-per-camera",
        "1",
        "--validation-level",
        "none",
        "--render-required",
        "false",
        "--mask-enabled",
        "false",
        "--mot-required",
        "false",
        "--pose-required",
        "false",
    ]
    assert audit_main(args) == 1
    report = json.loads(
        (root / "audit" / "soak_audit_report.json").read_text(encoding="utf-8")
    )

    assert report["checks"]["sync"]["status"] == "failed"
    assert any("同步" in error for error in report["errors"])


def test_audit_sync_unequal_annotation_lengths_fails_without_crashing(tmp_path):
    root = _write_audit_camera_fixture(tmp_path)
    camera_two = root / "Camera_02"
    shutil.copytree(root / "Camera_01", camera_two)
    second_frame = json.loads(
        (camera_two / "annotations.jsonl").read_text(encoding="utf-8")
    )
    second_frame["frame_index"] = 2
    with (camera_two / "annotations.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(second_frame) + "\n")
    camera_json = json.loads((camera_two / "camera.json").read_text(encoding="utf-8"))
    camera_json["extrinsics"]["world_location_m"] = [1.0, 0.0, 1.0]
    (camera_two / "camera.json").write_text(json.dumps(camera_json), encoding="utf-8")

    args = [
        "--input", str(root), "--expected-cameras", "2",
        "--expected-frames-per-camera", "1", "--validation-level", "none",
        "--render-required", "false", "--mask-enabled", "false",
        "--mot-required", "false", "--pose-required", "false",
    ]
    assert audit_main(args) == 1
    report = json.loads(
        (root / "audit" / "soak_audit_report.json").read_text(encoding="utf-8")
    )

    assert report["checks"]["sync"]["status"] == "failed"
    assert any("帧数不一致" in error for error in report["errors"])


def test_audit_pose_malformed_jsonl_fails_runtime_pose_check(tmp_path):
    root = _write_audit_camera_fixture(tmp_path)
    (root / "pose_session.json").write_text(
        json.dumps({"capture_complete": True}), encoding="utf-8"
    )
    (root / "pose_capture.jsonl").write_text(
        "not-json\n" + "{}\n" * 129, encoding="utf-8"
    )
    (root / "coco17_3d.jsonl").write_text(
        "{}\n" * 10, encoding="utf-8"
    )
    (root / "Camera_01" / "coco17_2d.jsonl").write_text(
        "{}\n" * 10, encoding="utf-8"
    )

    report = run_audit(
        root,
        require_render=False,
        require_mask=False,
        require_mot=False,
        require_pose=True,
        validation_level="none",
    )

    assert report["checks"]["runtime_pose"]["status"] == "failed"
    assert "pose_capture.jsonl" in " ".join(report["errors"])


def test_audit_pose_requires_coco17_2d_for_every_expected_camera(tmp_path):
    root = _write_audit_camera_fixture(tmp_path)
    camera_two = root / "Camera_02"
    shutil.copytree(root / "Camera_01", camera_two)
    camera_json = json.loads((camera_two / "camera.json").read_text(encoding="utf-8"))
    camera_json["extrinsics"]["world_location_m"] = [1.0, 0.0, 1.0]
    (camera_two / "camera.json").write_text(json.dumps(camera_json), encoding="utf-8")
    (root / "pose_session.json").write_text(
        json.dumps({"capture_complete": True}), encoding="utf-8"
    )
    (root / "pose_capture.jsonl").write_text(
        "{}\n" * 13 * 10, encoding="utf-8"
    )
    (root / "coco17_3d.jsonl").write_text(
        "{}\n" * 10, encoding="utf-8"
    )
    (root / "Camera_01" / "coco17_2d.jsonl").write_text(
        "{}\n" * 10, encoding="utf-8"
    )

    args = [
        "--input", str(root), "--expected-cameras", "2",
        "--expected-frames-per-camera", "1", "--validation-level", "none",
        "--render-required", "false", "--mask-enabled", "false",
        "--mot-required", "false", "--pose-required", "true",
    ]
    assert audit_main(args) == 1
    report = json.loads(
        (root / "audit" / "soak_audit_report.json").read_text(encoding="utf-8")
    )

    assert report["checks"]["runtime_pose"]["status"] == "failed"
    assert "Camera_02" in " ".join(report["errors"])
    assert "coco17_2d.jsonl" in " ".join(report["errors"])


def test_audit_rejects_legacy_17_point_final_pose_arrays(tmp_path):
    root = _write_audit_camera_fixture(tmp_path)
    (root / "pose_session.json").write_text(
        json.dumps({"capture_complete": True}), encoding="utf-8"
    )
    (root / "pose_capture.jsonl").write_text("{}\n" * (13 * 10), encoding="utf-8")
    (root / "coco17_3d.jsonl").write_text(
        "\n".join(json.dumps({"keypoints_3d_m": [[0.0, 0.0, 0.0]] * 17}) for _ in range(10)) + "\n",
        encoding="utf-8",
    )
    (root / "Camera_01" / "coco17_2d.jsonl").write_text(
        "\n".join(json.dumps({"keypoints_2d_px": [[0.0, 0.0]] * 17, "visible": [True] * 17}) for _ in range(10)) + "\n",
        encoding="utf-8",
    )

    report = run_audit(
        root,
        require_render=False,
        require_mask=False,
        require_mot=False,
        require_pose=True,
        validation_level="none",
    )

    assert report["passed"] is False
    assert report["checks"]["runtime_pose"]["status"] == "failed"
    assert "12 点肢体契约" in " ".join(report["errors"])


def test_audit_accepts_12_limb_points_and_preserves_13_raw_bones(tmp_path):
    root = _write_audit_camera_fixture(tmp_path)
    (root / "pose_session.json").write_text(
        json.dumps({"capture_complete": True}), encoding="utf-8"
    )
    (root / "pose_capture.jsonl").write_text("{}\n" * (13 * 10), encoding="utf-8")
    (root / "coco17_3d.jsonl").write_text(
        "\n".join(
            json.dumps({"keypoints_3d_m": [[0.0, 0.0, 0.0]] * 12}) for _ in range(10)
        ) + "\n",
        encoding="utf-8",
    )
    (root / "Camera_01" / "coco17_2d.jsonl").write_text(
        "\n".join(
            json.dumps({"keypoints_2d_px": [[0.0, 0.0]] * 12, "visible": [True] * 12})
            for _ in range(10)
        ) + "\n",
        encoding="utf-8",
    )

    report = run_audit(
        root,
        require_render=False,
        require_mask=False,
        require_mot=False,
        require_pose=True,
        validation_level="none",
    )

    assert report["passed"] is True
    assert report["pose_coco17"]["pose_capture_rows"] == 130
    assert report["pose_coco17"]["coco17_total_kp"] == 120


def test_audit_unrequested_render_zero_byte_is_ignored(tmp_path):
    root = _write_audit_camera_fixture(tmp_path)
    camera = root / "Camera_01"
    (camera / "mask").mkdir()
    (camera / "mask" / "000001.png").write_bytes(b"mask")
    (camera / "render_mask").mkdir()
    (camera / "render_mask" / "000000.exr").write_bytes(b"exr")
    (camera / "render").mkdir()
    (camera / "render" / "000000.png").write_bytes(b"")

    report = run_audit(
        root,
        require_render=False,
        require_mask=True,
        require_mot=False,
        require_pose=False,
        validation_level="none",
    )

    assert report["passed"] is True
    assert report["checks"]["render"]["status"] == "skipped"
    assert not any("零字节" in error for error in report["errors"])


def test_annotation_json_only_missing_mot_passes(tmp_path):
    root = make_annotation_fixture(tmp_path, mot=False)

    result = validate_annotation_result(
        root, workers=1, validation_level="full", require_mot=False
    )

    assert result.passed is True
    assert result.checks["mot_export"]["status"] == "skipped"


def test_annotation_required_mot_missing_fails(tmp_path):
    root = make_annotation_fixture(tmp_path, mot=False)

    result = validate_annotation_result(
        root, workers=1, validation_level="quick", require_mot=True
    )

    assert result.passed is False
    assert any("gt/gt.txt" in error for error in result.errors)


def test_annotation_required_mot_valid_passes(tmp_path):
    root = make_annotation_fixture(tmp_path, mot=True)

    result = validate_annotation_result(
        root, workers=1, validation_level="quick", require_mot=True
    )

    assert result.passed is True


def test_annotation_optional_mot_still_validates_existing_file(tmp_path):
    root = make_annotation_fixture(tmp_path, mot=True)
    (root / "Camera_01" / "gt" / "gt.txt").write_text(
        "not,a,valid,mot,row\n", encoding="utf-8"
    )

    result = validate_annotation_result(
        root, workers=1, validation_level="quick", require_mot=False
    )

    assert result.passed is False
    assert result.checks["mot_export"]["status"] == "failed"
    assert any("not,a,valid,mot,row" in error for error in result.errors)


def test_annotation_optional_mot_does_not_hide_invalid_existing_file(tmp_path):
    root = make_annotation_fixture(tmp_path, mot=False)
    shutil.copytree(root / "Camera_01", root / "Camera_02")
    (root / "Camera_02" / "gt" / "gt.txt").write_text(
        "not,a,valid,mot,row\n", encoding="utf-8"
    )

    result = validate_annotation_result(
        root, workers=1, validation_level="quick", require_mot=False
    )

    assert result.passed is False
    assert result.checks["mot_export"]["status"] == "failed"
    assert any("not,a,valid,mot,row" in error for error in result.errors)


@pytest.mark.parametrize("field_index", range(9))
@pytest.mark.parametrize("require_mot", [False, True])
@pytest.mark.parametrize("nonfinite_value", ["nan", "inf"])
def test_annotation_nonfinite_existing_mot_fails_for_every_field(
    tmp_path, field_index, require_mot, nonfinite_value
):
    root = make_annotation_fixture(tmp_path, mot=True)
    fields = ["1", "1", "1", "2", "10", "20", "1", "1", "1.00"]
    fields[field_index] = nonfinite_value
    (root / "Camera_01" / "gt" / "gt.txt").write_text(
        ",".join(fields) + "\n", encoding="utf-8"
    )

    result = validate_annotation_result(
        root, workers=1, validation_level="quick", require_mot=require_mot
    )

    assert result.passed is False
    assert result.checks["mot_export"]["status"] == "failed"
    assert any(
        f"第 1 行第 {field_index + 1} 列" in error for error in result.errors
    )


def test_dataset_regression_required_mot_missing_with_zero_expected_rows_fails(tmp_path):
    root = make_annotation_fixture(tmp_path, mot=False)
    annotation_path = root / "Camera_01" / "annotations.jsonl"
    frame = json.loads(annotation_path.read_text(encoding="utf-8"))
    frame["objects"] = []
    annotation_path.write_text(json.dumps(frame) + "\n", encoding="utf-8")

    errors = collect_dataset_regression_errors(root, require_mot=True)

    assert any("gt/gt.txt" in error for error in errors)


def test_annotation_json_only_missing_seqinfo_passes(tmp_path):
    root = make_annotation_fixture(tmp_path, mot=False)
    (root / "Camera_01" / "seqinfo.ini").unlink()

    result = validate_annotation_result(
        root, workers=1, validation_level="full", require_mot=False
    )

    assert result.passed is True
    assert result.checks["mot_export"]["status"] == "skipped"


def test_annotation_required_mot_missing_seqinfo_fails(tmp_path):
    root = make_annotation_fixture(tmp_path, mot=True)
    (root / "Camera_01" / "seqinfo.ini").unlink()

    result = validate_annotation_result(
        root, workers=1, validation_level="quick", require_mot=True
    )

    assert result.passed is False
    assert any("seqinfo.ini" in error for error in result.errors)


def test_result_errors_always_fail():
    result = ValidationResult()
    result.errors.append("broken")

    result.finalize()

    assert result.passed is False
    assert result.exit_code == 1


def test_result_warning_only_passes():
    result = ValidationResult()
    result.warnings.append("non-blocking")
    result.add_check(
        "optional", status="skipped", required=False, message="not enabled"
    )

    result.finalize()

    assert result.passed is True
    assert result.exit_code == 0


def test_required_failed_check_fails():
    result = ValidationResult()
    result.add_check(
        "render", status="failed", required=True, message="render failed"
    )

    result.finalize()

    assert result.passed is False
    assert result.exit_code == 1
    assert result.errors == ["render failed"]


def test_add_check_rejects_unknown_status():
    result = ValidationResult()

    with pytest.raises(ValueError):
        result.add_check("bad", status="unknown")


def test_add_check_does_not_duplicate_existing_error():
    result = ValidationResult(errors=["render failed"])

    result.add_check(
        "render", status=CheckStatus.FAILED, required=True, message="render failed"
    )

    assert result.errors == ["render failed"]


def test_result_to_dict_contains_only_json_safe_canonical_fields():
    result = ValidationResult()
    result.add_check(
        "detail",
        status=CheckStatus.PASSED,
        required=True,
        values=(1, 2),
        nested={"path": object()},
    )
    result.finalize()

    payload = result.to_dict()

    assert set(payload) == {"passed", "exit_code", "errors", "warnings", "checks"}
    assert json.dumps(payload)
    assert payload["checks"]["detail"]["status"] == "passed"
    assert payload["checks"]["detail"]["values"] == [1, 2]


def test_validation_result_from_canonical_report():
    report = {
        "passed": True,
        "exit_code": 0,
        "errors": [],
        "warnings": ["advisory"],
        "checks": {
            "render": {"status": "passed", "required": True},
            "pose": {"status": "skipped", "required": False},
        },
    }

    result = validation_result_from_report(report)

    assert result.passed is True
    assert result.exit_code == 0
    assert result.warnings == ["advisory"]
    assert result.checks == report["checks"]


def test_validation_result_accepts_minimal_canonical_success_report():
    result = validation_result_from_report({"passed": True})

    assert result.passed is True
    assert result.exit_code == 0
    assert result.errors == []


def test_validation_result_from_unfinalized_result_recomputes_failure():
    result = ValidationResult(errors=["broken"])

    normalized = validation_result_from_report(result)

    assert normalized.passed is False
    assert normalized.exit_code == 1
    assert normalized.errors == ["broken"]


def test_validation_result_from_legacy_report():
    result = validation_result_from_report({"ok": True, "failed_checks": []})

    assert result.passed is True
    assert result.exit_code == 0
    assert result.errors == []


@pytest.mark.parametrize(
    "report",
    [
        {},
        {"passed": True, "exit_code": 1, "errors": [], "warnings": [], "checks": {}},
        {"ok": True, "failed_checks": ["bad"]},
        {"passed": True, "ok": False},
    ],
)
def test_validation_result_from_malformed_or_ambiguous_report_fails_safe(report):
    result = validation_result_from_report(report)

    assert result.passed is False
    assert result.exit_code == 1
    assert result.errors


def _cleanup_resolved(tmp_path: Path, *, pose_enabled=False, render_enabled=True):
    return {
        "task_id": "cleanup-appearance-test",
        "repo_root": str(tmp_path),
        "postprocess": {"formats": ["json"], "yolo_pose": {"enabled": pose_enabled}},
        "ue_profile": {"appearance": {"mode": "seeded"}, "annotation_export": {
            "render_rgb": {"enabled": render_enabled},
            "instance_mask": {"enabled": False},
            "export_mot": False,
            "cameras": ["Camera_01"],
        }},
        "artifact_policy": {"profile": "research_minimal"},
    }


def test_cleanup_still_blocks_any_local_rollback_for_p1(tmp_path):
    root = tmp_path / "episode"
    transient = root / "Camera_01" / "render" / "frame.png"
    transient.parent.mkdir(parents=True)
    transient.write_bytes(b"render")
    _write_cleanup_summary(root)
    _write_successful_audit(root)
    rollback = (
        tmp_path / ".futsalmot" / "runtime" / "cleanup-appearance-test"
        / "appearance-rollback.json"
    )
    rollback.parent.mkdir(parents=True)
    rollback.write_text('{"state":"restored"}', encoding="utf-8")
    for report_path in (
        rollback.parent / "appearance-application.json",
        root / "provenance" / "appearance_application.json",
    ):
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps({"state": "restored"}), encoding="utf-8")

    result = apply_cleanup(root, ["Camera_01"], resolved=_cleanup_resolved(tmp_path))

    assert result["ok"] is False
    assert result["deleted_files"] == 0
    assert transient.is_file()
    assert rollback.is_file()
    assert any("baseline" in item for item in result["gate_problems"])


def _write_successful_audit(root: Path):
    audit = root / "audit"
    audit.mkdir(exist_ok=True)
    (audit / "soak_audit_report.json").write_text(json.dumps({
        "passed": True,
        "exit_code": 0,
        "errors": [],
        "warnings": [],
        "checks": {},
    }), encoding="utf-8")


def _write_cleanup_summary(root: Path, status="success"):
    (root / "render_summary.json").write_text(
        json.dumps({"status": status, "cameras": {"Camera_01": {"ok": True}}}),
        encoding="utf-8",
    )


def test_cleanup_disabled_pose_missing_session_is_not_a_gate(tmp_path):
    root = tmp_path / "episode"
    root.mkdir()
    _write_cleanup_summary(root)
    _write_successful_audit(root)
    assert _validation_gate(root, resolved=_cleanup_resolved(tmp_path)) == []


def test_cleanup_canonical_failed_audit_blocks_apply(tmp_path):
    root = tmp_path / "episode"
    root.mkdir()
    _write_cleanup_summary(root)
    audit = root / "audit"
    audit.mkdir()
    (audit / "soak_audit_report.json").write_text(json.dumps({
        "passed": False, "exit_code": 1, "errors": ["bad"],
        "warnings": [], "checks": {},
    }), encoding="utf-8")
    transient = root / "Camera_01" / "render"
    transient.mkdir(parents=True)
    marker = transient / "frame.png"
    marker.write_bytes(b"transient")

    result = apply_cleanup(root, ["Camera_01"], resolved=_cleanup_resolved(tmp_path))

    assert result["ok"] is False
    assert marker.exists()
    assert any("audit" in problem for problem in result["gate_problems"])


def test_cleanup_warnings_only_audit_is_non_blocking(tmp_path):
    root = tmp_path / "episode"
    root.mkdir()
    _write_cleanup_summary(root)
    audit = root / "audit"
    audit.mkdir()
    (audit / "soak_audit_report.json").write_text(json.dumps({
        "passed": True, "exit_code": 0, "errors": [],
        "warnings": ["warn"], "checks": {},
    }), encoding="utf-8")
    assert _validation_gate(root, resolved=_cleanup_resolved(tmp_path)) == []


def test_cleanup_minimal_canonical_success_audit_is_non_blocking(tmp_path):
    root = tmp_path / "episode"
    root.mkdir()
    _write_cleanup_summary(root)
    audit = root / "audit"
    audit.mkdir()
    (audit / "soak_audit_report.json").write_text(
        json.dumps({"passed": True}), encoding="utf-8"
    )

    assert _validation_gate(root, resolved=_cleanup_resolved(tmp_path)) == []


def test_cleanup_legacy_failed_audit_blocks_apply(tmp_path):
    root = tmp_path / "episode"
    root.mkdir()
    _write_cleanup_summary(root)
    audit = root / "audit"
    audit.mkdir()
    (audit / "soak_audit_report.json").write_text(json.dumps({
        "ok": False, "failed_checks": ["render"],
    }), encoding="utf-8")
    assert _validation_gate(root, resolved=_cleanup_resolved(tmp_path))


def test_cleanup_dry_run_does_not_delete_transient(tmp_path):
    root = tmp_path / "episode"
    camera = root / "Camera_01" / "render"
    camera.mkdir(parents=True)
    marker = camera / "frame.png"
    marker.write_bytes(b"transient")
    _write_cleanup_summary(root)
    _write_successful_audit(root)

    report = plan_cleanup(root, ["Camera_01"], resolved=_cleanup_resolved(tmp_path))

    assert report["dry_run"] is True
    assert report["gate_ok"] is True
    assert marker.exists()


@pytest.mark.parametrize(
    "session, expected_fragment",
    [
        (None, "pose_session.json"),
        ({"capture_complete": False}, "capture_complete"),
    ],
)
def test_cleanup_enabled_pose_blocks_missing_or_incomplete_session(
    tmp_path, session, expected_fragment
):
    root = tmp_path / "episode"
    root.mkdir()
    _write_cleanup_summary(root)
    _write_successful_audit(root)
    if session is not None:
        (root / "pose_session.json").write_text(
            json.dumps(session), encoding="utf-8"
        )

    result = apply_cleanup(
        root,
        ["Camera_01"],
        resolved=_cleanup_resolved(tmp_path, pose_enabled=True),
    )

    assert result["ok"] is False
    assert any(expected_fragment in problem for problem in result["gate_problems"])


def test_cleanup_enabled_pose_complete_allows_apply_and_keeps_canonical_files(tmp_path):
    root = tmp_path / "episode"
    camera = root / "Camera_01"
    transient = camera / "render"
    transient.mkdir(parents=True)
    marker = transient / "frame.png"
    marker.write_bytes(b"transient")
    (camera / "img1").mkdir()
    canonical = camera / "img1" / "000001.png"
    canonical.write_bytes(b"canonical")
    _write_cleanup_summary(root)
    _write_successful_audit(root)
    (root / "pose_session.json").write_text(
        json.dumps({"capture_complete": True}), encoding="utf-8"
    )

    result = apply_cleanup(
        root,
        ["Camera_01"],
        resolved=_cleanup_resolved(tmp_path, pose_enabled=True),
    )

    assert result["ok"] is True
    assert not marker.exists()
    assert canonical.exists()


def test_cleanup_malformed_audit_fails_safe(tmp_path):
    root = tmp_path / "episode"
    root.mkdir()
    _write_cleanup_summary(root)
    audit = root / "audit"
    audit.mkdir()
    (audit / "soak_audit_report.json").write_text("[]", encoding="utf-8")

    result = apply_cleanup(root, ["Camera_01"], resolved=_cleanup_resolved(tmp_path))

    assert result["ok"] is False
    assert any("audit" in problem for problem in result["gate_problems"])


def test_cleanup_malformed_render_summary_fails_safe(tmp_path):
    root = tmp_path / "episode"
    root.mkdir()
    (root / "render_summary.json").write_text("[]", encoding="utf-8")

    result = apply_cleanup(root, ["Camera_01"], resolved=_cleanup_resolved(tmp_path))

    assert result["ok"] is False
    assert any("render_summary" in problem for problem in result["gate_problems"])


@pytest.mark.parametrize("evidence", ["rollback", "local_report", "public_report"])
def test_cleanup_unresolved_appearance_transaction_preserves_recovery_evidence(
    tmp_path, evidence
):
    root = tmp_path / "episode"
    transient = root / "Camera_01" / "render" / "frame.png"
    transient.parent.mkdir(parents=True)
    transient.write_bytes(b"render")
    _write_cleanup_summary(root)
    _write_successful_audit(root)

    if evidence == "rollback":
        evidence_path = (
            tmp_path / ".futsalmot" / "runtime" / "cleanup-appearance-test"
            / "appearance-rollback.json"
        )
        evidence_path.parent.mkdir(parents=True)
        evidence_path.write_text("{}", encoding="utf-8")
    elif evidence == "local_report":
        evidence_path = (
            tmp_path / ".futsalmot" / "runtime" / "cleanup-appearance-test"
            / "appearance-application.json"
        )
        evidence_path.parent.mkdir(parents=True)
        evidence_path.write_text(json.dumps({"state": "rendering"}), encoding="utf-8")
    else:
        evidence_path = root / "provenance" / "appearance_application.json"
        evidence_path.parent.mkdir(parents=True)
        evidence_path.write_text(json.dumps({"state": "restore_failed"}), encoding="utf-8")

    result = apply_cleanup(
        root,
        ["Camera_01"],
        resolved=_cleanup_resolved(tmp_path),
    )

    assert result["ok"] is False
    assert result["deleted_files"] == 0
    assert transient.is_file()
    assert evidence_path.is_file()
    assert any("appearance" in problem.lower() for problem in result["gate_problems"])


def test_cleanup_blocks_rollback_even_when_reports_claim_restored(
    tmp_path,
):
    root = tmp_path / "episode"
    transient = root / "Camera_01" / "render" / "frame.png"
    transient.parent.mkdir(parents=True)
    transient.write_bytes(b"render")
    _write_cleanup_summary(root)
    _write_successful_audit(root)
    rollback_path = (
        tmp_path / ".futsalmot" / "runtime" / "cleanup-appearance-test"
        / "appearance-rollback.json"
    )
    rollback_path.parent.mkdir(parents=True)
    rollback_path.write_text("verified snapshot", encoding="utf-8")
    for report_path in (
        rollback_path.parent / "appearance-application.json",
        root / "provenance" / "appearance_application.json",
    ):
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps({"state": "restored"}), encoding="utf-8")

    result = apply_cleanup(
        root,
        ["Camera_01"],
        resolved=_cleanup_resolved(tmp_path),
    )

    assert result["ok"] is False
    assert result["deleted_files"] == 0
    assert transient.exists()
    assert rollback_path.is_file()
    assert (rollback_path.parent / "appearance-application.json").is_file()
    assert (root / "provenance" / "appearance_application.json").is_file()
    assert any("baseline" in problem for problem in result["gate_problems"])


def test_audit_required_pose_is_not_skipped_after_cleanup_applied(tmp_path):
    root = _write_audit_camera_fixture(tmp_path)
    (root / "dataset_manifest.json").write_text(
        json.dumps({"cleanup_status": "applied"}), encoding="utf-8"
    )

    rc = audit_main([
        "--input", str(root),
        "--expected-cameras", "1",
        "--expected-frames-per-camera", "1",
        "--validation-level", "none",
        "--render-required", "false",
        "--mask-enabled", "false",
        "--mot-required", "false",
        "--pose-skip", "true",
        "--pose-required", "true",
    ])

    report = json.loads(
        (root / "audit" / "soak_audit_report.json").read_text(encoding="utf-8")
    )
    assert rc == 1
    assert report["checks"]["runtime_pose"]["status"] == "failed"


def test_cleanup_without_audit_blocks_required_mot(tmp_path):
    root = tmp_path / "episode"
    root.mkdir()
    resolved = _cleanup_resolved(tmp_path, render_enabled=False)
    resolved["postprocess"]["formats"] = ["json", "mot"]
    resolved["ue_profile"]["annotation_export"]["export_mot"] = True

    result = apply_cleanup(root, ["Camera_01"], resolved=resolved)

    assert result["ok"] is False
    assert result["deleted_files"] == 0
    assert any("audit" in problem for problem in result["gate_problems"])


def test_cleanup_blocks_canonical_check_missing_required(tmp_path):
    root = tmp_path / "episode"
    root.mkdir()
    _write_cleanup_summary(root)
    audit = root / "audit"
    audit.mkdir()
    (audit / "soak_audit_report.json").write_text(json.dumps({
        "passed": True,
        "exit_code": 0,
        "errors": [],
        "warnings": [],
        "checks": {"mot_export": {"status": "failed"}},
    }), encoding="utf-8")

    result = apply_cleanup(root, ["Camera_01"], resolved=_cleanup_resolved(tmp_path))

    assert result["ok"] is False
    assert result["deleted_files"] == 0
    assert any("audit" in problem for problem in result["gate_problems"])


def test_postprocess_required_pose_skip_is_failure(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from grf_ue_bridge import task_requirements
    from grf_ue_bridge.workflows import task_postprocess

    monkeypatch.setattr(
        task_postprocess,
        "resolve_task_requirements",
        lambda resolved: task_requirements.TaskRequirements(requires_pose=True),
    )
    resolved = SimpleNamespace(
        task_id="pose-skip",
        dataset_episode_dir=str(tmp_path),
        trajectory_output=str(tmp_path),
        actor_mapping=str(tmp_path / "mapping.json"),
        postprocess={"yolo_pose": {"enabled": False}},
    )

    result = task_postprocess.run_postprocess(
        resolved,
        skip_cryptomatte=True,
        skip_annotate=True,
        skip_validate=True,
        skip_pose=True,
        skip_debug=True,
        print_fn=lambda message: None,
    )

    assert result == 1


def test_postprocess_forwards_resolved_requirements_to_validator(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from grf_ue_bridge.workflows import task_postprocess

    captured = {}

    def fake_annotate(*args, **kwargs):
        return 0

    def fake_validate(*args, **kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(task_postprocess, "resolve_task_requirements", lambda resolved: TaskRequirements(
        requires_mot=False,
        requires_instance_mask=True,
        requires_yolo_det=True,
        requires_yolo_seg=False,
        requires_pose=False,
    ))
    monkeypatch.setattr(
        "grf_ue_bridge.mask_annotator.annotate_masks_dir", fake_annotate
    )
    monkeypatch.setattr(
        "grf_ue_bridge.annotation_validator.validate_annotation_dir", fake_validate
    )
    resolved = SimpleNamespace(
        task_id="forwarding",
        dataset_episode_dir=str(tmp_path),
        trajectory_output=str(tmp_path),
        actor_mapping=str(tmp_path / "mapping.json"),
        postprocess={
            "formats": ["json"],
            "workers": 1,
            "chunk_size": 1,
            "clean_stale": True,
            "validation_level": "quick",
            "yolo_pose": {"enabled": False},
        },
    )

    result = task_postprocess.run_postprocess(
        resolved,
        skip_cryptomatte=True,
        skip_debug=True,
        print_fn=lambda message: None,
    )

    assert result == 0
    assert captured == {
        "workers": 1,
        "validation_level": "quick",
        "require_mot": False,
        "require_mask": True,
        "require_yolo_det": True,
        "require_yolo_seg": False,
    }


def test_cleanup_cli_apply_returns_nonzero_when_gate_blocks(tmp_path, pin_repo_root):
    from typer.testing import CliRunner
    from test_task_cli import _make_task_dir
    from grf_ue_bridge.cli import app

    task_file = _make_task_dir(tmp_path, cam_count=1, frames=1)
    episode = tmp_path / "ds" / "episode_cli_t1"
    episode.mkdir(parents=True)
    audit = episode / "audit"
    audit.mkdir()
    (audit / "soak_audit_report.json").write_text(json.dumps({
        "passed": False, "exit_code": 1, "errors": ["blocked"],
        "warnings": [], "checks": {},
    }), encoding="utf-8")

    result = CliRunner().invoke(app, ["task", "cleanup", str(task_file), "--apply"])

    assert result.exit_code == 1
    assert "blocked" in result.output


def test_cleanup_cli_without_audit_blocks_required_mot(tmp_path, pin_repo_root):
    from typer.testing import CliRunner
    from test_task_cli import _make_task_dir
    from grf_ue_bridge.cli import app

    task_file = _make_task_dir(tmp_path, cam_count=1, frames=1)
    task = json.loads(task_file.read_text(encoding="utf-8"))
    task["ue"]["annotation_export"]["export_mot"] = True
    task["postprocess"]["formats"] = ["json", "mot"]
    task_file.write_text(json.dumps(task), encoding="utf-8")
    (tmp_path / "ds" / "episode_cli_t1").mkdir(parents=True)

    result = CliRunner().invoke(app, [
        "task", "cleanup", str(task_file), "--apply"
    ])

    assert result.exit_code == 1
    assert "canonical audit" in result.output


def test_json_only_task_does_not_require_mot():
    source = {
        "ue_profile": {
            "annotation_export": {
                "render_rgb": {"enabled": True},
                "instance_mask": {"enabled": True},
                "export_mot": False,
            }
        },
        "postprocess": {"formats": ["json"]},
    }

    req = resolve_task_requirements(source)

    assert req.requires_mot is False


def test_pose_requirement_is_config_not_file_driven():
    source = {"postprocess": {"yolo_pose": {"enabled": False}}}

    assert resolve_task_requirements(source).requires_pose is False


def test_explicit_mot_format_requires_mot():
    source = {"postprocess": {"formats": ["json", "mot"]}}

    assert resolve_task_requirements(source).requires_mot is True


def test_missing_render_block_defaults_disabled():
    source = {"ue": {"annotation_export": {}}}

    assert resolve_task_requirements(source).requires_render is False


def test_requirement_resolution_uses_task_shape_and_defaults_render():
    source = {
        "ue": {
            "annotation_export": {
                "instance_mask": {"enabled": False},
                "export_mot": True,
            }
        },
        "postprocess": {},
    }

    req = resolve_task_requirements(source)

    assert isinstance(req, TaskRequirements)
    assert req.requires_render is False
    assert req.requires_instance_mask is False
    assert req.requires_mot is True
    assert req.requires_yolo_det is False
    assert req.requires_yolo_seg is False
    assert req.requires_pose is False


def test_requirement_resolution_reads_model_dump():
    class Source:
        def model_dump(self):
            return {
                "ue_profile": {
                    "annotation_export": {
                        "render_rgb": {"enabled": False},
                        "instance_mask": {"enabled": True},
                    }
                },
                "postprocess": {
                    "formats": ["json", "yolo-det", "yolo-seg"],
                    "yolo_pose": {"enabled": True},
                },
            }

    req = resolve_task_requirements(Source())

    assert req.requires_render is False
    assert req.requires_instance_mask is True
    assert req.requires_mot is False
    assert req.requires_yolo_det is True
    assert req.requires_yolo_seg is True
    assert req.requires_pose is True


def test_explicit_json_formats_override_export_mot_true():
    source = {
        "ue": {"annotation_export": {"export_mot": True}},
        "postprocess": {"formats": ["json"]},
    }

    assert resolve_task_requirements(source).requires_mot is False
