import json
import shutil
from pathlib import Path

import pytest

from grf_ue_bridge import annotation_validator as annotation_validator_module
from grf_ue_bridge.annotation_validator import validate_annotation_result
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
    rc = audit_main(args)
    assert rc in (0, 1)
    return json.loads(
        (root / "audit" / "soak_audit_report.json").read_text(encoding="utf-8")
    )


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
        "postprocess": {"formats": ["json"], "yolo_pose": {"enabled": pose_enabled}},
        "ue_profile": {"annotation_export": {
            "render_rgb": {"enabled": render_enabled},
            "instance_mask": {"enabled": False},
            "export_mot": False,
            "cameras": ["Camera_01"],
        }},
        "artifact_policy": {"profile": "research_minimal"},
    }


def _write_cleanup_summary(root: Path, status="success"):
    (root / "render_summary.json").write_text(
        json.dumps({"status": status, "cameras": {"Camera_01": {"ok": True}}}),
        encoding="utf-8",
    )


def test_cleanup_disabled_pose_missing_session_is_not_a_gate(tmp_path):
    root = tmp_path / "episode"
    root.mkdir()
    _write_cleanup_summary(root)
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
