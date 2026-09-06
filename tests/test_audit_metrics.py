"""Audit 轻量 metrics 测试。"""

from __future__ import annotations

import json

from grf_ue_bridge.validation_result import validation_result_from_report
from grf_ue_bridge.workflows.audit_metrics import calculate_metrics
from grf_ue_bridge.workflows.task_audit import main as audit_main


def _camera(root, name="Camera_01"):
    camera = root / name
    (camera / "img1").mkdir(parents=True)
    (camera / "mask").mkdir()
    (camera / "gt").mkdir()
    (camera / "camera.json").write_text(json.dumps({
        "image_width": 8,
        "image_height": 8,
        "intrinsics": {"width": 8, "height": 8, "fx": 5.0, "fy": 5.0, "cx": 4.0, "cy": 4.0},
        "extrinsics": {
            "world_location_m": [0.0, 0.0, 1.0],
            "forward": [1.0, 0.0, 0.0],
            "right": [0.0, 1.0, 0.0],
            "up": [0.0, 0.0, 1.0],
        },
    }), encoding="utf-8")
    return camera


def test_dataset_metrics_count_images_cameras_and_frames(tmp_path):
    camera = _camera(tmp_path)
    (camera / "img1" / "000001.png").write_bytes(b"x")
    (camera / "img1" / "000002.png").write_bytes(b"x")
    (camera / "annotations.jsonl").write_text("{}\n{}\n", encoding="utf-8")

    metrics = calculate_metrics(tmp_path)

    assert metrics["dataset"] == {
        "frame_count": 2,
        "image_count": 2,
        "camera_count": 1,
    }


def test_mot_metrics_count_tracks_and_frames(tmp_path):
    camera = _camera(tmp_path)
    (camera / "gt" / "gt.txt").write_text(
        "1,10,1,1,2,2,1,1,1\n2,10,1,1,2,2,1,1,1\n2,20,1,1,2,2,1,1,1\n",
        encoding="utf-8",
    )

    metrics = calculate_metrics(tmp_path)

    assert metrics["mot"]["track_count"] == 2
    assert metrics["mot"]["frame_count"] == 2
    assert metrics["mot"]["avg_track_length"] == 1.5


def test_mask_metrics_count_nonzero_instances(tmp_path):
    camera = _camera(tmp_path)
    (camera / "mask" / "000001.png").write_bytes(b"not-decoded")

    metrics = calculate_metrics(tmp_path)

    assert "mask" in metrics
    assert metrics["mask"]["instance_count"] is None
    assert metrics["errors"]


def test_mask_metrics_count_unique_nonzero_ids(tmp_path):
    from PIL import Image

    camera = _camera(tmp_path)
    Image.new("L", (2, 2), color=1).save(camera / "mask" / "000001.png")
    image = Image.new("L", (2, 2), color=2)
    image.save(camera / "mask" / "000002.png")

    metrics = calculate_metrics(tmp_path)

    assert metrics["mask"]["instance_count"] == 2


def test_metric_failure_does_not_change_validation_result(tmp_path):
    report = {
        "passed": True,
        "exit_code": 0,
        "errors": [],
        "warnings": [],
        "checks": {},
        "metrics": {"errors": ["failed to calculate mot statistics"]},
    }

    result = validation_result_from_report(report)

    assert result.passed is True


def test_legacy_audit_report_without_metrics_remains_readable():
    report = {
        "passed": True,
        "exit_code": 0,
        "errors": [],
        "warnings": [],
        "checks": {},
    }

    result = validation_result_from_report(report)

    assert result.passed is True


def test_audit_report_contains_metrics_without_changing_checks(tmp_path):
    camera = _camera(tmp_path)
    (camera / "img1" / "000001.png").write_bytes(b"x")
    (camera / "annotations.jsonl").write_text(
        json.dumps({"frame_index": 1, "objects": []}) + "\n", encoding="utf-8"
    )

    rc = audit_main([
        "--input", str(tmp_path),
        "--expected-cameras", "1",
        "--expected-frames-per-camera", "1",
        "--validation-level", "none",
        "--render-required", "false",
        "--mask-enabled", "false",
        "--mot-required", "false",
        "--pose-required", "false",
    ])

    report = json.loads(
        (tmp_path / "audit" / "soak_audit_report.json").read_text(encoding="utf-8")
    )
    assert rc == 0
    assert report["passed"] is True
    assert report["metrics"]["dataset"]["image_count"] == 1
    assert report["checks"]["render"]["status"] == "skipped"
