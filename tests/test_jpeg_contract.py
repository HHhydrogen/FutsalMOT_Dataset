import json
from types import SimpleNamespace
from pathlib import Path

import pytest
from PIL import Image

from grf_ue_bridge.config.models import ResolvedTask
from grf_ue_bridge.workflows.task_audit import check_camera, write_reports
from grf_ue_bridge.workflows.task_status import collect_status
from grf_ue_bridge.workflows.artifact_cleanup import apply_cleanup, collect_transient, plan_cleanup
from grf_ue_bridge.public_episode import encode_coco_rle


def _camera(root: Path) -> Path:
    cam = root / "Cam_01"
    cam.mkdir()
    (cam / "camera.json").write_text(json.dumps({"image_width": 2, "image_height": 1}))
    (cam / "seqinfo.ini").write_text("[Sequence]\n")
    (cam / "annotations.jsonl").write_text(json.dumps({"frame_index": 1}) + "\n")
    (cam / "img1").mkdir()
    Image.new("RGB", (2, 1), "black").save(cam / "img1" / "000001.jpg")
    (cam / "mask").mkdir()
    Image.new("L", (2, 1), 0).save(cam / "mask" / "000001.png")
    (cam / "render" / "nested").mkdir(parents=True)
    for suffix in ("png", "jpg", "jpeg"):
        Image.new("RGB", (2, 1), "black").save(cam / "render" / "nested" / f"000000.{suffix}")
    return cam


def test_task_status_counts_public_jpeg_and_all_transient_rgb(tmp_path):
    cam = _camera(tmp_path)
    (cam / "render_mask").mkdir()
    resolved = ResolvedTask(
        task_id="task", episode_name="episode", source_task_file="task.json",
        repo_root=str(tmp_path), ue_project_root=str(tmp_path), dataset_root=str(tmp_path),
        trajectory_output=str(tmp_path / "trajectory"), dataset_episode_dir=str(tmp_path),
    )

    status = collect_status(resolved)

    assert status["cameras"]["Cam_01"]["img1"] == 1
    assert status["cameras"]["Cam_01"]["render_rgb"] == 3


def test_task_status_counts_legacy_img1_suffixes(tmp_path):
    cam = _camera(tmp_path)
    for suffix in ("png", "jpeg"):
        Image.new("RGB", (2, 1), "black").save(cam / "img1" / f"000002.{suffix}")
    resolved = ResolvedTask(
        task_id="task", episode_name="episode", source_task_file="task.json",
        repo_root=str(tmp_path), ue_project_root=str(tmp_path), dataset_root=str(tmp_path),
        trajectory_output=str(tmp_path / "trajectory"), dataset_episode_dir=str(tmp_path),
    )
    assert collect_status(resolved)["cameras"]["Cam_01"]["img1"] == 3


def test_task_audit_counts_public_jpeg_and_transient_rgb(tmp_path):
    cam = _camera(tmp_path)
    errors = []
    stats = check_camera(cam, 1, [0], errors, [], mask_enabled=False)

    assert stats["img1_rgb"] == 1
    assert stats["render_rgb"] == 3
    assert not errors


def test_audit_markdown_uses_current_rgb_statistic_keys(tmp_path):
    args = SimpleNamespace(expected_cameras=1, expected_frames_per_camera=1)
    report = {
        "exit_code": 0, "passed": True, "errors": [], "warnings": [],
        "cameras": {"Cam_01": {
            "render_rgb": 7, "render_mask_exr": 2, "img1_rgb": 3,
            "mask_png": 3, "annotations_frames": 3, "det_txt": 3,
        "seg_txt": 3, "gt_txt_lines": 3, "img1_missing": [],
        "img1_dup": [], "zero_byte": 0,
        }},
        "sync": {}, "mapping": {}, "calibration": {},
        "render_summary": {}, "pose_coco17": {}, "cross_camera_identity": {},
        "validation": None,
    }
    _, markdown = write_reports(tmp_path, report, tmp_path / "audit", args)
    text = markdown.read_text(encoding="utf-8")
    assert "| Cam_01 | 7 | 2 | 3 | 3 |" in text


def test_cleanup_collects_all_yolo_rgb_suffixes(tmp_path):
    _camera(tmp_path)
    yolo = tmp_path / "yolo_pose" / "images"
    yolo.mkdir(parents=True)
    for suffix in ("png", "jpg", "jpeg"):
        (yolo / f"000001.{suffix}").write_bytes(b"x")

    transient = collect_transient(tmp_path, ["Cam_01"])

    assert all(str(yolo / f"000001.{suffix}") in transient for suffix in ("png", "jpg", "jpeg"))


def _write_real_public_fixture(root: Path):
    cam = root / "FutsalMOT_episode_01_C01"
    (cam / "img1").mkdir(parents=True)
    (cam / "gt").mkdir()
    Image.new("RGB", (2, 2), "black").save(cam / "img1" / "000001.jpg")
    (cam / "seqinfo.ini").write_text(
        "[Sequence]\nname=FutsalMOT_episode_01_C01\nimDir=img1\nframeRate=30\nseqLength=1\nimWidth=2\nimHeight=2\nimExt=.jpg\n",
        encoding="utf-8",
    )
    mot = "1,1,0,0,1,1,1,1,-1\n1,100,1,1,1,1,1,2,-1\n"
    (cam / "gt" / "gt.txt").write_text(mot, encoding="utf-8")
    mask = encode_coco_rle(__import__("numpy").array([[1, 0], [0, 11]], dtype="uint8"))
    mots = "1 1 1 2 2 " + mask["counts"] + "\n"
    mots += "1 100 2 2 2 " + encode_coco_rle(__import__("numpy").array([[0, 0], [0, 1]], dtype="uint8"))["counts"] + "\n"
    (cam / "gt" / "gt_mots.txt").write_text(mots, encoding="utf-8")
    (cam / "gt" / "gt_pose.json").write_text(json.dumps([
        {"frame_id": 1, "track_id": 1, "class_id": 1, "class_name": "player", "class": "player", "bbox": [0, 0, 1, 1], "keypoints": [[0, 0, 2]] * 17},
        {"frame_id": 1, "track_id": 100, "class_id": 2, "class_name": "ball", "class": "ball", "bbox": [1, 1, 1, 1], "keypoints": None},
    ]), encoding="utf-8")
    (root / "episode_manifest.json").write_text(json.dumps({
        "schema_version": 1, "episode_id": "episode_01",
        "trajectory_id": "episode_01",
        "sequences": [{"sequence_name": "FutsalMOT_episode_01_C01", "camera_id": "C01",
                       "relative_path": "FutsalMOT_episode_01_C01", "frame_count": 1,
                       "image_width": 2, "image_height": 2,
                       "modalities": ["mot", "pose_tracking", "mots"]}],
        "public_classes": ["player", "ball"], "class_id_policy": {"player": 1, "ball": 2},
        "track_id_policy": {"players": "L0..L4=1..5,R0..R4=6..10", "ball": 100},
    }), encoding="utf-8")
    (root / "render_summary.json").write_text(json.dumps({"status": "success"}), encoding="utf-8")
    (root / "pose_session.json").write_text(json.dumps({"capture_complete": True}), encoding="utf-8")
    return cam


def test_cleanup_preserves_public_outputs_and_removes_render_after_public_validation(tmp_path):
    cam = _write_real_public_fixture(tmp_path)
    (cam / "render_mask").mkdir()
    (cam / "render_mask" / "000000.exr").write_bytes(b"exr")
    report = plan_cleanup(tmp_path, ["FutsalMOT_episode_01_C01"])
    assert str(cam / "render_mask" / "000000.exr") in report["would_delete"]
    assert str(cam / "img1" / "000001.jpg") not in report["would_delete"]
    result = apply_cleanup(tmp_path, ["FutsalMOT_episode_01_C01"])
    assert result["ok"]
    assert not (cam / "render_mask").exists()
    assert (cam / "img1" / "000001.jpg").exists()
    assert (cam / "gt" / "gt_pose.json").exists()


def test_cleanup_blocks_when_public_validation_fails(tmp_path, monkeypatch):
    _write_real_public_fixture(tmp_path)
    (tmp_path / "FutsalMOT_episode_01_C01" / "render_mask").mkdir()
    (tmp_path / "FutsalMOT_episode_01_C01" / "render_mask" / "000000.exr").write_bytes(b"exr")
    calls = []

    def reject_public_output(episode_dir, resolved_task=None):
        calls.append(resolved_task)
        return type("Result", (), {"ok": False, "errors": ["bad public output"]})()

    monkeypatch.setattr("grf_ue_bridge.public_validator.validate_public_episode", reject_public_output)
    result = apply_cleanup(tmp_path, ["FutsalMOT_episode_01_C01"])
    assert not result["ok"]
    assert result["reason"] == "validation_gate_failed"
    assert calls == [None]


def test_cleanup_blocks_real_public_fixture_when_audit_report_fails(tmp_path):
    _write_real_public_fixture(tmp_path)
    cam = tmp_path / "FutsalMOT_episode_01_C01"
    (cam / "render_mask").mkdir()
    (cam / "render_mask" / "000000.exr").write_bytes(b"exr")
    audit = tmp_path / "audit"
    audit.mkdir()
    (audit / "soak_audit_report.json").write_text(json.dumps({"ok": False, "failed_checks": ["rgb"]}), encoding="utf-8")
    result = apply_cleanup(tmp_path, ["FutsalMOT_episode_01_C01"])
    assert not result["ok"]
    assert (cam / "render_mask" / "000000.exr").exists()
    assert any("audit" in problem for problem in result["gate_problems"])


def _write_resolved_cleanup_fixture(root: Path, annotations):
    cam = root / "FutsalMOT_episode_01_C01"
    (cam / "img1").mkdir(parents=True)
    Image.new("RGB", (2, 2), "black").save(cam / "img1" / "000001.jpg")
    (cam / "render").mkdir()
    (cam / "render" / "000000.jpg").write_bytes(b"render")
    (root / "render_summary.json").write_text(json.dumps({"status": "success"}), encoding="utf-8")
    return cam, ResolvedTask(
        task_id="task", episode_name="episode", source_task_file="task.json",
        repo_root=str(root), ue_project_root=str(root), dataset_root=str(root),
        trajectory_output=str(root / "trajectory"), dataset_episode_dir=str(root),
        config_v3={
            "annotations": annotations,
            "classes": ["player", "ball"],
        }
    )


def test_cleanup_allows_resolved_mot_without_pose_files(tmp_path):
    cam, resolved = _write_resolved_cleanup_fixture(tmp_path, ["mot"])

    report = plan_cleanup(tmp_path, [cam.name], resolved=resolved)
    result = apply_cleanup(tmp_path, [cam.name], resolved=resolved)

    assert report["gate_ok"]
    assert result["ok"]
    assert not (cam / "render").exists()


def test_cleanup_allows_resolved_mot_mots_without_pose_files(tmp_path):
    cam, resolved = _write_resolved_cleanup_fixture(tmp_path, ["mot", "mots"])

    report = plan_cleanup(tmp_path, [cam.name], resolved=resolved)
    result = apply_cleanup(tmp_path, [cam.name], resolved=resolved)

    assert report["gate_ok"]
    assert result["ok"]
    assert not (cam / "render").exists()


def test_cleanup_blocks_legacy_resolved_task_without_render_or_pose_files(tmp_path):
    cam = tmp_path / "Cam_01"
    (cam / "render").mkdir(parents=True)
    (cam / "render" / "000000.jpg").write_bytes(b"render")
    resolved = ResolvedTask(
        task_id="task", episode_name="episode", source_task_file="task.json",
        repo_root=str(tmp_path), ue_project_root=str(tmp_path), dataset_root=str(tmp_path),
        trajectory_output=str(tmp_path / "trajectory"), dataset_episode_dir=str(tmp_path),
    )

    result = apply_cleanup(tmp_path, [cam.name], resolved=resolved)

    assert not result["ok"]
    assert any("render_summary" in problem for problem in result["gate_problems"])
    assert any("pose_session" in problem for problem in result["gate_problems"])
    assert (cam / "render" / "000000.jpg").exists()


def test_cleanup_blocks_resolved_task_without_v3_contract(tmp_path):
    result = apply_cleanup(tmp_path, [], resolved={"task_id": "legacy"})

    assert not result["ok"]
    assert any("render_summary" in problem for problem in result["gate_problems"])
    assert any("pose_session" in problem for problem in result["gate_problems"])


def test_cleanup_blocks_v3_pose_without_pose_session(tmp_path):
    cam, resolved = _write_resolved_cleanup_fixture(tmp_path, ["pose"])

    result = apply_cleanup(tmp_path, [cam.name], resolved=resolved)

    assert not result["ok"]
    assert any("pose_session" in problem for problem in result["gate_problems"])
    assert (cam / "render" / "000000.jpg").exists()


def _write_real_v3_cleanup_fixture(root: Path, annotations):
    cam = _write_real_public_fixture(root)
    manifest_path = root / "episode_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sequences"][0]["modalities"] = annotations
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    if "mots" not in annotations:
        (cam / "gt" / "gt_mots.txt").unlink()
    if "pose" not in annotations:
        (cam / "gt" / "gt_pose.json").unlink()
        (root / "pose_session.json").unlink()
    (cam / "render").mkdir()
    (cam / "render" / "000000.jpg").write_bytes(b"render")
    resolved = ResolvedTask(
        task_id="task", episode_name="episode_01", source_task_file="task.json",
        repo_root=str(root), ue_project_root=str(root), dataset_root=str(root),
        trajectory_output=str(root / "trajectory"), dataset_episode_dir=str(root),
        config_v3={"annotations": annotations, "classes": ["player", "ball"]},
    )
    return cam, resolved


@pytest.mark.parametrize("annotations", [["mot"], ["mot", "mots"]])
def test_cleanup_real_v3_public_outputs_passes_without_pose_session(
    tmp_path, monkeypatch, annotations
):
    cam, resolved = _write_real_v3_cleanup_fixture(tmp_path, annotations)
    from grf_ue_bridge import public_validator

    calls = []
    real_validator = public_validator.validate_public_episode

    def validate_with_record(episode_dir, resolved_task=None):
        calls.append(resolved_task)
        return real_validator(episode_dir, resolved_task=resolved_task)

    monkeypatch.setattr(public_validator, "validate_public_episode", validate_with_record)
    result = apply_cleanup(tmp_path, [cam.name], resolved=resolved)

    assert result["ok"]
    assert calls == [resolved]
    assert not (cam / "render").exists()
    assert not (tmp_path / "pose_session.json").exists()


def test_cleanup_real_v3_pose_public_output_blocks_without_pose_session(tmp_path, monkeypatch):
    cam, resolved = _write_real_v3_cleanup_fixture(tmp_path, ["pose"])
    (tmp_path / "pose_session.json").unlink()
    from grf_ue_bridge import public_validator

    calls = []
    real_validator = public_validator.validate_public_episode

    def validate_with_record(episode_dir, resolved_task=None):
        calls.append(resolved_task)
        return real_validator(episode_dir, resolved_task=resolved_task)

    # Public GT validation also sees the requested Pose modality and fails first.
    monkeypatch.setattr(public_validator, "validate_public_episode", validate_with_record)
    result = apply_cleanup(tmp_path, [cam.name], resolved=resolved)

    assert not result["ok"]
    assert calls == [resolved]
    assert any("pose_session" in problem for problem in result["gate_problems"])
    assert (cam / "render" / "000000.jpg").exists()
