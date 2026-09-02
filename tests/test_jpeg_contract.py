import json
from types import SimpleNamespace
from pathlib import Path

from PIL import Image

from grf_ue_bridge.config.models import ResolvedTask
from grf_ue_bridge.workflows.task_audit import check_camera, write_reports
from grf_ue_bridge.workflows.task_status import collect_status
from grf_ue_bridge.workflows.artifact_cleanup import collect_transient


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
