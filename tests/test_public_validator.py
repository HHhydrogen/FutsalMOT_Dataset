import json
from pathlib import Path

from PIL import Image

from grf_ue_bridge.public_validator import validate_public_episode
from grf_ue_bridge.public_episode import encode_coco_rle
from grf_ue_bridge.workflows.task_audit import main as audit_main


def _rle(height, width, start=0, length=1):
    import numpy as np
    mask = np.zeros((height, width), dtype=np.uint8)
    flat = mask.T.reshape(-1)
    flat[start:start + length] = 1
    return encode_coco_rle(mask)


def _make_public_episode(root: Path):
    cam = root / "Cam_01"
    (cam / "img1").mkdir(parents=True)
    (cam / "gt").mkdir()
    for frame_id in (1, 2):
        Image.new("RGB", (4, 3), "black").save(cam / "img1" / f"{frame_id:06d}.jpg")
    (cam / "seqinfo.ini").write_text(
        "[Sequence]\nname=Cam_01\nimDir=img1\nframeRate=30\nseqLength=2\nimWidth=4\nimHeight=3\nimExt=.jpg\n",
        encoding="utf-8",
    )
    mot = []
    mots = []
    pose = []
    for frame_id in (1, 2):
        mot.extend([
            f"{frame_id},1,0,0,2,2,1,1,1",
            f"{frame_id},100,1,2,1,1,1,100,1",
        ])
        mots.extend([
            f"{frame_id} 1 1 3 4 {json.dumps(_rle(3, 4, 0, 4), separators=(',', ':'))}",
            f"{frame_id} 100 100 3 4 {json.dumps(_rle(3, 4, 4, 1), separators=(',', ':'))}",
        ])
        pose.extend([
            {"frame_id": frame_id, "track_id": 1, "class": "player",
             "bbox": [0, 0, 2, 2], "keypoints": [[1.0, 1.0, 2]] * 17},
            {"frame_id": frame_id, "track_id": 100, "class": "ball",
             "bbox": [2, 1, 1, 1], "keypoints": None},
        ])
    (cam / "gt" / "gt.txt").write_text("\n".join(mot) + "\n", encoding="utf-8")
    (cam / "gt" / "gt_mots.txt").write_text("\n".join(mots) + "\n", encoding="utf-8")
    (cam / "gt" / "gt_pose.json").write_text(json.dumps(pose), encoding="utf-8")
    (root / "episode_manifest.json").write_text(json.dumps({
        "schema_version": "futsalmot_public_episode_v1",
        "episode_id": "episode_01", "trajectory_id": "episode_01",
        "frame_count": 2, "dimensions": {"width": 4, "height": 3},
        "sequences": [{"name": "Cam_01", "frame_count": 2, "width": 4, "height": 3}],
        "public_classes": {"player": 1, "ball": 100},
    }), encoding="utf-8")


def test_valid_public_player_ball_episode_passes(tmp_path):
    _make_public_episode(tmp_path)

    result = validate_public_episode(tmp_path)

    assert result.ok
    assert result.errors == []
    assert result.stats["sequences"] == 1
    assert result.exit_code == 0
    assert int(result) == 0


def test_public_validator_rejects_cross_modal_frame_track_mismatch(tmp_path):
    _make_public_episode(tmp_path)
    mots = tmp_path / "Cam_01" / "gt" / "gt_mots.txt"
    mots.write_text(mots.read_text(encoding="utf-8").replace("2 100", "2 99"), encoding="utf-8")

    result = validate_public_episode(tmp_path)

    assert not result.ok
    assert result.exit_code == 1
    assert any("一致" in error or "mismatch" in error.lower() for error in result.errors)


def test_public_validator_rejects_missing_jpg_and_malformed_mot(tmp_path):
    _make_public_episode(tmp_path)
    (tmp_path / "Cam_01" / "img1" / "000002.jpg").unlink()
    (tmp_path / "Cam_01" / "gt" / "gt.txt").write_text("1,1,0,0,2\n", encoding="utf-8")

    result = validate_public_episode(tmp_path)

    assert not result.ok
    assert any("JPG" in error for error in result.errors)
    assert any("MOT" in error for error in result.errors)


def test_public_validator_rejects_ball_keypoints_and_bad_rle(tmp_path):
    _make_public_episode(tmp_path)
    pose_path = tmp_path / "Cam_01" / "gt" / "gt_pose.json"
    pose = json.loads(pose_path.read_text(encoding="utf-8"))
    pose[-1]["keypoints"] = []
    pose_path.write_text(json.dumps(pose), encoding="utf-8")
    mots_path = tmp_path / "Cam_01" / "gt" / "gt_mots.txt"
    mots_path.write_text(mots_path.read_text(encoding="utf-8").replace('"size":[3,4]', '"size":[9,9]', 1), encoding="utf-8")

    result = validate_public_episode(tmp_path)

    assert not result.ok
    assert any("keypoints" in error for error in result.errors)
    assert any("MOTS" in error for error in result.errors)


def test_public_validator_rejects_manifest_sequence_mismatch(tmp_path):
    _make_public_episode(tmp_path)
    manifest_path = tmp_path / "episode_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sequences"][0]["name"] = "Cam_02"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = validate_public_episode(tmp_path)

    assert not result.ok
    assert any("目录不存在" in error for error in result.errors)


def test_public_validator_returns_result_for_unreadable_annotation_files(tmp_path):
    _make_public_episode(tmp_path)
    for name in ("gt.txt", "gt_mots.txt"):
        (tmp_path / "Cam_01" / "gt" / name).write_bytes(b"\xff\xfe")

    result = validate_public_episode(tmp_path)

    assert not result.ok
    assert isinstance(result.errors, list)


def test_public_validator_rejects_invalid_manifest_types_and_counts(tmp_path):
    _make_public_episode(tmp_path)
    manifest_path = tmp_path / "episode_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["frame_count"] = True
    manifest["dimensions"] = {"width": 0, "height": 3}
    manifest["sequences"][0]["frame_count"] = 3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = validate_public_episode(tmp_path)

    assert not result.ok
    assert any("frame_count" in error or "尺寸" in error for error in result.errors)


def test_public_validator_rejects_fractional_and_out_of_range_ids(tmp_path):
    _make_public_episode(tmp_path)
    mot = tmp_path / "Cam_01" / "gt" / "gt.txt"
    mot.write_text("1.5,1,0,0,2,2,1,1,1\n3,1,0,0,2,2,1,1,1\n", encoding="utf-8")
    pose_path = tmp_path / "Cam_01" / "gt" / "gt_pose.json"
    pose = json.loads(pose_path.read_text(encoding="utf-8"))
    pose[0]["frame_id"] = 3
    pose[0]["track_id"] = True
    pose_path.write_text(json.dumps(pose), encoding="utf-8")

    result = validate_public_episode(tmp_path)

    assert not result.ok
    assert any("非法" in error or "MOT" in error for error in result.errors)


def test_public_validator_rejects_noncanonical_img1_files_and_mots_dimensions(tmp_path):
    _make_public_episode(tmp_path)
    img1 = tmp_path / "Cam_01" / "img1"
    (img1 / "1.jpg").write_bytes((img1 / "000001.jpg").read_bytes())
    (img1 / "000003.png").write_bytes(b"stale")
    mots = tmp_path / "Cam_01" / "gt" / "gt_mots.txt"
    line = mots.read_text(encoding="utf-8").splitlines()[0]
    mots.write_text(line.replace(" 3 4 ", " 9 4 ") + "\n", encoding="utf-8")

    result = validate_public_episode(tmp_path)

    assert not result.ok
    assert any("文件名" in error or "非 JPG" in error or "MOTS" in error for error in result.errors)


def test_public_task_audit_reports_validated_camera_count(tmp_path):
    _make_public_episode(tmp_path)

    assert audit_main([
        "--input", str(tmp_path), "--expected-cameras", "4",
        "--expected-frames-per-camera", "2", "--validation-level", "none",
    ]) == 0
    report = json.loads((tmp_path / "audit" / "soak_audit_report.json").read_text(encoding="utf-8"))

    assert list(report["cameras"]) == ["Cam_01"]


def test_public_validator_handles_malformed_counts_without_type_error(tmp_path):
    _make_public_episode(tmp_path)
    manifest_path = tmp_path / "episode_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["frame_count"] = "two"
    manifest["sequences"][0]["frame_count"] = "two"
    manifest["dimensions"] = {"width": 0, "height": -1}
    manifest["sequences"] = []
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = validate_public_episode(tmp_path)

    assert not result.ok
    assert any("frame_count" in error for error in result.errors)
    assert any("尺寸" in error for error in result.errors)


def test_public_validator_handles_seqinfo_interpolation_error(tmp_path):
    _make_public_episode(tmp_path)
    (tmp_path / "Cam_01" / "seqinfo.ini").write_text(
        "[Sequence]\nname=%(missing)s\nimDir=img1\nframeRate=30\nseqLength=2\nimWidth=4\nimHeight=3\nimExt=.jpg\n",
        encoding="utf-8",
    )

    result = validate_public_episode(tmp_path)

    assert not result.ok
    assert any("seqinfo" in error for error in result.errors)


def test_public_audit_report_contains_actual_public_stats(tmp_path):
    _make_public_episode(tmp_path)

    assert audit_main([
        "--input", str(tmp_path), "--expected-cameras", "1",
        "--expected-frames-per-camera", "2", "--validation-level", "none",
    ]) == 0
    report_text = (tmp_path / "audit" / "soak_audit_report.md").read_text(encoding="utf-8")

    assert "| Cam_01 | 0 | 0 | 2 | 0 | 2 | 0 | 0 | 4 |" in report_text
