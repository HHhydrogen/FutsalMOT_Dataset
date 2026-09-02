"""公共单 episode 输出契约测试。"""

import json
from pathlib import Path

import numpy as np
import pytest

from grf_ue_bridge.public_episode import (
    build_public_manifest,
    decode_coco_rle,
    encode_coco_rle,
    public_track_id,
    write_public_episode,
)


def test_rle_round_trip_non_square_and_validation():
    mask = np.zeros((2, 4), dtype=np.uint8)
    mask[0, 1] = 1
    mask[1, 2:4] = 1
    rle = encode_coco_rle(mask)
    assert decode_coco_rle(rle, 2, 4).tolist() == mask.tolist()
    with pytest.raises(ValueError):
        decode_coco_rle(rle, 3, 4)


def test_public_ids_include_ball_and_preserve_mask_id_policy():
    assert [public_track_id(x) for x in ("L0", "L4", "R0", "R4", "BALL")] == [1, 5, 6, 10, 100]


def test_writer_emits_player_ball_and_matching_identity_sets(tmp_path, monkeypatch):
    cam = tmp_path / "Cam_01"
    (cam / "render_mask").mkdir(parents=True)
    (cam / "img1").mkdir()
    (cam / "gt").mkdir()
    (cam / "camera.json").write_text(json.dumps({
        "camera_id": "Cam_01", "image_width": 8, "image_height": 6,
        "intrinsics": {"width": 8, "height": 6, "fx": 4, "fy": 4, "cx": 4, "cy": 3},
        "extrinsics": {"world_location_m": [0, 0, 0], "forward": [1, 0, 0],
                        "right": [0, 1, 0], "up": [0, 0, 1]},
    }), encoding="utf-8")
    (cam / "annotations.jsonl").write_text(json.dumps({
        "episode_id": "ep1", "camera_id": "Cam_01", "frame_index": 1,
        "objects": [{"entity_id": "L0", "track_id": 1}, {"entity_id": "BALL", "track_id": 100}],
    }) + "\n", encoding="utf-8")
    (cam / "pose_keypoints.jsonl").write_text("\n".join([
        json.dumps({"kind": "meta", "episode_id": "ep1", "camera_id": "Cam_01", "image_width": 8, "image_height": 6}),
        json.dumps({"kind": "frame", "frame_index": 1, "objects": [{"entity_id": "L0", "track_id": 1,
            "keypoints_world": [[4, 0, 1]] * 17}, {"entity_id": "BAD", "track_id": 999,
            "keypoints_world": [[None, None, None]] * 17}]}),
    ]) + "\n", encoding="utf-8")

    mask = np.zeros((6, 8), dtype=np.uint8)
    mask[1:4, 1:3] = 1
    mask[4:6, 6:8] = 11
    monkeypatch.setattr("grf_ue_bridge.public_episode._load_mask_for_frame", lambda *_: mask)

    result = write_public_episode(tmp_path, mapping={"L0": "Actor_L0", "BALL": "Ball"},
                                  sequence_configs=[{"sequence_name": "Cam_01", "camera_dir": cam}])
    assert result["episode_id"] == "ep1"
    gt = (cam / "gt" / "gt.txt").read_text(encoding="utf-8").splitlines()
    assert gt == ["1,1,1,1,2,3,1,1,1.00", "1,100,6,4,2,2,1,100,1.00"]
    pose = json.loads((cam / "gt" / "gt_pose.json").read_text(encoding="utf-8"))
    assert {(x["frame_id"], x["track_id"]) for x in pose} == {(1, 1), (1, 100)}
    mots = (cam / "gt" / "gt_mots.txt").read_text(encoding="utf-8").splitlines()
    assert len(mots) == 2 and mots[0].split()[1:3] == ["1", "1"]
    assert not (cam / "mask").exists()


def test_manifest_trajectory_id_and_dimensions():
    manifest = build_public_manifest("ep", [{"name": "Cam_01", "frame_count": 2,
                                             "width": 8, "height": 6}], 2, 8, 6)
    assert manifest["trajectory_id"] == "ep"
    assert manifest["dimensions"] == {"width": 8, "height": 6}


def test_invalid_keypoints_are_kept_as_zero_visibility(tmp_path, monkeypatch):
    cam = tmp_path / "Cam_01"
    cam.mkdir()
    (cam / "camera.json").write_text(json.dumps({"camera_id": "Cam_01", "image_width": 4, "image_height": 4,
        "intrinsics": {"width": 4, "height": 4, "fx": 2, "fy": 2, "cx": 2, "cy": 2},
        "extrinsics": {"world_location_m": [0, 0, 0], "forward": [1, 0, 0], "right": [0, 1, 0], "up": [0, 0, 1]}}), encoding="utf-8")
    (cam / "annotations.jsonl").write_text(json.dumps({"episode_id": "ep", "frame_index": 1,
        "objects": [{"entity_id": "L0"}]}) + "\n", encoding="utf-8")
    (cam / "pose_keypoints.jsonl").write_text(json.dumps({"kind": "frame", "frame_index": 1,
        "objects": [{"entity_id": "L0", "keypoints_world": [[None, 0, 1]] + [[4, 0, 1]] * 16}]}) + "\n", encoding="utf-8")
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0:2, 0:2] = 1
    monkeypatch.setattr("grf_ue_bridge.public_episode._load_mask_for_frame", lambda *_: mask)
    write_public_episode(tmp_path, mapping={"L0": "Actor_L0"}, sequence_configs=[{"camera_dir": cam}])
    pose = json.loads((cam / "gt" / "gt_pose.json").read_text(encoding="utf-8"))[0]
    assert pose["keypoints"][2] == 0


def test_invisible_frame_writes_empty_canonical_files(tmp_path, monkeypatch):
    cam = tmp_path / "Cam_01"
    cam.mkdir()
    (cam / "camera.json").write_text(json.dumps({"image_width": 3, "image_height": 2,
        "intrinsics": {"width": 3, "height": 2, "fx": 2, "fy": 2, "cx": 1, "cy": 1}, "extrinsics": {}}), encoding="utf-8")
    (cam / "annotations.jsonl").write_text(json.dumps({"episode_id": "ep", "frame_index": 1, "objects": []}) + "\n", encoding="utf-8")
    monkeypatch.setattr("grf_ue_bridge.public_episode._load_mask_for_frame", lambda *_: np.zeros((2, 3), dtype=np.uint8))
    write_public_episode(tmp_path, mapping={"L0": "Actor_L0"}, sequence_configs=[{"camera_dir": cam}])
    assert (cam / "gt" / "gt.txt").read_text(encoding="utf-8") == ""
    assert json.loads((cam / "gt" / "gt_pose.json").read_text(encoding="utf-8")) == []
