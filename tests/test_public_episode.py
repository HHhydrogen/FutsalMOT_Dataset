"""公共单 episode 输出契约测试。"""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from grf_ue_bridge.public_episode import (
    build_public_manifest,
    decode_coco_rle,
    encode_coco_rle,
    public_track_id,
    write_public_episode,
)

PUBLIC_MAPPING = {**{"L%d" % i: "Actor_L%d" % i for i in range(5)},
                  **{"R%d" % i: "Actor_R%d" % i for i in range(5)}, "BALL": "Ball"}


def test_rle_round_trip_non_square_and_validation():
    mask = np.zeros((2, 4), dtype=np.uint8)
    mask[0, 1] = 1
    mask[1, 2:4] = 1
    rle = encode_coco_rle(mask)
    assert decode_coco_rle(rle, 2, 4).tolist() == mask.tolist()
    with pytest.raises(ValueError):
        decode_coco_rle(rle, 3, 4)
    with pytest.raises(ValueError):
        decode_coco_rle({"size": [2, 4], "counts": ""}, 2, 4)


def test_package_import_works_without_ue_path_bootstrap(monkeypatch):
    import grf_ue_bridge.public_episode as module
    assert module.entity_id_to_mask_id("BALL") == 11


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

    result = write_public_episode(tmp_path, mapping=PUBLIC_MAPPING,
                                  sequence_configs=[{"sequence_name": "Cam_01", "camera_dir": cam}])
    assert result["episode_id"] == "ep1"
    public_cam = tmp_path / "FutsalMOT_ep1_C01"
    gt = (public_cam / "gt" / "gt.txt").read_text(encoding="utf-8").splitlines()
    assert gt == ["1,1,1,1,2,3,1,1,-1", "1,100,6,4,2,2,1,2,-1"]
    pose = json.loads((public_cam / "gt" / "gt_pose.json").read_text(encoding="utf-8"))
    assert {(x["frame_id"], x["track_id"], x["class_id"]) for x in pose} == {(1, 1, 1), (1, 100, 2)}
    assert pose[1]["class_name"] == "ball"
    assert pose[1]["keypoints"] is None
    mots = (public_cam / "gt" / "gt_mots.txt").read_text(encoding="utf-8").splitlines()
    assert len(mots) == 2 and mots[0].split()[1:3] == ["1", "1"]
    assert len(mots[0].split()) == 6
    assert mots[0].split()[-1] == encode_coco_rle(mask == 1)["counts"]
    assert not (public_cam / "mask").exists()


@pytest.mark.parametrize(
    "annotations, classes, expected_files, expected_tracks",
    [
        (["mot"], ["player", "ball"], {"gt.txt"}, {1, 100}),
        (["pose"], ["player", "ball"], {"gt_pose.json"}, {1, 100}),
        (["mots"], ["player", "ball"], {"gt_mots.txt"}, {1, 100}),
        (["mot", "pose", "mots"], ["player"], {"gt.txt", "gt_pose.json", "gt_mots.txt"}, {1}),
        (["mot", "mots"], ["player", "ball"], {"gt.txt", "gt_mots.txt"}, {1, 100}),
    ],
)
def test_writer_emits_only_requested_modalities_and_classes(
    tmp_path, monkeypatch, annotations, classes, expected_files, expected_tracks
):
    cam = tmp_path / "Cam_01"
    (cam / "img1").mkdir(parents=True)
    (cam / "camera.json").write_text(json.dumps({
        "image_width": 2, "image_height": 2,
        "intrinsics": {"width": 2, "height": 2, "fx": 1, "fy": 1, "cx": 1, "cy": 1},
        "extrinsics": {},
    }), encoding="utf-8")
    (cam / "annotations.jsonl").write_text(json.dumps({
        "episode_id": "ep", "frame_index": 1,
        "objects": [{"entity_id": "L0"}, {"entity_id": "BALL"}],
    }) + "\n", encoding="utf-8")
    (cam / "pose_keypoints.jsonl").write_text(json.dumps({
        "kind": "frame", "frame_index": 1,
        "objects": [{"entity_id": "L0", "keypoints_world": [[1, 0, 1]] * 17}],
    }) + "\n", encoding="utf-8")
    Image.new("RGB", (2, 2), "black").save(cam / "img1" / "000001.jpg")
    monkeypatch.setattr(
        "grf_ue_bridge.public_episode._load_mask_for_frame",
        lambda *_: np.array([[1, 0], [0, 11]], dtype=np.uint8),
    )

    manifest = write_public_episode(
        tmp_path, mapping=PUBLIC_MAPPING,
        sequence_configs=[{"camera_dir": cam}],
        annotations=annotations, classes=classes,
    )

    gt_dir = tmp_path / "FutsalMOT_ep_C01" / "gt"
    assert {path.name for path in gt_dir.iterdir()} == expected_files
    if "mot" in annotations:
        tracks = {int(line.split(",")[1]) for line in (gt_dir / "gt.txt").read_text().splitlines()}
        assert tracks == expected_tracks
    if "mots" in annotations:
        tracks = {int(line.split()[1]) for line in (gt_dir / "gt_mots.txt").read_text().splitlines()}
        assert tracks == expected_tracks
    if "pose" in annotations:
        tracks = {row["track_id"] for row in json.loads((gt_dir / "gt_pose.json").read_text())}
        assert tracks == expected_tracks
    assert manifest["public_classes"] == classes
    expected_modalities = ["pose_tracking" if value == "pose" else value for value in annotations]
    assert manifest["sequences"][0]["modalities"] == expected_modalities


def test_manifest_trajectory_id_and_approved_policy():
    manifest = build_public_manifest("ep", [{"name": "Cam_01", "frame_count": 2,
                                             "width": 8, "height": 6}], 2, 8, 6)
    assert manifest["trajectory_id"] == "ep"
    assert "frame_count" not in manifest
    assert "dimensions" not in manifest
    assert manifest["public_classes"] == ["player", "ball"]
    assert manifest["track_id_policy"] == {"players": "L0..L4=1..5,R0..R4=6..10", "ball": 100}
    assert manifest["class_id_policy"] == {"player": 1, "ball": 2}


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
    write_public_episode(tmp_path, mapping=PUBLIC_MAPPING, sequence_configs=[{"camera_dir": cam}])
    pose = json.loads((tmp_path / "FutsalMOT_ep_C01" / "gt" / "gt_pose.json").read_text(encoding="utf-8"))[0]
    assert pose["keypoints"][0][2] == 0


def test_offscreen_keypoints_are_zero_visibility(tmp_path, monkeypatch):
    cam = tmp_path / "Cam_01"
    cam.mkdir()
    (cam / "camera.json").write_text(json.dumps({"image_width": 4, "image_height": 4,
        "intrinsics": {"width": 4, "height": 4, "fx": 2, "fy": 2, "cx": 2, "cy": 2},
        "extrinsics": {"world_location_m": [0, 0, 0], "forward": [1, 0, 0], "right": [0, 1, 0], "up": [0, 0, 1]}}), encoding="utf-8")
    (cam / "annotations.jsonl").write_text(json.dumps({"episode_id": "ep", "frame_index": 1, "objects": [{"entity_id": "L0"}]}) + "\n", encoding="utf-8")
    (cam / "pose_keypoints.jsonl").write_text(json.dumps({"kind": "frame", "frame_index": 1,
        "objects": [{"entity_id": "L0", "keypoints_world": [[4, 100, 1]] * 17}]}) + "\n", encoding="utf-8")
    mask = np.zeros((4, 4), dtype=np.uint8); mask[0, 0] = 1
    monkeypatch.setattr("grf_ue_bridge.public_episode._load_mask_for_frame", lambda *_: mask)
    write_public_episode(tmp_path, mapping=PUBLIC_MAPPING, sequence_configs=[{"camera_dir": cam}])
    assert json.loads((tmp_path / "FutsalMOT_ep_C01" / "gt" / "gt_pose.json").read_text(encoding="utf-8"))[0]["keypoints"][0][2] == 0


def test_invisible_frame_writes_empty_canonical_files(tmp_path, monkeypatch):
    cam = tmp_path / "Cam_01"
    cam.mkdir()
    (cam / "camera.json").write_text(json.dumps({"image_width": 3, "image_height": 2,
        "intrinsics": {"width": 3, "height": 2, "fx": 2, "fy": 2, "cx": 1, "cy": 1}, "extrinsics": {}}), encoding="utf-8")
    (cam / "annotations.jsonl").write_text(json.dumps({"episode_id": "ep", "frame_index": 1, "objects": []}) + "\n", encoding="utf-8")
    monkeypatch.setattr("grf_ue_bridge.public_episode._load_mask_for_frame", lambda *_: np.zeros((2, 3), dtype=np.uint8))
    write_public_episode(tmp_path, mapping=PUBLIC_MAPPING, sequence_configs=[{"camera_dir": cam}])
    public_cam = tmp_path / "FutsalMOT_ep_C01"
    assert (public_cam / "gt" / "gt.txt").read_text(encoding="utf-8") == ""
    assert json.loads((public_cam / "gt" / "gt_pose.json").read_text(encoding="utf-8")) == []


def test_sequence_order_and_bbox_are_deterministic(tmp_path, monkeypatch):
    cams = []
    for name, x in (("Cam_B", 3), ("Cam_A", 0)):
        cam = tmp_path / name; cam.mkdir(); cams.append(cam)
        (cam / "camera.json").write_text(json.dumps({"image_width": 4, "image_height": 3,
            "intrinsics": {"width": 4, "height": 3, "fx": 2, "fy": 2, "cx": 2, "cy": 1}, "extrinsics": {}}), encoding="utf-8")
        (cam / "annotations.jsonl").write_text(json.dumps({"episode_id": "ep", "frame_index": 1, "objects": [{"entity_id": "L0"}]}) + "\n", encoding="utf-8")
    def mask_loader(camera_dir, frame_id, mapping, *args, **kwargs):
        m = np.zeros((3, 4), dtype=np.uint8); m[:, 3 if camera_dir.name == "Cam_B" else 0] = 1; return m
    monkeypatch.setattr("grf_ue_bridge.public_episode._load_mask_for_frame", mask_loader)
    result = write_public_episode(tmp_path, mapping=PUBLIC_MAPPING, sequence_configs=[{"camera_dir": cams[0]}, {"camera_dir": cams[1]}])
    assert [s["sequence_name"] for s in result["sequences"]] == ["FutsalMOT_ep_C01", "FutsalMOT_ep_C02"]


def test_writer_preserves_explicit_camera_identity_and_public_sequence_name(tmp_path, monkeypatch):
    cams = []
    for name in ("FrontCamera", "GoalCamera"):
        cam = tmp_path / name
        cam.mkdir()
        (cam / "img1").mkdir()
        (cam / "camera.json").write_text(json.dumps({
            "image_width": 2, "image_height": 2,
            "intrinsics": {"width": 2, "height": 2, "fx": 1, "fy": 1, "cx": 1, "cy": 1},
            "extrinsics": {},
        }), encoding="utf-8")
        (cam / "annotations.jsonl").write_text(json.dumps({
            "episode_id": "ep", "frame_index": 1, "objects": [{"entity_id": "L0"}]
        }) + "\n", encoding="utf-8")
        Image.new("RGB", (2, 2), "red").save(cam / "img1" / "000001.jpg")
        cams.append(cam)
    monkeypatch.setattr("grf_ue_bridge.public_episode._load_mask_for_frame",
                        lambda *_: np.array([[1, 0], [0, 0]], dtype=np.uint8))

    result = write_public_episode(tmp_path, mapping=PUBLIC_MAPPING, sequence_configs=[
        {"camera_id": "C07", "camera_actor": "GoalCamera", "public_sequence_name": "FutsalMOT_ep_C07", "camera_dir": cams[1]},
        {"camera_id": "C03", "camera_actor": "FrontCamera", "public_sequence_name": "FutsalMOT_ep_C03", "camera_dir": cams[0]},
    ])

    assert [item["camera_id"] for item in result["sequences"]] == ["C03", "C07"]
    assert [item["sequence_name"] for item in result["sequences"]] == [
        "FutsalMOT_ep_C03", "FutsalMOT_ep_C07"
    ]
    for camera_id in ("C03", "C07"):
        public_dir = tmp_path / f"FutsalMOT_ep_{camera_id}"
        assert public_dir.is_dir()
        assert f"name=FutsalMOT_ep_{camera_id}" in (public_dir / "seqinfo.ini").read_text()
    manifest = json.loads((tmp_path / "episode_manifest.json").read_text())
    assert [item["sequence_name"] for item in manifest["sequences"]] == [
        "FutsalMOT_ep_C03", "FutsalMOT_ep_C07"
    ]


def test_writer_rejects_duplicate_camera_actor_values(tmp_path):
    with pytest.raises(ValueError, match="camera_actor"):
        write_public_episode(tmp_path, mapping=PUBLIC_MAPPING, sequence_configs=[
            {"camera_id": "C03", "camera_actor": "SameCamera", "camera_dir": tmp_path / "a"},
            {"camera_id": "C07", "camera_actor": "SameCamera", "camera_dir": tmp_path / "b"},
        ])


def test_mapping_rejects_missing_extra_and_invalid_entities(tmp_path):
    with pytest.raises(ValueError):
        write_public_episode(tmp_path, mapping={"L0": "x"}, sequence_configs=[])
    with pytest.raises(ValueError):
        write_public_episode(tmp_path, mapping={**{"L%d" % i: "x" for i in range(5)}, **{"R%d" % i: "x" for i in range(5)}, "BALL": "x", "X": "x"}, sequence_configs=[])


def test_bbox_is_clipped_to_image_from_mask():
    from grf_ue_bridge.public_episode import _bbox
    mask = np.zeros((3, 4), dtype=np.uint8); mask[:, 0] = 1
    assert _bbox(mask, 1) == (0, 0, 1, 3)


def test_exr_mapping_uses_source_step_and_rejects_missing(tmp_path, monkeypatch):
    from grf_ue_bridge.public_episode import _load_mask_for_frame
    render = tmp_path / "render_mask"; render.mkdir()
    (render / "000000.exr").touch(); (render / "000003.exr").touch()
    seen = []
    monkeypatch.setattr("grf_ue_bridge.cryptomatte.load_cryptomatte", lambda path: (seen.append(path.name) or ({"Actor": "00000000"}, np.zeros((1, 1), dtype=np.float32))))
    monkeypatch.setattr("grf_ue_bridge.cryptomatte.build_mask", lambda *args: np.zeros((1, 1), dtype=np.uint8))
    _load_mask_for_frame(tmp_path, 2, {"L0": "Actor"}, {"source_step": 1}, {"source_step_seconds": 0.1, "playback_fps": 30})
    assert seen == ["000003.exr"]
    with pytest.raises(FileNotFoundError):
        _load_mask_for_frame(tmp_path, 2, {"L0": "Actor"}, {"source_step": 2}, {"source_step_seconds": 0.1, "playback_fps": 30})


def test_atomic_jpeg_conversion(tmp_path, monkeypatch):
    from PIL import Image
    from grf_ue_bridge.public_episode import _write_jpegs
    img = tmp_path / "img1"; img.mkdir(); Image.new("RGB", (2, 2), "red").save(img / "000001.png")
    _write_jpegs(tmp_path, 90)
    assert (img / "000001.jpg").exists()
    assert not list(img.glob("*.tmp"))


def test_jpeg_normalization_removes_png_and_jpeg_sources_and_is_idempotent(tmp_path):
    from PIL import Image
    img = tmp_path / "img1"
    img.mkdir()
    Image.new("RGBA", (2, 2), "red").save(img / "000001.png")
    Image.new("RGB", (2, 2), "blue").save(img / "000002.jpeg")

    from grf_ue_bridge.public_episode import _write_jpegs
    _write_jpegs(tmp_path, 90)
    first = {p.name: p.read_bytes() for p in img.iterdir()}
    assert sorted(first) == ["000001.jpg", "000002.jpg"]
    _write_jpegs(tmp_path, 90)
    assert sorted(p.name for p in img.iterdir()) == ["000001.jpg", "000002.jpg"]
    assert {p.name: p.read_bytes() for p in img.iterdir()} == first


def test_jpeg_normalization_rejects_normalized_name_collision_without_changes(tmp_path):
    from PIL import Image
    img = tmp_path / "img1"
    img.mkdir()
    Image.new("RGB", (2, 2), "red").save(img / "000001.jpg")
    Image.new("RGB", (2, 2), "blue").save(img / "1.png")
    before = {p.name: p.read_bytes() for p in img.iterdir()}

    from grf_ue_bridge.public_episode import _write_jpegs
    with pytest.raises(ValueError, match=r"000001\.jpg.*1\.png|1\.png.*000001\.jpg") as error:
        _write_jpegs(tmp_path, 90)

    assert "000001.jpg" in str(error.value)
    assert "1.png" in str(error.value)
    assert {p.name: p.read_bytes() for p in img.iterdir()} == before
    assert not list(img.glob("*.tmp"))


def test_write_public_episode_preflights_collision_before_any_episode_write(tmp_path, monkeypatch):
    cam = tmp_path / "Cam_01"
    (cam / "img1").mkdir(parents=True)
    (cam / "gt").mkdir()
    Image.new("RGB", (2, 2), "red").save(cam / "img1" / "000001.jpg")
    (cam / "camera.json").write_text(json.dumps({
        "camera_id": "Cam_01", "image_width": 2, "image_height": 2,
        "intrinsics": {"width": 2, "height": 2, "fx": 1, "fy": 1, "cx": 1, "cy": 1},
        "extrinsics": {"world_location_m": [0, 0, 0], "forward": [1, 0, 0],
                        "right": [0, 1, 0], "up": [0, 0, 1]},
    }), encoding="utf-8")
    (cam / "annotations.jsonl").write_text(json.dumps({
        "episode_id": "ep", "frame_index": 1, "objects": [{"entity_id": "L0"}]
    }) + "\n", encoding="utf-8")
    (cam / "pose_keypoints.jsonl").write_text("", encoding="utf-8")
    (cam / "gt" / "gt.txt").write_text("old gt\n", encoding="utf-8")
    (cam / "gt" / "gt_pose.json").write_text("old pose", encoding="utf-8")
    (cam / "gt" / "gt_mots.txt").write_text("old mots\n", encoding="utf-8")
    (cam / "seqinfo.ini").write_text("old seqinfo\n", encoding="utf-8")
    (tmp_path / "episode_manifest.json").write_text("old manifest", encoding="utf-8")
    cam2 = tmp_path / "Cam_02"
    import shutil
    shutil.copytree(cam, cam2)
    Image.new("RGB", (2, 2), "blue").save(cam2 / "img1" / "1.png")
    snapshot = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    monkeypatch.setattr("grf_ue_bridge.public_episode._load_mask_for_frame", lambda *_: np.zeros((2, 2), dtype=np.uint8))

    with pytest.raises(ValueError, match="规范化帧名冲突") as error:
        write_public_episode(tmp_path, mapping=PUBLIC_MAPPING, sequence_configs=[
            {"camera_dir": cam}, {"camera_dir": cam2}
        ])

    assert "000001.jpg" in str(error.value) and "1.png" in str(error.value)
    assert {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == snapshot


def test_public_manifest_uses_canonical_sequence_schema_and_camera_ids(tmp_path, monkeypatch):
    cam = tmp_path / "Cam_01"
    cam.mkdir()
    (cam / "img1").mkdir()
    (cam / "camera.json").write_text(json.dumps({
        "camera_id": "Cam_01", "image_width": 2, "image_height": 2,
        "intrinsics": {"width": 2, "height": 2, "fx": 1, "fy": 1, "cx": 1, "cy": 1},
        "extrinsics": {},
    }), encoding="utf-8")
    (cam / "annotations.jsonl").write_text(json.dumps({
        "episode_id": "ep", "frame_index": 1, "objects": [{"entity_id": "L0"}]
    }) + "\n", encoding="utf-8")
    Image.new("RGB", (2, 2), "red").save(cam / "img1" / "000001.jpg", quality=91)
    monkeypatch.setattr("grf_ue_bridge.public_episode._load_mask_for_frame",
                        lambda *_: np.array([[1, 0], [0, 0]], dtype=np.uint8))

    manifest = write_public_episode(tmp_path, mapping=PUBLIC_MAPPING,
                                    sequence_configs=[{"camera_dir": cam}])

    assert manifest["schema_version"] == 1
    assert manifest["public_classes"] == ["player", "ball"]
    assert manifest["track_id_policy"] == {"players": "L0..L4=1..5,R0..R4=6..10", "ball": 100}
    assert manifest["sequences"] == [{
        "sequence_name": "FutsalMOT_ep_C01", "camera_id": "C01",
        "relative_path": "FutsalMOT_ep_C01", "frame_count": 1,
        "image_width": 2, "image_height": 2,
        "modalities": ["mot", "pose_tracking", "mots"],
    }]
    assert (tmp_path / "FutsalMOT_ep_C01" / "img1" / "000001.jpg").read_bytes() == \
        (cam / "img1" / "000001.jpg").read_bytes()


def test_public_writer_rolls_back_all_cameras_on_late_failure(tmp_path, monkeypatch):
    cams = []
    for index in (1, 2):
        cam = tmp_path / f"Cam_{index:02d}"
        cam.mkdir()
        (cam / "img1").mkdir()
        (cam / "camera.json").write_text(json.dumps({
            "image_width": 2, "image_height": 2,
            "intrinsics": {"width": 2, "height": 2, "fx": 1, "fy": 1, "cx": 1, "cy": 1},
            "extrinsics": {},
        }), encoding="utf-8")
        (cam / "annotations.jsonl").write_text(json.dumps({
            "episode_id": "ep", "frame_index": 1, "objects": [{"entity_id": "L0"}]
        }) + "\n", encoding="utf-8")
        Image.new("RGB", (2, 2), "red").save(cam / "img1" / "000001.jpg")
        cams.append(cam)
    old = tmp_path / "FutsalMOT_ep_C01"
    (old / "gt").mkdir(parents=True)
    (old / "gt" / "gt.txt").write_text("old\n", encoding="utf-8")
    (tmp_path / "episode_manifest.json").write_text("old manifest", encoding="utf-8")
    monkeypatch.setattr("grf_ue_bridge.public_episode._load_mask_for_frame",
                        lambda camera_dir, *_: (_ for _ in ()).throw(RuntimeError("second camera"))
                        if camera_dir.name == "Cam_02" else np.array([[1, 0], [0, 0]], dtype=np.uint8))
    before = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}

    with pytest.raises(RuntimeError, match="second camera"):
        write_public_episode(tmp_path, mapping=PUBLIC_MAPPING,
                             sequence_configs=[{"camera_dir": c} for c in cams])

    assert {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


def test_player_pose_without_projection_is_seventeen_zero_triples(tmp_path, monkeypatch):
    cam = tmp_path / "Cam_01"
    cam.mkdir()
    (cam / "camera.json").write_text(json.dumps({
        "image_width": 2, "image_height": 2,
        "intrinsics": {"width": 2, "height": 2, "fx": 1, "fy": 1, "cx": 1, "cy": 1},
        "extrinsics": {},
    }), encoding="utf-8")
    (cam / "annotations.jsonl").write_text(json.dumps({
        "episode_id": "ep", "frame_index": 1, "objects": [{"entity_id": "L0"}]
    }) + "\n", encoding="utf-8")
    monkeypatch.setattr("grf_ue_bridge.public_episode._load_mask_for_frame",
                        lambda *_: np.array([[1, 0], [0, 0]], dtype=np.uint8))
    write_public_episode(tmp_path, mapping=PUBLIC_MAPPING, sequence_configs=[{"camera_dir": cam}])
    pose = json.loads((tmp_path / "FutsalMOT_ep_C01" / "gt" / "gt_pose.json").read_text())
    assert pose[0]["keypoints"] == [[0.0, 0.0, 0]] * 17
