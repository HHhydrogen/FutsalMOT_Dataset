"""v2 导出数学与契约测试，不代表真实 UE 渲染验收。"""

import copy

import pytest

from grf_ue_bridge.pose17_export import (
    COCO17_NAMES, binary_mask_geometry, build_coco, project_landmarks,
    validate_capture, validate_capture_coverage, validate_coco,
)


def capture(camera="CineCam_01"):
    return {
        "schema": "futsalmot_pose17_capture_v2", "keypoint_names": list(COCO17_NAMES),
        "seed": 42, "clip_id": "clip_000", "camera_id": camera,
        "capture_id": "render:42:0:" + camera, "render_sample_id": "render:42:0:" + camera,
        "frame_index": 0, "source_frame_index": 100, "sequence_frame_index": 0,
        "time_seconds": 10.0, "width": 4, "height": 3,
        "world_coordinates": "ue_left_handed_x_forward_y_right_z_up_m",
        "file_name": camera + "/rgb/000000.png", "mask_file_name": camera + "/mask/000000.png",
        "camera_calibration_ref": camera + "/camera.json",
        "objects": [{"entity_id": "L0", "track_id": 1, "mask_id": 1,
                     "keypoints": [{"world_m": [1.0, 0.0, 1.5], "uv_px": [1.5, 1.0],
                                    "source_method": "test_fixture", "visibility_state": "unknown",
                                    "visibility_method": "not_measured"} for _ in range(17)]}],
    }


MASK = [[0, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0]]


def test_projection_unit_axes_and_unknown_occlusion():
    camera = {"intrinsics": {"width": 100, "height": 100, "fx": 50, "fy": 50, "cx": 50, "cy": 50},
              "extrinsics": {"world_location_m": [0, 0, 0], "forward": [1, 0, 0],
                             "right": [0, 1, 0], "up": [0, 0, 1]}}
    worlds = [[10, 0, 0], [10, 2, 2], [10, 100, 0], [-1, 0, 0], None]
    points = project_landmarks([{"world_m": world, "source_method": "fixture"} for world in worlds], camera)
    assert points[0]["uv_px"] == [50, 50] and points[0]["visibility_state"] == "unknown"
    assert points[1]["uv_px"] == [60, 40]
    assert [p["visibility_state"] for p in points[2:]] == ["outside", "behind_camera", "missing"]
    camera["extrinsics"]["right"] = [1, 0, 0]
    with pytest.raises(ValueError, match="正交"):
        project_landmarks([], camera)


def test_geometry_exact_visible_bbox_area_and_column_rle():
    area, bbox, rle = binary_mask_geometry(MASK, 1)
    assert area == 3 and bbox == [1, 0, 2, 2]
    assert rle == {"size": [3, 4], "counts": [3, 2, 2, 1, 4]}


def test_unknown_is_not_occluded_or_visible_and_preserves_world():
    coco, sidecar = build_coco([capture()], [MASK])
    annotation = coco["annotations"][0]
    assert annotation["keypoints"] == [0] * 51
    assert annotation["num_keypoints"] == 0
    assert sidecar["objects"][0]["keypoints"][0]["world_m"] == [1, 0, 1.5]
    assert sidecar["objects"][0]["keypoints"][0]["uv_px"] == [1.5, 1]


def test_known_visibility_requires_evidence_and_counts_occluded():
    frame = capture()
    for state, point in zip(("visible", "occluded"), frame["objects"][0]["keypoints"]):
        point.update(visibility_state=state, visibility_method="surface_depth_test",
                     visibility_evidence_ref="depth_receipt:sha256:fixture")
    coco, _ = build_coco([frame], [MASK])
    assert coco["annotations"][0]["keypoints"][:6] == [1.5, 1, 2, 1.5, 1, 1]
    assert coco["annotations"][0]["num_keypoints"] == 2
    frame["objects"][0]["keypoints"][0]["visibility_method"] = "instance_mask_neighborhood"
    with pytest.raises(ValueError, match="mask"):
        validate_capture(frame)


@pytest.mark.parametrize("field,value", [
    ("render_sample_id", "wrong-frame"), ("keypoint_names", list(COCO17_NAMES)[5:]),
    ("frame_index", -1), ("width", 0), ("file_name", "../v1/rgb.png"),
    ("file_name", "G:/rgb.png"), ("time_seconds", float("nan")),
])
def test_bad_frame_rejected(field, value):
    frame = capture()
    frame[field] = value
    with pytest.raises(ValueError):
        validate_capture(frame)


def test_legacy_twelve_point_rejected():
    frame = capture()
    frame["objects"][0]["keypoints"] = frame["objects"][0]["keypoints"][5:]
    with pytest.raises(ValueError, match="17"):
        validate_capture(frame)


def test_empty_mask_preserves_entity_in_sidecar():
    coco, sidecar = build_coco([capture()], [[[0] * 4 for _ in range(3)]])
    assert not coco["annotations"]
    assert sidecar["objects"][0]["visible_mask_area"] == 0


def test_non_player_instance_must_be_explicitly_excluded():
    frame, mask = capture(), copy.deepcopy(MASK)
    frame["excluded_mask_ids"] = [11]
    mask[0][0] = 11
    coco, _ = build_coco([frame], [mask])
    assert coco["annotations"][0]["area"] == 3
    frame["excluded_mask_ids"] = [1]
    with pytest.raises(ValueError, match="排除"):
        validate_capture(frame)


def test_coverage_requires_each_entity_on_each_expected_frame():
    frame = capture()
    with pytest.raises(ValueError, match="十名"):
        validate_capture_coverage([frame], ["CineCam_01"], [0])
    template = frame["objects"][0]
    frame["objects"] = []
    for index, entity in enumerate([s + str(i) for s in "LR" for i in range(5)], 1):
        obj = copy.deepcopy(template)
        obj.update(entity_id=entity, track_id=index, mask_id=index)
        frame["objects"].append(obj)
    validate_capture_coverage([frame], ["CineCam_01"], [0])
    with pytest.raises(ValueError, match="帧集合"):
        validate_capture_coverage([frame], ["CineCam_01"], [0, 1])


def test_ids_stable_under_input_order_and_sync_rejected():
    first, second = capture(), capture("CineCam_02")
    assert build_coco([first, second], [MASK, MASK]) == build_coco([second, first], [MASK, MASK])
    second["objects"][0]["keypoints"][0]["world_m"][0] += .01
    with pytest.raises(ValueError, match="跨机位"):
        build_coco([first, second], [MASK, MASK])


@pytest.mark.parametrize("mutation", ["duplicate", "unmapped_mask", "mask_resolution", "track_id"])
def test_identity_and_mask_errors(mutation):
    frame, mask = capture(), copy.deepcopy(MASK)
    frames, masks = [frame], [mask]
    if mutation == "duplicate":
        frames, masks = [frame, frame], [mask, mask]
    elif mutation == "unmapped_mask":
        mask[0][0] = 7
    elif mutation == "mask_resolution":
        mask.pop()
    else:
        frame["objects"][0]["track_id"] = 6
    with pytest.raises(ValueError):
        build_coco(frames, masks)


@pytest.mark.parametrize("field,value", [("area", 4), ("bbox", [1, 0, 1, 1]),
                                        ("num_keypoints", 17), ("keypoints", [0] * 36)])
def test_validator_rejects_corruption(field, value):
    coco, _ = build_coco([capture()], [MASK])
    coco["annotations"][0][field] = value
    with pytest.raises(ValueError):
        validate_coco(coco)
