"""相机实际状态归一化和 JSONL 序列化测试。"""

import json

import pytest

from camera_projection import focal_length_to_horizontal_fov_deg
from camera_distribution import (
    LEFT_HALF_SIDELINE_HIGH_V1,
    normalize_camera_state,
    resolve_episode_camera_ids,
    resolve_partial_static_camera_profile,
    resolve_runtime_camera_selection,
    validate_partial_camera_coverage,
    write_camera_state_jsonl,
)


def _resolved_with_profiles_and_mapping(episode_profile="anchor_only", include_p01=False):
    camera_ids = ("C1", "C2", "C3", "C4", "C5")
    profiles = {
        camera_id: {"resolution": [1920, 1080]}
        for camera_id in camera_ids
    }
    mapping = {
        camera_id: {"actor": "CineCam_" + camera_id, "sequence": "LS_Cam_" + camera_id}
        for camera_id in camera_ids
    }
    if include_p01:
        profiles["P01"] = resolve_partial_static_camera_profile(
            "P01",
            {
                "type": "static_surveillance",
                "coverage": "partial_field",
                "region": "left_half",
            },
        )
        mapping["P01"] = {"actor": "CineCam_P01", "sequence": "LS_Cam_P01"}
    return {
        "simulation": {
            "camera": {
                "distribution": {
                    "anchors": list(camera_ids),
                    "episode_profile": episode_profile,
                },
                "profiles": profiles,
            }
        },
        "ue_profile": {"camera_mapping": mapping},
    }


def test_anchor_only_selects_canonical_anchors_in_order():
    resolved = _resolved_with_profiles_and_mapping(episode_profile="anchor_only")

    assert resolve_episode_camera_ids(resolved) == ["C1", "C2", "C3", "C4", "C5"]


def test_anchor_plus_one_partial_selects_p01_after_anchors():
    resolved = _resolved_with_profiles_and_mapping(
        episode_profile="anchor_plus_one_partial",
        include_p01=True,
    )

    assert resolve_episode_camera_ids(resolved) == [
        "C1", "C2", "C3", "C4", "C5", "P01"
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("coverage", "full_field"),
        ("region", "right_half"),
        ("placement_template", "wrong_template"),
    ],
)
def test_episode_selection_rejects_invalid_p01_semantics(field, value):
    resolved = _resolved_with_profiles_and_mapping(
        episode_profile="anchor_plus_one_partial",
        include_p01=True,
    )
    resolved["simulation"]["camera"]["profiles"]["P01"].update({
        "type": "static_surveillance",
        "coverage": "partial_field",
        "region": "left_half",
        "placement_template": "left_half_sideline_high_v1",
    })
    resolved["simulation"]["camera"]["profiles"]["P01"][field] = value

    with pytest.raises(ValueError, match=field):
        resolve_episode_camera_ids(resolved)


def test_camera_selection_is_repeatable():
    resolved = _resolved_with_profiles_and_mapping(
        episode_profile="anchor_plus_one_partial",
        include_p01=True,
    )

    assert resolve_episode_camera_ids(resolved) == resolve_episode_camera_ids(resolved)


def test_anchor_plus_two_partial_is_rejected_until_second_partial_exists():
    resolved = _resolved_with_profiles_and_mapping(
        episode_profile="anchor_plus_two_partial",
        include_p01=True,
    )

    with pytest.raises(ValueError, match="anchor_plus_two_partial|P02"):
        resolve_episode_camera_ids(resolved)


def test_runtime_selection_uses_episode_profile_for_canonical_entries():
    resolved = _resolved_with_profiles_and_mapping(
        episode_profile="anchor_plus_one_partial",
        include_p01=True,
    )

    selected = resolve_runtime_camera_selection(
        resolved,
        legacy_sequences=[],
        legacy_cameras=[],
    )

    assert [entry["camera_id"] for entry in selected["entries"]] == [
        "C1", "C2", "C3", "C4", "C5", "P01"
    ]


def test_episode_selection_rejects_noncanonical_anchor_contract():
    resolved = _resolved_with_profiles_and_mapping(
        episode_profile="anchor_plus_one_partial",
        include_p01=True,
    )
    resolved["simulation"]["camera"]["distribution"]["anchors"] = [
        "C1", "C2", "C3", "C4"
    ]

    with pytest.raises(ValueError, match="anchors.*C1..C5"):
        resolve_episode_camera_ids(resolved)


def _row(camera_id="C1", frame=0, source_step=None, time_seconds=None):
    if source_step is None:
        source_step = frame
    if time_seconds is None:
        time_seconds = frame * 0.1
    return normalize_camera_state(
        camera_id=camera_id,
        ue_actor="CineCam_01",
        sequence="LS_Cam_01",
        profile={
            "type": "static_surveillance",
            "coverage": "full_field",
            "resolution": [1920, 1080],
            "lens": {"focal_length_mm": 36.0, "horizontal_fov_deg": 70.0},
            "position_m": [1.0, -8.0, 8.0],
            "rotation_deg": [0.0, 45.0, 0.0],
            "height_m": 8.0,
            "distortion": None,
        },
        actual={
            "resolution": [1920, 1080],
            "focal_length_mm": 36.0,
            "horizontal_fov_deg": 70.0,
            "position_m": [1.0, -8.0, 8.0],
            "rotation_deg": [0.0, 45.0, 0.0],
            "height_m": 8.0,
            "distortion": {"model": "none"},
        },
        frame=frame,
        source_step=frame,
        time_seconds=time_seconds,
    )


def _profile():
    return {
        "type": "static_surveillance",
        "coverage": "full_field",
        "resolution": [1920, 1080],
        "lens": {"focal_length_mm": 36.0, "horizontal_fov_deg": 70.0},
        "position_m": [1.0, -8.0, 8.0],
        "rotation_deg": [0.0, 45.0, 0.0],
        "height_m": 8.0,
    }


def test_runtime_camera_selection_prefers_canonical_contract_over_legacy_fields():
    simulation = {
        "camera": {
            "distribution": {"anchors": ["C1", "C2", "C3", "C4", "C5"]},
            "profiles": {
                camera_id: {
                    "position_m": [index, 0.0, 8.0],
                    "resolution": [1920, 1080],
                    "lens": {"focal_length_mm": 15.0},
                }
                for index, camera_id in enumerate(("C1", "C2", "C3", "C4", "C5"), 1)
            },
        }
    }
    mapping = {
        camera_id: {"actor": f"CineCam_{index:02d}", "sequence": f"LS_Cam_{index:02d}"}
        for index, camera_id in enumerate(("C1", "C2", "C3", "C4", "C5"), 1)
    }

    selected = resolve_runtime_camera_selection(
        {"simulation": simulation, "ue_profile": {"camera_mapping": mapping}},
        legacy_sequences=[{"name": "Legacy", "camera_actor": "LegacyCamera"}],
        legacy_cameras=["LegacyCamera"],
    )

    assert [entry["camera_id"] for entry in selected["entries"]] == [
        "C1", "C2", "C3", "C4", "C5"
    ]
    assert [entry["name"] for entry in selected["sequences"]] == [
        "LS_Cam_01", "LS_Cam_02", "LS_Cam_03", "LS_Cam_04", "LS_Cam_05"
    ]
    assert selected["cameras"] == [
        "CineCam_01", "CineCam_02", "CineCam_03", "CineCam_04", "CineCam_05"
    ]
    assert selected["entries"][4]["profile"] == {
        "position_m": [5, 0.0, 8.0],
        "resolution": [1920, 1080],
        "lens": {"focal_length_mm": 15.0},
    }
    assert selected["resolution"] == [1920, 1080]
    assert selected["entries"][0]["profile"]["lens"] == {"focal_length_mm": 15.0}


def test_left_half_placement_is_deterministic():
    intent = {"type": "static_surveillance", "coverage": "partial_field", "region": "left_half"}

    first = resolve_partial_static_camera_profile("P01", intent)
    second = resolve_partial_static_camera_profile("P01", intent)

    assert first == second
    assert first["placement_template"] == LEFT_HALF_SIDELINE_HIGH_V1
    assert first["position_m"] == [-10.0, -25.0, 16.0]
    assert first["rotation_deg"] == pytest.approx([-32.47119229084849, 90.0, 0.0])


def test_left_half_placement_passes_geometry_validation():
    profile = resolve_partial_static_camera_profile(
        "P01",
        {"type": "static_surveillance", "coverage": "partial_field", "region": "left_half"},
    )

    assert validate_partial_camera_coverage(profile) is True


def test_left_half_placement_rejects_transform_outside_template():
    profile = resolve_partial_static_camera_profile(
        "P01",
        {"type": "static_surveillance", "coverage": "partial_field", "region": "left_half"},
    )
    profile["position_m"][0] += 1.0

    with pytest.raises(ValueError, match="placement template"):
        validate_partial_camera_coverage(profile)


def test_runtime_selection_keeps_p01_unmapped_and_applies_only_canonical_anchors():
    simulation = {
        "camera": {
            "distribution": {"anchors": ["C1", "C2", "C3", "C4", "C5"]},
            "profiles": {
                camera_id: {
                    "position_m": [index, 0.0, 8.0],
                    "resolution": [1920, 1080],
                    "lens": {"focal_length_mm": 15.0},
                }
                for index, camera_id in enumerate(("C1", "C2", "C3", "C4", "C5"), 1)
            },
        }
    }
    simulation["camera"]["profiles"]["P01"] = resolve_partial_static_camera_profile(
        "P01",
        {"type": "static_surveillance", "coverage": "partial_field", "region": "left_half"},
    )
    mapping = {
        camera_id: {"actor": f"CineCam_{index:02d}", "sequence": f"LS_Cam_{index:02d}"}
        for index, camera_id in enumerate(("C1", "C2", "C3", "C4", "C5"), 1)
    }

    selected = resolve_runtime_camera_selection(
        {"simulation": simulation, "ue_profile": {"camera_mapping": mapping}},
        legacy_sequences=[],
        legacy_cameras=[],
    )

    assert [entry["camera_id"] for entry in selected["entries"]] == [
        "C1", "C2", "C3", "C4", "C5"
    ]


def test_runtime_selection_supports_p01_mapping_when_explicitly_selected():
    p01 = resolve_partial_static_camera_profile(
        "P01",
        {"type": "static_surveillance", "coverage": "partial_field", "region": "left_half"},
    )
    selected = resolve_runtime_camera_selection(
        {
            "simulation": {"camera": {"profiles": {"P01": p01}}},
            "ue_profile": {
                "camera_mapping": {
                    "P01": {"actor": "CineCam_P01", "sequence": "LS_Cam_P01"}
                }
            },
        },
        legacy_sequences=[],
        legacy_cameras=[],
        camera_ids=["P01"],
    )

    assert selected["entries"] == [{
        "camera_id": "P01",
        "camera_actor": "CineCam_P01",
        "name": "LS_Cam_P01",
        "profile": p01,
    }]


def test_anchor_selection_allows_p01_mapping_without_selecting_p01():
    profiles = {
        camera_id: {
            "position_m": [index, 0.0, 8.0],
            "resolution": [1920, 1080],
            "lens": {"focal_length_mm": 15.0},
        }
        for index, camera_id in enumerate(("C1", "C2", "C3", "C4", "C5"), 1)
    }
    profiles["P01"] = resolve_partial_static_camera_profile(
        "P01",
        {"type": "static_surveillance", "coverage": "partial_field", "region": "left_half"},
    )
    mapping = {
        camera_id: {"actor": f"CineCam_{index:02d}", "sequence": f"LS_Cam_{index:02d}"}
        for index, camera_id in enumerate(("C1", "C2", "C3", "C4", "C5"), 1)
    }
    mapping["P01"] = {"actor": "CineCam_P01", "sequence": "LS_Cam_P01"}

    selected = resolve_runtime_camera_selection(
        {
            "simulation": {
                "camera": {
                    "distribution": {"anchors": ["C1", "C2", "C3", "C4", "C5"]},
                    "profiles": profiles,
                }
            },
            "ue_profile": {"camera_mapping": mapping},
        },
        legacy_sequences=[],
        legacy_cameras=[],
    )

    assert [entry["camera_id"] for entry in selected["entries"]] == [
        "C1", "C2", "C3", "C4", "C5"
    ]


@pytest.mark.parametrize(
    ("focal_length_mm", "expected_fov"),
    [(15.0, 76.7584), (10.0, 99.8219)],
)
def test_horizontal_fov_is_derived_from_focal_length_and_sensor_width(
    focal_length_mm, expected_fov
):
    assert focal_length_to_horizontal_fov_deg(focal_length_mm) == pytest.approx(
        expected_fov, abs=1e-3
    )


def test_runtime_camera_selection_rejects_mixed_mrq_resolutions():
    simulation = {
        "camera": {
            "distribution": {"anchors": ["C1", "C2", "C3", "C4", "C5"]},
            "profiles": {
                camera_id: {
                    "position_m": [index, 0.0, 8.0],
                    "resolution": [2560, 1440] if camera_id == "C5" else [1920, 1080],
                }
                for index, camera_id in enumerate(("C1", "C2", "C3", "C4", "C5"), 1)
            },
        }
    }
    mapping = {
        camera_id: {"actor": f"CineCam_{index:02d}", "sequence": f"LS_Cam_{index:02d}"}
        for index, camera_id in enumerate(("C1", "C2", "C3", "C4", "C5"), 1)
    }

    with pytest.raises(ValueError, match="resolution.*统一"):
        resolve_runtime_camera_selection(
            {"simulation": simulation, "ue_profile": {"camera_mapping": mapping}},
            legacy_sequences=[],
            legacy_cameras=[],
        )


def test_runtime_camera_selection_keeps_legacy_fallback():
    selected = resolve_runtime_camera_selection(
        {"ue_profile": {}},
        legacy_sequences=[{"name": "Legacy", "camera_actor": "LegacyCamera"}],
        legacy_cameras=["LegacyCamera"],
    )

    assert selected == {
        "entries": [],
        "sequences": [{"name": "Legacy", "camera_actor": "LegacyCamera"}],
        "cameras": ["LegacyCamera"],
        "resolution": None,
        "canonical": False,
    }


def test_normalize_camera_state_uses_actual_state_and_zero_based_indices():
    row = _row()

    assert row["schema"] == "futsalmot_camera_state"
    assert row["version"] == 1
    assert row["camera_id"] == "C1"
    assert row["ue_actor"] == "CineCam_01"
    assert row["resolution"] == [1920, 1080]
    assert row["focal_length_mm"] == 36.0
    assert row["horizontal_fov_deg"] == 70.0
    assert row["frame"] == 0
    assert row["source_step"] == 0
    assert row["position_m"] == [1.0, -8.0, 8.0]
    assert row["rotation_deg"] == [0.0, 45.0, 0.0]


@pytest.mark.parametrize(
    "resolution",
    [
        None,
        1920,
        object(),
        "1920x1080",
        [1920],
        [1920, 1080, 1],
        [True, 1080],
        [1920, False],
        [float("nan"), 1080],
        [1920, float("inf")],
        [0, 1080],
        [1920, -1],
        [1920.5, 1080],
    ],
)
def test_normalize_camera_state_rejects_invalid_actual_resolution(resolution):
    actual = {
        "resolution": resolution,
        "focal_length_mm": 36.0,
        "horizontal_fov_deg": 70.0,
        "position_m": [1.0, -8.0, 8.0],
        "rotation_deg": [0.0, 45.0, 0.0],
        "height_m": 8.0,
    }

    with pytest.raises(ValueError, match="resolution"):
        _normalize_with_actual(actual)


def test_normalize_camera_state_rejects_non_sequence_resolution_before_conversion():
    class NonSequenceResolution:
        def __iter__(self):
            raise TypeError("resolution must not be converted")

    actual = {
        "resolution": NonSequenceResolution(),
        "focal_length_mm": 36.0,
        "horizontal_fov_deg": 70.0,
        "position_m": [1.0, -8.0, 8.0],
        "rotation_deg": [0.0, 45.0, 0.0],
        "height_m": 8.0,
    }

    with pytest.raises(ValueError, match="resolution"):
        _normalize_with_actual(actual)


@pytest.mark.parametrize(
    "resolution",
    [
        None,
        1920,
        object(),
        "1920x1080",
        [1920],
        [1920, 1080, 1],
        [True, 1080],
        [1920, False],
        [float("nan"), 1080],
        [1920, float("inf")],
        [0, 1080],
        [1920, -1],
        [1920.5, 1080],
    ],
)
def test_write_camera_state_jsonl_rejects_invalid_serialized_resolution(tmp_path, resolution):
    rows = [_row(camera) for camera in ("C1", "C2", "C3", "C4", "C5")]
    rows[0]["resolution"] = resolution

    with pytest.raises(ValueError, match="resolution"):
        write_camera_state_jsonl(tmp_path / "camera_state.jsonl", rows)


@pytest.mark.parametrize("value", ["36.0", True, float("nan"), float("inf")])
def test_normalize_camera_state_rejects_non_finite_actual_lens_values(value):
    actual = {
        "resolution": [1920, 1080],
        "focal_length_mm": value,
        "horizontal_fov_deg": 70.0,
        "position_m": [1.0, -8.0, 8.0],
        "rotation_deg": [0.0, 45.0, 0.0],
        "height_m": 8.0,
    }

    with pytest.raises(ValueError, match="focal_length_mm"):
        normalize_camera_state(
            "C1", "CineCam_01", "LS_Cam_01", _profile(), actual, 0, 0, 0.0
        )


@pytest.mark.parametrize("field", ["height_m", "horizontal_fov_deg"])
@pytest.mark.parametrize("value", ["8.0", True, float("nan"), float("inf")])
def test_normalize_camera_state_rejects_non_finite_actual_scalar(field, value):
    actual = {
        "resolution": [1920, 1080],
        "focal_length_mm": 36.0,
        "horizontal_fov_deg": 70.0,
        "position_m": [1.0, -8.0, 8.0],
        "rotation_deg": [0.0, 45.0, 0.0],
        "height_m": 8.0,
    }
    actual[field] = value

    with pytest.raises(ValueError, match=field):
        _normalize_with_actual(actual)


@pytest.mark.parametrize("field", ["position_m", "rotation_deg"])
@pytest.mark.parametrize("value", ["1.0", True, float("nan"), float("inf")])
def test_normalize_camera_state_rejects_malformed_actual_vector_elements(field, value):
    actual = {
        "resolution": [1920, 1080],
        "focal_length_mm": 36.0,
        "horizontal_fov_deg": 70.0,
        "position_m": [1.0, -8.0, 8.0],
        "rotation_deg": [0.0, 45.0, 0.0],
        "height_m": 8.0,
    }
    actual[field][1] = value

    with pytest.raises(ValueError, match=field):
        _normalize_with_actual(actual)


def _normalize_with_actual(actual):
    return normalize_camera_state(
        "C1",
        "CineCam_01",
        "LS_Cam_01",
        {
            "type": "static_surveillance",
            "coverage": "full_field",
            "resolution": [1920, 1080],
            "lens": {"focal_length_mm": 36.0, "horizontal_fov_deg": 70.0},
            "position_m": [1.0, -8.0, 8.0],
            "rotation_deg": [0.0, 45.0, 0.0],
            "height_m": 8.0,
        },
        actual,
        0,
        0,
        0.0,
    )


@pytest.mark.parametrize("field", ["height_m", "focal_length_mm", "horizontal_fov_deg"])
@pytest.mark.parametrize("value", ["8.0", True, float("nan"), float("inf")])
def test_write_camera_state_jsonl_rejects_malformed_numeric_rows(tmp_path, field, value):
    rows = [_row(camera) for camera in ("C1", "C2", "C3", "C4", "C5")]
    rows[0][field] = value

    with pytest.raises(ValueError, match=field):
        write_camera_state_jsonl(tmp_path / "camera_state.jsonl", rows)


@pytest.mark.parametrize("field", ["position_m", "rotation_deg"])
@pytest.mark.parametrize("value", ["1.0", True, float("nan"), float("inf")])
def test_write_camera_state_jsonl_rejects_malformed_numeric_vectors(tmp_path, field, value):
    rows = [_row(camera) for camera in ("C1", "C2", "C3", "C4", "C5")]
    rows[0][field][1] = value

    with pytest.raises(ValueError, match=field):
        write_camera_state_jsonl(tmp_path / "camera_state.jsonl", rows)


@pytest.mark.parametrize(
    "field,value",
    [
        ("sequence", ""),
        ("type", None),
        ("coverage", None),
        ("height_m", None),
        ("frame", -1),
        ("source_step", -1),
        ("time_seconds", -0.1),
        ("time_seconds", float("inf")),
    ],
)
def test_write_camera_state_jsonl_rejects_invalid_required_row_fields(tmp_path, field, value):
    row = _row()
    row[field] = value

    with pytest.raises(ValueError, match=field):
        write_camera_state_jsonl(tmp_path / "camera_state.jsonl", [row] + [_row(camera) for camera in ("C2", "C3", "C4", "C5")])


@pytest.mark.parametrize("field,expected", [("schema", "wrong"), ("version", 2)])
def test_write_camera_state_jsonl_rejects_invalid_schema_metadata(tmp_path, field, expected):
    rows = [_row(camera) for camera in ("C1", "C2", "C3", "C4", "C5")]
    rows[0][field] = expected

    with pytest.raises(ValueError, match=field):
        write_camera_state_jsonl(tmp_path / "camera_state.jsonl", rows)


@pytest.mark.parametrize(
    "actual_field,actual_value",
    [
        ("resolution", [1280, 720]),
        ("focal_length_mm", 40.0),
        ("horizontal_fov_deg", 60.0),
        ("position_m", [2.0, -8.0, 8.0]),
        ("rotation_deg", [0.0, 40.0, 0.0]),
        ("height_m", 7.0),
    ],
)
def test_normalize_camera_state_rejects_profile_actual_mismatch(actual_field, actual_value):
    actual = {
        "resolution": [1920, 1080],
        "focal_length_mm": 36.0,
        "horizontal_fov_deg": 70.0,
        "position_m": [1.0, -8.0, 8.0],
        "rotation_deg": [0.0, 45.0, 0.0],
        "height_m": 8.0,
    }
    if actual_field in ("focal_length_mm", "horizontal_fov_deg"):
        actual[actual_field] = actual_value
    else:
        actual[actual_field] = actual_value

    with pytest.raises(ValueError, match=actual_field):
        normalize_camera_state(
            camera_id="C1",
            ue_actor="CineCam_01",
            sequence="LS_Cam_01",
            profile=_profile(),
            actual=actual,
            frame=0,
            source_step=0,
            time_seconds=0.0,
        )


@pytest.mark.parametrize("field", ["camera_id", "ue_actor", "resolution", "position_m"])
def test_normalize_camera_state_rejects_missing_required_actual_state(field):
    row = _row()
    if field in ("camera_id", "ue_actor"):
        row[field] = ""
    else:
        row[field] = None

    with pytest.raises(ValueError, match=field):
        write_camera_state_jsonl("unused.jsonl", [row])


def test_write_camera_state_jsonl_rejects_missing_actual_lens_value(tmp_path):
    row = _row()
    row["focal_length_mm"] = None
    row["horizontal_fov_deg"] = None

    with pytest.raises(ValueError, match="focal_length_mm"):
        write_camera_state_jsonl(tmp_path / "camera_state.jsonl", [row])


def test_write_camera_state_jsonl_orders_by_frame_then_canonical_camera_id(tmp_path):
    rows = [
        _row(camera, frame)
        for camera, frame in (("C5", 1), ("C5", 0), ("C2", 0), ("C1", 0), ("C4", 0), ("C3", 0),
                              ("C1", 1), ("C2", 1), ("C3", 1), ("C4", 1))
    ]
    path = tmp_path / "camera_state.jsonl"

    write_camera_state_jsonl(path, rows)

    written = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [(row["frame"], row["camera_id"]) for row in written] == [
        (0, "C1"),
            (0, "C2"),
            (0, "C3"),
            (0, "C4"),
            (0, "C5"),
            (1, "C1"),
        (1, "C2"),
        (1, "C3"),
        (1, "C4"),
        (1, "C5"),
    ]


@pytest.mark.parametrize(
    "rows",
    [
        [_row(camera) for camera in ("C1", "C2", "C3", "C4")],
        [_row(camera) for camera in ("C1", "C2", "C3", "C4", "C5")] + [_row("C1")],
    ],
)
def test_write_camera_state_jsonl_rejects_incomplete_or_duplicate_frame_sets(tmp_path, rows):
    with pytest.raises(ValueError, match="C1..C5|duplicate"):
        write_camera_state_jsonl(tmp_path / "camera_state.jsonl", rows)


def test_write_camera_state_jsonl_allows_empty_rows(tmp_path):
    path = tmp_path / "camera_state.jsonl"

    write_camera_state_jsonl(path, [])

    assert path.read_text(encoding="utf-8") == ""
