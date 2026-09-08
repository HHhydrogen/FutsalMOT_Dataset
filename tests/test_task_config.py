"""task schema（单 config）测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from grf_ue_bridge.config import loader
from grf_ue_bridge.config.models import (
    AuditTaskConfig,
    CameraMappingConfig,
    DatasetTaskConfig,
    PostprocessTaskConfig,
)


def _write_task(path: Path, **over):
    data = {
        "schema": "futsalmot_dataset_task",
        "version": 2,
        "task_id": "t1",
        "episode_name": "episode_t1",
        "dataset_root": "G:/DS",
        "ue_project_root": "D:/UE",
        "export": {
            "scenario": "5_vs_5", "seed": 42, "num_steps": 300, "playback_fps": 30,
        },
        "ue": {
            "actor_mapping": "ue/actor_mapping.example.json",
            "sequences": [{"name": "LS_Cam_01", "camera_actor": "CineCam_01"}],
            "annotation_export": {"cameras": ["CineCam_01"]},
        },
        "postprocess": {
            "include_ball": True,
            "workers": 4,
            "chunk_size": 50,
            "png_compress_level": 1,
            "formats": ["json", "mot", "yolo-det", "yolo-seg"],
            "clean_stale": True,
            "validation_level": "full",
        },
        "audit": {"expected_cameras": 1, "expected_frames_per_camera": 300},
    }
    data.update(over)
    (path / "task.json").write_text(json.dumps(data), encoding="utf-8")
    return path / "task.json"


def _camera_task_fields():
    profiles = {}
    for index, camera_id in enumerate(("C1", "C2", "C3", "C4", "C5")):
        profiles[camera_id] = {
            "type": "static_surveillance",
            "coverage": "full_field",
            "resolution": [1920, 1080],
            "lens": {"focal_length_mm": 10.0 if camera_id == "C5" else 15.0},
            "position_m": [float(index), -8.0, 8.0],
            "rotation_deg": [0.0, 45.0, 0.0],
            "height_m": 8.0,
            "distortion": None,
        }
    return {
        "simulation": {
            "camera": {
                "distribution": {
                    "system": "dual",
                    "anchors": ["C1", "C2", "C3", "C4", "C5"],
                    "static_surveillance_ratio": [0.70, 0.80],
                    "dynamic_broadcast_ratio": [0.20, 0.30],
                },
                "profiles": profiles,
            }
        },
        "ue": {
            "camera_mapping": {
                camera_id: {
                    "actor": f"CineCam_{index:02d}",
                    "sequence": f"LS_Cam_{index:02d}",
                }
                for index, camera_id in enumerate(("C1", "C2", "C3", "C4", "C5"), 1)
            }
        },
    }


def _partial_camera_task_fields():
    fields = _camera_task_fields()
    fields["simulation"]["camera"]["profiles"]["P01"] = {
        "type": "static_surveillance",
        "coverage": "partial_field",
        "region": "left_half",
    }
    fields["ue"]["camera_mapping"]["P01"] = {
        "actor": "CineCam_P01",
        "sequence": "LS_Cam_P01",
    }
    return fields


def _p01_mapping_fields():
    fields = _partial_camera_task_fields()
    fields["ue"]["camera_mapping"]["P01"] = {
        "actor": "CineCam_P01",
        "sequence": "LS_Cam_P01",
    }
    return fields


class TestTaskSchema:
    def test_valid_task(self, tmp_path):
        tf = _write_task(tmp_path)
        t = loader.load_task_config(tf)
        assert t.task_id == "t1"
        assert t.episode_name == "episode_t1"
        assert t.dataset_root == "G:/DS"
        assert t.ue_project_root == "D:/UE"
        assert t.export.num_steps == 300
        assert t.ue.annotation_export["cameras"] == ["CineCam_01"]
        assert t.postprocess.validation_level == "full"

    def test_bad_schema_rejected(self, tmp_path):
        tf = _write_task(tmp_path, schema="other")
        with pytest.raises(ValueError):
            loader.load_task_config(tf)

    def test_bad_task_id_rejected(self, tmp_path):
        tf = _write_task(tmp_path, task_id="bad id!")
        with pytest.raises(Exception):
            loader.load_task_config(tf)

    def test_bad_episode_name_rejected(self, tmp_path):
        tf = _write_task(tmp_path, episode_name="has/slash")
        with pytest.raises(Exception):
            loader.load_task_config(tf)

    def test_bad_workers_rejected(self, tmp_path):
        tf = _write_task(tmp_path, postprocess={"workers": 999})
        with pytest.raises(Exception):
            loader.load_task_config(tf)

    def test_unsupported_format_rejected(self, tmp_path):
        tf = _write_task(tmp_path, postprocess={"formats": ["json", "coco"]})
        with pytest.raises(ValueError, match="格式"):
            loader.load_task_config(tf)  # load 时即调用 validate_formats

    def test_missing_machine_path_fails(self, tmp_path):
        # 去掉 dataset_root（必填）→ load 即报错；validate_task 返回问题
        tf = _write_task(tmp_path, dataset_root=None)
        from grf_ue_bridge.config import resolver
        problems = resolver.validate_task(tf)
        assert problems  # 解析失败（dataset_root 缺失）

    def test_missing_export_block_fails(self, tmp_path):
        tf = _write_task(tmp_path)
        task = json.loads(tf.read_text(encoding="utf-8"))
        del task["export"]
        tf.write_text(json.dumps(task), encoding="utf-8")
        with pytest.raises(Exception):
            loader.load_task_config(tf)

    def test_export_scenario_overrides(self, tmp_path):
        tf = _write_task(
            tmp_path,
            export={
                "scenario": "5_vs_5", "seed": 42, "num_steps": 300, "playback_fps": 30,
                "game_duration": 10000,
                "left_team_difficulty": 0.6, "right_team_difficulty": 0.6,
            },
        )
        t = loader.load_task_config(tf)
        assert t.export.game_duration == 10000
        assert t.export.left_team_difficulty == 0.6
        assert t.export.right_team_difficulty == 0.6

    def test_export_defaults_none(self, tmp_path):
        # 缺省（null）= 用场景默认，不覆盖
        tf = _write_task(tmp_path)
        t = loader.load_task_config(tf)
        assert t.export.game_duration is None
        assert t.export.left_team_difficulty is None
        assert t.export.right_team_difficulty is None

    def test_bad_difficulty_rejected(self, tmp_path):
        tf = _write_task(tmp_path, export={
            "scenario": "5_vs_5", "num_steps": 300,
            "left_team_difficulty": 1.5,  # 越界 >1
        })
        with pytest.raises(Exception):
            loader.load_task_config(tf)

    def test_defaults(self):
        pp = PostprocessTaskConfig()
        assert pp.workers == 4
        assert pp.validation_level == "full"
        assert "yolo-seg" in pp.formats
        au = AuditTaskConfig()
        assert au.expected_cameras == 4
        assert au.expected_frames_per_camera == 300

    def test_valid_five_anchor_camera_task(self, tmp_path):
        tf = _write_task(tmp_path, **_camera_task_fields())
        t = loader.load_task_config(tf)
        assert t.simulation.camera.distribution.anchors == ["C1", "C2", "C3", "C4", "C5"]
        assert set(t.simulation.camera.profiles) == {"C1", "C2", "C3", "C4", "C5"}
        assert isinstance(t.ue.camera_mapping["C1"], CameraMappingConfig)
        assert t.ue.camera_mapping["C5"].actor == "CineCam_05"

    @pytest.mark.parametrize(
        "episode_profile",
        ["anchor_only"],
    )
    def test_camera_distribution_accepts_episode_profile(self, tmp_path, episode_profile):
        fields = _camera_task_fields()
        fields["simulation"]["camera"]["distribution"]["episode_profile"] = episode_profile
        tf = _write_task(tmp_path, **fields)

        task = loader.load_task_config(tf)

        assert task.simulation.camera.distribution.episode_profile == episode_profile

    def test_camera_distribution_defaults_to_anchor_only(self, tmp_path):
        task = loader.load_task_config(_write_task(tmp_path, **_camera_task_fields()))

        assert task.simulation.camera.distribution.episode_profile == "anchor_only"

    def test_camera_distribution_rejects_unknown_episode_profile(self, tmp_path):
        fields = _camera_task_fields()
        fields["simulation"]["camera"]["distribution"]["episode_profile"] = "random_cameras"
        tf = _write_task(tmp_path, **fields)

        with pytest.raises(ValueError, match="episode_profile"):
            loader.load_task_config(tf)

    def test_legacy_task_without_simulation_remains_valid(self, tmp_path):
        tf = _write_task(tmp_path)
        t = loader.load_task_config(tf)
        assert t.simulation is None
        assert t.ue.camera_mapping is None

    @pytest.mark.parametrize(
        ("block", "field"),
        [("lens", "unexpected"), ("profile", "unexpected"), ("distribution", "unexpected")],
    )
    def test_anchor_rejects_unknown_camera_fields(self, tmp_path, block, field):
        fields = _camera_task_fields()
        if block == "lens":
            fields["simulation"]["camera"]["profiles"]["C1"]["lens"][field] = 1
        elif block == "profile":
            fields["simulation"]["camera"]["profiles"]["C1"][field] = 1
        else:
            fields["simulation"]["camera"]["distribution"][field] = 1
        tf = _write_task(tmp_path, **fields)
        with pytest.raises(ValueError, match=f"{block}.*{field}"):
            loader.load_task_config(tf)

    def test_anchor_rejects_unknown_mapping_field(self, tmp_path):
        fields = _camera_task_fields()
        fields["ue"]["camera_mapping"]["C1"]["unexpected"] = 1
        tf = _write_task(tmp_path, **fields)
        with pytest.raises(ValueError, match="C1.*unexpected"):
            loader.load_task_config(tf)

    def test_anchor_requires_camera_mapping(self, tmp_path):
        fields = _camera_task_fields()
        del fields["ue"]["camera_mapping"]
        tf = _write_task(tmp_path, **fields)
        with pytest.raises(ValueError, match="camera_mapping"):
            loader.load_task_config(tf)

    @pytest.mark.parametrize("mapping_change", ["unknown", "missing", "incomplete"])
    def test_anchor_requires_exact_canonical_mapping_set(self, tmp_path, mapping_change):
        fields = _camera_task_fields()
        mapping = fields["ue"]["camera_mapping"]
        if mapping_change == "unknown":
            mapping["C6"] = mapping.pop("C5")
        elif mapping_change == "missing":
            del mapping["C5"]
        else:
            mapping.pop("C5")
            mapping["C6"] = {
                "actor": "CineCam_06",
                "sequence": "LS_Cam_06",
            }
        tf = _write_task(tmp_path, **fields)
        with pytest.raises(ValueError, match="C1.*C2.*C3.*C4.*C5"):
            loader.load_task_config(tf)

    @pytest.mark.parametrize("field", ["type", "coverage"])
    def test_anchor_rejects_unsupported_profile_values(self, tmp_path, field):
        fields = _camera_task_fields()
        fields["simulation"]["camera"]["profiles"]["C1"][field] = (
            "broadcast" if field == "type" else "partial"
        )
        tf = _write_task(tmp_path, **fields)
        with pytest.raises(ValueError, match=field):
            loader.load_task_config(tf)

    @pytest.mark.parametrize("resolution", [[0, 1080], [1920, 0], [1920], [1920, 1080, 1]])
    def test_anchor_rejects_invalid_resolution(self, tmp_path, resolution):
        fields = _camera_task_fields()
        fields["simulation"]["camera"]["profiles"]["C1"]["resolution"] = resolution
        tf = _write_task(tmp_path, **fields)
        with pytest.raises(ValueError, match="resolution"):
            loader.load_task_config(tf)

    @pytest.mark.parametrize(
        ("camera_id", "resolution"),
        [
            ("C1", [3840, 2160]),
            ("C5", [1280, 720]),
            ("C5", [2560, 1080]),
        ],
    )
    def test_anchor_rejects_resolution_outside_camera_allowlist(
        self, tmp_path, camera_id, resolution
    ):
        fields = _camera_task_fields()
        fields["simulation"]["camera"]["profiles"][camera_id]["resolution"] = resolution
        tf = _write_task(tmp_path, **fields)

        with pytest.raises(ValueError, match=f"resolution.*{camera_id}"):
            loader.load_task_config(tf)

    def test_anchor_allows_legacy_c5_resolution(self, tmp_path):
        fields = _camera_task_fields()
        fields["simulation"]["camera"]["profiles"]["C5"]["resolution"] = [1920, 1080]
        tf = _write_task(tmp_path, **fields)

        task = loader.load_task_config(tf)

        assert task.simulation.camera.profiles["C5"].resolution == [1920, 1080]

    def test_anchor_defaults_missing_resolution(self, tmp_path):
        fields = _camera_task_fields()
        for profile in fields["simulation"]["camera"]["profiles"].values():
            del profile["resolution"]
        tf = _write_task(tmp_path, **fields)

        task = loader.load_task_config(tf)

        assert {
            camera_id: profile.resolution
            for camera_id, profile in task.simulation.camera.profiles.items()
        } == {camera_id: [1920, 1080] for camera_id in ("C1", "C2", "C3", "C4", "C5")}

    @pytest.mark.parametrize("field", ["position_m", "rotation_deg"])
    def test_anchor_rejects_non_three_dimensional_vectors(self, tmp_path, field):
        fields = _camera_task_fields()
        fields["simulation"]["camera"]["profiles"]["C1"][field] = [1.0, 2.0]
        tf = _write_task(tmp_path, **fields)
        with pytest.raises(ValueError, match="三维向量"):
            loader.load_task_config(tf)

    def test_anchor_rejects_height_mismatch(self, tmp_path):
        fields = _camera_task_fields()
        fields["simulation"]["camera"]["profiles"]["C1"]["height_m"] = 7.0
        tf = _write_task(tmp_path, **fields)
        with pytest.raises(ValueError, match="height_m"):
            loader.load_task_config(tf)

    def test_anchor_rejects_fov_only_lens_definition(self, tmp_path):
        fields = _camera_task_fields()
        fields["simulation"]["camera"]["profiles"]["C1"]["lens"] = {
            "horizontal_fov_deg": 72.0
        }
        tf = _write_task(tmp_path, **fields)
        with pytest.raises(ValueError, match="focal_length_mm"):
            loader.load_task_config(tf)

    def test_partial_p01_profile_is_resolved_from_coverage_intent(self, tmp_path):
        tf = _write_task(tmp_path, **_partial_camera_task_fields())

        task = loader.load_task_config(tf)
        profile = task.simulation.camera.profiles["P01"]

        assert profile.coverage == "partial_field"
        assert profile.region == "left_half"
        assert profile.position_m == [-10.0, -25.0, 16.0]
        assert profile.height_m == 16.0
        assert profile.lens.focal_length_mm == 12.0
        assert profile.resolution == [1920, 1080]

    def test_p01_profile_requires_p01_mapping(self, tmp_path):
        fields = _partial_camera_task_fields()
        del fields["ue"]["camera_mapping"]["P01"]
        tf = _write_task(tmp_path, **fields)

        with pytest.raises(ValueError, match="P01"):
            loader.load_task_config(tf)

    def test_p01_profile_and_mapping_are_accepted(self, tmp_path):
        tf = _write_task(tmp_path, **_p01_mapping_fields())

        task = loader.load_task_config(tf)

        assert task.ue.camera_mapping["P01"].actor == "CineCam_P01"

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("coverage", "full_field", "coverage"),
            ("region", "right_half", "region"),
            ("placement_template", "wrong_template", "placement_template"),
        ],
    )
    def test_p01_rejects_non_implemented_partial_semantics(
        self, tmp_path, field, value, message
    ):
        fields = _partial_camera_task_fields()
        fields["simulation"]["camera"]["profiles"]["P01"][field] = value
        tf = _write_task(tmp_path, **fields)

        with pytest.raises(ValueError, match=message):
            loader.load_task_config(tf)

    def test_anchor_plus_one_partial_requires_p01_mapping(self, tmp_path):
        fields = _partial_camera_task_fields()
        del fields["ue"]["camera_mapping"]["P01"]
        fields["simulation"]["camera"]["distribution"]["episode_profile"] = (
            "anchor_plus_one_partial"
        )
        tf = _write_task(tmp_path, **fields)

        with pytest.raises(ValueError, match="P01"):
            loader.load_task_config(tf)

    def test_anchor_only_allows_optional_p01_profile_and_mapping(self, tmp_path):
        task = loader.load_task_config(_write_task(tmp_path, **_partial_camera_task_fields()))

        assert task.simulation.camera.distribution.episode_profile == "anchor_only"

    def test_anchor_plus_two_partial_is_rejected_until_second_partial_exists(self, tmp_path):
        fields = _partial_camera_task_fields()
        fields["simulation"]["camera"]["distribution"]["episode_profile"] = (
            "anchor_plus_two_partial"
        )
        tf = _write_task(tmp_path, **fields)

        with pytest.raises(ValueError, match="anchor_plus_two_partial|P02"):
            loader.load_task_config(tf)

    def test_unknown_camera_mapping_id_is_rejected(self, tmp_path):
        fields = _camera_task_fields()
        fields["ue"]["camera_mapping"]["P02"] = {
            "actor": "CineCam_P02",
            "sequence": "LS_Cam_P02",
        }
        tf = _write_task(tmp_path, **fields)

        with pytest.raises(ValueError, match="未知.*P02"):
            loader.load_task_config(tf)

    def test_full_field_profile_rejects_region(self, tmp_path):
        fields = _camera_task_fields()
        fields["simulation"]["camera"]["profiles"]["C1"]["region"] = "left_half"
        tf = _write_task(tmp_path, **fields)

        with pytest.raises(ValueError, match="full_field.*region"):
            loader.load_task_config(tf)

    def test_unknown_profile_id_is_rejected(self, tmp_path):
        fields = _camera_task_fields()
        fields["simulation"]["camera"]["profiles"]["P02"] = {
            "type": "static_surveillance",
            "coverage": "full_field",
            "resolution": [1920, 1080],
            "lens": {"focal_length_mm": 15.0},
            "position_m": [0.0, 0.0, 8.0],
            "rotation_deg": [0.0, 0.0, 0.0],
            "height_m": 8.0,
        }
        tf = _write_task(tmp_path, **fields)

        with pytest.raises(ValueError, match="只允许额外 P01"):
            loader.load_task_config(tf)

    @pytest.mark.parametrize(
        ("camera_id", "expected"),
        [("C1", 15.0), ("C2", 15.0), ("C3", 15.0), ("C4", 15.0), ("C5", 10.0)],
    )
    def test_anchor_defaults_focal_length_when_lens_is_omitted(
        self, tmp_path, camera_id, expected
    ):
        fields = _camera_task_fields()
        for profile in fields["simulation"]["camera"]["profiles"].values():
            profile.pop("lens")
        tf = _write_task(tmp_path, **fields)

        task = loader.load_task_config(tf)

        assert task.simulation.camera.profiles[camera_id].lens.focal_length_mm == expected

    @pytest.mark.parametrize(
        ("camera_id", "focal_length"),
        [("C1", 12.0), ("C1", 15.0), ("C1", 18.0), ("C5", 8.0), ("C5", 10.0), ("C5", 12.0)],
    )
    def test_anchor_allows_canonical_focal_lengths(
        self, tmp_path, camera_id, focal_length
    ):
        fields = _camera_task_fields()
        fields["simulation"]["camera"]["profiles"][camera_id]["lens"] = {
            "focal_length_mm": focal_length
        }
        tf = _write_task(tmp_path, **fields)

        task = loader.load_task_config(tf)

        assert task.simulation.camera.profiles[camera_id].lens.focal_length_mm == focal_length

    @pytest.mark.parametrize(
        ("camera_id", "focal_length"),
        [("C1", 8.0), ("C1", 20.0), ("C5", 15.0), ("C5", 6.0)],
    )
    def test_anchor_rejects_focal_length_outside_camera_allowlist(
        self, tmp_path, camera_id, focal_length
    ):
        fields = _camera_task_fields()
        fields["simulation"]["camera"]["profiles"][camera_id]["lens"] = {
            "focal_length_mm": focal_length
        }
        tf = _write_task(tmp_path, **fields)

        with pytest.raises(ValueError, match=f"focal_length_mm.*{camera_id}"):
            loader.load_task_config(tf)

    def test_legacy_fov_is_only_a_consistency_assertion(self, tmp_path):
        fields = _camera_task_fields()
        fields["simulation"]["camera"]["profiles"]["C1"]["lens"] = {
            "focal_length_mm": 15.0,
            "horizontal_fov_deg": 76.7584,
        }
        tf = _write_task(tmp_path, **fields)

        task = loader.load_task_config(tf)

        assert task.simulation.camera.profiles["C1"].lens.focal_length_mm == 15.0

    def test_legacy_fov_conflict_is_rejected(self, tmp_path):
        fields = _camera_task_fields()
        fields["simulation"]["camera"]["profiles"]["C1"]["lens"] = {
            "focal_length_mm": 15.0,
            "horizontal_fov_deg": 72.0,
        }
        tf = _write_task(tmp_path, **fields)

        with pytest.raises(ValueError, match="horizontal_fov_deg.*focal_length_mm"):
            loader.load_task_config(tf)

    @pytest.mark.parametrize(
        ("mapping", "missing_field"),
        [({"actor": "CineCam_01"}, "sequence"),
         ({"sequence": "LS_Cam_01"}, "actor")],
    )
    def test_anchor_rejects_malformed_camera_mapping(
        self, tmp_path, mapping, missing_field
    ):
        fields = _camera_task_fields()
        fields["ue"]["camera_mapping"]["C1"] = mapping
        tf = _write_task(tmp_path, **fields)
        with pytest.raises(ValueError, match=f"C1.*{missing_field}"):
            loader.load_task_config(tf)
