"""task schema（单 config）测试。"""

from __future__ import annotations

import json
import typing
from pathlib import Path

import pytest
from pydantic import ValidationError

from grf_ue_bridge.config import loader
from grf_ue_bridge.config.models import (
    AuditTaskConfig,
    DatasetTaskConfig,
    LocalMachineConfig,
    PostprocessTaskConfig,
    TaskConfigV3,
)
from grf_ue_bridge.config.paths import LOCAL_CONFIG_ENV, TASK_V3_SCHEMA


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


class TestTaskSchema:
    def test_v2_load_emits_config_v2_deprecation_warning(self, tmp_path):
        tf = _write_task(tmp_path)
        with pytest.warns(DeprecationWarning, match="Config v2"):
            loader.load_task_config(tf)

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


def _v3_task(**over):
    data = {
        "schema": "futsalmot_task",
        "version": 3,
        "episode_id": "0001",
        "simulation": {"scenario": "5_vs_5", "seed": 42, "steps": 300},
        "cameras": {"C01": "CineCam_01"},
        "output": {
            "fps": 30,
            "resolution": [1920, 1080],
            "annotations": ["mot", "pose", "mots"],
            "classes": ["player", "ball"],
        },
        "debug": False,
    }
    data.update(over)
    return data


class TestTaskConfigV3:
    def test_loader_type_hints_resolve_at_runtime(self):
        hints = typing.get_type_hints(loader.load_task_config)
        assert hints["return"] == typing.Union[DatasetTaskConfig, TaskConfigV3]

    def test_valid_v3_task(self):
        task = TaskConfigV3(**_v3_task())
        assert task.schema_ == TASK_V3_SCHEMA
        assert task.episode_id == "0001"
        assert task.simulation.steps == 300
        assert task.output.resolution == [1920, 1080]

    def test_loader_dispatches_v3(self, tmp_path):
        tf = tmp_path / "task.json"
        tf.write_text(json.dumps(_v3_task()), encoding="utf-8")
        task = loader.load_task_config(tf)
        assert isinstance(task, TaskConfigV3)

    def test_v3_rejects_float_version(self):
        data = _v3_task(version=3.0)
        with pytest.raises(ValidationError) as error:
            TaskConfigV3(**data)
        assert error.value.errors()[0]["loc"] == ("version",)

    @pytest.mark.parametrize(
        "changes",
        [{"schema": None}, {"schema": "other"}, {"version": None}, {"version": 2}],
    )
    def test_v3_requires_exact_schema_and_version(self, changes):
        data = _v3_task()
        data.update(changes)
        with pytest.raises(ValidationError) as error:
            TaskConfigV3(**data)
        assert any(item["loc"][-1] == next(iter(changes)) for item in error.value.errors())

    @pytest.mark.parametrize("field", ["schema", "version"])
    def test_v3_requires_schema_and_version_fields(self, field):
        data = _v3_task()
        del data[field]
        with pytest.raises(ValidationError) as error:
            TaskConfigV3(**data)
        assert error.value.errors()[0]["loc"][-1] == field

    @pytest.mark.parametrize("version", [None, 2, 4])
    def test_loader_rejects_invalid_v3_version(self, tmp_path, version):
        data = _v3_task(version=version)
        tf = tmp_path / "task.json"
        tf.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match="version"):
            loader.load_task_config(tf)

    @pytest.mark.parametrize("field", ["dataset_root", "ue_project_root", "export", "ue", "postprocess", "audit", "artifact_policy", "task_id", "episode_name"])
    def test_v3_rejects_legacy_top_level_fields(self, field):
        with pytest.raises(ValidationError, match=field):
            TaskConfigV3(**_v3_task(**{field: "legacy"}))

    @pytest.mark.parametrize("episode_id", ["../bad", "has/slash", "", "a b"])
    def test_v3_rejects_unsafe_episode_id(self, episode_id):
        with pytest.raises(ValidationError, match="episode_id"):
            TaskConfigV3(**_v3_task(episode_id=episode_id))

    @pytest.mark.parametrize(
        "changes",
        [
            {"simulation": {"scenario": "unknown", "steps": 1}},
            {"simulation": {"scenario": "5_vs_5", "steps": 0}},
            {"cameras": {"cam1": "CineCam_01"}},
            {"cameras": {"C01": ""}},
            {"output": {"fps": 30, "resolution": [0, 1080], "annotations": ["mot"], "classes": ["player"]}},
            {"output": {"fps": 30, "resolution": [1920, 1080], "annotations": [], "classes": ["player"]}},
            {"output": {"fps": 30, "resolution": [1920, 1080], "annotations": ["boxes"], "classes": ["player"]}},
            {"output": {"fps": 30, "resolution": [1920, 1080], "annotations": ["mot"], "classes": ["person"]}},
        ],
    )
    def test_v3_rejects_invalid_nested_values(self, changes):
        data = _v3_task()
        for key, value in changes.items():
            data[key] = value
        with pytest.raises(ValidationError):
            TaskConfigV3(**data)

    @pytest.mark.parametrize(
        "field, value",
        [
            ("annotations", ["mot", "mot"]),
            ("classes", ["player", "player"]),
        ],
    )
    def test_v3_rejects_duplicate_output_entries(self, field, value):
        data = _v3_task()
        data["output"][field] = value

        with pytest.raises(ValidationError, match=field):
            TaskConfigV3(**data)

    def test_v3_rejects_duplicate_camera_actors(self):
        with pytest.raises(ValidationError, match="actor"):
            TaskConfigV3(**_v3_task(cameras={"C03": "FrontCamera", "C07": "FrontCamera"}))


class TestLocalMachineConfig:
    def test_accepts_machine_fields_only(self):
        config = LocalMachineConfig(dataset_root="D:/dataset", ue_project_root="D:/ue")
        assert config.dataset_root == "D:/dataset"
        assert config.ue_project_root == "D:/ue"
        assert LOCAL_CONFIG_ENV == "FUTSALMOT_LOCAL_CONFIG"

    @pytest.mark.parametrize("field", ["episode_id", "cameras", "fps", "annotations", "classes", "debug"])
    def test_rejects_task_fields(self, field):
        with pytest.raises(ValidationError, match=field):
            LocalMachineConfig(dataset_root="D:/dataset", ue_project_root="D:/ue", **{field: False})

    def test_example_is_placeholder_only(self):
        example = Path(__file__).parents[1] / "configs" / "local.machine.example.json"
        data = json.loads(example.read_text(encoding="utf-8"))
        assert data == {"dataset_root": "<DATASET_ROOT>", "ue_project_root": "<UE_PROJECT_ROOT>"}
        assert "configs/local.machine.json" in (Path(__file__).parents[1] / ".gitignore").read_text(encoding="utf-8")
