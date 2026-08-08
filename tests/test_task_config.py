"""task schema（单 config）测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from grf_ue_bridge.config import loader
from grf_ue_bridge.config.models import (
    AuditTaskConfig,
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

    def test_defaults(self):
        pp = PostprocessTaskConfig()
        assert pp.workers == 4
        assert pp.validation_level == "full"
        assert "yolo-seg" in pp.formats
        au = AuditTaskConfig()
        assert au.expected_cameras == 4
        assert au.expected_frames_per_camera == 300
