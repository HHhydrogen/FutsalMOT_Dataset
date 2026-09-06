"""Unified Run Manifest 的纯 Python 测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from grf_ue_bridge.run_manifest import (
    create_run_manifest,
    load_run_manifest,
    run_manifest_path,
    save_run_manifest,
    task_spec_hash,
    update_artifacts,
    update_validation,
)


def test_create_manifest_has_schema_and_source(tmp_path):
    task = tmp_path / "task.json"
    task.write_text('{"task_id":"t1","dataset_root":"${DATA}"}', encoding="utf-8")

    manifest = create_run_manifest("t1", task, tmp_path)

    assert manifest["schema"] == "futsalmot_run_manifest"
    assert manifest["version"] == 1
    assert manifest["task_id"] == "t1"
    assert manifest["source"]["task_hash"] == task_spec_hash(task)
    assert "code_commit" in manifest["source"]
    assert "ue_commit" in manifest["source"]
    assert manifest["runtime"]["started_at"]
    assert manifest["runtime"]["finished_at"] is None


def test_manifest_path_is_runtime_path(tmp_path):
    assert run_manifest_path(tmp_path, "task-1") == (
        tmp_path / ".futsalmot" / "runtime" / "task-1" / "run_manifest.json"
    )


def test_task_hash_is_raw_task_content(tmp_path):
    task = tmp_path / "task.json"
    task.write_bytes(b'{"dataset_root":"/machine/a"}\n')

    first = task_spec_hash(task)
    task.write_bytes(b'{"dataset_root":"/machine/b"}\n')

    assert first != task_spec_hash(task)


def test_update_validation_uses_audit_passed(tmp_path):
    task = tmp_path / "task.json"
    task.write_text("{}", encoding="utf-8")
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"passed": False}), encoding="utf-8")
    manifest = create_run_manifest("t1", task, tmp_path)

    update_validation(manifest, audit)

    assert manifest["validation"] == {"passed": False, "audit_report": str(audit)}


def test_artifact_summary_failure_is_recorded_without_raise(tmp_path):
    task = tmp_path / "task.json"
    task.write_text("{}", encoding="utf-8")
    manifest = create_run_manifest("t1", task, tmp_path)

    update_artifacts(manifest, tmp_path / "missing-episode")

    assert manifest["artifacts"].get("errors")


def test_manifest_atomic_save_and_schema_validation(tmp_path):
    task = tmp_path / "task.json"
    task.write_text("{}", encoding="utf-8")
    manifest = create_run_manifest("t1", task, tmp_path)
    path = tmp_path / "run_manifest.json"

    save_run_manifest(manifest, path)

    assert load_run_manifest(path)["task_id"] == "t1"
    assert not list(tmp_path.glob(".run_manifest.*.tmp"))


def test_load_rejects_unsupported_manifest_version(tmp_path):
    path = tmp_path / "run_manifest.json"
    path.write_text(json.dumps({"schema": "futsalmot_run_manifest", "version": 999}), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported run manifest version"):
        load_run_manifest(path)


def test_artifact_summary_counts_camera_files(tmp_path):
    task = tmp_path / "task.json"
    task.write_text("{}", encoding="utf-8")
    episode = tmp_path / "episode" / "Camera_01"
    (episode / "img1").mkdir(parents=True)
    (episode / "img1" / "000001.png").write_bytes(b"png")
    (episode / "annotations.jsonl").write_text("{}\n", encoding="utf-8")
    (episode / "camera.json").write_text("{}", encoding="utf-8")
    manifest = create_run_manifest("t1", task, tmp_path)

    update_artifacts(manifest, episode.parent)

    assert manifest["artifacts"]["cameras"]["count"] == 1
    assert manifest["artifacts"]["images"]["count"] == 1
    assert manifest["artifacts"]["annotations"]["count"] == 1
