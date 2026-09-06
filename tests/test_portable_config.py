"""Portable Config 路径解析和 task CLI 集成测试。"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest
from typer.testing import CliRunner

from grf_ue_bridge.cli import app
from grf_ue_bridge.config import resolver


def _make_task(tmp_path: Path, dataset_root: str, ue_project_root: str) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    task = {
        "schema": "futsalmot_dataset_task",
        "version": 2,
        "task_id": "portable_t1",
        "episode_name": "episode_portable_t1",
        "dataset_root": dataset_root,
        "ue_project_root": ue_project_root,
        "export": {"scenario": "5_vs_5", "seed": 42, "num_steps": 300,
                    "playback_fps": 30},
        "ue": {"actor_mapping": "ue/actor_mapping.example.json",
               "sequences": [{"name": "LS_Cam_01", "camera_actor": "CineCam_01"}],
               "annotation_export": {"cameras": ["CineCam_01"],
                                     "image_width": 1920, "image_height": 1080}},
        "postprocess": {"workers": 1, "validation_level": "full"},
        "audit": {"expected_cameras": 1, "expected_frames_per_camera": 300},
    }
    path = repo / "task.json"
    path.write_text(json.dumps(task), encoding="utf-8")
    return path


def _write_local(repo_root: Path, dataset_root: str, ue_project_root: str) -> Path:
    config = repo_root / ".futsalmot" / "local.json"
    config.parent.mkdir()
    config.write_text(json.dumps({"paths": {
        "dataset_root": dataset_root,
        "ue_project_root": ue_project_root,
    }}), encoding="utf-8")
    return config


def _make_ue_root(tmp_path: Path) -> Path:
    root = tmp_path / "ue"
    root.mkdir()
    (root / "FutsalMOT.uproject").write_text("{}", encoding="utf-8")
    return root


def test_environment_expands_task_placeholders(tmp_path, monkeypatch):
    ue = _make_ue_root(tmp_path)
    monkeypatch.setenv("FUTSALMOT_DATASET_ROOT", str(tmp_path / "env-data"))
    monkeypatch.setenv("FUTSALMOT_UE_ROOT", str(ue))
    task_file = _make_task(tmp_path, "${FUTSALMOT_DATASET_ROOT}", "${FUTSALMOT_UE_ROOT}")

    resolved = resolver.resolve_task(task_file)

    assert Path(resolved.dataset_root) == (tmp_path / "env-data").resolve()
    assert Path(resolved.ue_project_root) == ue.resolve()


def test_ue_project_root_environment_alias_is_supported(tmp_path, monkeypatch):
    ue = _make_ue_root(tmp_path)
    monkeypatch.setenv("FUTSALMOT_DATASET_ROOT", str(tmp_path / "env-data"))
    monkeypatch.setenv("FUTSALMOT_UE_PROJECT_ROOT", str(ue))
    task_file = _make_task(tmp_path, "${FUTSALMOT_DATASET_ROOT}", "${FUTSALMOT_UE_PROJECT_ROOT}")

    resolved = resolver.resolve_task(task_file)

    assert Path(resolved.ue_project_root) == ue.resolve()


def test_local_config_overrides_environment_and_task(tmp_path, monkeypatch):
    local_data = tmp_path / "local-data"
    local_ue = _make_ue_root(tmp_path)
    monkeypatch.setenv("FUTSALMOT_DATASET_ROOT", str(tmp_path / "env-data"))
    monkeypatch.setenv("FUTSALMOT_UE_ROOT", str(tmp_path / "env-ue"))
    task_file = _make_task(tmp_path, "${FUTSALMOT_DATASET_ROOT}", "${FUTSALMOT_UE_ROOT}")
    _write_local(task_file.parent, str(local_data), str(local_ue))

    resolved = resolver.resolve_task(task_file)

    assert Path(resolved.dataset_root) == local_data.resolve()
    assert Path(resolved.ue_project_root) == local_ue.resolve()


def test_local_config_missing_required_path_fails(tmp_path):
    ue = _make_ue_root(tmp_path)
    task_file = _make_task(tmp_path, "legacy-data", str(ue))
    config = task_file.parent / ".futsalmot" / "local.json"
    config.parent.mkdir()
    config.write_text(json.dumps({"paths": {"dataset_root": "local-data"}}), encoding="utf-8")

    with pytest.raises(ValueError, match="required path: ue_project_root"):
        resolver.resolve_task(task_file)


def test_explicit_cli_resolver_overrides_local_config(tmp_path):
    local_ue = _make_ue_root(tmp_path)
    cli_ue = tmp_path / "cli-ue"
    cli_ue.mkdir()
    (cli_ue / "Cli.uproject").write_text("{}", encoding="utf-8")
    task_file = _make_task(tmp_path, "legacy-data", str(local_ue))
    _write_local(task_file.parent, "local-data", str(local_ue))

    resolved = resolver.resolve_task(
        task_file,
        dataset_root=str(tmp_path / "cli-data"),
        ue_project_root=str(cli_ue),
    )

    assert Path(resolved.dataset_root) == (tmp_path / "cli-data").resolve()
    assert Path(resolved.ue_project_root) == cli_ue.resolve()


def test_legacy_absolute_paths_warn(tmp_path):
    ue = _make_ue_root(tmp_path)
    dataset = (tmp_path / "legacy-data").resolve()
    task_file = _make_task(tmp_path, str(dataset), str(ue))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolved = resolver.resolve_task(task_file)

    assert Path(resolved.dataset_root) == dataset
    assert any("deprecated absolute path" in str(item.message) for item in caught)


def test_missing_runtime_variable_fails(tmp_path):
    task_file = _make_task(tmp_path, "${UNKNOWN_ROOT}", "${UNKNOWN_UE_ROOT}")

    with pytest.raises(ValueError, match="missing runtime path variable"):
        resolver.resolve_task(task_file)


def test_validate_allows_missing_dataset_root_but_requires_uproject(tmp_path):
    task_file = _make_task(tmp_path, str(tmp_path / "not-created"), str(tmp_path / "missing-ue"))

    problems = resolver.validate_task(task_file)

    assert any(".uproject" in problem for problem in problems)
    assert not any("dataset_root" in problem and "不存在" in problem for problem in problems)


def test_validate_accepts_missing_dataset_root_with_valid_ue(tmp_path):
    ue = _make_ue_root(tmp_path)
    task_file = _make_task(tmp_path, str(tmp_path / "not-created"), str(ue))

    assert not resolver.validate_task(task_file)


def test_cli_validate_accepts_path_overrides(tmp_path):
    ue = _make_ue_root(tmp_path)
    task_file = _make_task(tmp_path, "${MISSING_DATASET}", "${MISSING_UE}")
    runner = CliRunner()

    result = runner.invoke(app, [
        "task", "validate", str(task_file),
        "--dataset-root", str(tmp_path / "cli-data"),
        "--ue-project-root", str(ue),
    ])

    assert result.exit_code == 0, result.output


def test_cli_resolve_uses_local_config(tmp_path, monkeypatch):
    ue = _make_ue_root(tmp_path)
    task_file = _make_task(tmp_path, "legacy-data", str(ue))
    local_data = tmp_path / "local-data"
    _write_local(task_file.parent, str(local_data), str(ue))
    monkeypatch.chdir(task_file.parent)

    result = CliRunner().invoke(app, ["task", "resolve", str(task_file)])

    assert result.exit_code == 0, result.output
    assert str(local_data.resolve()) in result.output


def test_portable_resolved_task_round_trip_keeps_absolute_paths(tmp_path, monkeypatch):
    ue = _make_ue_root(tmp_path)
    dataset = tmp_path / "portable-data"
    monkeypatch.setenv("FUTSALMOT_DATASET_ROOT", str(dataset))
    monkeypatch.setenv("FUTSALMOT_UE_ROOT", str(ue))
    task_file = _make_task(
        tmp_path, "${FUTSALMOT_DATASET_ROOT}", "${FUTSALMOT_UE_ROOT}"
    )

    resolved = resolver.resolve_task(task_file)
    runtime_file = resolver.save_resolved_task(resolved, Path(resolved.repo_root))
    loaded = resolver.load_resolved_task(runtime_file)

    assert Path(loaded.dataset_root) == dataset.resolve()
    assert Path(loaded.ue_project_root) == ue.resolve()
    assert Path(loaded.dataset_episode_dir).is_absolute()


def test_export_creates_missing_dataset_root(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace
    from grf_ue_bridge.workflows import task_export

    dataset_root = tmp_path / "created-at-export"
    ue = _make_ue_root(tmp_path)
    task_file = _make_task(tmp_path, str(dataset_root), str(ue))
    resolved = resolver.resolve_task(task_file)

    class FakeExportConfig:
        seed = 1
        scenario = "5_vs_5"
        num_steps = 1
        trajectory_time_scale = 1.0
        render = False
        game_duration = None
        left_team_difficulty = 0
        right_team_difficulty = 0
        number_of_left_players_agent_controls = 0
        number_of_right_players_agent_controls = 0

    monkeypatch.setitem(
        sys.modules,
        "grf_ue_bridge.config.models",
        SimpleNamespace(ExportConfig=lambda **kwargs: FakeExportConfig()),
    )
    monkeypatch.setitem(
        sys.modules,
        "grf_ue_bridge.exporter",
        SimpleNamespace(
            compute_source_steps=lambda cfg: 1,
            export_episode=lambda cfg, result, traj: traj.mkdir(
                parents=True, exist_ok=True
            ),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "grf_ue_bridge.grf_runner",
        SimpleNamespace(run_episode=lambda **kwargs: {}),
    )
    monkeypatch.setattr(task_export, "_write_provenance", lambda resolved, traj: None)

    assert task_export.run_export(resolved, print_fn=lambda message: None) == 0
    assert dataset_root.is_dir()
