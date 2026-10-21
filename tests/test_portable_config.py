"""Portable Config 路径解析和 task CLI 集成测试。"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest
from typer.testing import CliRunner

from grf_ue_bridge.cli import app
from grf_ue_bridge.appearance import APPEARANCE_POLICY, build_appearance_manifest
from grf_ue_bridge.config import resolver
from grf_ue_bridge.seeds import derive_episode_seeds


def _appearance_catalog() -> dict:
    return {
        "mode": "seeded",
        "policy": APPEARANCE_POLICY,
        "catalog_id": "mpfb_players_003_009_v1",
        "models": [
            {
                "player_id": player_id,
                "mesh": f"/Game/Players/Player_{player_id:03d}",
                "anim_class": "/Game/Core/ABP_RuntimeRetarget",
                "skeleton": "/Game/Core/SKEL_MPFB_Base",
            }
            for player_id in range(3, 10)
        ],
        "skins": [
            {
                "skin_id": f"skin-{index:02d}",
                "texture": f"/Game/Textures/T_Skin_{index:02d}",
            }
            for index in range(1, 10)
        ],
        "kits": [
            {
                "set_id": f"{index:02d}",
                "jersey_material": f"/Game/Materials/M_Jersey_Set_{index:02d}",
                "shorts_material": f"/Game/Materials/M_Shorts_Set_{index:02d}",
            }
            for index in range(1, 6)
        ],
        "body_texture_parameter": "SkinTexture",
    }


def _appearance_meta(*, episode_id: str = "episode_portable_t1") -> dict:
    entities = [
        {
            "id": entity_id,
            "team": "left" if entity_id.startswith("L") else "right",
            "is_goalkeeper": entity_id in {"L2", "R1"},
        }
        for entity_id in (
            "L0", "L1", "L2", "L3", "L4",
            "R0", "R1", "R2", "R3", "R4",
        )
    ]
    return {
        "schema": "grf_ue_episode",
        "version": 1,
        "episode_id": episode_id,
        "source": {"seed": 42},
        "randomness": derive_episode_seeds(42).model_dump(),
        "timing": {
            "source_step_seconds": 0.1,
            "playback_fps": 10,
            "num_steps": 1,
        },
        "field": {
            "length_m": 40.0,
            "width_m": 20.0,
        },
        "entities": entities + [{"id": "BALL", "team": None, "is_goalkeeper": False}],
    }


def _appearance_frame() -> dict:
    return {
        "step": 0,
        "time_seconds": 0.0,
        "score": [0, 0],
        "ball": {"position_m": [0.0, 0.0, 0.11]},
        "players": [
            {
                "id": entity_id,
                "position_m": [0.0, 0.0, 0.0],
            }
            for entity_id in (
                "L0", "L1", "L2", "L3", "L4",
                "R0", "R1", "R2", "R3", "R4",
            )
        ],
        "ball_owned_team": -1,
        "ball_owned_player": -1,
        "game_mode": 0,
    }


def _make_task(
    tmp_path: Path,
    dataset_root: str,
    ue_project_root: str,
    *,
    appearance: bool = False,
) -> Path:
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
    if appearance:
        task["ue"]["appearance"] = _appearance_catalog()
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
    resolved = resolver.resolve_task(
        task_file, dataset_root=str(dataset_root), ue_project_root=str(ue)
    )

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
    assert task_export.run_export(resolved, print_fn=lambda message: None) == 0
    assert dataset_root.is_dir()
    provenance = Path(resolved.trajectory_output) / "provenance"
    assert (provenance / "ue-profile.json").is_file()
    assert (provenance / "export-profile.json").is_file()
    assert not (provenance / "appearance.json").exists()


def test_export_writes_appearance_manifest_after_provenance(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace
    from grf_ue_bridge.workflows import task_export

    dataset_root = tmp_path / "appearance-data"
    ue = _make_ue_root(tmp_path)
    task_file = _make_task(
        tmp_path, str(dataset_root), str(ue), appearance=True
    )
    resolved = resolver.resolve_task(
        task_file, dataset_root=str(dataset_root), ue_project_root=str(ue)
    )

    class FakeExportConfig:
        seed = 42
        scenario = "5_vs_5"
        num_steps = 1
        trajectory_time_scale = 1.0
        render = False
        game_duration = None
        left_team_difficulty = 0
        right_team_difficulty = 0
        number_of_left_players_agent_controls = 0
        number_of_right_players_agent_controls = 0

    def fake_export(_config, _result, traj):
        traj.mkdir(parents=True, exist_ok=True)
        (traj / "meta.json").write_text(
            json.dumps(_appearance_meta(episode_id=traj.name)), encoding="utf-8"
        )
        (traj / "frames.jsonl").write_text(
            json.dumps(_appearance_frame()) + "\n", encoding="utf-8"
        )

    monkeypatch.setitem(
        sys.modules,
        "grf_ue_bridge.config.models",
        SimpleNamespace(ExportConfig=lambda **kwargs: FakeExportConfig()),
    )
    monkeypatch.setitem(
        sys.modules,
        "grf_ue_bridge.exporter",
        SimpleNamespace(compute_source_steps=lambda cfg: 1, export_episode=fake_export),
    )
    monkeypatch.setitem(
        sys.modules,
        "grf_ue_bridge.grf_runner",
        SimpleNamespace(run_episode=lambda **kwargs: {}),
    )

    assert task_export.run_export(resolved, print_fn=lambda message: None) == 0

    episode_dir = Path(resolved.trajectory_output)
    manifest_path = episode_dir / "provenance" / "appearance.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = build_appearance_manifest(
        _appearance_meta(episode_id=episode_dir.name),
        episode_dir / "frames.jsonl",
        resolved.ue_profile["appearance"],
    )
    assert manifest["episode_id"] == episode_dir.name
    assert manifest["root_seed"] == 42
    assert manifest["catalog_fingerprint"] == expected["catalog_fingerprint"]
    assert manifest["trajectory_sha256"] == expected["trajectory_sha256"]
    assert manifest["fingerprint"] == expected["fingerprint"]
    assert (episode_dir / "provenance" / "ue-profile.json").is_file()


@pytest.mark.parametrize("failure", ["bad_roster", "missing_frames", "corrupt_frames"])
def test_appearance_export_failure_removes_only_appearance_manifest(
    monkeypatch, tmp_path, failure
):
    import sys
    from types import SimpleNamespace
    from grf_ue_bridge.workflows import task_export

    dataset_root = tmp_path / f"appearance-failure-{failure}"
    ue = _make_ue_root(tmp_path)
    task_file = _make_task(
        tmp_path, str(dataset_root), str(ue), appearance=True
    )
    resolved = resolver.resolve_task(
        task_file, dataset_root=str(dataset_root), ue_project_root=str(ue)
    )

    class FakeExportConfig:
        seed = 42
        scenario = "5_vs_5"
        num_steps = 1
        trajectory_time_scale = 1.0
        render = False
        game_duration = None
        left_team_difficulty = 0
        right_team_difficulty = 0
        number_of_left_players_agent_controls = 0
        number_of_right_players_agent_controls = 0

    def fake_export(_config, _result, traj):
        provenance = traj / "provenance"
        provenance.mkdir(parents=True, exist_ok=True)
        (provenance / "keep.json").write_text("keep", encoding="utf-8")
        (provenance / "appearance.json").write_text(
            '{"stale":true}', encoding="utf-8"
        )
        meta = _appearance_meta(episode_id=traj.name)
        if failure == "bad_roster":
            meta["entities"] = [
                entity for entity in meta["entities"] if entity.get("id") != "L2"
            ]
        (traj / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        if failure == "corrupt_frames":
            (traj / "frames.jsonl").write_bytes(b"not json\n")
        elif failure != "missing_frames":
            (traj / "frames.jsonl").write_text(
                json.dumps(_appearance_frame()) + "\n", encoding="utf-8"
            )

    monkeypatch.setitem(
        sys.modules,
        "grf_ue_bridge.config.models",
        SimpleNamespace(ExportConfig=lambda **kwargs: FakeExportConfig()),
    )
    monkeypatch.setitem(
        sys.modules,
        "grf_ue_bridge.exporter",
        SimpleNamespace(compute_source_steps=lambda cfg: 1, export_episode=fake_export),
    )
    monkeypatch.setitem(
        sys.modules,
        "grf_ue_bridge.grf_runner",
        SimpleNamespace(run_episode=lambda **kwargs: {}),
    )

    with pytest.raises(ValueError):
        task_export.run_export(resolved, print_fn=lambda message: None)

    provenance = Path(resolved.trajectory_output) / "provenance"
    assert not (provenance / "appearance.json").exists()
    assert (provenance / "keep.json").read_text(encoding="utf-8") == "keep"


@pytest.fixture(autouse=True)
def isolate_runtime_machine_paths(monkeypatch, tmp_path):
    """路径优先级测试只读取临时仓库，避免开发机 local.json 或环境变量污染。"""
    from grf_ue_bridge.config import paths

    monkeypatch.setattr(paths, "default_repo_root", lambda: tmp_path / "repo")
    for name in ("FUTSALMOT_DATASET_ROOT", "FUTSALMOT_UE_ROOT", "FUTSALMOT_UE_PROJECT_ROOT"):
        monkeypatch.delenv(name, raising=False)
