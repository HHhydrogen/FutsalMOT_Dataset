"""run_task episode appearance 事务编排测试。"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ue import run_task


class FakeAppearanceTransaction:
    def __init__(self):
        self.restore_calls = []
        self.callback_ownership_transferred = False
        self.restore_result = None

    def mark_rendering(self):
        return None

    def restore(self, final_status, render_result=None):
        self.restore_calls.append((final_status, render_result))
        self.restore_result = {"ok": True, "state": "restored"}
        return self.restore_result


def test_legacy_mode_runs_without_an_appearance_transaction():
    calls = []

    result = run_task._run_with_optional_appearance(
        None, lambda: calls.append("downstream") or 17
    )

    assert result == 17
    assert calls == ["downstream"]


def test_pose_finalize_does_not_enable_appearance_application():
    assert not run_task._appearance_enabled_for_mode({"appearance": {}} , "pose-finalize")
    assert run_task._appearance_enabled_for_mode({"appearance": {}}, "full")


def test_static_player_visuals_and_seeded_appearance_are_mutually_exclusive():
    with pytest.raises(ValueError, match="不能同时启用"):
        run_task._validate_appearance_exclusivity(
            {"mode": "seeded"}, {"default_profile": "mpfb_player_003"}
        )


def test_synchronous_mode_restores_after_success_and_preserves_result(monkeypatch):
    transaction = FakeAppearanceTransaction()
    calls = []
    monkeypatch.setattr(run_task, "_apply_prepared_appearance", lambda _prepared: transaction)

    result = run_task._run_prepared_appearance_synchronously(
        object(), lambda: calls.append("downstream") or 23
    )

    assert result == 23
    assert calls == ["downstream"]
    assert transaction.restore_calls == [("success", None)]


@pytest.mark.parametrize("mode", ["sequence", "annotations"])
def test_sync_modes_restore_after_exception_and_preserve_exception(monkeypatch, mode):
    transaction = FakeAppearanceTransaction()
    failure = RuntimeError("downstream failed")
    monkeypatch.setattr(run_task, "_apply_prepared_appearance", lambda _prepared: transaction)
    with pytest.raises(RuntimeError, match="downstream failed"):
        run_task._run_prepared_appearance_synchronously(
            object(), lambda: (_ for _ in ()).throw(failure)
        )

    assert transaction.restore_calls == [("failed", None)]


@pytest.mark.parametrize("mode", ["sequence", "annotations"])
def test_sync_modes_apply_before_downstream_and_restore_after_return(monkeypatch, mode):
    transaction = FakeAppearanceTransaction()
    events = []
    monkeypatch.setattr(
        run_task,
        "_apply_prepared_appearance",
        lambda prepared: events.append(("apply", mode)) or transaction,
    )

    result = run_task._run_prepared_appearance_synchronously(
        {"mode": mode}, lambda: events.append(("downstream", mode)) or 0
    )

    assert result == 0
    assert events == [("apply", mode), ("downstream", mode)]
    assert transaction.restore_calls == [("success", None)]


def test_appearance_inputs_are_validated_before_all_ten_preflight(tmp_path, monkeypatch):
    episode_dir = tmp_path / "episode"
    provenance = episode_dir / "provenance"
    provenance.mkdir(parents=True)
    (episode_dir / "meta.json").write_text("{}", encoding="utf-8")
    (episode_dir / "frames.jsonl").write_text('{"step":0}\n', encoding="utf-8")
    (provenance / "appearance.json").write_text("{}", encoding="utf-8")
    events = []
    adapter = SimpleNamespace(
        validate_appearance_inputs=lambda *_args: events.append("validate") or [],
        preflight_episode_appearance=lambda *_args: events.append("preflight")
        or {"ok": True, "actors": [{"entity_id": "L0", "actor_path": "/Game/Actor.L0"}]},
    )
    monkeypatch.setitem(sys.modules, "episode_appearance", adapter)

    prepared = run_task._prepare_episode_appearance(
        {"appearance": {"mode": "seeded"}},
        "sequence",
        episode_dir,
        {"L0": "Player_L0"},
        tmp_path,
        "task_8",
    )

    assert prepared["manifest"] == {}
    assert events == ["validate", "preflight"]
    assert prepared["public_report_path"] == (
        episode_dir / "provenance" / "appearance_application.json"
    )


def test_existing_runtime_recovery_evidence_blocks_preflight_overwrite(tmp_path, monkeypatch):
    episode_dir = tmp_path / "episode"
    provenance = episode_dir / "provenance"
    provenance.mkdir(parents=True)
    (episode_dir / "meta.json").write_text("{}", encoding="utf-8")
    (episode_dir / "frames.jsonl").write_text('{"step":0}\n', encoding="utf-8")
    (provenance / "appearance.json").write_text("{}", encoding="utf-8")
    rollback_path, _ = run_task._appearance_runtime_paths(tmp_path, "task_8")
    rollback_path.parent.mkdir(parents=True)
    rollback_path.write_text("first snapshot", encoding="utf-8")
    adapter = SimpleNamespace(
        validate_appearance_inputs=lambda *_args: [],
        preflight_episode_appearance=lambda *_args: pytest.fail(
            "stale runtime evidence must be rejected before UE preflight"
        ),
    )
    monkeypatch.setitem(sys.modules, "episode_appearance", adapter)

    with pytest.raises(RuntimeError, match="拒绝覆盖"):
        run_task._prepare_episode_appearance(
            {"appearance": {"mode": "seeded"}},
            "sequence",
            episode_dir,
            {"L0": "Player_L0"},
            tmp_path,
            "task_8",
        )

    assert rollback_path.read_text(encoding="utf-8") == "first snapshot"


def _write_runtime_recovery_fixture(tmp_path, state="applied"):
    episode_dir = tmp_path / "episode"
    provenance = episode_dir / "provenance"
    provenance.mkdir(parents=True)
    (episode_dir / "meta.json").write_text("{}", encoding="utf-8")
    (episode_dir / "frames.jsonl").write_text('{"step":0}\n', encoding="utf-8")
    manifest = {
        "fingerprint": "manifest-fingerprint",
        "catalog_fingerprint": "catalog-fingerprint",
        "trajectory_sha256": "trajectory-hash",
        "episode_id": "episode",
        "catalog_id": "catalog",
        "body_texture_parameter": "SkinTexture",
        "entities": [
            {
                "entity_id": entity_id,
                "mesh": "/Game/Appearance/mesh-{}".format(entity_id),
                "anim_class": "/Game/Appearance/anim_class-{}".format(entity_id),
                "skeleton": "/Game/Appearance/skeleton-{}".format(entity_id),
                "skin_texture": "/Game/Appearance/skin_texture-{}".format(entity_id),
                "jersey_material": "/Game/Appearance/jersey_material-{}".format(entity_id),
                "shorts_material": "/Game/Appearance/shorts_material-{}".format(entity_id),
            }
            for entity_id in ("L0", "L1", "L2", "L3", "L4", "R0", "R1", "R2", "R3", "R4")
        ],
    }
    (provenance / "appearance.json").write_text(json.dumps(manifest), encoding="utf-8")
    rollback_path, report_path = run_task._appearance_runtime_paths(tmp_path, "task_8")
    rollback_path.parent.mkdir(parents=True)
    rollback = {"schema": "rollback", "transaction_id": "transaction-1"}
    report = {
        "schema": "application",
        "version": 1,
        "state": state,
        "manifest_fingerprint": "manifest-fingerprint",
        "catalog_fingerprint": "catalog-fingerprint",
        "trajectory_sha256": "trajectory-hash",
        "transaction_id": "transaction-1",
        "episode_id": "episode",
        "catalog_id": "catalog",
        "actors": [
            {
                "entity_id": entity_id,
                "actor_label": "Player_{}".format(entity_id),
                "apply_status": "applied",
                "restore_status": "not_started",
                "assets": {
                    field: "/Game/Appearance/{}-{}".format(field, entity_id)
                    for field in (
                        "mesh", "anim_class", "skeleton", "skin_texture",
                        "jersey_material", "shorts_material",
                    )
                },
                "slots": {
                    "body_texture_parameter": "SkinTexture",
                    "Body": "/Game/Appearance/skin_texture-{}".format(entity_id),
                    "Jersey": "/Game/Appearance/jersey_material-{}".format(entity_id),
                    "Shorts": "/Game/Appearance/shorts_material-{}".format(entity_id),
                },
            }
            for entity_id in ("L0", "L1", "L2", "L3", "L4", "R0", "R1", "R2", "R3", "R4")
        ],
        "mrq_jobs": [],
        "mrq_pending": state == "rendering",
        "errors": [],
        "warnings": [],
    }
    rollback_path.write_text(json.dumps(rollback), encoding="utf-8")
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return episode_dir, rollback_path, report_path, rollback, report


@pytest.mark.parametrize("state", ["applied", "rendering", "restore_failed"])
def test_valid_local_unresolved_evidence_is_restored_published_and_retired(
    tmp_path, monkeypatch, state
):
    episode_dir, rollback_path, report_path, rollback, report = _write_runtime_recovery_fixture(
        tmp_path, state
    )
    events = []

    class Adapter:
        @staticmethod
        def validate_appearance_inputs(*_args):
            return []

        @staticmethod
        def validate_local_recovery_evidence(snapshot, local_report, manifest, catalog):
            assert snapshot == rollback
            assert local_report["state"] == state
            assert manifest["fingerprint"] == "manifest-fingerprint"
            assert catalog["catalog_id"] == "catalog"
            return []

        @staticmethod
        def preflight_episode_appearance_recovery(_snapshot, _mapping, verify_restored=False):
            events.append("baseline-readback" if verify_restored else "recovery-preflight")
            return {"ok": True, "actors": []}

        @staticmethod
        def restore_episode_appearance(snapshot_path, local_report_path, final_status, **_kwargs):
            events.append(("restore", final_status, Path(snapshot_path).is_file()))
            restored = json.loads(Path(local_report_path).read_text(encoding="utf-8"))
            restored.update(state="restored", mrq_pending=False)
            Path(local_report_path).write_text(json.dumps(restored), encoding="utf-8")
            return {"ok": True, "state": "restored"}

        @staticmethod
        def read_application_report(path):
            return json.loads(Path(path).read_text(encoding="utf-8"))

        @staticmethod
        def write_application_report(path, value):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(json.dumps(value), encoding="utf-8")

        @staticmethod
        def preflight_episode_appearance(*_args):
            events.append("new-application-preflight")
            return {"ok": True, "actors": []}

    monkeypatch.setitem(sys.modules, "episode_appearance", Adapter)
    monkeypatch.setitem(sys.modules, "render_episode", SimpleNamespace(_ACTIVE_RENDER=None))

    prepared = run_task._prepare_episode_appearance(
        {"appearance": {"catalog_id": "catalog"}},
        "sequence",
        episode_dir,
        {"L0": "Player_L0"},
        tmp_path,
        "task_8",
    )

    public_report = episode_dir / "provenance" / "appearance_application.json"
    assert events[0] == "recovery-preflight"
    assert events[-2:] == ["baseline-readback", "new-application-preflight"]
    if state == "restored":
        assert len(events) == 3
    else:
        assert events[1] == ("restore", "failed", True)
    assert json.loads(public_report.read_text(encoding="utf-8"))["state"] == "restored"
    assert not rollback_path.exists()
    assert not report_path.exists()
    assert prepared["manifest"]["fingerprint"] == "manifest-fingerprint"


def test_rerun_after_restored_local_evidence_was_retired_is_allowed(tmp_path, monkeypatch):
    episode_dir, rollback_path, report_path, _, _ = _write_runtime_recovery_fixture(
        tmp_path, "restored"
    )

    class Adapter:
        @staticmethod
        def validate_appearance_inputs(*_args):
            return []

        @staticmethod
        def validate_local_recovery_evidence(*_args):
            return []

        @staticmethod
        def preflight_episode_appearance_recovery(*_args):
            return {"ok": True, "actors": []}

        @staticmethod
        def restore_episode_appearance(*_args, **_kwargs):
            pytest.fail("already-restored local evidence must not restore twice")

        @staticmethod
        def read_application_report(path):
            return json.loads(Path(path).read_text(encoding="utf-8"))

        @staticmethod
        def write_application_report(path, value):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(json.dumps(value), encoding="utf-8")

        @staticmethod
        def preflight_episode_appearance(*_args):
            return {"ok": True, "actors": []}

        @staticmethod
        def validate_local_recovery_evidence(*_args):
            return []

    monkeypatch.setitem(sys.modules, "episode_appearance", Adapter)
    monkeypatch.setitem(sys.modules, "render_episode", SimpleNamespace(_ACTIVE_RENDER=None))
    args = (
        {"appearance": {"catalog_id": "catalog"}},
        "sequence",
        episode_dir,
        {"L0": "Player_L0"},
        tmp_path,
        "task_8",
    )

    run_task._prepare_episode_appearance(*args)
    second = run_task._prepare_episode_appearance(*args)

    assert second is not None
    assert not rollback_path.exists()
    assert not report_path.exists()


@pytest.mark.parametrize("state", ["applied", "restore_failed"])
def test_failed_local_recovery_refuses_new_apply_and_preserves_evidence(
    tmp_path, monkeypatch, state
):
    episode_dir, rollback_path, report_path, _, _ = _write_runtime_recovery_fixture(
        tmp_path, state
    )
    original_snapshot = rollback_path.read_bytes()
    events = []

    class Adapter:
        @staticmethod
        def validate_appearance_inputs(*_args):
            return []

        @staticmethod
        def validate_local_recovery_evidence(*_args):
            return []

        @staticmethod
        def preflight_episode_appearance_recovery(*_args):
            return {"ok": True, "actors": []}

        @staticmethod
        def restore_episode_appearance(*_args, **_kwargs):
            events.append("restore")
            return {"ok": False, "state": "restore_failed", "errors": ["readback mismatch"]}

        @staticmethod
        def read_application_report(path):
            return json.loads(Path(path).read_text(encoding="utf-8"))

        @staticmethod
        def write_application_report(path, value):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(json.dumps(value), encoding="utf-8")

        @staticmethod
        def preflight_episode_appearance(*_args):
            events.append("new-application-preflight")
            return {"ok": True, "actors": []}

        @staticmethod
        def validate_local_recovery_evidence(*_args):
            return []

    monkeypatch.setitem(sys.modules, "episode_appearance", Adapter)
    monkeypatch.setitem(sys.modules, "render_episode", SimpleNamespace(_ACTIVE_RENDER=None))

    with pytest.raises(RuntimeError, match="恢复失败"):
        run_task._prepare_episode_appearance(
            {"appearance": {"catalog_id": "catalog"}},
            "sequence",
            episode_dir,
            {"L0": "Player_L0"},
            tmp_path,
            "task_8",
        )

    assert events == ["restore"]
    assert rollback_path.read_bytes() == original_snapshot
    assert report_path.is_file()


def test_local_recovery_identity_mismatch_refuses_without_retiring_evidence(
    tmp_path, monkeypatch
):
    episode_dir, rollback_path, report_path, _, _ = _write_runtime_recovery_fixture(
        tmp_path, "applied"
    )
    original_snapshot = rollback_path.read_bytes()

    class Adapter:
        @staticmethod
        def validate_appearance_inputs(*_args):
            return []

        @staticmethod
        def validate_local_recovery_evidence(*_args):
            return ["rollback transaction_id mismatch"]

        @staticmethod
        def preflight_episode_appearance_recovery(*_args):
            pytest.fail("mismatched evidence must fail before UE recovery gate")

    monkeypatch.setitem(sys.modules, "episode_appearance", Adapter)

    with pytest.raises(RuntimeError, match="恢复证据无效"):
        run_task._prepare_episode_appearance(
            {"appearance": {"catalog_id": "catalog"}},
            "sequence",
            episode_dir,
            {"L0": "Player_L0"},
            tmp_path,
            "task_8",
        )

    assert rollback_path.read_bytes() == original_snapshot
    assert report_path.is_file()


def test_local_evidence_retirement_failure_keeps_both_files_and_blocks_rerun(
    tmp_path, monkeypatch
):
    episode_dir, rollback_path, report_path, _, _ = _write_runtime_recovery_fixture(
        tmp_path, "restored"
    )
    original_snapshot = rollback_path.read_bytes()
    original_report = report_path.read_bytes()
    original_unlink = Path.unlink

    def fail_rollback_unlink(path, *args, **kwargs):
        if str(path) == str(report_path):
            raise PermissionError("local report is locked")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_rollback_unlink)

    class Adapter:
        @staticmethod
        def validate_appearance_inputs(*_args):
            return []

        @staticmethod
        def validate_local_recovery_evidence(*_args):
            return []

        @staticmethod
        def preflight_episode_appearance_recovery(*_args):
            return {"ok": True, "actors": []}

        @staticmethod
        def read_application_report(path):
            return json.loads(Path(path).read_text(encoding="utf-8"))

        @staticmethod
        def write_application_report(path, value):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(json.dumps(value), encoding="utf-8")

        @staticmethod
        def preflight_episode_appearance(*_args):
            pytest.fail("failed evidence retirement must block new apply")

        @staticmethod
        def preflight_episode_appearance_recovery(*_args):
            return {"ok": True, "actors": []}

    monkeypatch.setitem(sys.modules, "episode_appearance", Adapter)
    monkeypatch.setitem(sys.modules, "render_episode", SimpleNamespace(_ACTIVE_RENDER=None))

    with pytest.raises(RuntimeError, match="恢复证据清理失败"):
        run_task._prepare_episode_appearance(
            {"appearance": {"catalog_id": "catalog"}},
            "sequence",
            episode_dir,
            {"L0": "Player_L0"},
            tmp_path,
            "task_8",
        )

    assert rollback_path.read_bytes() == original_snapshot
    assert report_path.read_bytes() == original_report


def test_unresolved_public_report_from_another_transaction_blocks_local_recovery(
    tmp_path, monkeypatch
):
    episode_dir, rollback_path, report_path, _, _ = _write_runtime_recovery_fixture(
        tmp_path, "applied"
    )
    public_report = episode_dir / "provenance" / "appearance_application.json"
    public_report.write_text(
        json.dumps({"schema": "futsalmot_appearance_application", "state": "rendering", "transaction_id": "other"}),
        encoding="utf-8",
    )

    class Adapter:
        @staticmethod
        def validate_appearance_inputs(*_args):
            return []

        @staticmethod
        def validate_local_recovery_evidence(*_args):
            return []

        @staticmethod
        def preflight_episode_appearance_recovery(*_args):
            pytest.fail("different unresolved public transaction must block recovery")

    monkeypatch.setitem(sys.modules, "episode_appearance", Adapter)

    with pytest.raises(RuntimeError, match="公共 report"):
        run_task._prepare_episode_appearance(
            {"appearance": {"catalog_id": "catalog"}},
            "sequence",
            episode_dir,
            {"L0": "Player_L0"},
            tmp_path,
            "task_8",
        )

    assert rollback_path.is_file()
    assert report_path.is_file()


def test_unresolved_public_report_with_stale_manifest_identity_blocks_recovery(
    tmp_path, monkeypatch
):
    episode_dir, rollback_path, report_path, rollback, _ = _write_runtime_recovery_fixture(
        tmp_path, "applied"
    )
    public_report = episode_dir / "provenance" / "appearance_application.json"
    public_report.write_text(
        json.dumps({
            "schema": "futsalmot_appearance_application",
            "version": 1,
            "state": "rendering",
            "transaction_id": rollback["transaction_id"],
            "manifest_fingerprint": "stale-manifest",
            "catalog_fingerprint": "catalog-fingerprint",
            "trajectory_sha256": "trajectory-hash",
            "episode_id": "episode",
            "catalog_id": "catalog",
        }),
        encoding="utf-8",
    )

    class Adapter:
        @staticmethod
        def validate_appearance_inputs(*_args):
            return []

        @staticmethod
        def validate_local_recovery_evidence(*_args):
            return []

        @staticmethod
        def _validate_report(_report):
            return None

        @staticmethod
        def preflight_episode_appearance_recovery(*_args):
            pytest.fail("stale public identity must block recovery before UE gate")

    monkeypatch.setitem(sys.modules, "episode_appearance", Adapter)

    with pytest.raises(RuntimeError, match="公共 report.*manifest_fingerprint"):
        run_task._prepare_episode_appearance(
            {"appearance": {"catalog_id": "catalog"}},
            "sequence",
            episode_dir,
            {"L0": "Player_L0"},
            tmp_path,
            "task_8",
        )

    assert rollback_path.is_file()
    assert report_path.is_file()


def test_unknown_public_report_state_blocks_replacement(tmp_path, monkeypatch):
    episode_dir = tmp_path / "episode"
    provenance = episode_dir / "provenance"
    provenance.mkdir(parents=True)
    (episode_dir / "meta.json").write_text("{}", encoding="utf-8")
    (episode_dir / "frames.jsonl").write_text('{"step":0}\n', encoding="utf-8")
    (provenance / "appearance.json").write_text("{}", encoding="utf-8")
    public_report = provenance / "appearance_application.json"
    public_report.write_text('{"schema":"unknown","state":"future"}', encoding="utf-8")
    adapter = SimpleNamespace(
        validate_appearance_inputs=lambda *_args: [],
        preflight_episode_appearance=lambda *_args: pytest.fail(
            "unknown public report state must fail before UE preflight"
        ),
    )
    monkeypatch.setitem(sys.modules, "episode_appearance", adapter)

    with pytest.raises(RuntimeError, match="拒绝覆盖"):
        run_task._prepare_episode_appearance(
            {"appearance": {"mode": "seeded"}},
            "sequence",
            episode_dir,
            {"L0": "Player_L0"},
            tmp_path,
            "task_8",
        )

    assert public_report.read_text(encoding="utf-8") == (
        '{"schema":"unknown","state":"future"}'
    )


def test_render_sync_completion_restores_before_return_without_ownership_transfer():
    transaction = FakeAppearanceTransaction()
    captured = {}

    def submit(*, on_complete):
        captured["callback"] = on_complete
        return {
            "callback_ownership_transferred": False,
            "terminal_result": {"status": "skipped", "jobs": []},
        }

    result = run_task._submit_render_with_appearance(submit, transaction)

    assert callable(captured["callback"])
    assert result["callback_ownership_transferred"] is False
    assert transaction.restore_calls == [("no_render", {"status": "skipped", "jobs": []})]


def test_render_callback_ownership_signal_transfers_restore_to_callback():
    transaction = FakeAppearanceTransaction()
    captured = {}

    def submit(*, on_complete):
        captured["callback"] = on_complete
        return {"callback_ownership_transferred": True}

    result = run_task._submit_render_with_appearance(submit, transaction)

    assert result["callback_ownership_transferred"] is True
    assert callable(captured["callback"])
    assert transaction.restore_calls == []

    captured["callback"]({"status": "cancelled", "jobs": []})

    assert transaction.restore_calls == [
        ("cancelled", {"status": "cancelled", "jobs": []})
    ]


def test_synchronous_completion_during_submit_does_not_transfer_ownership():
    transaction = FakeAppearanceTransaction()

    def submit(*, on_complete):
        on_complete({"status": "success", "jobs": []})
        return {"callback_ownership_transferred": True}

    result = run_task._submit_render_with_appearance(submit, transaction)

    assert result["callback_ownership_transferred"] is False
    assert transaction.callback_ownership_transferred is False
    assert transaction.restore_calls == [("success", {"status": "success", "jobs": []})]


def test_render_callback_setup_exception_restores_synchronously():
    transaction = FakeAppearanceTransaction()

    def submit(*, on_complete):
        raise TypeError("on_complete is unsupported")

    with pytest.raises(TypeError, match="unsupported"):
        run_task._submit_render_with_appearance(submit, transaction)

    assert transaction.restore_calls == [("failed", None)]


def test_terminal_render_result_is_written_before_restore_and_publication(tmp_path):
    rollback_path = tmp_path / "rollback.json"
    report_path = tmp_path / "local-report.json"
    public_report_path = tmp_path / "episode" / "appearance_application.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("{}", encoding="utf-8")
    events = []
    report = {"state": "rendering", "mrq_pending": True, "errors": [], "actors": []}

    class Adapter:
        @staticmethod
        def read_application_report(_path):
            return dict(report)

        @staticmethod
        def write_application_report(path, value):
            report.clear()
            report.update(value)
            events.append(("write", Path(path)))
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(json.dumps(value), encoding="utf-8")

        @staticmethod
        def restore_episode_appearance(_rollback, _report, status, render_result=None):
            events.append(("restore", status, dict(report), render_result))
            report["state"] = "restored"
            return {"ok": True, "state": "restored"}

    transaction = run_task._AppearanceTransaction(
        Adapter(), rollback_path, report_path, public_report_path
    )
    terminal_result = {
        "status": "success",
        "jobs": [{"camera_id": "CamA", "job_name": "Seq_CamA", "status": "success"}],
        "cameras": {"CamA": {"ok": True, "img1_frames": 2}},
    }

    result = transaction.restore("success", terminal_result)

    restore_event = next(event for event in events if event[0] == "restore")
    assert restore_event[1] == "success"
    assert restore_event[2]["render_result"]["status"] == "success"
    assert restore_event[2]["mrq_jobs"] == [
        {"camera_id": "CamA", "status": "success"}
    ]
    assert result == {"ok": True, "state": "restored"}
    assert json.loads(public_report_path.read_text(encoding="utf-8"))["state"] == "restored"


def test_terminal_render_diagnostics_survive_in_public_application_report(tmp_path):
    report_path = tmp_path / "runtime" / "appearance_application.json"
    public_report_path = tmp_path / "episode" / "provenance" / "appearance_application.json"
    report = {"state": "rendering", "mrq_pending": True, "errors": [], "actors": []}
    report_path.parent.mkdir(parents=True)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    local_path = r"C:\agent\episode output\render_summary.json"
    summary_write_error = "OSError: [Errno 28] No space left: {!r}".format(local_path)
    terminal_result = {
        "status": "failed",
        "reason": "render_summary 写入失败: {}".format(summary_write_error),
        "mrq_errors": ["render_summary 写入失败: {}".format(summary_write_error)],
        "summary_write_error": summary_write_error,
        "jobs": [],
        "cameras": {},
    }

    class Adapter:
        @staticmethod
        def read_application_report(_path):
            return dict(report)

        @staticmethod
        def write_application_report(path, value):
            report.clear()
            report.update(value)
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(json.dumps(value), encoding="utf-8")

        @staticmethod
        def restore_episode_appearance(_rollback, _report, _status, render_result=None):
            report.update(state="restored", mrq_pending=False)
            report["render_result"] = dict(render_result)
            return {"ok": True, "state": "restored"}

    transaction = run_task._AppearanceTransaction(
        Adapter(), tmp_path / "rollback.json", report_path, public_report_path
    )

    transaction.restore("failed", terminal_result)

    published = json.loads(public_report_path.read_text(encoding="utf-8"))
    render_result = published["render_result"]
    assert "No space left" in render_result["reason"]
    assert "No space left" in render_result["mrq_errors"][0]
    assert "No space left" in render_result["summary_write_error"]
    assert local_path not in json.dumps(render_result)
    json.dumps(render_result, allow_nan=False)


def test_restore_failure_is_published_with_terminal_render_result(tmp_path):
    report_path = tmp_path / "appearance_application.json"
    public_report_path = tmp_path / "episode" / "appearance_application.json"
    report = {"state": "rendering", "mrq_pending": True, "errors": [], "actors": []}
    report_path.write_text(json.dumps(report), encoding="utf-8")

    class Adapter:
        @staticmethod
        def read_application_report(_path):
            return dict(report)

        @staticmethod
        def write_application_report(path, value):
            report.clear()
            report.update(value)
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(json.dumps(value), encoding="utf-8")

        @staticmethod
        def restore_episode_appearance(_rollback, _report, _status, render_result=None):
            assert render_result["status"] == "failed"
            return {"ok": False, "state": "restore_failed", "errors": ["readback mismatch"]}

    transaction = run_task._AppearanceTransaction(
        Adapter(), tmp_path / "rollback.json", report_path, public_report_path
    )

    result = transaction.restore("failed", {"status": "failed", "jobs": []})

    assert result["state"] == "restore_failed"
    published = json.loads(public_report_path.read_text(encoding="utf-8"))
    assert published["state"] == "restore_failed"
    assert published["render_result"]["status"] == "failed"


def test_restore_report_publish_failure_is_exposed_and_retryable(tmp_path):
    report_path = tmp_path / "appearance_application.json"
    public_report_path = tmp_path / "episode" / "appearance_application.json"
    report = {
        "state": "rendering",
        "mrq_pending": True,
        "errors": [],
        "actors": [],
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    events = []

    class Adapter:
        fail_publication = True

        @staticmethod
        def read_application_report(_path):
            return dict(report)

        @staticmethod
        def write_application_report(path, value):
            report.clear()
            report.update(value)
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(json.dumps(value), encoding="utf-8")

        @staticmethod
        def restore_episode_appearance(_rollback, _report, _status):
            events.append("restore")
            report.update(state="restored", mrq_pending=False)
            return {"ok": True, "state": "restored"}

    adapter = Adapter()
    original_write = adapter.write_application_report

    def write_with_publication_failure(path, value):
        if Path(path) == public_report_path and adapter.fail_publication:
            raise OSError("public report unavailable")
        original_write(path, value)

    adapter.write_application_report = write_with_publication_failure
    transaction = run_task._AppearanceTransaction(
        adapter, tmp_path / "rollback.json", report_path, public_report_path
    )

    result = transaction.restore("success")

    assert result["ok"] is False
    assert result["state"] == "restore_failed"
    assert report["state"] == "restore_failed"
    assert any("public report unavailable" in error for error in report["errors"])

    adapter.fail_publication = False
    result = transaction.restore("success")

    assert result["ok"] is False
    assert json.loads(public_report_path.read_text(encoding="utf-8"))["state"] == "restore_failed"
    assert events == ["restore"]


def test_runtime_appearance_evidence_uses_distinct_task_local_names(tmp_path):
    rollback_path, report_path = run_task._appearance_runtime_paths(
        tmp_path, "task_8"
    )

    assert rollback_path.parent == tmp_path / ".futsalmot" / "runtime" / "task_8"
    assert report_path.parent == rollback_path.parent
    assert rollback_path.name == "appearance-rollback.json"
    assert report_path.name == "appearance-application.json"
    assert rollback_path != report_path


def test_run_task_reloads_episode_appearance_adapter():
    source_path = Path(__file__).resolve().parents[1] / "ue" / "run_task.py"
    source = source_path.read_text(encoding="utf-8")
    module_block = source.split("_UE_MODULE_NAMES = (", 1)[1].split(")", 1)[0]

    assert '"episode_appearance"' in module_block
