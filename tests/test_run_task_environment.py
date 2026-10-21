"""run_task environment 与 appearance 事务的 MRQ 生命周期测试。"""

import json
from pathlib import Path
from types import SimpleNamespace

from ue import run_task


class _FakeTransaction:
    def __init__(self, name, events, error=None):
        self.name = name
        self.events = events
        self.error = error
        self.restore_result = None
        self.render_deferred = False
        self.callback_ownership_transferred = False

    def mark_rendering(self):
        self.events.append((self.name, "mark_rendering"))

    def record_downstream_failure(self, exc):
        self.events.append((self.name, "downstream_failure", type(exc).__name__))

    def restore(self, reason, render_result=None):
        self.events.append((self.name, "restore", reason, render_result))
        if self.error is not None:
            raise self.error
        self.restore_result = {"ok": True, "state": "restored"}
        return self.restore_result


def test_combined_transaction_marks_both_before_submit_and_restores_appearance_first():
    """environment 和 appearance 共享 callback ownership，恢复顺序固定。"""
    events = []
    environment = _FakeTransaction("environment", events)
    appearance = _FakeTransaction("appearance", events)
    combined = run_task._combine_environment_appearance_transactions(
        environment, appearance
    )
    captured = {}

    def submit(*, on_complete):
        captured["callback"] = on_complete
        return {"callback_ownership_transferred": True}

    result = run_task._submit_render_with_appearance(submit, combined)

    assert result["callback_ownership_transferred"] is True
    assert events == [
        ("environment", "mark_rendering"),
        ("appearance", "mark_rendering"),
    ]

    terminal = {
        "status": "success",
        "environment_restore_ready": True,
        "appearance_restore_ready": True,
        "jobs": [],
    }
    captured["callback"](terminal)

    assert [event[:2] for event in events[-2:]] == [
        ("appearance", "restore"),
        ("environment", "restore"),
    ]
    assert events[-1][3] is terminal


def test_environment_render_options_forward_only_portable_metadata():
    """render 只接收环境 policy、fingerprint、preset 和已验证状态。"""
    prepared = {
        "manifest": {
            "policy": "futsalmot_environment_v1_5",
            "fingerprint": "d" * 64,
            "preset": "indoor_default",
            "parameters": {"must_not": "cross_boundary"},
        },
        "preflight": {"ok": True},
    }

    options = run_task._render_environment_options(prepared)

    assert options == {
        "environment_active": True,
        "environment_policy": "futsalmot_environment_v1_5",
        "environment_manifest_fingerprint": "d" * 64,
        "environment_preset": "indoor_default",
        "environment_preflight_verified": True,
        "environment_transaction_state": "applied",
    }
    assert "parameters" not in options


def test_combined_transaction_uses_terminal_failure_when_submit_raises():
    """MRQ 提交异常时两笔事务都从同一同步失败路径恢复。"""
    events = []
    combined = run_task._combine_environment_appearance_transactions(
        _FakeTransaction("environment", events),
        _FakeTransaction("appearance", events),
    )

    def submit(*, on_complete):
        raise RuntimeError("submit failed")

    try:
        run_task._submit_render_with_appearance(submit, combined)
    except RuntimeError as exc:
        assert str(exc) == "submit failed"
    else:
        raise AssertionError("submit failure must be re-raised")

    assert [event[:2] for event in events] == [
        ("environment", "mark_rendering"),
        ("appearance", "mark_rendering"),
        ("appearance", "restore"),
        ("environment", "restore"),
    ]


def test_combined_restore_isolates_transaction_failures_and_calls_both_restores():
    """一笔事务恢复异常时，另一笔仍必须收到同一终态恢复调用。"""
    events = []
    combined = run_task._combine_environment_appearance_transactions(
        _FakeTransaction("environment", events),
        _FakeTransaction("appearance", events, RuntimeError("appearance restore failed")),
    )

    result = combined.restore("failed", {"status": "failed", "jobs": []})

    assert [event[:2] for event in events] == [
        ("appearance", "restore"),
        ("environment", "restore"),
    ]
    assert result["ok"] is False
    assert any("appearance restore failed" in error for error in result["errors"])


def test_environment_only_terminal_failure_restores_from_callback():
    """仅启用 environment 时，非 ready 终态仍必须进入 environment restore。"""
    events = []
    environment = _FakeTransaction("environment", events)
    captured = {}

    def submit(*, on_complete):
        captured["callback"] = on_complete
        return {"callback_ownership_transferred": True}

    result = run_task._submit_render_with_appearance(submit, environment)
    assert result["callback_ownership_transferred"] is True

    captured["callback"](
        {
            "status": "failed",
            "environment_restore_ready": False,
            "jobs": [],
        }
    )

    assert events[-1][:2] == ("environment", "restore")


def test_pre_render_failure_restores_already_applied_episode_transaction():
    events = []
    transaction = _FakeTransaction("environment", events)

    result = run_task._fail_with_episode_transaction(
        transaction,
        "player_visuals preflight failed",
        "player_visuals_preflight_failed",
    )

    assert result == 2
    assert events == [
        ("environment", "downstream_failure", "RuntimeError"),
        ("environment", "restore", "player_visuals_preflight_failed", None),
    ]


def _environment_transaction_fixture(tmp_path):
    class Adapter:
        @staticmethod
        def restore_episode_environment(snapshot_path, report_path, payload):
            return {"ok": True, "state": "restored", "status": None, "errors": [], "warnings": []}

        @staticmethod
        def _read_json(path):
            return json.loads(Path(path).read_text(encoding="utf-8"))

        @staticmethod
        def _write_json(path, value):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(json.dumps(value), encoding="utf-8")

    snapshot = tmp_path / "runtime" / "environment-rollback.json"
    report = tmp_path / "runtime" / "environment-application.json"
    public = tmp_path / "provenance" / "environment_application.json"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_text("{}", encoding="utf-8")
    report.write_text(
        json.dumps({"state": "rendering", "errors": [], "warnings": []}), encoding="utf-8"
    )
    return Adapter, snapshot, report, public


def test_environment_restore_publishes_terminal_report_then_retires_local_evidence(tmp_path):
    Adapter, snapshot, report, public = _environment_transaction_fixture(tmp_path)
    transaction = run_task._EnvironmentTransaction(Adapter, snapshot, report, public)

    result = transaction.restore("mrq_complete", {"status": "success"})

    assert result["ok"] is True
    assert not snapshot.exists()
    assert not report.exists()
    published = json.loads(public.read_text(encoding="utf-8"))
    assert published["state"] == "restored"
    assert published["restore_status"] == "restored"


def test_environment_restore_retirement_failure_keeps_evidence_and_fails_closed(
    tmp_path, monkeypatch
):
    Adapter, snapshot, report, public = _environment_transaction_fixture(tmp_path)

    def fail_retire(rollback_path, report_path):
        raise RuntimeError("retire failed")

    monkeypatch.setattr(run_task, "_retire_environment_runtime_evidence", fail_retire)
    transaction = run_task._EnvironmentTransaction(Adapter, snapshot, report, public)

    result = transaction.restore("mrq_complete", {"status": "success"})

    assert result["ok"] is False
    assert result["status"] == "MANUAL_REQUIRED"
    assert snapshot.exists()
    assert report.exists()

def test_environment_restore_records_portable_render_result(tmp_path):
    Adapter, snapshot, report, public = _environment_transaction_fixture(tmp_path)
    transaction = run_task._EnvironmentTransaction(Adapter, snapshot, report, public)
    render_result = {
        "status": "success",
        "environment_restore_ready": True,
        "cameras": {
            "CineCam_01": {"status": "success"},
            "CineCam_02": {"status": "success"},
        },
    }

    result = transaction.restore("mrq_complete", render_result)

    assert result["ok"] is True
    published = json.loads(public.read_text(encoding="utf-8"))
    assert published["render_result"]["status"] == "success"
    assert published["render_result"]["environment_restore_ready"] is True
    assert {
        record["camera_id"] for record in published["render_result"]["cameras"]
    } == {"CineCam_01", "CineCam_02"}
