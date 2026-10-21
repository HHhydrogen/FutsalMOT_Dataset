"""MRQ completion callback 的生命周期测试。"""

import json
import sys
from types import SimpleNamespace

import pytest

from ue import render_episode


def _pipeline(tmp_path, callback, jobs=None):
    return render_episode._AsyncRenderPipeline(
        jobs or [], "episode_callback", tmp_path, [0, 3], on_complete=callback
    )


@pytest.mark.parametrize(
    ("status", "camera_status"),
    [
        ("success", "success"),
        ("failed", "failed"),
        ("cancelled", "cancelled"),
    ],
)
def test_terminal_finalize_calls_callback_once_after_summary(
    tmp_path, status, camera_status
):
    calls = []
    pipeline = _pipeline(
        tmp_path,
        lambda result: calls.append(
            (result, json.loads(pipeline.summary_path().read_text(encoding="utf-8")))
        ),
        [{"name": "Seq_CamA", "cam_id": "CamA", "kind": "rgb"}],
    )

    pipeline._finalize(
        failed=status == "failed",
        status=status,
        per_camera={
            "CamA": {
                "ok": camera_status == "success",
                "expected_frames": 2,
                "img1_frames": 2,
            }
        },
        reason="terminal test",
    )
    pipeline._finalize(failed=True, status="failed", reason="duplicate signal")

    assert len(calls) == 1
    result, summary = calls[0]
    assert summary["status"] == status
    assert result["status"] == status
    assert result["cameras"]["CamA"]["img1_frames"] == 2
    assert result["jobs"] == [
        {"camera_id": "CamA", "job_name": "Seq_CamA", "status": camera_status}
    ]
    json.dumps(result, allow_nan=False)


def test_terminal_job_statuses_use_each_jobs_camera_output_counts(tmp_path):
    calls = []
    jobs = [
        {"name": "Seq_CamA", "cam_id": "CamA", "kind": "rgb"},
        {"name": "Seq_CamA_MASK", "cam_id": "CamA", "kind": "mask"},
    ]
    pipeline = _pipeline(tmp_path, calls.append, jobs)

    pipeline._finalize(
        failed=False,
        status="partial",
        per_camera={
            "CamA": {
                "ok": False,
                "expected_frames": 3,
                "img1_frames": 3,
                "mask_frames": 2,
            }
        },
    )

    assert calls[0]["jobs"] == [
        {"camera_id": "CamA", "job_name": "Seq_CamA", "status": "success"},
        {"camera_id": "CamA", "job_name": "Seq_CamA_MASK", "status": "failed"},
    ]


def test_finished_executor_success_uses_terminal_callback(tmp_path, monkeypatch):
    calls = []
    pipeline = _pipeline(tmp_path, calls.append)
    monkeypatch.setattr(
        pipeline,
        "_copy_and_finalize",
        lambda: pipeline._finalize(
            failed=False,
            status="success",
            per_camera={
                "CamA": {"ok": True, "expected_frames": 2, "img1_frames": 2}
            },
        ),
    )

    pipeline._on_finished(SimpleNamespace(), True)

    assert len(calls) == 1
    assert calls[0]["status"] == "success"


def test_finished_executor_error_uses_terminal_callback(tmp_path):
    calls = []
    pipeline = _pipeline(tmp_path, calls.append)

    pipeline._on_finished(SimpleNamespace(), False)

    assert len(calls) == 1
    assert calls[0]["status"] == "failed"


def test_fatal_executor_error_finalizes_callback_once(tmp_path):
    calls = []
    pipeline = _pipeline(tmp_path, calls.append)

    pipeline._on_errored(SimpleNamespace(), object(), True, ["fatal"])
    pipeline._on_finished(SimpleNamespace(), False)

    assert len(calls) == 1
    assert calls[0]["status"] == "failed"
    assert calls[0]["mrq_errors"] == ["fatal"]


def test_fatal_error_waits_for_executor_terminal_signal(tmp_path, monkeypatch):
    calls = []
    pipeline = _pipeline(tmp_path, calls.append)
    pipeline.executor = SimpleNamespace()
    monkeypatch.setattr(pipeline, "_executor_rendering", lambda: False)
    monkeypatch.setattr(pipeline, "_all_keep_frames_present", lambda: False)
    monkeypatch.setattr(pipeline, "_total_render_files", lambda: 0)
    ticks = []
    monkeypatch.setitem(
        sys.modules,
        "unreal",
        SimpleNamespace(
            register_slate_post_tick_callback=lambda callback: ticks.append(callback) or 12,
            unregister_slate_post_tick_callback=lambda _handle: None,
        ),
    )

    pipeline._on_errored(SimpleNamespace(), object(), True, ["fatal"])

    assert calls == []
    assert pipeline.finished is False
    pipeline._start_completion_watchdog()
    assert ticks[0](0.016) is False
    assert calls[0]["status"] == "failed"


def test_finished_cancelled_executor_reports_cancelled(tmp_path):
    calls = []
    pipeline = _pipeline(tmp_path, calls.append)
    executor = SimpleNamespace(is_cancelled=lambda: True)

    pipeline._on_finished(executor, False)

    assert len(calls) == 1
    assert calls[0]["status"] == "cancelled"


def test_watchdog_completion_calls_callback_once(tmp_path, monkeypatch):
    calls = []
    pipeline = _pipeline(tmp_path, calls.append)
    pipeline.executor = SimpleNamespace()
    monkeypatch.setattr(pipeline, "_executor_rendering", lambda: False)
    monkeypatch.setattr(pipeline, "_all_keep_frames_present", lambda: True)
    monkeypatch.setattr(pipeline, "_total_render_files", lambda: 2)
    monkeypatch.setattr(
        pipeline,
        "_copy_and_finalize",
        lambda: pipeline._finalize(
            failed=False,
            status="success",
            per_camera={"CamA": {"ok": True}},
        ),
    )
    ticks = []
    fake_unreal = SimpleNamespace(
        register_slate_post_tick_callback=lambda callback: ticks.append(callback) or 7,
        unregister_slate_post_tick_callback=lambda _handle: None,
    )
    monkeypatch.setitem(sys.modules, "unreal", fake_unreal)

    pipeline._start_completion_watchdog()
    assert ticks[0](0.016) is False
    pipeline._finalize(failed=True, status="failed")

    assert len(calls) == 1
    assert calls[0]["status"] == "success"


def test_watchdog_cancellation_calls_callback_with_cancelled_terminal_status(
    tmp_path, monkeypatch
):
    calls = []
    pipeline = _pipeline(tmp_path, calls.append)
    pipeline.executor = SimpleNamespace(is_cancelled=lambda: True)
    monkeypatch.setattr(pipeline, "_executor_rendering", lambda: False)
    monkeypatch.setattr(pipeline, "_all_keep_frames_present", lambda: False)
    monkeypatch.setattr(pipeline, "_total_render_files", lambda: 0)
    ticks = []
    monkeypatch.setitem(
        sys.modules,
        "unreal",
        SimpleNamespace(
            register_slate_post_tick_callback=lambda callback: ticks.append(callback) or 8,
            unregister_slate_post_tick_callback=lambda _handle: None,
        ),
    )

    pipeline._start_completion_watchdog()

    assert ticks[0](0.016) is False
    assert len(calls) == 1
    assert calls[0]["status"] == "cancelled"


def test_watchdog_cancellation_finalizes_when_rendering_state_is_unavailable(
    tmp_path, monkeypatch
):
    calls = []
    pipeline = _pipeline(tmp_path, calls.append)
    pipeline.executor = SimpleNamespace(is_cancelled=lambda: True)
    monkeypatch.setattr(pipeline, "_executor_rendering", lambda: None)
    monkeypatch.setattr(pipeline, "_all_keep_frames_present", lambda: False)
    monkeypatch.setattr(pipeline, "_total_render_files", lambda: 0)
    ticks = []
    monkeypatch.setitem(
        sys.modules,
        "unreal",
        SimpleNamespace(
            register_slate_post_tick_callback=lambda callback: ticks.append(callback) or 10,
            unregister_slate_post_tick_callback=lambda _handle: None,
        ),
    )

    pipeline._start_completion_watchdog()

    assert ticks[0](0.016) is False
    assert calls[0]["status"] == "cancelled"


def test_watchdog_reports_cancelled_before_output_inspection(tmp_path, monkeypatch):
    calls = []
    pipeline = _pipeline(tmp_path, calls.append)
    pipeline.executor = SimpleNamespace(is_cancelled=lambda: True)
    monkeypatch.setattr(
        pipeline,
        "_executor_rendering",
        lambda: (_ for _ in ()).throw(RuntimeError("executor state unavailable")),
    )
    monkeypatch.setattr(
        pipeline,
        "_all_keep_frames_present",
        lambda: (_ for _ in ()).throw(RuntimeError("output scan unavailable")),
    )
    ticks = []
    monkeypatch.setitem(
        sys.modules,
        "unreal",
        SimpleNamespace(
            register_slate_post_tick_callback=lambda callback: ticks.append(callback) or 13,
            unregister_slate_post_tick_callback=lambda _handle: None,
        ),
    )

    pipeline._start_completion_watchdog()

    assert ticks[0](0.016) is False
    assert calls[0]["status"] == "cancelled"


def test_watchdog_inspection_exception_finalizes_as_failed(tmp_path, monkeypatch):
    calls = []
    pipeline = _pipeline(tmp_path, calls.append)
    pipeline.executor = SimpleNamespace()
    monkeypatch.setattr(pipeline, "_executor_rendering", lambda: False)
    monkeypatch.setattr(
        pipeline,
        "_all_keep_frames_present",
        lambda: (_ for _ in ()).throw(ValueError("ambiguous output")),
    )
    ticks = []
    monkeypatch.setitem(
        sys.modules,
        "unreal",
        SimpleNamespace(
            register_slate_post_tick_callback=lambda callback: ticks.append(callback) or 11,
            unregister_slate_post_tick_callback=lambda _handle: None,
        ),
    )

    pipeline._start_completion_watchdog()

    assert ticks[0](0.016) is False
    assert len(calls) == 1
    assert calls[0]["status"] == "failed"


def test_watchdog_finalizes_executor_error_after_executor_stops(tmp_path, monkeypatch):
    calls = []
    pipeline = _pipeline(tmp_path, calls.append)
    pipeline.executor = SimpleNamespace()
    pipeline._on_errored(SimpleNamespace(), object(), False, ["executor failed"])
    monkeypatch.setattr(pipeline, "_executor_rendering", lambda: False)
    monkeypatch.setattr(pipeline, "_all_keep_frames_present", lambda: False)
    monkeypatch.setattr(pipeline, "_total_render_files", lambda: 0)
    ticks = []
    monkeypatch.setitem(
        sys.modules,
        "unreal",
        SimpleNamespace(
            register_slate_post_tick_callback=lambda callback: ticks.append(callback) or 9,
            unregister_slate_post_tick_callback=lambda _handle: None,
        ),
    )

    pipeline._start_completion_watchdog()

    assert ticks[0](0.016) is False
    assert len(calls) == 1
    assert calls[0]["status"] == "failed"
    assert calls[0]["mrq_errors"] == ["executor failed"]


def test_submission_failure_finalizes_before_return(tmp_path, monkeypatch):
    calls = []
    pipeline = _pipeline(
        tmp_path,
        lambda result: calls.append(
            (result, pipeline.summary_path().is_file())
        ),
        [{
            "name": "Seq_CamA",
            "cam_id": "CamA",
            "kind": "rgb",
            "seq_asset_path": "/Game/Sequences/Seq_CamA",
            "render_dir": tmp_path / "render",
            "img1_dir": tmp_path / "img1",
        }],
    )
    monkeypatch.setattr(render_episode, "_clear_queue", lambda _queue: None)
    monkeypatch.setattr(render_episode, "_build_mrq_job", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(render_episode, "_submit_render", lambda *_args: False)

    transferred = pipeline.start(
        object(), object(), SimpleNamespace(), "/Game/Map", 64, 64, 30, "{frame_number}", 6
    )

    assert transferred is False
    assert len(calls) == 1
    assert calls[0][1] is True
    assert calls[0][0]["status"] == "failed"
    assert calls[0][0]["jobs"] == [
        {"camera_id": "CamA", "job_name": "Seq_CamA", "status": "failed"}
    ]


@pytest.mark.parametrize("watchdog_registration", ["missing", "raises"])
def test_start_does_not_submit_without_any_completion_source(
    tmp_path, monkeypatch, watchdog_registration
):
    calls = []
    delegate = SimpleNamespace(add_callable=lambda _callback: None)
    executor = SimpleNamespace(on_executor_finished_delegate=delegate)
    unreal_module = SimpleNamespace()
    if watchdog_registration == "raises":
        unreal_module.register_slate_post_tick_callback = lambda _callback: (
            _ for _ in ()
        ).throw(RuntimeError("watchdog unavailable"))
    monkeypatch.setitem(sys.modules, "unreal", unreal_module)
    pipeline = _pipeline(
        tmp_path,
        calls.append,
        [{
            "name": "Seq_CamA",
            "cam_id": "CamA",
            "kind": "rgb",
            "seq_asset_path": "/Game/Sequences/Seq_CamA",
            "render_dir": tmp_path / "render",
            "img1_dir": tmp_path / "img1",
        }],
    )
    submitted = []
    monkeypatch.setattr(render_episode, "_clear_queue", lambda _queue: None)
    monkeypatch.setattr(render_episode, "_build_mrq_job", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        render_episode, "_submit_render", lambda *_args: submitted.append(True) or True
    )
    delegate.add_callable = lambda _callback: (_ for _ in ()).throw(
        RuntimeError("finished delegate unavailable")
    )

    transferred = pipeline.start(
        object(), object(), executor, "/Game/Map", 64, 64, 30, "{frame_number}", 6
    )

    assert transferred is False
    assert submitted == []
    assert len(calls) == 1
    assert calls[0]["status"] == "failed"
    assert "completion" in calls[0]["reason"].lower()


def test_start_returns_no_ownership_when_submit_finishes_synchronously(
    tmp_path, monkeypatch
):
    calls = []
    ticks = []

    class Delegate:
        def add_callable(self, callback):
            self.callback = callback

    delegate = Delegate()
    executor = SimpleNamespace(on_executor_finished_delegate=delegate)
    monkeypatch.setitem(
        sys.modules,
        "unreal",
        SimpleNamespace(
            register_slate_post_tick_callback=lambda callback: ticks.append(callback) or 3,
            unregister_slate_post_tick_callback=lambda _handle: None,
        ),
    )
    pipeline = _pipeline(
        tmp_path,
        calls.append,
        [{
            "name": "Seq_CamA",
            "cam_id": "CamA",
            "kind": "rgb",
            "seq_asset_path": "/Game/Sequences/Seq_CamA",
            "render_dir": tmp_path / "render",
            "img1_dir": tmp_path / "img1",
        }],
    )
    monkeypatch.setattr(render_episode, "_clear_queue", lambda _queue: None)
    monkeypatch.setattr(render_episode, "_build_mrq_job", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        pipeline,
        "_copy_and_finalize",
        lambda: pipeline._finalize(failed=False, status="success"),
    )

    def submit(_subsystem, _executor):
        delegate.callback(executor, True)
        return True

    monkeypatch.setattr(render_episode, "_submit_render", submit)

    transferred = pipeline.start(
        object(), object(), executor, "/Game/Map", 64, 64, 30, "{frame_number}", 6
    )

    assert transferred is False
    assert ticks == []
    assert len(calls) == 1
    assert calls[0]["status"] == "success"


def test_summary_write_failure_still_calls_callback_once_with_failed_result(
    tmp_path, monkeypatch, capsys
):
    calls = []
    pipeline = _pipeline(tmp_path, calls.append)
    monkeypatch.setattr(
        render_episode,
        "write_json_atomic",
        lambda *_args: (_ for _ in ()).throw(OSError("disk full")),
    )

    pipeline._finalize(failed=False, status="success")
    pipeline._finalize(failed=True, status="failed", reason="duplicate")

    assert len(calls) == 1
    assert calls[0]["status"] == "failed"
    assert "disk full" in calls[0]["reason"]
    assert calls[0]["summary_write_error"] == "OSError: disk full"
    assert "disk full" in capsys.readouterr().out


def test_setup_exception_finalizes_callback_before_reraising(tmp_path, monkeypatch):
    calls = []
    pipeline = _pipeline(tmp_path, calls.append)
    monkeypatch.setattr(
        render_episode,
        "_clear_queue",
        lambda _queue: (_ for _ in ()).throw(RuntimeError("queue setup failed")),
    )

    with pytest.raises(RuntimeError, match="queue setup failed"):
        pipeline.start(
            object(), object(), SimpleNamespace(), "/Game/Map", 64, 64, 30,
            "{frame_number}", 6,
        )

    assert len(calls) == 1
    assert calls[0]["status"] == "failed"
    assert (pipeline.summary_path()).is_file()


def test_public_setup_exception_calls_callback_before_reraising(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "unreal", SimpleNamespace())
    monkeypatch.setattr(render_episode, "_ACTIVE_RENDER", None)
    monkeypatch.setattr(render_episode, "resolve_preset", lambda _cfg: (None, None))
    monkeypatch.setattr(
        render_episode,
        "resolve_output_resolution",
        lambda _cfg: (_ for _ in ()).throw(ValueError("bad resolution")),
    )
    calls = []

    with pytest.raises(ValueError, match="bad resolution"):
        render_episode.render_sequences(
            [], {"render_rgb": {"enabled": True}}, "/Game/Sequences", tmp_path,
            tmp_path / "output", on_complete=calls.append,
        )

    assert len(calls) == 1
    assert calls[0]["status"] == "failed"
    assert (
        tmp_path / "output" / tmp_path.name / "render_summary.json"
    ).is_file()


def test_callback_exception_does_not_break_render_finalization(tmp_path, capsys):
    def broken_callback(_result):
        raise RuntimeError("appearance restore failed")

    pipeline = _pipeline(tmp_path, broken_callback)

    pipeline._finalize(failed=False, status="success")

    assert pipeline.finished is True
    assert pipeline.summary_path().is_file()
    assert "appearance restore failed" in capsys.readouterr().out


def test_disabled_render_calls_callback_synchronously_and_preserves_old_callers(
    tmp_path, monkeypatch
):
    monkeypatch.setitem(sys.modules, "unreal", SimpleNamespace())
    monkeypatch.setattr(render_episode, "_ACTIVE_RENDER", None)
    calls = []

    result = render_episode.render_sequences(
        [], {"render_rgb": {"enabled": False}}, "/Game/Sequences", tmp_path,
        tmp_path / "output", on_complete=calls.append,
    )
    old_result = render_episode.render_sequences(
        [], {"render_rgb": {"enabled": False}}, "/Game/Sequences", tmp_path,
        tmp_path / "output",
    )

    assert result["callback_ownership_transferred"] is False
    assert result["terminal_result"]["status"] == "skipped"
    assert calls == [result["terminal_result"]]
    assert old_result is None
    summary = json.loads(
        (tmp_path / "output" / tmp_path.name / "render_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["status"] == "skipped"


def test_empty_sequence_list_calls_callback_synchronously(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "unreal", SimpleNamespace())
    monkeypatch.setattr(render_episode, "_ACTIVE_RENDER", None)
    monkeypatch.setattr(render_episode, "resolve_preset", lambda _cfg: (None, None))
    monkeypatch.setattr(
        render_episode,
        "load_episode",
        lambda _path: ({"episode_id": "episode_callback", "timing": {"source_step_seconds": 0.1}}, []),
    )
    monkeypatch.setattr(render_episode, "resolve_output_resolution", lambda _cfg: (64, 64))
    monkeypatch.setattr(render_episode, "_apply_cv_gt_camera_post_process", lambda *_args: None)
    calls = []

    result = render_episode.render_sequences(
        [], {"render_rgb": {"enabled": True}}, "/Game/Sequences", tmp_path,
        tmp_path / "output", on_complete=calls.append,
    )

    assert result["callback_ownership_transferred"] is False
    assert calls == [result["terminal_result"]]
    assert result["terminal_result"]["status"] == "skipped"


def test_appearance_first_frame_actor_bake_is_temporary_and_never_saves(monkeypatch):
    calls = []
    monkeypatch.setattr(render_episode, "load_mapping", lambda _path: {"L0": "Player_L0"})
    monkeypatch.setattr(render_episode, "find_all_actors", lambda _mapping: {"L0": object()})
    monkeypatch.setattr(render_episode, "gk_entity_ids_from_meta", lambda _meta: set())
    monkeypatch.setattr(
        render_episode,
        "apply_preview_frame",
        lambda *_args, **_kwargs: calls.append("bake"),
    )
    monkeypatch.setattr(render_episode, "_save_current_level", lambda: calls.append("save"))

    baked = render_episode._bake_first_frame_for_render(
        {"timing": {}}, [{"step": 0}], "mapping.json", appearance_active=True
    )

    assert baked is True
    assert calls == ["bake"]


def test_appearance_first_frame_bake_excludes_ball_from_transaction_mutation(monkeypatch):
    player = object()
    ball = object()
    captured = {}
    monkeypatch.setattr(
        render_episode,
        "load_mapping",
        lambda _path: {"L0": "Player_L0", "BALL": "Ball_01"},
    )
    monkeypatch.setattr(
        render_episode,
        "find_all_actors",
        lambda _mapping: {"L0": player, "BALL": ball},
    )
    monkeypatch.setattr(render_episode, "gk_entity_ids_from_meta", lambda _meta: set())
    monkeypatch.setattr(
        render_episode,
        "apply_preview_frame",
        lambda actors, *_args, **_kwargs: captured.update(actors),
    )
    monkeypatch.setattr(
        render_episode, "_save_current_level", lambda: pytest.fail("must not save")
    )

    baked = render_episode._bake_first_frame_for_render(
        {"timing": {}}, [{"step": 0}], "mapping.json", appearance_active=True
    )

    assert baked is True
    assert captured == {"L0": player}


def test_appearance_first_frame_bake_without_players_returns_false(monkeypatch):
    monkeypatch.setattr(
        render_episode,
        "load_mapping",
        lambda _path: {"BALL": "Ball_01"},
    )
    monkeypatch.setattr(
        render_episode, "find_all_actors", lambda _mapping: {"BALL": object()}
    )
    monkeypatch.setattr(
        render_episode,
        "apply_preview_frame",
        lambda *_args, **_kwargs: pytest.fail("no actor should be baked"),
    )

    assert render_episode._bake_first_frame_for_render(
        {"timing": {}}, [{"step": 0}], "mapping.json", appearance_active=True
    ) is False


def test_appearance_first_frame_bake_does_not_swallow_actor_mutation_error(monkeypatch):
    monkeypatch.setattr(render_episode, "load_mapping", lambda _path: {"L0": "Player_L0"})
    monkeypatch.setattr(render_episode, "find_all_actors", lambda _mapping: {"L0": object()})
    monkeypatch.setattr(render_episode, "gk_entity_ids_from_meta", lambda _meta: set())
    monkeypatch.setattr(
        render_episode,
        "apply_preview_frame",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("actor mutation failed")),
    )
    monkeypatch.setattr(render_episode, "_save_current_level", lambda: pytest.fail("must not save"))

    with pytest.raises(RuntimeError, match="actor mutation failed"):
        render_episode._bake_first_frame_for_render(
            {"timing": {}}, [{"step": 0}], "mapping.json", appearance_active=True
        )


def test_appearance_without_mapping_needs_no_first_frame_bake():
    assert render_episode._bake_first_frame_for_render(
        {"timing": {}}, [{"step": 0}], None, appearance_active=True
    ) is False


def test_completion_callback_alone_preserves_legacy_camera_save(monkeypatch):
    calls = []
    monkeypatch.setattr(
        render_episode,
        "camera_post_process_overrides",
        lambda *_args: {"post_process_blend_weight": 1.0},
    )
    monkeypatch.setattr(render_episode, "find_actor", lambda _name: object())
    monkeypatch.setattr(
        render_episode,
        "_set_config_overrides",
        lambda *_args: None,
    )
    monkeypatch.setattr(render_episode, "_save_current_level", lambda: calls.append("save"))
    monkeypatch.setitem(
        sys.modules,
        "unreal",
        SimpleNamespace(
            CineCameraComponent=object,
            PostProcessSettings=lambda: SimpleNamespace(set_editor_property=lambda *_args: None),
        ),
    )
    camera = SimpleNamespace(
        get_components_by_class=lambda _class: [
            SimpleNamespace(
                get_editor_property=lambda _name: None,
                set_editor_property=lambda *_args: None,
            )
        ]
    )
    monkeypatch.setattr(render_episode, "find_actor", lambda _name: camera)

    render_episode._apply_cv_gt_camera_post_process(
        [{"name": "Seq_CamA", "camera_actor": "CamA"}],
        object(),
        object(),
    )

    assert calls == ["save"]


def test_unsaved_camera_override_path_does_not_mutate_camera(monkeypatch):
    calls = []
    monkeypatch.setattr(
        render_episode,
        "camera_post_process_overrides",
        lambda *_args: {"post_process_blend_weight": 1.0},
    )
    monkeypatch.setitem(
        sys.modules,
        "unreal",
        SimpleNamespace(
            CineCameraComponent=object,
            PostProcessSettings=lambda: SimpleNamespace(
                set_editor_property=lambda *_args: calls.append("set")
            ),
        ),
    )
    monkeypatch.setattr(
        render_episode,
        "find_actor",
        lambda _name: SimpleNamespace(
            get_components_by_class=lambda _class: [
                SimpleNamespace(
                    get_editor_property=lambda _name: None,
                    set_editor_property=lambda *_args: calls.append("set"),
                )
            ]
        ),
    )
    monkeypatch.setattr(render_episode, "_set_config_overrides", lambda *_args: None)
    monkeypatch.setattr(render_episode, "_save_current_level", lambda: calls.append("save"))

    render_episode._apply_cv_gt_camera_post_process(
        [{"name": "Seq_CamA", "camera_actor": "CamA"}],
        object(),
        object(),
        save_level=False,
    )

    assert calls == ["set", "set"]


def test_non_appearance_keeps_first_frame_bake_and_save(monkeypatch):
    calls = []
    monkeypatch.setattr(render_episode, "load_mapping", lambda _path: {"L0": "Player_L0"})
    monkeypatch.setattr(render_episode, "find_all_actors", lambda _mapping: {"L0": object()})
    monkeypatch.setattr(render_episode, "gk_entity_ids_from_meta", lambda _meta: set())
    monkeypatch.setattr(
        render_episode,
        "apply_preview_frame",
        lambda *_args, **_kwargs: calls.append("bake"),
    )
    monkeypatch.setattr(render_episode, "_save_current_level", lambda: calls.append("save"))

    baked = render_episode._bake_first_frame_for_render(
        {"timing": {}}, [{"step": 0}], "mapping.json", appearance_active=False
    )

    assert baked is True
    assert calls == ["bake", "save"]


def test_appearance_rejects_save_dependent_stencil_path_before_actor_mutation(
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(render_episode, "load_mapping", lambda _path: calls.append("load"))
    monkeypatch.setattr(render_episode, "find_all_actors", lambda _mapping: calls.append("find"))
    monkeypatch.setattr(render_episode, "_assign_custom_stencil", lambda _actors: calls.append("stencil"))
    monkeypatch.setattr(render_episode, "_save_current_level", lambda: calls.append("save"))

    with pytest.raises(RuntimeError, match="appearance.*stencil.*保存"):
        render_episode._prepare_stencil_for_render("mapping.json", appearance_active=True)

    assert calls == []


def test_non_appearance_stencil_path_retains_assignment_and_save(tmp_path, monkeypatch):
    calls = []
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(render_episode, "_ensure_custom_depth_stencil_enabled", lambda: calls.append("enable"))
    monkeypatch.setattr(render_episode, "load_mapping", lambda _path: {"L0": "Player_L0"})
    monkeypatch.setattr(render_episode, "find_all_actors", lambda _mapping: {"L0": object()})
    monkeypatch.setattr(render_episode, "_assign_custom_stencil", lambda _actors: calls.append("stencil"))
    monkeypatch.setattr(render_episode, "_save_current_level", lambda: calls.append("save"))

    render_episode._prepare_stencil_for_render(str(mapping_path), appearance_active=False)

    assert calls == ["enable", "stencil", "save"]


def test_async_environment_callback_observes_summary_before_restore_handoff(tmp_path):
    """异步终态 callback 只能在 summary 落盘和 restore readiness 计算后触发。"""
    observations = []
    pipeline = render_episode._AsyncRenderPipeline(
        [],
        "episode_environment",
        tmp_path,
        [],
        on_complete=lambda result: observations.append(
            (result["environment_restore_ready"], pipeline.summary_path().is_file())
        ),
        environment_active=True,
        environment_policy="futsalmot_environment_v1_5",
        environment_manifest_fingerprint="c" * 64,
        environment_preset="indoor_default",
        environment_preflight_verified=True,
        environment_transaction_state="rendering",
    )

    pipeline._finalize(failed=False, status="success")

    assert observations == [(True, True)]


def test_active_environment_gate_fails_before_mrq_and_writes_summary(
    tmp_path, monkeypatch
):
    """未确认 environment preflight 时必须在 MRQ 访问前写入失败终态。"""
    monkeypatch.setattr(render_episode, "_ACTIVE_RENDER", None)
    monkeypatch.setitem(sys.modules, "unreal", SimpleNamespace())
    monkeypatch.setattr(
        render_episode,
        "load_episode",
        lambda _path: (
            {"episode_id": "episode_environment", "timing": {"source_step_seconds": 0.1}},
            [{"step": 0}],
        ),
    )
    monkeypatch.setattr(
        render_episode,
        "_mrq_subsystem",
        lambda: pytest.fail("environment gate 失败时不得访问 MRQ"),
    )
    callbacks = []

    result = render_episode.render_sequences(
        [{"name": "LS_Cam_01", "camera_actor": "CineCam_01"}],
        {"render_rgb": {"enabled": True}},
        "/Game/FutsalMOT/Sequences",
        tmp_path,
        tmp_path / "output",
        on_complete=callbacks.append,
        environment_active=True,
        environment_policy="futsalmot_environment_v1_5",
        environment_manifest_fingerprint="a" * 64,
        environment_preset="indoor_default",
        environment_preflight_verified=False,
        environment_transaction_state="applied",
    )

    terminal = result["terminal_result"]
    summary = json.loads(
        (tmp_path / "output" / "episode_environment" / "render_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert terminal["status"] == "failed"
    assert "MANUAL_REQUIRED" in terminal["reason"]
    assert terminal["environment_restore_ready"] is False
    assert summary["environment_restore_ready"] is False
    assert callbacks == [terminal]


@pytest.mark.parametrize(
    ("fingerprint", "preset"),
    [("", "indoor_default"), ("a" * 64, "")],
)
def test_active_environment_gate_rejects_missing_metadata_before_mrq(
    tmp_path, monkeypatch, fingerprint, preset
):
    """environment 指纹或 preset 缺失时必须保持 fail-closed。"""
    monkeypatch.setattr(render_episode, "_ACTIVE_RENDER", None)
    monkeypatch.setitem(sys.modules, "unreal", SimpleNamespace())
    monkeypatch.setattr(
        render_episode,
        "load_episode",
        lambda _path: (
            {"episode_id": "episode_environment", "timing": {"source_step_seconds": 0.1}},
            [{"step": 0}],
        ),
    )
    monkeypatch.setattr(
        render_episode,
        "_mrq_subsystem",
        lambda: pytest.fail("invalid environment metadata must not access MRQ"),
    )
    callbacks = []

    result = render_episode.render_sequences(
        [{"name": "LS_Cam_01", "camera_actor": "CineCam_01"}],
        {"render_rgb": {"enabled": True}},
        "/Game/FutsalMOT/Sequences",
        tmp_path,
        tmp_path / "output",
        on_complete=callbacks.append,
        environment_active=True,
        environment_policy="futsalmot_environment_v1_5",
        environment_manifest_fingerprint=fingerprint,
        environment_preset=preset,
        environment_preflight_verified=True,
        environment_transaction_state="rendering",
    )

    assert result["terminal_result"]["status"] == "failed"
    assert "MANUAL_REQUIRED" in result["terminal_result"]["reason"]
    assert callbacks == [result["terminal_result"]]


def test_environment_active_render_never_saves_level_for_setup():
    assert render_episode._should_save_level(False, False) is True
    assert render_episode._should_save_level(True, False) is False
    assert render_episode._should_save_level(False, True) is False
    assert render_episode._should_save_level(True, True) is False


def test_bake_first_frame_skips_level_save_during_environment(monkeypatch):
    calls = []
    monkeypatch.setattr(render_episode, "_save_current_level", lambda: calls.append("save"))
    monkeypatch.setattr(render_episode, "load_mapping", lambda path: {"L0": "Player_L0"})
    monkeypatch.setattr(render_episode, "find_all_actors", lambda mapping: {"L0": object()})
    monkeypatch.setattr(render_episode, "gk_entity_ids_from_meta", lambda meta: [])
    monkeypatch.setattr(render_episode, "apply_preview_frame", lambda *a, **k: None)

    saved = render_episode._bake_first_frame_for_render(
        {"gk_entity_ids": []},
        [{"step": 0}],
        "mapping.json",
        environment_active=True,
    )

    assert saved is True
    assert calls == []

    render_episode._bake_first_frame_for_render(
        {"gk_entity_ids": []},
        [{"step": 0}],
        "mapping.json",
        appearance_active=False,
        environment_active=False,
    )
    assert calls == ["save"]

def test_active_environment_metadata_reaches_terminal_and_summary_after_finalize(tmp_path):
    """active environment 的便携 metadata 必须在终态摘要和 job 中一致。"""
    callbacks = []
    pipeline = render_episode._AsyncRenderPipeline(
        [{"name": "Seq_CamA", "cam_id": "CamA", "kind": "rgb"}],
        "episode_environment",
        tmp_path,
        [0, 3],
        on_complete=lambda result: callbacks.append(
            (result, json.loads(pipeline.summary_path().read_text(encoding="utf-8")))
        ),
        environment_active=True,
        environment_policy="futsalmot_environment_v1_5",
        environment_manifest_fingerprint="b" * 64,
        environment_preset="indoor_default",
        environment_preflight_verified=True,
        environment_transaction_state="rendering",
    )

    pipeline._finalize(
        failed=False,
        status="success",
        per_camera={"CamA": {"ok": True, "expected_frames": 2, "img1_frames": 2}},
    )

    terminal, summary = callbacks[0]
    for value in (terminal, summary, terminal["jobs"][0], summary["jobs"][0]):
        assert value["environment_policy"] == "futsalmot_environment_v1_5"
        assert value["environment_manifest_fingerprint"] == "b" * 64
        assert value["environment_preset"] == "indoor_default"
    assert terminal["environment_restore_ready"] is True
    assert summary["environment_restore_ready"] is True
    assert summary["cameras"]["CamA"]["environment_policy"] == (
        "futsalmot_environment_v1_5"
    )
