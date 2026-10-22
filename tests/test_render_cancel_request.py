"""模拟 UE 5.8 无取消属性、native 取消同步重入 finished(False) 的顺序。"""

from types import SimpleNamespace

import pytest

from test_render_cancel_restore_timing import LEVEL, rig


@pytest.fixture
def render_module():
    from ue import render_episode
    return render_episode


def no_cancel_flag_executor(rig, cancel=None):
    executor = SimpleNamespace(is_rendering=lambda: rig.state['executor_rendering'])
    if cancel is not None:
        executor.cancel_all_jobs = cancel
    rig.pipeline.executor = executor
    return executor


def finish_teardown(rig):
    rig.state.update(world=LEVEL, pie=False, mrq=False, executor_rendering=False)
    rig.tick()
    assert not rig.calls
    rig.tick()


@pytest.mark.parametrize('success', (False, True))
def test_request_cancel_registers_wait_before_native_synchronous_finished(rig, success):
    p = rig.pipeline
    native_calls = []
    def cancel():
        native_calls.append('native')
        assert p._cancel_wait is not None
        p._on_finished(p.executor, success)
        assert not p.finished and not rig.calls
    no_cancel_flag_executor(rig, cancel)
    assert p.request_cancel() is True
    assert native_calls == ['native'] and not rig.calls
    assert p.request_cancel() is False
    assert native_calls == ['native']
    finish_teardown(rig)
    assert len(rig.calls) == 1 and rig.calls[0]['status'] == 'cancelled'
    assert p.request_cancel() is False


def test_unclassified_finished_false_waits_then_preserves_failed_reason(rig):
    p = rig.pipeline
    no_cancel_flag_executor(rig)
    p._on_finished(p.executor, False)
    assert not p.finished and not rig.calls
    assert p._cancel_wait is not None
    finish_teardown(rig)
    assert len(rig.calls) == 1
    assert rig.calls[0]['status'] == 'failed'
    assert rig.calls[0]['reason'] == 'MRQ finished 委托报告 success=False'


def test_failed_wait_cannot_be_reclassified_by_late_success(rig):
    p = rig.pipeline
    no_cancel_flag_executor(rig)
    p._on_finished(p.executor, False)
    p._on_finished(p.executor, True)
    p._copy_and_finalize()
    p._finalize(failed=False)
    assert not p.finished and not rig.calls
    finish_teardown(rig)
    assert rig.calls[0]['status'] == 'failed'
    assert rig.calls[0]['reason'] == 'MRQ finished 委托报告 success=False'


def test_native_cancel_failure_is_failed_only_after_safe_world_recovery(rig):
    p = rig.pipeline
    def cancel():
        assert p._cancel_wait is not None
        raise RuntimeError('native cancel failed')
    no_cancel_flag_executor(rig, cancel)
    with pytest.raises(RuntimeError, match='native cancel failed'):
        p.request_cancel()
    assert not p.finished and not rig.calls
    finish_teardown(rig)
    assert rig.calls[0]['status'] == 'failed'
    assert 'native cancel failed' in rig.calls[0]['reason']


def test_registration_failure_still_requests_native_cancel_without_restore(rig):
    native_calls = []
    p = rig.pipeline
    no_cancel_flag_executor(rig, lambda: native_calls.append(True))
    rig.fake.register_slate_post_tick_callback = lambda callback: None
    assert p.request_cancel() is True
    assert native_calls == [True]
    assert p.finished and not rig.calls
    assert p.terminal_result['status'] == 'failed'
    assert p.terminal_result['restore_deferred'] is True
    assert 'MANUAL_REQUIRED' in p.terminal_result['reason']


def test_request_cancel_without_native_api_preserves_unrequested_state(rig):
    p = rig.pipeline
    no_cancel_flag_executor(rig)
    with pytest.raises(RuntimeError, match='cancel_all_jobs'):
        p.request_cancel()
    assert p._cancel_wait is None and not p.finished and not rig.calls


def test_request_cancel_without_submitted_map_does_not_call_native(rig):
    native_calls = []
    p = rig.pipeline
    no_cancel_flag_executor(rig, lambda: native_calls.append(True))
    p._restore_map_path = None
    with pytest.raises(RuntimeError, match='地图'):
        p.request_cancel()
    assert not native_calls and p._cancel_wait is None


def test_unclassified_failure_timeout_retains_original_reason_without_restore(rig):
    p = rig.pipeline
    no_cancel_flag_executor(rig)
    p._on_finished(p.executor, False)
    rig.state['time'] = 121.
    rig.tick()
    assert not rig.calls and p.finished
    assert p.terminal_result['status'] == 'failed'
    assert p.terminal_result['restore_deferred'] is True
    assert 'MANUAL_REQUIRED' in p.terminal_result['reason']
    assert 'MRQ finished 委托报告 success=False' in p.terminal_result['reason']


def test_watchdog_error_waits_for_world_without_reclassifying_failure(rig):
    p = rig.pipeline
    no_cancel_flag_executor(rig)
    p.error_messages.append('render failure')
    assert p._start_completion_watchdog()
    rig.state['executor_rendering'] = False
    rig.tick()
    assert not p.finished and not rig.calls
    finish_teardown(rig)
    assert rig.calls[0]['status'] == 'failed'
    assert rig.calls[0]['reason'] == 'MRQ executor error: render failure'
