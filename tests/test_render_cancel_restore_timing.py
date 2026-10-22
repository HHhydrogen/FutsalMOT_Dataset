"""取消 delegate 可早于 Editor World 恢复；只用 CPU 模拟真实清理顺序。"""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import weakref

import pytest


LEVEL = '/Game/FutsalMOT/Maps/L_FutsalCourt'


@pytest.fixture(params=('working', 'frozen'))
def render_module(request):
    if request.param == 'working':
        from ue import render_episode
        return render_episode
    path = Path(__file__).resolve().parents[4] / 'Saved/WeatherPushReview/dataset/ue/render_episode.py'
    if not path.is_file():
        pytest.skip('冻结 runtime 为本机生成物')
    spec = importlib.util.spec_from_file_location('_frozen_cancel_render', str(path))
    module = importlib.util.module_from_spec(spec)
    old_path = sys.path[:]
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
    return module


@pytest.fixture
def rig(tmp_path, monkeypatch, render_module):
    state = {'world': None, 'pie': True, 'mrq': True, 'executor_rendering': True, 'time': 0.}
    worlds = weakref.WeakSet()
    class World:
        def get_path_name(self):
            return state['world'] + '.L_FutsalCourt'
    def editor_world():
        if state['world'] is None:
            return None
        value = World()
        worlds.add(value)
        return value
    ticks = {}
    removed = []
    def register(callback):
        handle = len(ticks) + 1
        ticks[handle] = callback
        return handle
    def unregister(handle):
        removed.append(handle)
        ticks.pop(handle, None)
    subsystem = SimpleNamespace(is_rendering=lambda: state['mrq'], get_active_executor=lambda: object() if state['mrq'] else None)
    fake = SimpleNamespace(EditorLevelLibrary=SimpleNamespace(get_editor_world=editor_world,
        get_pie_worlds=lambda include_simulating: [object()] if state['pie'] else []),
        MoviePipelineQueueSubsystem=object(), get_editor_subsystem=lambda cls: subsystem,
        register_slate_post_tick_callback=register, unregister_slate_post_tick_callback=unregister)
    monkeypatch.setitem(sys.modules, 'unreal', fake)
    monkeypatch.setattr(render_module.time, 'monotonic', lambda: state['time'])
    monkeypatch.setattr(render_module.time, 'sleep', lambda delay: pytest.fail('Slate tick 不得 sleep'))
    calls = []
    pipeline = render_module._AsyncRenderPipeline([], 'cancel_episode', tmp_path, [0], on_complete=calls.append)
    pipeline.executor = SimpleNamespace(is_cancelled=lambda: True, is_rendering=lambda: state['executor_rendering'])
    # 实际生产由 _start 保存提交时的字符串地图路径，不持有 UWorld。
    pipeline._restore_map_path = LEVEL
    def tick():
        for callback in list(ticks.values()):
            callback(.016)
    return SimpleNamespace(state=state, worlds=worlds, ticks=ticks, removed=removed,
        pipeline=pipeline, calls=calls, tick=tick, fake=fake, module=render_module)


def test_cancel_delegate_waits_for_editor_world_and_stable_inactive_state(rig):
    p = rig.pipeline
    p._on_finished(p.executor, False)
    assert rig.calls == [] and p.finished is False
    assert len(rig.ticks) == 1
    rig.tick()
    rig.state['world'] = LEVEL
    rig.tick()
    assert not rig.calls
    rig.state.update(pie=False, mrq=False, executor_rendering=False)
    rig.tick()
    assert not rig.calls
    rig.tick()
    assert len(rig.calls) == 1 and rig.calls[0]['status'] == 'cancelled'
    assert not rig.ticks and rig.removed
    assert len(rig.worlds) == 0
    p._on_finished(p.executor, False)
    assert len(rig.calls) == 1


def test_watchdog_cancel_uses_same_wait_and_ignores_output_scans(rig, monkeypatch):
    p = rig.pipeline
    monkeypatch.setattr(p, '_all_keep_frames_present', lambda: pytest.fail('取消等待不应扫描输出'))
    assert p._start_completion_watchdog() is True
    rig.tick()
    assert not rig.calls and not p.finished
    assert len(rig.ticks) == 1
    rig.state.update(world=LEVEL, pie=False, mrq=False, executor_rendering=False)
    rig.tick(); rig.tick()
    assert len(rig.calls) == 1 and rig.calls[0]['status'] == 'cancelled'


def test_wrong_map_and_transient_ready_do_not_allow_restore(rig):
    p = rig.pipeline
    p._on_finished(p.executor, False)
    rig.state.update(world='/Game/Other', pie=False, mrq=False, executor_rendering=False)
    rig.tick(); rig.tick()
    assert not rig.calls
    rig.state['world'] = LEVEL
    rig.tick()
    rig.state['world'] = None
    rig.tick()
    rig.state['world'] = LEVEL
    rig.tick()
    assert not rig.calls
    rig.tick()
    assert len(rig.calls) == 1


def test_cancel_timeout_fails_closed_without_calling_transaction_restore(rig):
    p = rig.pipeline
    p._on_finished(p.executor, False)
    rig.state['time'] = 1000.
    rig.tick()
    assert not rig.calls
    assert p.finished is True
    assert p.terminal_result['status'] == 'failed'
    assert 'MANUAL_REQUIRED' in p.terminal_result['reason']
    assert p.terminal_result['restore_deferred'] is True
    assert not rig.ticks


def test_registration_failure_does_not_restore_during_world_teardown(rig):
    rig.fake.register_slate_post_tick_callback = lambda callback: None
    rig.pipeline._on_finished(rig.pipeline.executor, False)
    assert not rig.calls
    assert rig.pipeline.finished is True
    assert rig.pipeline.terminal_result['status'] == 'failed'
    assert 'MANUAL_REQUIRED' in rig.pipeline.terminal_result['reason']


def test_pending_cancel_cannot_be_bypassed_by_late_success_or_force_finalize(rig):
    p = rig.pipeline
    p._on_finished(p.executor, False)
    p._on_finished(p.executor, True)
    p._copy_and_finalize()
    p._finalize(failed=True, reason='另一路完成信号')
    assert not rig.calls and not p.finished
    assert len(rig.ticks) == 1
    rig.state.update(world=LEVEL, pie=False, mrq=False, executor_rendering=False)
    # 即使 executor 清除了取消标志，已登记的等待也必须继续。
    p.executor.is_cancelled = lambda: False
    rig.tick(); rig.tick()
    assert len(rig.calls) == 1 and rig.calls[0]['status'] == 'cancelled'


def test_submit_captures_map_before_synchronous_cancel_delegate(rig, monkeypatch, tmp_path):
    p = rig.pipeline
    p._restore_map_path = None
    p.jobs = [{'name':'Seq', 'cam_id':'Cam', 'seq_asset_path':'/Game/Seq',
               'render_dir':tmp_path/'render', 'img1_dir':tmp_path/'img1'}]
    bound = []
    p.executor.on_executor_finished_delegate = SimpleNamespace(add_callable=bound.append)
    monkeypatch.setattr(rig.module, '_clear_queue', lambda queue: None)
    monkeypatch.setattr(rig.module, '_build_mrq_job', lambda *args, **kwargs: object())
    def submit(subsystem, executor):
        bound[0](executor, False)
        return True
    monkeypatch.setattr(rig.module, '_submit_render', submit)
    assert p.start(object(), object(), p.executor, LEVEL, 3840, 2160, 10, '{frame_number}', 6) is True
    assert p._restore_map_path == LEVEL
    assert not rig.calls and not p.finished and len(rig.ticks) == 1
    rig.state.update(world=LEVEL, pie=False, mrq=False, executor_rendering=False)
    rig.tick(); rig.tick()
    assert len(rig.calls) == 1


def test_unavailable_runtime_api_fails_closed_at_timeout(rig):
    rig.fake.EditorLevelLibrary.get_pie_worlds = lambda include_simulating: None
    rig.state.update(world=LEVEL, pie=False, mrq=False, executor_rendering=False)
    p = rig.pipeline
    p._on_finished(p.executor, False)
    rig.tick(); rig.tick()
    assert not rig.calls and not p.finished
    rig.state['time'] = 121.
    rig.tick()
    assert not rig.calls and p.terminal_result['restore_deferred'] is True
