"""仅在终态移除跨语言回调及job引用，活动MRQ不得被资源清理打断。"""
import gc
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ue'))
from production_session import release_finished_renderers


def fixture_modules(monkeypatch, *, rendering=False):
    calls = []
    subsystem = SimpleNamespace(is_rendering=lambda: rendering)
    unreal = SimpleNamespace(EditorLevelLibrary=SimpleNamespace(get_pie_worlds=lambda _: []),
        MoviePipelineQueueSubsystem=object(), get_editor_subsystem=lambda _: subsystem,
        SystemLibrary=SimpleNamespace(collect_garbage=lambda: calls.append('ue_gc')))
    render = SimpleNamespace(_ACTIVE_RENDER=None, _mrq_get_queue=lambda _: 'queue',
        _clear_queue=lambda queue: calls.append(('clear', queue)))
    monkeypatch.setitem(sys.modules, 'unreal', unreal)
    monkeypatch.setitem(sys.modules, 'render_episode', render)
    monkeypatch.setattr(gc, 'collect', lambda: calls.append('python_gc'))
    return calls


def test_terminal_renderer_releases_delegate_and_native_jobs(monkeypatch):
    calls = fixture_modules(monkeypatch)
    delegate = SimpleNamespace(remove_callable=lambda callback: calls.append(('remove', callback)))
    executor = SimpleNamespace(on_executor_finished_delegate=delegate, on_executor_errored_delegate=delegate,
        get_path_name=lambda: '/Engine/Transient.TestExecutor')
    cls = type('_AsyncRenderPipeline', (), {'__module__': 'render_episode'})
    pipeline = cls()
    pipeline.finished = True
    pipeline.executor = executor
    pipeline.on_complete = object()
    pipeline._on_finished = 'finished_callback'
    pipeline._on_errored = 'error_callback'
    pipeline.jobs = [{'job': object(), 'cam_id': 'CineCam_01'}]
    pipeline.requested_jobs = list(pipeline.jobs)
    monkeypatch.setattr(gc, 'get_objects', lambda: [pipeline])
    assert release_finished_renderers() == ['/Engine/Transient.TestExecutor']
    assert pipeline.executor is None and pipeline.on_complete is None and pipeline.jobs == []
    assert pipeline.requested_jobs == [{'cam_id': 'CineCam_01'}]
    assert calls == [('remove', 'finished_callback'), ('remove', 'error_callback'), ('clear', 'queue'), 'python_gc', 'ue_gc']


def test_active_render_rejects_cleanup_before_any_mutation(monkeypatch):
    calls = fixture_modules(monkeypatch, rendering=True)
    with pytest.raises(AssertionError):
        release_finished_renderers()
    assert calls == []
