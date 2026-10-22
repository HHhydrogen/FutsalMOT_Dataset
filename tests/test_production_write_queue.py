"""激进写队列必须在低内存时只收小容量，活动渲染不重建线程池。"""
import importlib.util
from pathlib import Path
import sys
import types


def test_pressure_reduces_capacity_and_keeps_it_for_editor_session(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setitem(sys.modules, 'unreal', types.SimpleNamespace(
        SystemLibrary=types.SimpleNamespace(execute_console_command=lambda world, cmd: commands.append(cmd)),
        register_slate_post_tick_callback=lambda fn: 1,
        unregister_slate_post_tick_callback=lambda handle: None))
    monkeypatch.setitem(sys.modules, 'render_episode', types.SimpleNamespace(_ACTIVE_RENDER=object()))
    monkeypatch.setitem(sys.modules, 'production_session', types.SimpleNamespace(dump=lambda *args: None))
    path = Path(__file__).resolve().parents[1] / 'ue/production_write_queue.py'
    spec = importlib.util.spec_from_file_location('test_queue_runtime', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'commit_headroom_gib', lambda: 20)
    entry = {'task_id':'seed1044', 'episode':str(tmp_path)}
    policy = {'revision':7, 'image_write_concurrency':12, 'image_write_queue_size':12}
    assert module.configure(entry, policy)['ImageWriteQueue.MaxQueueSize'] == 12
    commands.clear()
    monkeypatch.setattr(module, 'commit_headroom_gib', lambda: 8)
    module._tick(0)
    assert commands == ['ImageWriteQueue.MaxQueueSize 10']
    module._last_tick = 0
    monkeypatch.setattr(module, 'commit_headroom_gib', lambda: 7)
    module._tick(0)
    assert commands[-1] == 'ImageWriteQueue.MaxQueueSize 8'
    monkeypatch.setattr(module, 'commit_headroom_gib', lambda: 20)
    assert module.configure(entry, policy)['ImageWriteQueue.MaxQueueSize'] == 8
    next_entry = dict(entry, task_id='seed1048')
    configured = module.configure(next_entry, policy)
    assert configured['ImageWriteQueue.MaxQueueSize'] == 8
    assert configured['ImageWriteQueue.MaxConcurrency'] == 12
    assert module._state['target_capacity'] == 12
    assert module._state['events'][0]['previous_task_id'] == 'seed1044'
    assert module._state['events'][0]['new_capacity'] == 8
