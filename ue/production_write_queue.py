"""正式MRQ的写队列内存回退；只缩小容量，不在活动渲染中重建线程池。"""
import ctypes
from datetime import datetime, timezone
import time
from pathlib import Path
import unreal


class PerformanceInfo(ctypes.Structure):
    _fields_ = [('cb', ctypes.c_ulong)] + [(name, ctypes.c_size_t) for name in (
        'CommitTotal', 'CommitLimit', 'CommitPeak', 'PhysicalTotal', 'PhysicalAvailable',
        'SystemCache', 'KernelTotal', 'KernelPaged', 'KernelNonpaged', 'PageSize')] + [
        ('HandleCount', ctypes.c_ulong), ('ProcessCount', ctypes.c_ulong), ('ThreadCount', ctypes.c_ulong)]


def commit_headroom_gib():
    value = PerformanceInfo()
    value.cb = ctypes.sizeof(value)
    if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(value), value.cb):
        raise ctypes.WinError()
    return (value.CommitLimit - value.CommitTotal) * value.PageSize / 1024**3


_handle = None
_state = None
_last_tick = 0.0


def _save():
    from production_session import dump
    dump(Path(_state['episode']) / 'provenance/write_queue_runtime.json', _state)


def _tick(_delta):
    global _handle, _last_tick
    if time.monotonic() - _last_tick < 0.5:
        return
    _last_tick = time.monotonic()
    import render_episode
    if render_episode._ACTIVE_RENDER is None:
        if _handle is not None:
            unreal.unregister_slate_post_tick_callback(_handle)
            _handle = None
        _save()
        return
    try:
        headroom = commit_headroom_gib()
        _state['minimum_commit_headroom_gib'] = min(_state['minimum_commit_headroom_gib'], headroom)
        capacity = _state['capacity']
        desired = min(capacity, 8 if headroom < 7.5 else (10 if headroom < 8.5 else capacity))
        if desired < capacity:
            unreal.SystemLibrary.execute_console_command(None, 'ImageWriteQueue.MaxQueueSize ' + str(desired))
            _state['capacity'] = desired
            _state['events'].append({'utc': datetime.now(timezone.utc).isoformat(),
                'commit_headroom_gib': round(headroom, 3), 'previous_capacity': capacity,
                'new_capacity': desired, 'reason': '提交内存余量下降，当前seed只收小队列'})
            _save()
    except Exception as exc:
        # 资源读回失败时收回已知可用的8容量；保留原调度器6GiB最终保护。
        unreal.SystemLibrary.execute_console_command(None, 'ImageWriteQueue.MaxQueueSize 8')
        _state['capacity'] = 8
        _state['events'].append({'utc': datetime.now(timezone.utc).isoformat(), 'error': str(exc),
            'new_capacity': 8})
        if _handle is not None:
            unreal.unregister_slate_post_tick_callback(_handle)
            _handle = None
        _save()


def configure(entry, policy):
    """在空闲机位边界设置线程池；同一Editor内保留已触发的容量回退。"""
    global _state, _handle, _last_tick
    if _handle is not None:
        unreal.unregister_slate_post_tick_callback(_handle)
        _handle = None
    concurrency = int(policy['image_write_concurrency'])
    target = int(policy['image_write_queue_size'])
    if _state is None or _state['task_id'] != entry['task_id']:
        previous = _state
        capacity = min(target, previous['capacity']) if previous is not None else target
        _state = {'schema': 'futsalmot_write_queue_runtime', 'version': 1,
            'task_id': entry['task_id'], 'episode': entry['episode'], 'policy_revision': policy['revision'],
            'concurrency': concurrency, 'target_capacity': target, 'capacity': capacity,
            'minimum_commit_headroom_gib': commit_headroom_gib(), 'events': []}
        if capacity < target:
            _state['events'].append({'utc': datetime.now(timezone.utc).isoformat(),
                'previous_task_id': previous['task_id'], 'new_capacity': capacity,
                'reason': '同一Editor继承已触发的内存回退，避免新seed启动时重置大队列'})
    _state['capacity'] = min(_state['capacity'], target)
    unreal.SystemLibrary.execute_console_command(None, 'ImageWriteQueue.MaxQueueSize ' + str(_state['capacity']))
    unreal.SystemLibrary.execute_console_command(None, 'ImageWriteQueue.MaxConcurrency ' + str(concurrency))
    _last_tick = 0.0
    _save()
    return {'ImageWriteQueue.MaxConcurrency': concurrency, 'ImageWriteQueue.MaxQueueSize': _state['capacity']}


def watch():
    global _handle
    if _handle is None:
        _handle = unreal.register_slate_post_tick_callback(_tick)
