"""真实目录计数证明 watchdog 在渲染中不再逐 tick 全扫；无需 UE。"""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture(params=('working', 'frozen'))
def render_module(request):
    if request.param == 'working':
        from ue import render_episode
        return render_episode
    path = Path(__file__).resolve().parents[4] / 'Saved/WeatherPushReview/dataset/ue/render_episode.py'
    if not path.is_file():
        pytest.skip('冻结 runtime 为本机生成物')
    spec = importlib.util.spec_from_file_location('_frozen_scan_render', str(path))
    module = importlib.util.module_from_spec(spec)
    old_path = sys.path[:]
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
    return module


@pytest.fixture
def rig(tmp_path, monkeypatch, render_module):
    state = {'rendering': True, 'time': 10., 'scans': 0, 'error': None}
    render_dir = tmp_path / 'render'
    render_dir.mkdir()
    for index in (0, 1):
        (render_dir / ('%06d.png' % index)).write_bytes(b'png')
    original = Path.rglob
    def counted(path, pattern):
        state['scans'] += 1
        if state['error'] is not None:
            raise state['error']
        return original(path, pattern)
    monkeypatch.setattr(Path, 'rglob', counted)
    monkeypatch.setattr(render_module.time, 'monotonic', lambda: state['time'])
    p = render_module._AsyncRenderPipeline([], 'scan_episode', tmp_path, [0, 1])
    p.jobs = [{'name': 'seq', 'cam_id': 'cam', 'kind': 'rgb', 'render_dir': render_dir}]
    p.executor = SimpleNamespace(is_rendering=lambda: state['rendering'])
    return SimpleNamespace(p=p, state=state, directory=render_dir)


def test_rendering_skips_complete_frame_scan(rig):
    for _ in range(30):
        assert rig.p._all_keep_frames_present() is False
    assert rig.state['scans'] == 0


def test_rendering_counts_real_files_at_most_every_two_seconds(rig):
    assert rig.p._total_render_files() == 2
    assert rig.state['scans'] == 2
    (rig.directory / '000002.exr').write_bytes(b'exr')
    for offset in (.01, .5, 1., 1.999):
        rig.state['time'] = 10. + offset
        assert rig.p._total_render_files() == 2
    assert rig.state['scans'] == 2
    rig.state['time'] = 12.
    assert rig.p._total_render_files() == 3
    assert rig.state['scans'] == 4


@pytest.mark.parametrize('rendering', (None, False))
def test_inactive_or_unknown_always_checks_real_files_and_invalidates_cache(rig, rendering):
    assert rig.p._total_render_files() == 2
    if rendering is None:
        rig.p.executor = SimpleNamespace()
    else:
        rig.state['rendering'] = rendering
    (rig.directory / '000002.exr').write_bytes(b'exr')
    assert rig.p._total_render_files() == 3
    (rig.directory / '000002.exr').unlink()
    assert rig.p._total_render_files() == 2
    assert rig.state['scans'] == 6
    rig.p.executor = SimpleNamespace(is_rendering=lambda: True)
    assert rig.p._total_render_files() == 2
    assert rig.state['scans'] == 8


def test_missing_final_frame_is_checked_immediately_after_rendering_stops(rig):
    assert rig.p._all_keep_frames_present() is False
    (rig.directory / '000001.png').unlink()
    rig.state['rendering'] = False
    assert rig.p._all_keep_frames_present() is False
    (rig.directory / '000001.png').write_bytes(b'png')
    assert rig.p._all_keep_frames_present() is True
    # 无 FinalImage 前缀时先查 FinalImage，再走 legacy PNG 兜底，各遍历一次。
    assert rig.state['scans'] == 4


def test_scan_error_is_not_cached_or_suppressed(rig):
    rig.state['error'] = OSError('目录不可读')
    with pytest.raises(OSError, match='目录不可读'):
        rig.p._total_render_files()
    rig.state['error'] = None
    assert rig.p._total_render_files() == 2
    assert rig.state['scans'] == 3


def test_executor_error_is_not_hidden_by_cached_count(rig):
    assert rig.p._total_render_files() == 2
    def broken():
        raise RuntimeError('executor 状态失败')
    rig.p.executor = SimpleNamespace(is_rendering=broken)
    for method in (rig.p._total_render_files, rig.p._all_keep_frames_present):
        with pytest.raises(RuntimeError, match='executor 状态失败'):
            method()
