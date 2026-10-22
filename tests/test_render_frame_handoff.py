"""渲染帧同卷硬链接、跨卷复制和原子覆盖的真实文件回归。"""

import errno
import os
import shutil

import pytest

from ue import render_episode


def test_same_volume_frame_uses_shared_inode_and_survives_source_unlink(tmp_path):
    source = tmp_path / 'source.png'
    destination = tmp_path / 'destination.png'
    source.write_bytes(b'complete frame')
    render_episode._link_or_copy_frame_atomic(source, destination)
    assert os.path.samefile(source, destination)
    assert source.stat().st_ino == destination.stat().st_ino
    source.unlink()
    assert destination.read_bytes() == b'complete frame'


def test_cross_volume_falls_back_to_byte_identical_copy(tmp_path, monkeypatch):
    source = tmp_path / 'source.exr'
    destination = tmp_path / 'destination.exr'
    source.write_bytes(bytes(range(256)) * 100)
    copied = []
    original = shutil.copy2
    def cross_volume(*args, **kwargs):
        raise OSError(errno.EXDEV, '跨卷链接')
    def recorded_copy(src, dst):
        copied.append((src, dst))
        return original(src, dst)
    monkeypatch.setattr(render_episode.os, 'link', cross_volume)
    monkeypatch.setattr(render_episode.shutil, 'copy2', recorded_copy)
    render_episode._link_or_copy_frame_atomic(source, destination)
    assert len(copied) == 1
    assert source.read_bytes() == destination.read_bytes()
    assert not os.path.samefile(source, destination)
    assert source.stat().st_mtime_ns == destination.stat().st_mtime_ns


@pytest.mark.parametrize('existing', (False, True))
def test_failed_copy_preserves_source_and_previous_destination_without_partial_files(tmp_path, monkeypatch, existing):
    source = tmp_path / 'source.png'
    destination = tmp_path / 'destination.png'
    source.write_bytes(b'complete frame')
    if existing:
        destination.write_bytes(b'old frame')
    before = set(tmp_path.iterdir())
    def unavailable_link(*args, **kwargs):
        raise OSError(errno.EXDEV, '跨卷链接')
    def partial_copy(src, dst):
        dst.write_bytes(b'partial')
        raise OSError('复制中断')
    monkeypatch.setattr(render_episode.os, 'link', unavailable_link)
    monkeypatch.setattr(render_episode.shutil, 'copy2', partial_copy)
    with pytest.raises(OSError, match='复制中断'):
        render_episode._link_or_copy_frame_atomic(source, destination)
    assert source.read_bytes() == b'complete frame'
    assert set(tmp_path.iterdir()) == before
    if existing:
        assert destination.read_bytes() == b'old frame'
    else:
        assert not destination.exists()


def test_existing_destination_is_atomically_replaced_after_new_frame_ready(tmp_path, monkeypatch):
    source = tmp_path / 'source.exr'
    destination = tmp_path / 'destination.exr'
    source.write_bytes(b'new frame')
    destination.write_bytes(b'old frame')
    replace = os.replace
    calls = []
    def checked_replace(staged, target):
        assert destination.read_bytes() == b'old frame'
        assert staged.read_bytes() == b'new frame'
        calls.append((staged, target))
        return replace(staged, target)
    monkeypatch.setattr(render_episode.os, 'replace', checked_replace)
    render_episode._link_or_copy_frame_atomic(source, destination)
    assert len(calls) == 1
    assert os.path.samefile(source, destination)
    assert destination.read_bytes() == b'new frame'
    assert len(list(tmp_path.iterdir())) == 2


@pytest.mark.parametrize('fallback_copy', (False, True))
def test_failed_atomic_replace_keeps_both_old_target_and_source(tmp_path, monkeypatch, fallback_copy):
    source = tmp_path / 'source.exr'
    destination = tmp_path / 'destination.exr'
    source.write_bytes(b'new frame')
    destination.write_bytes(b'old frame')
    def refused(*args):
        raise PermissionError('目标占用')
    if fallback_copy:
        def unavailable_link(*args, **kwargs):
            raise OSError(errno.EXDEV, '跨卷链接')
        monkeypatch.setattr(render_episode.os, 'link', unavailable_link)
    monkeypatch.setattr(render_episode.os, 'replace', refused)
    with pytest.raises(PermissionError, match='目标占用'):
        render_episode._link_or_copy_frame_atomic(source, destination)
    assert source.read_bytes() == b'new frame'
    assert destination.read_bytes() == b'old frame'
    assert len(list(tmp_path.iterdir())) == 2


def test_same_file_retains_copy2_error_contract(tmp_path):
    source = tmp_path / 'source.png'
    source.write_bytes(b'frame')
    with pytest.raises(shutil.SameFileError):
        render_episode._link_or_copy_frame_atomic(source, source)
    assert source.read_bytes() == b'frame'


def test_rgb_mapping_keeps_source_and_uses_hard_links(tmp_path):
    render_dir = tmp_path / 'render'
    render_dir.mkdir()
    for index in (0, 3):
        (render_dir / ('FinalImage.%06d.png' % index)).write_bytes(str(index).encode())
    destination = tmp_path / 'img1'
    assert render_episode.copy_rendered_frames(render_dir, destination, [0, 3, 6]) == 2
    for frame, index in ((1, 0), (2, 3)):
        assert os.path.samefile(render_dir / ('FinalImage.%06d.png' % index), destination / ('%06d.png' % frame))


def test_repeated_rgb_handoff_keeps_completed_links_and_finishes_remaining_frames(tmp_path):
    render_dir = tmp_path / 'render'
    render_dir.mkdir()
    first = render_dir / 'FinalImage.000000.png'
    second = render_dir / 'FinalImage.000003.png'
    first.write_bytes(b'first frame')
    destination = tmp_path / 'img1'
    assert render_episode.copy_rendered_frames(render_dir, destination, [0, 3]) == 1
    first_inode = (destination / '000001.png').stat().st_ino
    second.write_bytes(b'second frame')
    assert render_episode.copy_rendered_frames(render_dir, destination, [0, 3]) == 2
    assert (destination / '000001.png').stat().st_ino == first_inode
    assert os.path.samefile(second, destination / '000002.png')


def test_repeated_exr_copy_can_resume_before_source_cleanup(tmp_path):
    render_dir = tmp_path / 'render'
    render_dir.mkdir()
    source = render_dir / 'ObjectID.000000.exr'
    source.write_bytes(b'complete exr')
    destination = tmp_path / 'render_mask'
    assert render_episode.copy_object_id_frames(render_dir, destination, [0]) == 1
    assert render_episode.archive_object_id_frames(render_dir, destination, [0]) == 1
    assert not source.exists()
    assert (destination / '000000.exr').read_bytes() == b'complete exr'


def test_exr_archive_keeps_render_frame_numbers_and_removes_only_transferred_source(tmp_path):
    render_dir = tmp_path / 'render'
    render_dir.mkdir()
    sources = {}
    for index in (0, 3, 7):
        path = render_dir / ('ObjectID.%06d.exr' % index)
        path.write_bytes(('frame %d' % index).encode())
        sources[index] = (path, path.stat().st_ino)
    destination = tmp_path / 'render_mask'
    assert render_episode.archive_object_id_frames(render_dir, destination, [0, 3, 6]) == 2
    for index in (0, 3):
        path, inode = sources[index]
        archived = destination / ('%06d.exr' % index)
        assert not path.exists()
        assert archived.stat().st_ino == inode
        assert archived.read_bytes() == ('frame %d' % index).encode()
    assert sources[7][0].is_file()


def test_archive_copy_failure_does_not_delete_any_source(tmp_path, monkeypatch):
    render_dir = tmp_path / 'render'
    render_dir.mkdir()
    for index in (0, 3):
        (render_dir / ('ObjectID.%06d.exr' % index)).write_bytes(b'frame')
    def unavailable_link(*args, **kwargs):
        raise OSError(errno.EXDEV, '跨卷链接')
    def broken_copy(src, dst):
        dst.write_bytes(b'partial')
        raise OSError('复制中断')
    monkeypatch.setattr(render_episode.os, 'link', unavailable_link)
    monkeypatch.setattr(render_episode.shutil, 'copy2', broken_copy)
    destination = tmp_path / 'render_mask'
    with pytest.raises(OSError, match='复制中断'):
        render_episode.archive_object_id_frames(render_dir, destination, [0, 3])
    assert len(list(render_dir.glob('*.exr'))) == 2
    assert list(destination.iterdir()) == []


def test_archive_failure_after_first_transfer_keeps_every_source_and_can_resume(tmp_path, monkeypatch):
    render_dir = tmp_path / 'render'
    render_dir.mkdir()
    for index in (0, 3):
        (render_dir / ('ObjectID.%06d.exr' % index)).write_bytes(('frame %d' % index).encode())
    original = render_episode._link_or_copy_frame_atomic
    calls = []
    def interrupted(source, destination):
        calls.append(source)
        if len(calls) == 2:
            raise OSError('第二帧中断')
        return original(source, destination)
    monkeypatch.setattr(render_episode, '_link_or_copy_frame_atomic', interrupted)
    destination = tmp_path / 'render_mask'
    with pytest.raises(OSError, match='第二帧中断'):
        render_episode.archive_object_id_frames(render_dir, destination, [0, 3])
    assert len(list(render_dir.glob('*.exr'))) == 2
    assert (destination / '000000.exr').read_bytes() == b'frame 0'
    assert not (destination / '000003.exr').exists()
    monkeypatch.setattr(render_episode, '_link_or_copy_frame_atomic', original)
    assert render_episode.archive_object_id_frames(render_dir, destination, [0, 3]) == 2
    assert list(render_dir.iterdir()) == []
    assert (destination / '000003.exr').read_bytes() == b'frame 3'
