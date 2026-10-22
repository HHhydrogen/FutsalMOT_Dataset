"""跨卷图像模拟仍用真实临时文件和硬链接，验证共享物理内容及故障原子性。"""
import errno
import hashlib
import importlib.util
import os
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('sim2real_cache_pipeline',Path(__file__).with_name('pipeline.py'))
pipeline=importlib.util.module_from_spec(spec);spec.loader.exec_module(pipeline)


def simulate_cross_volume(tmp_path,monkeypatch,extra_volume=False):
    source_dir=tmp_path/'g_canonical';source_dir.mkdir()
    research=tmp_path/'d_research';research.mkdir()
    outside=tmp_path/'e_elsewhere';outside.mkdir()
    real_link=os.link
    def cross_link(source,target):
        source,target=Path(source),Path(target)
        if source.parent==source_dir or (extra_volume and 'image_cache' in source.parts and target.parent==outside):
            raise OSError(errno.EXDEV,'模拟不同卷')
        return real_link(source,target)
    monkeypatch.setattr(pipeline,'ROOT',research)
    monkeypatch.setattr(pipeline.os,'link',cross_link)
    return source_dir,research,outside


def test_two_formats_share_one_cross_volume_physical_image(tmp_path,monkeypatch):
    source_dir,research,_=simulate_cross_volume(tmp_path,monkeypatch)
    contents=bytes(range(256))*5000
    first_source=source_dir/'a.png';first_source.write_bytes(contents)
    second_source=source_dir/'b.png';second_source.write_bytes(contents)
    copies=[];real_copy=pipeline.shutil.copyfileobj
    def copy(reader,writer,**kwargs):
        copies.append(writer.name)
        return real_copy(reader,writer,**kwargs)
    monkeypatch.setattr(pipeline.shutil,'copyfileobj',copy)
    yolo=research/'datasets/yolo/images/frame.png'
    mot=research/'datasets/mot/img1/frame.png'
    assert pipeline.link_or_copy(first_source,yolo)=='cache_hardlink'
    assert pipeline.link_or_copy(second_source,mot)=='cache_hardlink'
    cache=list((research/'shared/image_cache').rglob('*'))
    files=[path for path in cache if path.is_file()]
    assert len(files)==len(copies)==1
    assert os.path.samefile(yolo,mot) and os.path.samefile(yolo,files[0])
    assert len({path.stat().st_ino for path in (yolo,mot,files[0])})==1
    assert yolo.read_bytes()==mot.read_bytes()==contents
    assert files[0].name==hashlib.sha256(contents).hexdigest()


def test_same_volume_keeps_direct_hardlink_without_cache(tmp_path,monkeypatch):
    monkeypatch.setattr(pipeline,'ROOT',tmp_path/'research')
    source=tmp_path/'source.png';source.write_bytes(b'original pixels')
    target=tmp_path/'target.png'
    assert pipeline.link_or_copy(source,target)=='hardlink'
    assert os.path.samefile(source,target)
    assert not (pipeline.ROOT/'shared').exists()


@pytest.mark.parametrize('conflict',['target','cache'])
def test_conflicting_existing_content_is_never_overwritten(tmp_path,monkeypatch,conflict):
    source_dir,research,_=simulate_cross_volume(tmp_path,monkeypatch)
    source=source_dir/'source.png';source.write_bytes(b'correct bytes')
    target=research/'yolo.png'
    if conflict=='target':bad=target
    else:
        fingerprint=pipeline.digest(source)
        bad=research/'shared/image_cache'/fingerprint[:2]/fingerprint
    bad.parent.mkdir(parents=True,exist_ok=True);bad.write_bytes(b'conflicting bytes')
    with pytest.raises(ValueError,match='拒绝覆盖'):pipeline.link_or_copy(source,target)
    assert bad.read_bytes()==b'conflicting bytes'
    if conflict=='cache':assert not target.exists()


def test_interrupted_copy_leaves_no_visible_image_or_cache(tmp_path,monkeypatch):
    source_dir,research,_=simulate_cross_volume(tmp_path,monkeypatch)
    source=source_dir/'source.png';source.write_bytes(b'full file')
    target=research/'yolo.png'
    def interrupted(reader,writer,**kwargs):
        writer.write(b'partial')
        raise OSError('模拟复制中断')
    monkeypatch.setattr(pipeline.shutil,'copyfileobj',interrupted)
    with pytest.raises(OSError,match='模拟复制中断'):pipeline.link_or_copy(source,target)
    assert not target.exists()
    assert not [path for path in (research/'shared/image_cache').rglob('*') if path.is_file()]


def test_destination_on_another_volume_has_explicit_copy_fallback(tmp_path,monkeypatch,capsys):
    source_dir,research,outside=simulate_cross_volume(tmp_path,monkeypatch,extra_volume=True)
    source=source_dir/'source.png';source.write_bytes(b'unchanged image')
    target=outside/'frame.png'
    assert pipeline.link_or_copy(source,target)=='copy_fallback'
    assert 'IMAGE_COPY_FALLBACK' in capsys.readouterr().out
    cached=next(path for path in (research/'shared/image_cache').rglob('*') if path.is_file())
    assert target.read_bytes()==source.read_bytes()==cached.read_bytes()
    assert not os.path.samefile(cached,target)


def test_filesystem_without_hardlinks_uses_atomic_rename(tmp_path,monkeypatch,capsys):
    research=tmp_path/'research';monkeypatch.setattr(pipeline,'ROOT',research)
    source=tmp_path/'source.png';source.write_bytes(b'exact file')
    def unavailable(*args,**kwargs):raise OSError('无硬链接支持')
    monkeypatch.setattr(pipeline.os,'link',unavailable)
    target=research/'images/frame.png'
    assert pipeline.link_or_copy(source,target)=='copy_fallback'
    assert target.read_bytes()==source.read_bytes()
    assert not list(research.rglob('*.lock')) and not list(research.rglob('*.tmp_*'))
