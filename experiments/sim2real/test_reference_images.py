"""正式图像目录引用不复制像素，时间采样仍按seed/相机均衡。"""
import os
from pathlib import Path
import pytest
import pipeline


@pytest.mark.skipif(os.name!='nt',reason='Windows目录联接')
def test_reference_directory_preserves_identity_and_rejects_other_target(tmp_path):
    source=tmp_path/'canonical';source.mkdir()
    image=source/'000001.png';image.write_bytes(b'original')
    target=tmp_path/'derived/images/train/seed_00001001_clip_000_CineCam_01'
    pipeline.reference_image_directory(source,target)
    assert os.path.samefile(image,target/'000001.png')
    pipeline.reference_image_directory(source,target)
    other=tmp_path/'other';other.mkdir()
    with pytest.raises(ValueError,match='不同来源'):
        pipeline.reference_image_directory(other,target)
    # 清理联接本身，防止测试清理器遍历源目录。
    os.rmdir(target)


def test_reference_paths_keep_balanced_temporal_selection():
    files=[f'D:/images/train/seed_{seed:08d}_clip_000_CineCam_01/{frame:06d}.png'
        for seed in (1001,1002) for frame in range(1,101)]
    selected,receipt=pipeline.temporal_sample(files,20,42)
    assert len(selected)==20 and len(receipt)==2
    assert {row['selected'] for row in receipt.values()}=={10}


@pytest.mark.skipif(os.name!='nt',reason='Windows目录联接')
def test_training_list_preserves_reference_path_for_yolo_labels(tmp_path):
    source=tmp_path/'canonical';source.mkdir()
    (source/'000001.png').write_bytes(b'original')
    output=tmp_path/'derived';output.mkdir()
    link=output/'images/train/seed_00001001_clip_000_CineCam_01'
    pipeline.reference_image_directory(source,link)
    image=link/'000001.png'
    pipeline.write_lists(output,{'train':[image],'val':[],'test':[]})
    assert (output/'train.txt').read_text().strip()==image.absolute().as_posix()
    os.rmdir(link)
