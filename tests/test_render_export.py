"""渲染帧选择/对齐纯函数的测试（render_episode.py）。"""

import json

from render_episode import (
    copy_rendered_frames,
    map_rendered_to_annotation,
    recover_render_to_img1,
    select_rendered_frame_indices,
)


class TestSelectRenderedFrameIndices:
    def test_300_steps_30fps(self):
        idx = select_rendered_frame_indices(300, 0.1, 30)
        assert len(idx) == 300
        assert idx[0] == 0
        assert idx[1] == 3
        assert idx[2] == 6
        assert idx[298] == 894
        assert idx[299] == 897
        assert all(idx[i] == 3 * i for i in range(300))

    def test_generic_fps(self):
        idx = select_rendered_frame_indices(5, 0.1, 10)
        assert idx == [0, 1, 2, 3, 4]


class TestMapRenderedToAnnotation:
    def test_basic(self):
        m = map_rendered_to_annotation([0, 3, 6, 9], [0, 3, 6, 9])
        assert m == {1: 0, 2: 3, 3: 6, 4: 9}

    def test_missing_frames_skipped(self):
        # 渲染输出缺帧 3、9：对应 annotation 帧被跳过
        m = map_rendered_to_annotation([0, 6], [0, 3, 6, 9])
        assert m == {1: 0, 3: 6}

    def test_unsorted_input(self):
        m = map_rendered_to_annotation([9, 0, 3, 6], [0, 3, 6, 9])
        assert m == {1: 0, 2: 3, 3: 6, 4: 9}


class TestCopyRenderedFrames:
    def test_copy_names(self, tmp_path):
        render_dir = tmp_path / "render"
        img1 = tmp_path / "img1"
        render_dir.mkdir()
        for n in (0, 3, 6):
            (render_dir / f"{n:06d}.png").write_bytes(b"x")
        copied = copy_rendered_frames(render_dir, img1, [0, 3, 6])
        assert copied == 3
        assert (img1 / "000001.png").exists()
        assert (img1 / "000002.png").exists()
        assert (img1 / "000003.png").exists()
        assert not (img1 / "000004.png").exists()

    def test_subdir_scan(self, tmp_path):
        # 渲染输出可能在子目录（MRQ 会建 shot 子目录），递归扫描应能找到
        render_dir = tmp_path / "render" / "sub"
        img1 = tmp_path / "img1"
        render_dir.mkdir(parents=True)
        (render_dir / "000000.png").write_bytes(b"x")
        copied = copy_rendered_frames(render_dir.parent, img1, [0])
        assert copied == 1
        assert (img1 / "000001.png").exists()


def _make_episode(tmp_path, num_steps=3, step_seconds=0.1, fps=30):
    """构造最小 episode（meta.json + frames.jsonl）并返回 (episode_dir, out_dir)。"""
    episode = tmp_path / "episode_demo"
    episode.mkdir()
    meta = {
        "episode_id": "episode_demo",
        "timing": {
            "source_step_seconds": step_seconds,
            "playback_fps": fps,
            "num_steps": num_steps,
        },
    }
    (episode / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (episode / "frames.jsonl").write_text(
        "\n".join(json.dumps({"step": i}) for i in range(num_steps)) + "\n",
        encoding="utf-8",
    )
    return episode, tmp_path / "dataset"


class TestRecoverRenderToImg1:
    def test_recover_writes_img1_and_summary(self, tmp_path):
        # num_steps=3, step=0.1s, fps=30 -> keep_indices = [0, 3, 6]
        episode, out = _make_episode(tmp_path, num_steps=3)
        cam_out = out / "episode_demo" / "Cam_01"
        render_dir = cam_out / "render"
        render_dir.mkdir(parents=True)
        for n in (0, 3, 6, 9):  # 渲染出 4 帧，只保留 3 帧
            (render_dir / f"{n:06d}.png").write_bytes(b"x")

        seqs = [{"name": "LS_Cam", "camera_actor": "Cam_01"}]
        ann = {"render_rgb": {"frame_rate": 30}}
        status, per_cam = recover_render_to_img1(seqs, ann, episode, out)

        assert status == "success"
        assert per_cam["Cam_01"]["img1_frames"] == 3
        assert per_cam["Cam_01"]["ok"] is True
        assert (cam_out / "img1" / "000001.png").exists()
        assert (cam_out / "img1" / "000002.png").exists()
        assert (cam_out / "img1" / "000003.png").exists()
        assert not (cam_out / "img1" / "000004.png").exists()
        summary = json.loads((out / "episode_demo" / "render_summary.json").read_text(encoding="utf-8"))
        assert summary["status"] == "success"
        assert summary["total_img1_frames"] == 3

    def test_recover_missing_render_dir(self, tmp_path):
        episode, out = _make_episode(tmp_path, num_steps=3)
        seqs = [{"name": "LS_Cam", "camera_actor": "Cam_01"}]
        ann = {"render_rgb": {"frame_rate": 30}}
        status, per_cam = recover_render_to_img1(seqs, ann, episode, out)
        assert status == "failed"
        assert per_cam == {}

    def test_recover_partial(self, tmp_path):
        # 渲染只输出 1 帧（000000），keep_indices=[0,3,6] 只能命中 1 帧 -> partial
        episode, out = _make_episode(tmp_path, num_steps=3)
        cam_out = out / "episode_demo" / "Cam_01"
        (cam_out / "render").mkdir(parents=True)
        (cam_out / "render" / "000000.png").write_bytes(b"x")

        seqs = [{"name": "LS_Cam", "camera_actor": "Cam_01"}]
        ann = {"render_rgb": {"frame_rate": 30}}
        status, per_cam = recover_render_to_img1(seqs, ann, episode, out)
        assert status == "partial"
        assert per_cam["Cam_01"]["img1_frames"] == 1
        assert per_cam["Cam_01"]["ok"] is False
