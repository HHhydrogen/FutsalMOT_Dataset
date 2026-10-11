"""渲染帧选择/对齐纯函数的测试（render_episode.py）。"""

import json

from render_episode import (
    archive_object_id_frames,
    combined_file_name_format,
    copy_mask_frames,
    copy_object_id_frames,
    copy_rendered_frames,
    find_final_image_files,
    find_mask_files,
    find_object_id_files,
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


class TestCombinedRenderPassFiles:
    def test_combined_file_name_format_contains_render_pass_and_frame(self):
        assert combined_file_name_format("{frame_number}") == "{render_pass}{frame_number}"
        assert combined_file_name_format("{render_pass}{frame_number}") == "{render_pass}{frame_number}"

    def test_find_final_image_files_ignores_burnin_overlay(self, tmp_path):
        render_dir = tmp_path / "render"
        render_dir.mkdir()
        (render_dir / "FinalImage.000000.png").write_bytes(b"x")
        (render_dir / "BurnInOverlay.000000.png").write_bytes(b"x")
        (render_dir / "FinalImage.000001.png").write_bytes(b"x")
        found = find_final_image_files(render_dir)
        assert sorted(found) == [0, 1]
        assert found[0].name == "FinalImage.000000.png"

    def test_find_object_id_files_uses_only_exr(self, tmp_path):
        render_dir = tmp_path / "render"
        render_dir.mkdir()
        (render_dir / "FinalImage.000000.png").write_bytes(b"x")
        (render_dir / "000000.exr").write_bytes(b"x")
        (render_dir / "ActorHitProxyMask.000000.png").write_bytes(b"x")
        found = find_object_id_files(render_dir)
        assert sorted(found) == [0]
        assert found[0].suffix == ".exr"

    def test_copy_object_id_frames_preserves_render_frame_numbers(self, tmp_path):
        render_dir = tmp_path / "render"
        mask_dir = tmp_path / "render_mask"
        render_dir.mkdir()
        for n in (0, 3, 6):
            (render_dir / f"ActorHitProxyMask.{n:06d}.exr").write_bytes(b"x")
        copied = copy_object_id_frames(render_dir, mask_dir, [0, 3, 6])
        assert copied == 3
        assert (mask_dir / "000000.exr").exists()
        assert (mask_dir / "000003.exr").exists()
        assert (mask_dir / "000006.exr").exists()

    def test_archive_object_id_frames_moves_sources_after_copy(self, tmp_path):
        render_dir = tmp_path / "render"
        mask_dir = tmp_path / "render_mask"
        render_dir.mkdir()
        for n in (0, 3):
            (render_dir / f"ActorHitProxyMask.{n:06d}.exr").write_bytes(b"exr")
        (render_dir / "FinalImage.000000.png").write_bytes(b"png")

        archived = archive_object_id_frames(render_dir, mask_dir, [0, 3, 6])

        assert archived == 2
        assert (mask_dir / "000000.exr").read_bytes() == b"exr"
        assert (mask_dir / "000003.exr").read_bytes() == b"exr"
        assert not (render_dir / "ActorHitProxyMask.000000.exr").exists()
        assert not (render_dir / "ActorHitProxyMask.000003.exr").exists()
        assert (render_dir / "FinalImage.000000.png").exists()

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


class TestFindMaskFiles:
    def test_single_group(self, tmp_path):
        d = tmp_path / "render_mask"
        d.mkdir()
        (d / "000000.png").write_bytes(b"x")
        (d / "000003.png").write_bytes(b"x")
        found = find_mask_files(d)
        assert found == {0: d / "000000.png", 3: d / "000003.png"}

    def test_multi_group_prefers_depth_prefix(self, tmp_path):
        d = tmp_path / "render_mask"
        d.mkdir()
        # 假设 job 同时输出 Beauty 与 CustomDepth（{render_pass} 前缀）
        for prefix in ("Beauty", "CustomDepth"):
            for n in (0, 3):
                (d / f"{prefix}_{n:06d}.png").write_bytes(b"x")
        found = find_mask_files(d)
        assert sorted(found.keys()) == [0, 3]
        assert all("CustomDepth" in str(p) for p in found.values())

    def test_multi_group_fallback_most_frames(self, tmp_path):
        d = tmp_path / "render_mask"
        d.mkdir()
        for n in (0, 3, 6):
            (d / f"A_{n:06d}.png").write_bytes(b"x")
        (d / "B_000000.png").write_bytes(b"x")
        found = find_mask_files(d)
        assert sorted(found.keys()) == [0, 3, 6]  # A 组帧数更多

    def test_exr_object_id_pass(self, tmp_path):
        # Object ID Pass → Cryptomatte multilayer EXR（UE 5.8 生产路径）
        d = tmp_path / "render_mask"
        d.mkdir()
        for n in (0, 3, 6):
            (d / f"{n:06d}.exr").write_bytes(b"x")
        found = find_mask_files(d)
        assert sorted(found.keys()) == [0, 3, 6]
        assert all(p.suffix == ".exr" for p in found.values())


class TestCopyMaskFrames:
    def test_copy_names(self, tmp_path):
        render_mask = tmp_path / "render_mask"
        mask_dir = tmp_path / "mask"
        render_mask.mkdir()
        for n in (0, 3, 6):
            (render_mask / f"{n:06d}.png").write_bytes(b"x")
        copied = copy_mask_frames(render_mask, mask_dir, [0, 3, 6])
        assert copied == 3
        assert (mask_dir / "000001.png").exists()
        assert (mask_dir / "000003.png").exists()
        assert not (mask_dir / "000004.png").exists()

    def test_exr_counts_aligned_without_copying(self, tmp_path):
        # Object ID EXR 源：统计对齐帧数，但不复制成 mask/*.png（解码由 P1 完成）
        render_mask = tmp_path / "render_mask"
        mask_dir = tmp_path / "mask"
        render_mask.mkdir()
        for n in (0, 3, 6):
            (render_mask / f"{n:06d}.exr").write_bytes(b"x")
        copied = copy_mask_frames(render_mask, mask_dir, [0, 3, 6])
        assert copied == 3
        assert not mask_dir.exists()  # 不落任何 mask/*.png

    def test_exr_partial_aligned(self, tmp_path):
        render_mask = tmp_path / "render_mask"
        mask_dir = tmp_path / "mask"
        render_mask.mkdir()
        for n in (0, 6):  # 缺 3
            (render_mask / f"{n:06d}.exr").write_bytes(b"x")
        copied = copy_mask_frames(render_mask, mask_dir, [0, 3, 6])
        assert copied == 2
        assert not mask_dir.exists()


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

    def test_recover_mask_too(self, tmp_path):
        # render/ 与 render_mask/ 都存在 → img1/ 与 mask/ 都恢复，ok 需两者都齐
        episode, out = _make_episode(tmp_path, num_steps=3)
        cam_out = out / "episode_demo" / "Cam_01"
        (cam_out / "render").mkdir(parents=True)
        (cam_out / "render_mask").mkdir(parents=True)
        for n in (0, 3, 6):
            (cam_out / "render" / f"{n:06d}.png").write_bytes(b"x")
            (cam_out / "render_mask" / f"{n:06d}.png").write_bytes(b"x")

        seqs = [{"name": "LS_Cam", "camera_actor": "Cam_01"}]
        ann = {"render_rgb": {"frame_rate": 30}}
        status, per_cam = recover_render_to_img1(seqs, ann, episode, out)
        assert status == "success"
        assert per_cam["Cam_01"]["img1_frames"] == 3
        assert per_cam["Cam_01"]["mask_frames"] == 3
        assert per_cam["Cam_01"]["ok"] is True
        assert (cam_out / "mask" / "000001.png").exists()
        assert (cam_out / "mask" / "000002.png").exists()
        assert (cam_out / "mask" / "000003.png").exists()

    def test_recover_mask_missing_ok_is_false(self, tmp_path):
        # render/ 齐全但 render_mask/ 缺帧 → 整体 partial
        episode, out = _make_episode(tmp_path, num_steps=3)
        cam_out = out / "episode_demo" / "Cam_01"
        (cam_out / "render").mkdir(parents=True)
        (cam_out / "render_mask").mkdir(parents=True)
        for n in (0, 3, 6):
            (cam_out / "render" / f"{n:06d}.png").write_bytes(b"x")
        (cam_out / "render_mask" / "000000.png").write_bytes(b"x")  # mask 只 1 帧

        seqs = [{"name": "LS_Cam", "camera_actor": "Cam_01"}]
        ann = {"render_rgb": {"frame_rate": 30}}
        status, per_cam = recover_render_to_img1(seqs, ann, episode, out)
        assert status == "partial"
        assert per_cam["Cam_01"]["img1_frames"] == 3
        assert per_cam["Cam_01"]["mask_frames"] == 1
        assert per_cam["Cam_01"]["ok"] is False

    def test_recover_exr_mask_success(self, tmp_path):
        # render_mask/ 为 Object ID EXR：恢复后 mask_frames 统计对齐帧数，不落 mask/*.png
        episode, out = _make_episode(tmp_path, num_steps=3)
        cam_out = out / "episode_demo" / "Cam_01"
        (cam_out / "render").mkdir(parents=True)
        (cam_out / "render_mask").mkdir(parents=True)
        for n in (0, 3, 6):
            (cam_out / "render" / f"{n:06d}.png").write_bytes(b"x")
            (cam_out / "render_mask" / f"{n:06d}.exr").write_bytes(b"x")

        seqs = [{"name": "LS_Cam", "camera_actor": "Cam_01"}]
        ann = {"render_rgb": {"frame_rate": 30}}
        status, per_cam = recover_render_to_img1(seqs, ann, episode, out)
        assert status == "success"
        assert per_cam["Cam_01"]["img1_frames"] == 3
        assert per_cam["Cam_01"]["mask_frames"] == 3
        assert per_cam["Cam_01"]["mask_source"] == "object_id_exr"
        assert per_cam["Cam_01"]["ok"] is True
        assert (cam_out / "mask").exists() is False  # EXR 源不直接生成 mask/*.png
        summary = json.loads((out / "episode_demo" / "render_summary.json").read_text(encoding="utf-8"))
        assert summary["status"] == "success"

    def test_recover_exr_mask_missing_partial(self, tmp_path):
        # render_mask/ 缺 1 帧 EXR → 整体 partial，不被当作 success
        episode, out = _make_episode(tmp_path, num_steps=3)
        cam_out = out / "episode_demo" / "Cam_01"
        (cam_out / "render").mkdir(parents=True)
        (cam_out / "render_mask").mkdir(parents=True)
        for n in (0, 3, 6):
            (cam_out / "render" / f"{n:06d}.png").write_bytes(b"x")
        for n in (0, 3):  # mask 缺 000006.exr
            (cam_out / "render_mask" / f"{n:06d}.exr").write_bytes(b"x")

        seqs = [{"name": "LS_Cam", "camera_actor": "Cam_01"}]
        ann = {"render_rgb": {"frame_rate": 30}}
        status, per_cam = recover_render_to_img1(seqs, ann, episode, out)
        assert status == "partial"
        assert per_cam["Cam_01"]["mask_frames"] == 2
        assert per_cam["Cam_01"]["ok"] is False
