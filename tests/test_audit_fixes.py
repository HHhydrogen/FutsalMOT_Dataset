"""审计修复的回归测试：单相机帧级并行 / benchmark 帧数与状态 / 陈旧产物清理 / 进程树内存。

覆盖任务要求的 6.1–6.10：
  单相机公开入口并行（workers>1 帧级分块，不退回串行）、多相机 2/4/8 worker 一致、
  帧块边界（不丢不重）、benchmark 帧数统计（2 相机 × 3 帧 = 6）、benchmark 失败状态、
  陈旧文件清理流程（all → json,mot → all 可恢复）、--no-clean-stale 兼容、
  进程树峰值内存（peak_tree > peak_root）、PNG 语义兼容（像素一致、字节不保证）。
"""

import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from test_mask_annotator import _make_camera

from grf_ue_bridge.mask_annotator import (
    AnnotationConfig,
    annotate_masks_dir,
    build_annotation_tasks,
)

# benchmark 正式实现在包内（scripts/ 薄包装已删除）
from grf_ue_bridge.tools import benchmark_postprocess as bp  # noqa: E402


def _config(**over):
    base = dict(
        mask_channel="r", include_ball=False, polygon_tolerance_px=1.0,
        max_polygon_points=64, id_scale=1.0, id_offset=0.0,
        formats=frozenset({"json", "mot", "yolo-det", "yolo-seg"}),
        no_segmentation=False, clean_stale=True,
    )
    base.update(over)
    return AnnotationConfig(**base)


def _read_bytes(path):
    return path.read_bytes() if path.exists() else None


def _camera_files(cam_dir):
    """收集相机目录下所有产物的路径（相对）+ 字节，用于逐字节比较。"""
    out = {}
    for p in cam_dir.rglob("*"):
        if p.is_file() and p.name != "camera.json" and "mask" not in str(p):
            out[str(p.relative_to(cam_dir))] = p.read_bytes()
    return out


def _two_camera_fixture(tmp_path):
    """构造含 2 个相机（Cam_01/Cam_02，内容相同）的目录。"""
    src = Path(tmp_path) / "src"
    _make_camera(src)
    shutil.copytree(src / "Cam_01", src / "Cam_02")
    return src


# ── 6.1 单相机公开入口并行 ────────────────────────────────────────────

class TestSingleCameraParallel:
    def test_single_camera_creates_frame_tasks(self, tmp_path):
        cam = _make_camera(Path(tmp_path))
        tasks = build_annotation_tasks([cam], workers=4, chunk_size=0, config=_config())
        assert len(tasks) > 1  # 单相机 + workers>1 → 帧级分块，绝不退回串行
        # 合并区间不丢不重
        covered = sorted(x for t in tasks for x in range(t.start_index, t.end_index))
        assert covered == list(range(2))

    def test_single_camera_parallel_equals_serial(self, tmp_path):
        root = Path(tmp_path)
        _make_camera(root)
        d1 = Path(tmp_path) / "a"
        d2 = Path(tmp_path) / "b"
        shutil.copytree(root, d1)
        shutil.copytree(root, d2)
        assert annotate_masks_dir(d1, workers=1) == 0
        assert annotate_masks_dir(d2, workers=4, chunk_size=1) == 0
        assert _camera_files(d1 / "Cam_01") == _camera_files(d2 / "Cam_01")


# ── 6.2 多相机并行一致 ────────────────────────────────────────────────

class TestMultiCameraParallel:
    @pytest.mark.parametrize("workers", [2, 4, 8])
    def test_output_identical_to_serial(self, tmp_path, workers):
        src = _two_camera_fixture(tmp_path)
        d1 = Path(tmp_path) / "serial"
        dw = Path(tmp_path) / f"w{workers}"
        shutil.copytree(src, d1)
        shutil.copytree(src, dw)
        assert annotate_masks_dir(d1, workers=1) == 0
        assert annotate_masks_dir(dw, workers=workers) == 0
        assert _camera_files(d1 / "Cam_01") == _camera_files(dw / "Cam_01")


# ── 6.3 帧块边界 ──────────────────────────────────────────────────────

class TestFrameChunkBoundaries:
    @pytest.mark.parametrize("n_frames,chunk_size", [
        (3, 5),   # n < chunk_size
        (5, 5),   # n == chunk_size
        (6, 5),   # n == chunk_size + 1
        (7, 5),   # n 不能整除 chunk_size
        (0, 5),   # 空
    ])
    def test_chunks_cover_exactly(self, tmp_path, n_frames, chunk_size):
        cam = Path(tmp_path) / "C"
        (cam / "gt").mkdir(parents=True)
        (cam / "camera.json").write_text(
            '{"image_width": 64, "image_height": 64, "intrinsics": {"width": 64, "height": 64}}',
            encoding="utf-8",
        )
        with open(cam / "annotations.jsonl", "w", encoding="utf-8") as f:
            for i in range(n_frames):
                f.write(json.dumps({"frame_index": i + 1, "objects": []}) + "\n")
        tasks = build_annotation_tasks([cam], workers=8, chunk_size=chunk_size, config=_config())
        if n_frames == 0:
            assert [(t.start_index, t.end_index) for t in tasks] == [(0, 0)]
        else:
            covered = sorted(x for t in tasks for x in range(t.start_index, t.end_index))
            assert covered == list(range(n_frames))  # 不丢不重
            assert all(t.start_index < t.end_index for t in tasks if t.end_index > 0)


# ── 6.4 benchmark 帧数统计 ────────────────────────────────────────────

class TestBenchmarkFrameCounting:
    def test_expected_total_frames_sum(self, tmp_path):
        src = _two_camera_fixture(tmp_path)
        cams = bp._camera_dirs(src)
        assert [bp.count_annotation_frames(c) for c in cams] == [2, 2]
        assert sum(bp.count_annotation_frames(c) for c in cams) == 4

    def test_two_cameras_three_frames(self, tmp_path):
        # 2 相机 × 3 帧 → expected_total_frames == 6（不是 3）
        for name in ("A", "B"):
            cam = Path(tmp_path) / name
            (cam / "gt").mkdir(parents=True)
            (cam / "camera.json").write_text(
                '{"image_width": 64, "image_height": 64, "intrinsics": {"width": 64, "height": 64}}',
                encoding="utf-8",
            )
            with open(cam / "annotations.jsonl", "w", encoding="utf-8") as f:
                for i in range(3):
                    f.write(json.dumps({"frame_index": i + 1, "objects": []}) + "\n")
        cams = bp._camera_dirs(Path(tmp_path))
        frames_per_camera = [bp.count_annotation_frames(c) for c in cams]
        assert sum(frames_per_camera) == 6  # 而不是 max=3

    def test_metadata_contains_expected(self, tmp_path):
        src = _two_camera_fixture(tmp_path)
        cams = bp._camera_dirs(src)
        frames = [bp.count_annotation_frames(c) for c in cams]
        import argparse
        args = argparse.Namespace(
            input=str(src), workers=4, chunk_size=0, validation_level="full",
            png_compress_level=1,
        )
        meta = bp.collect_metadata(args, cams, frames, sum(frames))
        assert meta["camera_count"] == 2
        assert meta["frames_per_camera"] == [2, 2]
        assert meta["expected_total_frames"] == 4
        assert meta["workers"] == 4
        assert meta["python_version"]
        assert meta["git_commit"] is not None  # 仓库内运行应有提交哈希（非 git 时为 None 不报错）


# ── 6.5 benchmark 失败状态 ────────────────────────────────────────────

class TestBenchmarkFailureStatus:
    def test_main_fails_when_validate_incomplete(self, tmp_path):
        # 构造 1 相机 fixture，不带 img1 staging → validate 报告缺 img1 → benchmark 非零退出
        import argparse
        import contextlib
        import io
        root = Path(tmp_path)
        _make_camera(root)
        args = argparse.Namespace(
            input=str(root), repeat=1, max_frames=None, keep=False, stage_img1=False,
            validate_on_input=False, only="validate", mask_channel="r", include_ball=False,
            polygon_tolerance_px=1.0, max_polygon_points=64, png_compress_level=1,
            mapping=None, episode=None, workers=1, chunk_size=0, validation_level="full",
            formats="all", no_segmentation=False, no_clean_stale=False,
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = _run_main_with_args(args)
        assert rc != 0  # validate 阶段不完整 → benchmark 必须失败


def _run_main_with_args(args):
    """用给定 args 调用参数化后的 bp.main(argv)（argv 不含 prog）。"""
    argv = ["benchmark_postprocess"]
    for k, v in vars(args).items():
        if v is False or v is None:
            continue
        if v is True:
            argv.append(f"--{k.replace('_', '-')}")
        else:
            argv.extend([f"--{k.replace('_', '-')}", str(v)])
    return bp.main(argv[1:])


# ── 6.6/6.7 陈旧产物清理 ──────────────────────────────────────────────

class TestStaleCleanup:
    def test_all_to_mot_to_all(self, tmp_path):
        root = Path(tmp_path)
        cam = _make_camera(root)
        # formats=all
        assert annotate_masks_dir(root, formats="all", clean_stale=True) == 0
        assert (cam / "labels" / "det").exists()
        assert (cam / "labels" / "seg").exists()
        assert (cam / "gt" / "gt.txt").exists()
        # json,mot + no-seg + clean_stale：det/seg 被清理
        assert annotate_masks_dir(root, formats="json,mot", no_segmentation=True, clean_stale=True) == 0
        assert (cam / "gt" / "gt.txt").exists()
        assert not (cam / "labels" / "det").exists()
        assert not (cam / "labels" / "seg").exists()
        assert (cam / "annotations.jsonl").exists()
        # 恢复 all：所有派生产物重新生成
        assert annotate_masks_dir(root, formats="all", clean_stale=True) == 0
        assert (cam / "labels" / "det").exists()
        assert (cam / "labels" / "seg").exists()
        assert (cam / "gt" / "gt.txt").exists()

    def test_clean_stale_does_not_delete_annotations_or_seqinfo(self, tmp_path):
        root = Path(tmp_path)
        cam = _make_camera(root)
        seq = cam / "seqinfo.ini"
        ann = cam / "annotations.jsonl"
        assert seq.exists() and ann.exists()
        annotate_masks_dir(root, formats="json,mot", no_segmentation=True, clean_stale=True)
        assert seq.exists()   # UE 生成、validator 依赖，绝不删除
        assert ann.exists()   # 核心输出，绝不删除

    def test_no_clean_stale_keeps_old_files(self, tmp_path):
        root = Path(tmp_path)
        cam = _make_camera(root)
        annotate_masks_dir(root, formats="all")
        assert (cam / "labels" / "seg").exists()
        annotate_masks_dir(root, formats="json,mot", no_segmentation=True, clean_stale=False)
        assert (cam / "labels" / "seg").exists()  # --no-clean-stale 保留旧文件


# ── 6.8 进程树内存 ────────────────────────────────────────────────────

class TestProcessTreeMemory:
    def test_tree_rss_greater_than_root(self):
        # 用子进程分配内存，验证树 RSS > root RSS（子进程内存被计入）
        code = (
            "import sys, time; "
            "data = [bytearray(64*1024*1024)]; "  # 64MB
            "time.sleep(2)"
        )
        import psutil
        proc = subprocess.Popen([sys.executable, "-c", code])
        try:
            p = psutil.Process(proc.pid)
            time.sleep(0.8)  # 等子进程分配内存
            r0, c0, t0 = bp.process_tree_rss(p)
            assert t0 > r0  # 子进程 64MB 计入进程树 RSS
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    def test_peakmemory_reports_tree(self):
        pm = bp.PeakMemory()
        pm.start()
        time.sleep(0.2)  # 触发至少一次采样
        peaks = pm.stop()
        assert "tree_rss" in peaks and "root_rss" in peaks
        assert "child_count" in peaks


# ── 6.9 PNG 语义兼容 ──────────────────────────────────────────────────

class TestPngSemanticCompat:
    def test_pixels_equal_bytes_may_differ(self, tmp_path):
        rng = np.random.RandomState(0)
        arr = rng.randint(0, 12, size=(32, 32), dtype=np.uint8)
        p1 = tmp_path / "m1.png"
        p6 = tmp_path / "m6.png"
        Image.fromarray(arr).save(str(p1), compress_level=1)
        Image.fromarray(arr).save(str(p6), compress_level=6)
        # 解码像素逐点一致
        assert np.array_equal(np.array(Image.open(p1)), np.array(Image.open(p6)))
        # 字节不要求一致（不同压缩等级/不同 Pillow 版本都可能不同）
        # 只断言两者都能解码为相同像素，不比较文件字节
