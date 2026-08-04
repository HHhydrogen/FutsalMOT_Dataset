"""后处理基准工具：测量 cryptomatte-to-mask / annotate-masks / validate-annotations 的性能。

用法（Windows PowerShell）：

    uv run python scripts/benchmark_postprocess.py `
      --input G:/FutsalMOT_Dataset/episode_demo `
      --repeat 3

    uv run python scripts/benchmark_postprocess.py `
      --input G:/FutsalMOT_Dataset/episode_0001 `
      --repeat 1 --max-frames 30 --workers 4

报告的三个命令级总耗时（time.perf_counter，多次 repeat 取平均）：
  cryptomatte-to-mask   —— 每相机一次 convert_render_mask_dir（含 EXR 解码 /
                            ID 映射 / PNG 写入；--workers 透传并行）
  annotate-masks        —— 整个目录一次 annotate_masks_dir（--workers 并行，
                            --formats / --no-segmentation 透传）
  validate-annotations  —— 整个目录一次 validate_annotation_dir（--workers /
                            --validation-level 透传）

另报告 annotate 的串行语义逐阶段分解（mask_read / quantize / bbox_count /
polygon，仅定位瓶颈用）、峰值 RSS（psutil 轮询）、处理帧数、每秒处理帧数。

工作方式：在临时目录里 staging 一份 camera.json / annotations.jsonl / mask PNG 子集，
在原目录读取 render_mask/*.exr（体积大，不复制），所有写入都落在临时目录，绝不修改真实数据集。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
_UE_DIR = _REPO_ROOT / "ue"
for p in (str(_UE_DIR), str(_REPO_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)


def _camera_dirs(root: Path) -> List[Path]:
    """递归找出所有含 camera.json 的 camera 子目录（与 annotation_validator 一致）。"""
    if (root / "camera.json").exists():
        return [root]
    return sorted(d.parent for d in root.rglob("camera.json"))


def _annotation_frame_indices(cam_dir: Path) -> List[int]:
    """按序读 annotations.jsonl 的 frame_index。文件缺失返回 []。"""
    p = cam_dir / "annotations.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(int(json.loads(line).get("frame_index", 0)))
            except (json.JSONDecodeError, ValueError):
                continue
    return out


def _stage_camera(cam_dir: Path, work_dir: Path, max_frames: Optional[int],
                  copy_mask: bool = True, stage_img1: bool = False) -> Path:
    """把单个 camera 的最小输入复制到 work_dir 下，返回 work 里的 camera 目录。

    复制 camera.json / annotations.jsonl / seqinfo.ini；若 copy_mask，则复制
    mask/ 中前 max_frames 个 annotation 帧号对应的 PNG（mask 图很小）；
    stage_img1 时同时复制 img1/ 的对应 RGB 帧（validate 的 RGB↔mask 一致性需要）。
    render_mask/*.exr 不复制（体积大，基准在原目录读取）。
    """
    work_cam = work_dir / cam_dir.name
    work_cam.mkdir(parents=True, exist_ok=True)
    for fname in ("camera.json", "annotations.jsonl", "seqinfo.ini"):
        src = cam_dir / fname
        if src.exists():
            shutil.copy2(src, work_cam / fname)
    if copy_mask:
        mask_src = cam_dir / "mask"
        if mask_src.exists():
            (work_cam / "mask").mkdir(parents=True, exist_ok=True)
            indices = _annotation_frame_indices(cam_dir)
            if max_frames:
                indices = indices[:max_frames]
            for fi in indices:
                src = mask_src / f"{fi:06d}.png"
                if src.exists():
                    shutil.copy2(src, work_cam / "mask" / f"{fi:06d}.png")
    if stage_img1:
        img1_src = cam_dir / "img1"
        if img1_src.exists():
            (work_cam / "img1").mkdir(parents=True, exist_ok=True)
            indices = _annotation_frame_indices(cam_dir)
            if max_frames:
                indices = indices[:max_frames]
            for fi in indices:
                src = img1_src / f"{fi:06d}.png"
                if src.exists():
                    shutil.copy2(src, work_cam / "img1" / f"{fi:06d}.png")
    return work_cam


class PeakMemory:
    """后台线程轮询进程 RSS 峰值（psutil；不可用时静默返回 0）。"""

    def __init__(self) -> None:
        self._peak = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        try:
            import psutil
            self._proc = psutil.Process()
        except Exception:
            self._proc = None

    def start(self) -> None:
        if self._proc is None:
            return
        self._peak = 0
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                self._peak = max(self._peak, self._proc.memory_info().rss)
            except Exception:
                pass
            self._stop.wait(0.05)

    def stop(self) -> int:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=0.5)
        return self._peak

    def peak_mb(self) -> float:
        return self._peak / (1024 * 1024)


class Timer:
    """累加计时器：记录每个阶段的累计耗时与调用次数。"""

    def __init__(self) -> None:
        self.acc: Dict[str, float] = {}
        self.count: Dict[str, int] = {}

    def run(self, phase: str, fn: Callable[[], None]) -> None:
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        self.acc[phase] = self.acc.get(phase, 0.0) + dt
        self.count[phase] = self.count.get(phase, 0) + 1

    def run_ret(self, phase: str, fn: Callable[[], object]) -> object:
        t0 = time.perf_counter()
        ret = fn()
        dt = time.perf_counter() - t0
        self.acc[phase] = self.acc.get(phase, 0.0) + dt
        self.count[phase] = self.count.get(phase, 0) + 1
        return ret

    def snapshot(self) -> Dict[str, float]:
        return dict(self.acc)


# ── annotate-masks 阶段计时（定位瓶颈用）───────────────────────────────

def _upgrade_object_timed(obj: dict, mask_ids, stats, timer: Timer,
                          polygon_tolerance_px: float, max_polygon_points: int,
                          id_scale: float, id_offset: float) -> None:
    """在 object 上做与 mask_annotator._upgrade_object 等价的 mask→bbox/分割升级，但分阶段计时。

    stats 为单帧 compute_instance_stats（新实现）结果；为 None 时回退到旧的
    逐对象 decode+nonzero 路径（baseline 用）。只计时，不改写 obj。
    """
    from annotation_utils import entity_id_to_mask_id
    from instance_mask import merge_to_single_ring, ring_to_yolo_flat

    entity_id = obj.get("entity_id")
    try:
        mask_id = entity_id_to_mask_id(entity_id)
    except (ValueError, TypeError):
        return
    width, height = mask_ids.shape[1], mask_ids.shape[0]

    def bbox_count_old() -> Tuple[Optional[list], int]:
        from instance_mask import decode_mask_pixels, mask_to_bbox, visible_pixel_count
        binary = decode_mask_pixels(mask_ids, mask_id, id_scale, id_offset)
        bb = mask_to_bbox(binary)
        return (list(bb) if bb is not None else None), (visible_pixel_count(binary) if bb is not None else 0)

    st = stats.get(mask_id) if stats is not None else None
    if st is not None and st.pixel_count > 0:
        bbox = list(st.bbox_xyxy)
        vpc = st.pixel_count
        roi_binary = mask_ids[st.roi_slice] == mask_id
        x0, y0 = int(st.bbox_xyxy[0]), int(st.bbox_xyxy[1])
    else:
        bb, vpc = timer.run_ret("bbox_count", bbox_count_old)
        if bb is None or vpc == 0:
            return
        bbox = bb
        x0, y0 = int(bbox[0]), int(bbox[1])
        roi_binary = None

    # polygon 阶段
    def poly_work() -> None:
        from instance_mask import decode_mask_pixels, mask_to_polygons_with_areas
        if roi_binary is not None:
            polys, comp_areas = mask_to_polygons_with_areas(roi_binary, polygon_tolerance_px, max_polygon_points)
            polys = [[(px + x0, py + y0) for px, py in poly] for poly in polys]
        else:
            binary = decode_mask_pixels(mask_ids, mask_id, id_scale, id_offset)
            polys, comp_areas = mask_to_polygons_with_areas(binary, polygon_tolerance_px, max_polygon_points)
        if polys:
            ring, _meta = merge_to_single_ring(polys, comp_areas)
            ring_to_yolo_flat(ring, width, height)

    timer.run("polygon", poly_work)


def _bench_annotate(work_cam: Path, mask_channel: str, include_ball: bool,
                    polygon_tolerance_px: float, max_polygon_points: int,
                    id_scale: float, id_offset: float,
                    expected_frames: int) -> Tuple[Dict[str, float], int]:
    """分阶段计时 annotate-masks 的逐帧工作（只读 mask + 计算，不写盘）。"""
    from instance_mask import load_mask_array, quantize_mask_pixels

    ann_path = work_cam / "annotations.jsonl"
    if not ann_path.exists():
        return {}, 0
    frames = [json.loads(l) for l in ann_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    mask_dir = work_cam / "mask"
    timer = Timer()
    n = 0
    # 新实现的单帧实例统计（compute_instance_stats）可用时每帧只算一次
    try:
        from instance_mask import compute_instance_stats
        stats_available = True
    except (ImportError, AttributeError):
        stats_available = False
    for frame in frames:
        fi = int(frame.get("frame_index", 0))
        mask_path = mask_dir / f"{fi:06d}.png"
        if not mask_path.exists():
            continue
        mask_img = None

        def do_read() -> None:
            nonlocal mask_img
            mask_img = load_mask_array(mask_path, mask_channel)

        timer.run("mask_read", do_read)
        quantized = None
        stats = None

        def do_quantize() -> None:
            nonlocal quantized, stats
            quantized = quantize_mask_pixels(mask_img, id_scale, id_offset)
            if stats_available:
                from instance_mask import compute_instance_stats
                stats = compute_instance_stats(quantized)

        timer.run("quantize", do_quantize)
        # bbox+count / polygon：逐对象（stats 复用同一帧结果）
        for obj in frame.get("objects", []):
            _upgrade_object_timed(obj, quantized, stats, timer,
                                  polygon_tolerance_px, max_polygon_points, id_scale, id_offset)
        n += 1
    return timer.snapshot(), n




# ── 报告 ───────────────────────────────────────────────────────────────

def _fmt(seconds: float) -> str:
    return f"{seconds:8.3f}"


def _report_run(title: str, totals: Dict[str, float], breakdown: Dict[str, float],
                frames: int, peak_mb: float) -> None:
    print(f"\n=== {title} ===")
    pipeline = sum(totals.values())
    print(f"  cryptomatte-to-mask : {_fmt(totals.get('cryptomatte', 0.0))} s")
    print(f"  annotate-masks      : {_fmt(totals.get('annotate', 0.0))} s")
    print(f"  validate-annotations: {_fmt(totals.get('validate', 0.0))} s")
    print(f"  pipeline(total)     : {_fmt(pipeline)} s")
    if breakdown:
        print("  annotate compute breakdown (serial semantics):")
        for k in ("mask_read", "quantize", "bbox_count", "polygon"):
            if breakdown.get(k, 0.0) > 0:
                print(f"    {k:<12} {_fmt(breakdown[k])} s")
        print(f"    {'sum':<12} {_fmt(sum(breakdown.values()))} s")
    print(f"  frames              : {frames}")
    print(f"  fps (pipeline)      : {_fmt(frames / pipeline) if pipeline > 0 else 0.0}")
    print(f"  peak_rss            : {_fmt(peak_mb)} MB")


def _avg_totals(runs: List[Dict[str, float]]) -> Dict[str, float]:
    agg: Dict[str, float] = {}
    for r in runs:
        for k, v in r.items():
            agg[k] = agg.get(k, 0.0) + v
    n = max(1, len(runs))
    return {k: v / n for k, v in agg.items()}


# ── 主流程 ─────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="后处理性能基准工具")
    parser.add_argument("--input", required=True, help="episode 或 camera 目录（含 camera.json）")
    parser.add_argument("--repeat", type=int, default=3, help="重复次数（默认 3）")
    parser.add_argument("--max-frames", type=int, default=None, help="每相机最多处理的 annotation 帧数")
    parser.add_argument("--keep", action="store_true", help="保留临时工作目录")
    parser.add_argument("--stage-img1", action="store_true",
                        help="同时复制 img1/ RGB 子集（validate 需要 RGB↔mask 一致性校验）")
    parser.add_argument("--only", choices=["cryptomatte", "annotate", "validate"],
                        default=None, help="只跑某个阶段（默认全部）")
    parser.add_argument("--mask-channel", default="r")
    parser.add_argument("--include-ball", action="store_true")
    parser.add_argument("--polygon-tolerance-px", type=float, default=1.0)
    parser.add_argument("--max-polygon-points", type=int, default=64)
    parser.add_argument("--png-compress-level", type=int, default=1)
    parser.add_argument("--mapping", default=None, help="actor 映射 JSON（cryptomatte 用）")
    parser.add_argument("--episode", default=None, help="episode 目录（cryptomatte 读时序用）")
    parser.add_argument("--workers", type=int, default=0,
                        help="命令并行 worker 数（0=自动，1=串行；透传给三个命令）")
    parser.add_argument("--chunk-size", type=int, default=0, help="帧分块大小（透传）")
    parser.add_argument("--validation-level", choices=["full", "quick"], default="full")
    parser.add_argument("--formats", default="all", help="annotate 导出的格式（透传）")
    parser.add_argument("--no-segmentation", action="store_true")
    args = parser.parse_args()

    root = Path(args.input).resolve()
    cams = _camera_dirs(root)
    if not cams:
        print(f"ERROR: {root} 下没有 camera 子目录（缺少 camera.json）")
        return 1

    # 解析 cryptomatte 配置（与 cli 相同逻辑）
    mapping_dict = None
    num_steps = step_sec = fps = None
    cfg = {}
    cfg_path = _REPO_ROOT / "ue_import_config.json"
    if cfg_path.exists():
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg = {k: v for k, v in raw.items() if not k.startswith("comment_")}
    if args.only in (None, "cryptomatte"):
        import json as _json
        from grf_ue_bridge.cryptomatte import convert_render_mask_dir  # noqa: F401 (校验依赖可 import)
        from dataset_export import load_episode, load_mapping

        mapping_path = Path(args.mapping) if args.mapping else Path(cfg.get("mapping", ""))
        episode_path = Path(args.episode) if args.episode else Path(cfg.get("episode", ""))
        if not mapping_path.exists() or not episode_path.exists():
            print("ERROR: 需要 --mapping 与 --episode（或 ue_import_config.json 中的映射/episode）")
            return 1
        meta, _frames = load_episode(episode_path)
        num_steps = int(meta["timing"]["num_steps"])
        step_sec = float(meta["timing"]["source_step_seconds"])
        fps = int(meta["timing"].get("playback_fps", 30))
        mapping_dict = load_mapping(mapping_path)

    peak = PeakMemory()
    run_totals: List[Dict[str, float]] = []
    run_breakdowns: List[Dict[str, float]] = []
    run_frames: List[int] = []
    run_peaks: List[float] = []

    with tempfile.TemporaryDirectory(prefix="grfue_bench_") as tmp:
        work_root = Path(tmp)
        # staging：每相机复制 camera.json / annotations.jsonl / seqinfo / mask（+可选 img1）
        work_cams: Dict[str, Path] = {}
        for cam in cams:
            work_cams[cam.name] = _stage_camera(cam, work_root, args.max_frames,
                                                copy_mask=True, stage_img1=args.stage_img1)
        for rep in range(args.repeat):
            totals: Dict[str, float] = {}
            breakdown: Dict[str, float] = {}
            total_frames = 0
            peak.start()
            # 1) cryptomatte-to-mask：每相机一次完整转换（含 EXR 解码 / 映射 / PNG 写入）
            if args.only in (None, "cryptomatte") and mapping_dict is not None:
                crypto_acc = 0.0
                for cam in cams:
                    if not (cam / "render_mask").exists():
                        continue
                    work_cam = work_cams[cam.name]
                    t0 = time.perf_counter()
                    status, per = convert_render_mask_dir(
                        cam / "render_mask", mapping_dict, work_cam / "mask",
                        num_steps, step_sec, fps,
                        png_compress_level=args.png_compress_level,
                        workers=args.workers, chunk_size=args.chunk_size,
                    )
                    crypto_acc += time.perf_counter() - t0
                    total_frames = max(total_frames, len(per))
                totals["cryptomatte"] = crypto_acc
            # 2) annotate-masks：整个 work_root 一次（相机级/帧级并行由 --workers 控制）
            if args.only in (None, "annotate"):
                from grf_ue_bridge.mask_annotator import annotate_masks_dir
                t0 = time.perf_counter()
                annotate_masks_dir(
                    work_root,
                    mask_channel=args.mask_channel,
                    include_ball=args.include_ball,
                    polygon_tolerance_px=args.polygon_tolerance_px,
                    max_polygon_points=args.max_polygon_points,
                    id_scale=1.0, id_offset=0.0,
                    workers=args.workers, chunk_size=args.chunk_size,
                    formats=args.formats, no_segmentation=args.no_segmentation,
                )
                totals["annotate"] = time.perf_counter() - t0
                # 逐阶段分解（串行语义，只为定位瓶颈）
                for name, work_cam in work_cams.items():
                    ph, n = _bench_annotate(work_cam, args.mask_channel, args.include_ball,
                                            args.polygon_tolerance_px, args.max_polygon_points,
                                            1.0, 0.0, args.max_frames)
                    for k, v in ph.items():
                        breakdown[k] = breakdown.get(k, 0.0) + v
                    total_frames = max(total_frames, n)
            # 3) validate-annotations：整个 work_root 一次
            if args.only in (None, "validate"):
                from grf_ue_bridge.annotation_validator import validate_annotation_dir
                t0 = time.perf_counter()
                validate_annotation_dir(work_root, workers=args.workers,
                                        validation_level=args.validation_level)
                totals["validate"] = time.perf_counter() - t0
                total_frames = max(total_frames, len(work_cams) * (args.max_frames or 1))
            peak.stop()
            run_totals.append(totals)
            run_breakdowns.append(breakdown)
            run_frames.append(total_frames)
            run_peaks.append(peak.peak_mb())
            if args.keep:
                print(f"workdir (repeat {rep}): {work_root}")

    # 汇总报告
    print("\n================================================")
    print("benchmark_postprocess 汇总（同一台机器，同一份数据）")
    print(f"input: {root}   cameras: {len(cams)}   repeat: {args.repeat}   max_frames: {args.max_frames}"
          f"   workers: {args.workers}   validation_level: {args.validation_level}"
          f"   formats: {args.formats}{'  --no-segmentation' if args.no_segmentation else ''}")
    avg_totals = _avg_totals(run_totals)
    avg_breakdown = _avg_totals(run_breakdowns)
    avg_frames = sum(run_frames) / max(1, len(run_frames))
    avg_peak = sum(run_peaks) / max(1, len(run_peaks))
    _report_run("ALL CAMERAS (avg per repeat)", avg_totals, avg_breakdown,
                int(avg_frames), avg_peak)
    return 0


if __name__ == "__main__":
    sys.exit(main())
