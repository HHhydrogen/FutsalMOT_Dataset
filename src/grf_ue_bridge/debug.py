"""debug 可视化：bbox / 彩色 mask / pose 关节点 三套图集 + 自动拼接视频。

统一入口 `debug_annotations_dir`（对输出目录下所有 camera 全量渲染）：
  1. bbox overlay            -> debug/{frame:06d}_bbox.png
  2. 彩色 mask               -> debug/{frame:06d}_mask_color.png
  3. pose 关节点             -> debug/pose/{frame:06d}.png（仅点 + 骨架连线，无文字标注，YOLO 风格）
  4. 三套图集各拼接为 mp4     -> video_bbox.mp4 / video_mask.mp4 / video_pose.mp4

task 集成：`postprocess.debug.enabled=true` 时 `task postprocess` 自动全量执行；
也可 `grf-ue debug <annotation_dir>` 手动执行（对单 camera 可 `grf-ue annotate-overlay` /
`grf-ue pose-overlay` / `grf-ue make-video` 单独跑）。

纯 Python + pillow + numpy + opencv-python（cv2 仅视频需要，缺失时图像仍可渲染）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 把仓库的 ue/ 目录加入 sys.path（与 tests/conftest.py 一致），以便 import 纯模块
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_UE_DIR = _REPO_ROOT / "ue"
if str(_UE_DIR) not in sys.path:
    sys.path.insert(0, str(_UE_DIR))

from instance_mask import load_mask_array, mask_to_color_image  # noqa: E402


def _camera_dirs(annotation_dir: Path) -> List[Path]:
    """递归找出所有含 camera.json 的 camera 子目录（与 annotation_validator 一致）。"""
    return sorted(d.parent for d in annotation_dir.rglob("camera.json"))


# ── bbox overlay ─────────────────────────────────────────────────────

def draw_frame_overlay(img, objects: list, include_ball: bool):
    """把一帧 objects 的 bbox + 标签画到 PIL Image 上，返回新 Image（overlay / 视频共用）。"""
    from PIL import Image, ImageDraw

    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    for obj in objects:
        if not obj.get("in_frame"):
            continue
        if obj.get("class") == "ball" and not include_ball:
            continue
        xmin, ymin, xmax, ymax = obj["bbox_xyxy"]
        color = (0, 255, 0) if obj.get("class") == "player" else (255, 128, 0)
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=2)
        label = f"{obj['entity_id']} #{obj['track_id']}"
        draw.text((xmin, max(0, ymin - 14)), label, fill=color)
    return img


def render_overlay_dir(cam_dir: Path, include_ball: bool = False,
                       mask_color: bool = True,
                       print_fn=print) -> Tuple[int, int]:
    """渲染 bbox overlay（+ 可选彩色 mask）到 debug/。返回 (bbox 帧数, mask 帧数)。

    bbox 读 annotations.jsonl（mask-primary）叠加到 img1/ 对应帧 -> debug/{frame}_bbox.png；
    mask_color=True 时把 mask/*.png 转成彩色可视化 -> debug/{frame}_mask_color.png
    （仅查看，不改写 mask 数据契约）。
    """
    from PIL import Image

    ann_path = cam_dir / "annotations.jsonl"
    img_dir = cam_dir / "img1"
    if not ann_path.exists() or not img_dir.exists():
        print_fn(f"ERROR: 缺少 annotations.jsonl 或 img1/: {cam_dir}")
        return 0, 0

    out_dir = cam_dir / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    drawn = 0
    with open(ann_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                frame = json.loads(line)
            except json.JSONDecodeError:
                continue
            fi = frame.get("frame_index")
            if not isinstance(fi, int):
                continue
            img_path = img_dir / f"{fi:06d}.png"
            if not img_path.exists():
                continue
            img = draw_frame_overlay(Image.open(img_path), frame.get("objects", []), include_ball)
            img.save(out_dir / f"{fi:06d}_bbox.png")
            drawn += 1

    mask_drawn = 0
    if mask_color:
        mask_dir = cam_dir / "mask"
        if not mask_dir.is_dir():
            print_fn(f"  (无 mask/ 目录，跳过彩色 mask 输出: {cam_dir})")
        else:
            for p in sorted(mask_dir.glob("*.png")):
                arr = load_mask_array(p, "r")
                col = Image.fromarray(mask_to_color_image(arr))
                col.save(out_dir / f"{p.stem}_mask_color.png")
                mask_drawn += 1
    return drawn, mask_drawn


# ── 视频编码 ─────────────────────────────────────────────────────────

def _read_seqinfo_fps(cam_dir: Path) -> Optional[int]:
    """从 seqinfo.ini 读取 frameRate（缺省 None）。"""
    p = cam_dir / "seqinfo.ini"
    if not p.exists():
        return None
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip().lower().startswith("framerat"):
                return int(line.split("=", 1)[1].strip())
    except Exception:  # noqa: BLE001
        return None
    return None


def _require_cv2(print_fn=print):
    """校验 opencv-python 可用，缺失时提示并返回 None。"""
    try:
        import cv2  # noqa: F401
        return cv2
    except ImportError:
        print_fn("需要 opencv-python：请运行 `uv sync --extra video` 或 `uv pip install opencv-python`")
        return None


def encode_video(image_paths: List[Path], out_path: Path, fps: float,
                 size: Tuple[int, int]) -> int:
    """把有序图片列表编码为 mp4（cv2，mp4v）。返回写入帧数。"""
    import numpy as np
    from PIL import Image

    cv2 = _require_cv2()
    if cv2 is None:
        return 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (int(size[0]), int(size[1])))
    written = 0
    for p in image_paths:
        img = Image.open(p)
        writer.write(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
        written += 1
    writer.release()
    return written


def make_video(cam_dir: Path, fps: Optional[int] = None, out: Optional[Path] = None,
               plain: bool = False, include_ball: bool = False,
               max_frames: Optional[int] = None, print_fn=print) -> int:
    """把 img1/ 帧编码成 mp4 标注视频（默认叠加 bbox；多视角 = 每相机跑一次）。

    帧顺序取 annotations.jsonl 的 frame_index（默认），否则按 img1/ 文件名排序。
    """
    import numpy as np
    from PIL import Image

    img_dir = cam_dir / "img1"
    ann_path = cam_dir / "annotations.jsonl"
    if not img_dir.is_dir():
        print_fn(f"ERROR: 缺 img1/: {cam_dir}")
        return 1
    if fps is None:
        fps = _read_seqinfo_fps(cam_dir) or 30
    out_path = out or (cam_dir / f"video_{fps}fps.mp4")
    out_path = Path(out_path)

    # 帧列表：annotations（默认，带 bbox 信息）或 img1 文件名
    frames: List[dict] = []
    if ann_path.exists() and not plain:
        with open(ann_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        frames.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    else:
        nums = []
        for p in img_dir.glob("*.png"):
            digits = "".join(ch for ch in p.stem if ch.isdigit())
            if digits:
                nums.append(int(digits))
        frames = [{"frame_index": n} for n in sorted(nums)]
    if max_frames:
        frames = frames[:max_frames]

    first = next((fr for fr in frames if (img_dir / f"{fr['frame_index']:06d}.png").exists()), None)
    if first is None:
        print_fn(f"ERROR: 无可用的 img1 帧: {img_dir}")
        return 1
    with Image.open(img_dir / f"{first['frame_index']:06d}.png") as im:
        W, H = im.size

    paths = [img_dir / f"{fr['frame_index']:06d}.png" for fr in frames if
             (img_dir / f"{fr['frame_index']:06d}.png").exists()]
    written = encode_video(paths, out_path, float(fps), (W, H))
    print_fn(f"视频已写: {out_path}（{written} 帧 @ {fps}fps，{'bbox 叠加' if not plain else '原图'}）")
    return 0 if written else 1


# ── 全量 debug：三套图集 + 三个视频 ──────────────────────────────────

def _frame_number(path: Path) -> int:
    """从 debug 图片文件名解析主帧号（000001_bbox / 000001_mask_color / 000001）。"""
    stem = path.stem
    digits = "".join(ch for ch in stem.split("_")[0] if ch.isdigit())
    return int(digits) if digits else -1


def _debug_image_sets(cam_dir: Path) -> Dict[str, List[Path]]:
    """按图集分组 debug 图片（已按帧号排序）。"""
    debug_dir = cam_dir / "debug"
    sets: Dict[str, List[Path]] = {"bbox": [], "mask": [], "pose": []}
    if debug_dir.is_dir():
        for p in debug_dir.glob("*.png"):
            stem = p.stem
            if stem.endswith("_bbox"):
                sets["bbox"].append(p)
            elif stem.endswith("_mask_color"):
                sets["mask"].append(p)
        pose_dir = debug_dir / "pose"
        if pose_dir.is_dir():
            sets["pose"] = [p for p in pose_dir.glob("*.png")]
    for key in sets:
        sets[key] = sorted(sets[key], key=_frame_number)
    return sets


def render_camera_debug(cam_dir: Path, cfg: Optional[dict] = None,
                        print_fn=print) -> Dict[str, int]:
    """对单个 camera 全量渲染三套 debug 图集。返回 {'bbox': n, 'mask': n, 'pose': n}。

    cfg 可选字段：include_ball / pose_dot_radius / pose_edge_width。
    pose 图集仅当存在 pose_keypoints.jsonl 时生成（点 + 骨架连线，无文字标注）。
    """
    cfg = cfg or {}
    counts: Dict[str, int] = {}
    counts["bbox"], counts["mask"] = render_overlay_dir(
        cam_dir, include_ball=bool(cfg.get("include_ball", False)),
        mask_color=True, print_fn=print_fn,
    )
    if (cam_dir / "pose_keypoints.jsonl").exists():
        from grf_ue_bridge.pose_annotator import pose_overlay_dir

        counts["pose"] = pose_overlay_dir(
            cam_dir,
            dot_radius=int(cfg.get("pose_dot_radius", 5)),
            edge_width=int(cfg.get("pose_edge_width", 3)),
            keypoint_names=False,  # 只画点 + 骨架连线，不标注文字
        )
    else:
        print_fn(f"  (无 pose_keypoints.jsonl，跳过 pose 图集: {cam_dir} —— 需启用 postprocess.yolo_pose)")
        counts["pose"] = 0
    return counts


def make_debug_videos(cam_dir: Path, cfg: Optional[dict] = None,
                      fps: Optional[int] = None, print_fn=print) -> Dict[str, Tuple[Path, int]]:
    """把三套 debug 图集各拼接为 mp4：video_bbox.mp4 / video_mask.mp4 / video_pose.mp4。"""
    cfg = cfg or {}
    if fps is None:
        fps = int(cfg.get("video_fps") or 0) or None
    if fps is None:
        fps = _read_seqinfo_fps(cam_dir) or 30
    from PIL import Image

    sets = _debug_image_sets(cam_dir)
    results: Dict[str, Tuple[Path, int]] = {}
    for name, out_name in (("bbox", "video_bbox.mp4"), ("mask", "video_mask.mp4"),
                           ("pose", "video_pose.mp4")):
        paths = sets.get(name) or []
        if not paths:
            print_fn(f"  (跳过 {name} 视频：无 debug 图片)")
            continue
        with Image.open(paths[0]) as im:
            size = im.size
        out_path = cam_dir / out_name
        n = encode_video(paths, out_path, float(fps), size)
        print_fn(f"  视频 {name}: {out_path.name}（{n} 帧 @ {fps}fps）")
        if n:
            results[name] = (out_path, n)
    return results


def debug_annotations_dir(annotation_dir: Path, cfg: Optional[dict] = None,
                          make_videos: Optional[bool] = None,
                          print_fn=print) -> int:
    """对输出目录下所有 camera 全量渲染 debug 图集并（可选）拼接视频。返回退出码。

    cfg 可选字段：include_ball / make_videos / video_fps / pose_dot_radius / pose_edge_width。
    make_videos：None = 用 cfg.make_videos（默认 true）；True/False = 显式覆盖。
    """
    cfg = dict(cfg or {})
    if make_videos is None:
        make_videos = bool(cfg.get("make_videos", True))
    cam_dirs = _camera_dirs(annotation_dir)
    if not cam_dirs:
        print_fn(f"ERROR: {annotation_dir} 下没有 camera 子目录（缺少 camera.json）")
        return 1
    failures = 0
    for cam_dir in cam_dirs:
        try:
            counts = render_camera_debug(cam_dir, cfg, print_fn=print_fn)
            print_fn(f"  [OK] {cam_dir.name}: bbox={counts['bbox']} mask={counts['mask']} pose={counts['pose']} -> debug/")
            if make_videos:
                make_debug_videos(cam_dir, cfg, print_fn=print_fn)
        except Exception as e:  # noqa: BLE001
            print_fn(f"  ERROR: {cam_dir.name}: {e}")
            failures += 1
    if failures:
        print_fn(f"debug 可视化完成，但有 {failures} 个 camera 目录失败")
        return 1
    print_fn("debug 可视化完成")
    return 0
