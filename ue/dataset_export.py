"""把标注序列化为 JSONL / MOTChallenge 格式。纯 Python，不依赖 unreal。

本模块提供：
- 原子写入辅助（先写临时文件再替换）。
- episode 数据与 actor 映射的加载（从 import_grf_episode 移入，供 UE 脚本复用）。
- MOTChallenge gt.txt / seqinfo.ini 的构建。

MOT 约定：
- frame 从 1 开始，与内部 annotation 的 frame_index（1 基）一致。
- x,y,w,h 为整数像素，满足 x>=0、y>=0、w>=1、h>=1、x+w<=W、y+h<=H。
- conf 固定为 1。
- class：球员 = 1（MOT16/17 的 pedestrian），球 = 100（自定义类别）。
- visibility：MOT 语义为遮挡程度（0..1）。本仓库第一版不建模真实遮挡，
  通过 mot_visibility_mode 控制（见 build_mot_gt）。
"""

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# MOTChallenge 类别编号
MOT_CLASS_PLAYER = 1  # pedestrian / 球员
MOT_CLASS_BALL = 100  # 自定义类别：球（标准 MOT 无对应类别）

# visibility 模式
VIS_MODE_UNOCCLUDED = "unoccluded"  # 默认：写 1.0，文档化"未建模遮挡"
VIS_MODE_TRUNCATION = "truncation"  # 裁剪面积 / 原始面积，仅基于图像边界截断


def ensure_dir(path: Path) -> Path:
    """创建目录（含父目录），返回该目录。"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_text_atomic(path: Path, text: str) -> None:
    """原子写入文本：先写 .tmp 再 os.replace。"""
    ensure_dir(path.parent)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def write_json_atomic(path: Path, data) -> None:
    """原子写入 JSON。"""
    ensure_dir(path.parent)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def write_jsonl_atomic(path: Path, lines: Sequence[dict]) -> None:
    """原子写入 JSONL（每行一个 JSON 对象）。"""
    ensure_dir(path.parent)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def load_episode(episode_dir: Path):
    """从 episode 目录加载 meta.json 和 frames.jsonl。

    显式 UTF-8：meta.json 可能包含中文字段（如 coordinate_transform 说明）。
    """
    with open(episode_dir / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    frames = []
    with open(episode_dir / "frames.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    return meta, frames


def load_mapping(path: Path) -> dict:
    """加载 actor 映射 JSON。"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def mot_int_bbox(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    """把裁剪后的浮点 bbox 转成满足 MOT 约束的整数 (x, y, w, h)。

    保证 x>=0、y>=0、w>=1、h>=1、x+w<=width、y+h<=height。
    """
    xmin = max(0.0, float(xmin))
    ymin = max(0.0, float(ymin))
    xmax = min(float(width), float(xmax))
    ymax = min(float(height), float(ymax))
    x = int(math.floor(xmin))
    y = int(math.floor(ymin))
    x2 = int(math.ceil(xmax))
    y2 = int(math.ceil(ymax))
    # 保证宽/高至少 1 且在图像内
    x2 = max(x2, x + 1)
    y2 = max(y2, y + 1)
    x2 = min(x2, width)
    y2 = min(y2, height)
    if x2 <= x:
        x = max(0, x2 - 1)
    if y2 <= y:
        y = max(0, y2 - 1)
    return (x, y, max(1, x2 - x), max(1, y2 - y))


def format_mot_line(
    frame: int,
    track_id: int,
    x: int,
    y: int,
    w: int,
    h: int,
    conf: float,
    class_id: int,
    visibility: float,
) -> str:
    """格式化一行 MOTChallenge gt.txt。

    frame 从 1 开始；x/y/w/h 为整数像素；visibility 输出两位小数。
    """
    return (
        f"{int(frame)},{int(track_id)},{int(x)},{int(y)},{int(w)},{int(h)},"
        f"{conf},{int(class_id)},{visibility:.2f}"
    )


def _visibility_for(obj: dict, mode: str) -> float:
    """根据 visibility 模式计算 MOT visibility 值。"""
    if mode == VIS_MODE_UNOCCLUDED:
        return 1.0
    if mode == VIS_MODE_TRUNCATION:
        raw = obj.get("raw_bbox_xywh")
        clipped = obj.get("bbox_xywh")
        if not raw or not clipped:
            return 1.0
        raw_area = max(0.0, float(raw[2]) * float(raw[3]))
        clipped_area = max(0.0, float(clipped[2]) * float(clipped[3]))
        if raw_area <= 0.0:
            return 0.0
        return max(0.0, min(1.0, clipped_area / raw_area))
    raise ValueError(f"未知 mot_visibility_mode: {mode!r}")


def build_mot_gt(
    per_frame_objects: Sequence[Sequence[dict]],
    image_width: int,
    image_height: int,
    include_ball: bool,
    visibility_mode: str = VIS_MODE_UNOCCLUDED,
) -> List[str]:
    """按帧汇总 objects，生成 MOT gt.txt 的行列表。

    per_frame_objects 的每个元素是该帧的 object 列表；object 需包含：
      in_frame   : bool（不在画面中的对象不写入 MOT）
      class      : 'player' 或 'ball'（ball 仅在 include_ball=True 时输出）
      track_id   : int
      bbox_xyxy  : 裁剪后的浮点 (xmin, ymin, xmax, ymax)
      raw_bbox_xywh : 原始浮点 (x, y, w, h)，仅 truncation visibility 使用
      bbox_xywh  : 裁剪后的浮点 (x, y, w, h)，仅 truncation visibility 使用

    只有 in_frame=true 且有合法 bbox_xyxy 的对象会写入 MOT；
    bbox_source="not_visible"（mask 存在但实体不可见）的对象 in_frame=false，
    天然被排除。
    """
    rows: List[str] = []
    for frame_index_1based, objects in enumerate(per_frame_objects, start=1):
        for obj in objects:
            if not obj.get("in_frame"):
                continue
            cls = obj.get("class", "player")
            if cls == "ball" and not include_ball:
                continue
            xyxy = obj.get("bbox_xyxy")
            # 防御：in_frame=true 但 bbox 缺失/非法（schema 不一致）时跳过而非崩溃
            if not isinstance(xyxy, (list, tuple)) or len(xyxy) != 4:
                continue
            x, y, w, h = mot_int_bbox(
                xyxy[0], xyxy[1], xyxy[2], xyxy[3], image_width, image_height
            )
            class_id = MOT_CLASS_PLAYER if cls == "player" else MOT_CLASS_BALL
            vis = _visibility_for(obj, visibility_mode)
            rows.append(
                format_mot_line(
                    frame_index_1based,
                    obj["track_id"],
                    x, y, w, h,
                    1,
                    class_id,
                    vis,
                )
            )
    return rows


def build_seqinfo(
    sequence_name: str,
    im_dir: str,
    frame_rate: int,
    seq_length: int,
    im_width: int,
    im_height: int,
    im_ext: str = ".jpg",
) -> str:
    """生成 MOTChallenge seqinfo.ini 文本。"""
    return (
        "[Sequence]\n"
        f"name={sequence_name}\n"
        f"imDir={im_dir}\n"
        f"frameRate={frame_rate}\n"
        f"seqLength={seq_length}\n"
        f"imWidth={im_width}\n"
        f"imHeight={im_height}\n"
        f"imExt={im_ext}\n"
    )
