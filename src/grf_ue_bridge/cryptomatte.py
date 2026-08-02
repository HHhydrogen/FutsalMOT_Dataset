"""从 Object ID Pass 渲染的 Cryptomatte multilayer EXR 提取逐实体 mask（P1 纯逻辑）。

输入：MRQ `MoviePipelineObjectIdRenderPass` + EXR(multilayer) 输出的 `render_mask/*.exr`
      （UE 5.8 实测）。
输出：`mask/{frame_index:06d}.png`，每个实体像素 = mask_id
      （`L0..L4→1..5`、`R0..R4→6..10`、`BALL→11`，背景 0），与现有
      `annotate-masks` / validator 的 mask 契约完全一致。

UE 5.8 实测编码：
  - manifest 在 EXR header 的 `cryptomatte/<hash>/manifest`（JSON：{actor_label: "hex_id"}）。
  - 实体 ID 以 float32 存于 `RGBA` 层的 R 通道；`hex_id` 是该 float32 位模式的**大端**十六进制。
  - 逐实体 mask = (R 通道 == float32(hex_id))（float32 精确相等）。

依赖：openexr + numpy。
"""

import json
import struct
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

# 把仓库的 ue/ 目录加入 sys.path（与 conftest 一致），以便 import 纯模块
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_UE_DIR = _REPO_ROOT / "ue"
if str(_UE_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_UE_DIR))

from annotation_utils import entity_id_to_mask_id  # noqa: E402
from dataset_export import ensure_dir, write_json_atomic  # noqa: E402
from render_episode import (  # noqa: E402
    map_rendered_to_annotation,
    select_rendered_frame_indices,
)


def hex_id_to_float(hex_id: str) -> float:
    """manifest 的 8 位十六进制 ID → float32（大端位模式）。"""
    return struct.unpack(">f", bytes.fromhex(hex_id))[0]


def load_cryptomatte(exr_path) -> Tuple[dict, np.ndarray]:
    """读一个 Cryptomatte EXR，返回 (manifest: {actor_label: hex_id}, id_channel: float32 R)。

    id_channel 是 (H, W) float32，像素值 == 实体的 float32 ID（0 = 背景）。
    """
    import OpenEXR

    f = OpenEXR.File(str(exr_path))
    # manifest：取 header 里第一个 cryptomatte/*/manifest
    manifest = {}
    for key, val in f.header().items():
        if key.startswith("cryptomatte/") and key.endswith("/manifest"):
            manifest = json.loads(val)
            break
    if not manifest:
        raise ValueError(f"{exr_path}: 无 cryptomatte manifest")
    # ID 通道：优先 RGBA.R；否则在各层 R 通道里找非零且能匹配 manifest 的
    layers = f.channels()
    if "RGBA" in layers:
        px = layers["RGBA"].pixels
        if px is not None and px.ndim == 3 and px.shape[2] >= 1:
            return manifest, np.ascontiguousarray(px[:, :, 0], dtype=np.float32)
    # fallback：扫描各层 R 通道
    ids = None
    for name, ch in layers.items():
        px = getattr(ch, "pixels", None)
        if px is None or px.ndim != 3 or px.shape[2] < 1:
            continue
        cand = np.ascontiguousarray(px[:, :, 0], dtype=np.float32)
        uniq = set(np.unique(cand).tolist())
        matches = sum(hex_id_to_float(h) in uniq for h in manifest.values())
        if ids is None or matches > 0:
            ids = cand
    if ids is None:
        raise ValueError(f"{exr_path}: 找不到 ID 通道")
    return manifest, ids


def entity_names_from_mapping(mapping: dict) -> Dict[str, str]:
    """entity_id → manifest 名（actor 标签）。"""
    return {eid: label for eid, label in mapping.items()}


def convert_frame(
    exr_path: Path,
    mapping: dict,
    out_png: Path,
) -> Dict[str, int]:
    """把一个 Cryptomatte EXR 帧转成 mask PNG（每个实体像素 = mask_id）。

    返回 {entity_id: 像素数}。mapping: entity_id → actor label（manifest 名）。
    """
    from PIL import Image

    manifest, ids = load_cryptomatte(exr_path)
    h, w = ids.shape
    out = np.zeros((h, w), dtype=np.uint8)
    counts: Dict[str, int] = {}
    for entity_id, label in mapping.items():
        hex_id = manifest.get(label)
        if hex_id is None:
            counts[entity_id] = 0
            continue
        mask = ids == np.float32(hex_id_to_float(hex_id))
        n = int(mask.sum())
        counts[entity_id] = n
        if n > 0:
            out[mask] = entity_id_to_mask_id(entity_id)
    ensure_dir(out_png.parent)
    Image.fromarray(out, mode="L").save(str(out_png))
    return counts


def convert_render_mask_dir(
    render_mask_dir: Path,
    mapping: dict,
    mask_dir: Path,
    num_steps: int,
    source_step_seconds: float,
    playback_fps: int,
) -> Tuple[str, Dict[str, dict]]:
    """把 render_mask/*.exr 全部转成 mask/{frame_index:06d}.png。

    帧映射与 RGB 一致：annotation frame_index = step + 1 ↔ render 帧 round(step*step*fps)。
    返回 (status, per_frame)。status ∈ {"success","partial","failed"}。
    """
    keep_indices = select_rendered_frame_indices(num_steps, source_step_seconds, playback_fps)
    # 解析 .exr 帧号
    rendered = {}
    for p in render_mask_dir.glob("*.exr"):
        digits = "".join(ch for ch in p.stem if ch.isdigit())
        if digits:
            rendered[int(digits)] = p
    if not rendered:
        return "failed", {}
    mapping_ann = map_rendered_to_annotation(sorted(rendered.keys()), keep_indices)
    per_frame: Dict[str, dict] = {}
    total = 0
    for frame_index, render_num in mapping_ann.items():
        counts = convert_frame(rendered[render_num], mapping, mask_dir / f"{frame_index:06d}.png")
        per_frame[str(frame_index)] = {"render_frame": render_num, "pixel_counts": counts}
        total += 1
    status = "success" if total == len(keep_indices) else "partial"
    return status, per_frame
