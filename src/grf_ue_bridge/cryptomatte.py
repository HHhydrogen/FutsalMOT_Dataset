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

性能说明：
  - EXR 用 PIZ 压缩（整帧解压）；openexr Python 绑定只提供整层 `pixels`（RGBA 4 通道），
    无按通道选择性解码（`InputFile` 已废弃）。因此解码成本是固有的，靠多进程并行摊薄。
  - ID 匹配用 float32 位模式 → uint32 精确整数比较（`ids.view(np.uint32)`），
    比浮点比较更快且 bit-exact。

依赖：openexr + numpy。
"""

import json
import struct
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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


def _resolve_actor_plan(mapping: dict) -> List[Tuple[str, str, int]]:
    """预先解析 entity_id → (actor label, mask_id)，整批转换只算一次。

    返回 [(entity_id, label, mask_id), ...]，与 mapping 的键顺序一致（保证
    counts 字典的插入顺序确定）。
    """
    return [(eid, label, entity_id_to_mask_id(eid)) for eid, label in mapping.items()]


def build_mask(
    manifest: dict,
    ids: np.ndarray,
    mapping: dict,
    plan: Optional[List[Tuple[str, str, int]]] = None,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """把 Cryptomatte ID 通道映射成 mask_id 标签图。

    返回 (out, counts)：out 为 (H, W) uint8，每个实体像素 = mask_id；counts 为
    {entity_id: 像素数}。mapping: entity_id → actor label（manifest 名）。

    位模式精确匹配：manifest 的 8 位十六进制 ID 是 float32 Actor ID 的**大端**
    位模式，直接按位解释为 uint32（`ids.view(np.uint32)`）后做整数相等比较——
    比浮点比较更快且保证 bit-exact（绝无宽松浮点误差导致的 ID 混淆）。
    plan 为 _resolve_actor_plan(mapping) 的缓存（同一批 EXR 只解析一次）。
    """
    if plan is None:
        plan = _resolve_actor_plan(mapping)
    h, w = ids.shape
    out = np.zeros((h, w), dtype=np.uint8)
    counts: Dict[str, int] = {}
    ids_bits = ids.view(np.uint32)  # float32 位模式按位解释为 uint32（视图，不复制）
    for entity_id, label, mask_id in plan:
        hex_id = manifest.get(label)
        if hex_id is None:
            counts[entity_id] = 0
            continue
        try:
            target = np.uint32(int(hex_id, 16))  # 8 位十六进制 → 大端 uint32 位模式
        except ValueError:
            # 防御：非标准 manifest 值回退到 float32 精确比较
            target = np.float32(hex_id_to_float(hex_id))
            mask = ids == target
        else:
            mask = ids_bits == target
        n = int(mask.sum())
        counts[entity_id] = n
        if n > 0:
            out[mask] = mask_id
    return out, counts


def save_mask_png(mask_arr: np.ndarray, out_png: Path, png_compress_level: int = 1) -> None:
    """把 mask_id 标签图写成单通道 L 模式 PNG。

    compress_level: PNG zlib 压缩等级 0–9。实例 ID 图是离散小整数，等级 1 已足够
    （与默认 9 相比解码后逐像素一致，写入显著更快）。不改变像素值。
    """
    from PIL import Image

    ensure_dir(out_png.parent)
    Image.fromarray(mask_arr, mode="L").save(
        str(out_png), compress_level=png_compress_level, optimize=False
    )


def convert_frame(
    exr_path: Path,
    mapping: dict,
    out_png: Path,
    png_compress_level: int = 1,
    plan: Optional[List[Tuple[str, str, int]]] = None,
) -> Dict[str, int]:
    """把一个 Cryptomatte EXR 帧转成 mask PNG（每个实体像素 = mask_id）。

    返回 {entity_id: 像素数}。mapping: entity_id → actor label（manifest 名）。
    """
    manifest, ids = load_cryptomatte(exr_path)
    out, counts = build_mask(manifest, ids, mapping, plan=plan)
    save_mask_png(out, out_png, png_compress_level=png_compress_level)
    return counts


def convert_render_mask_dir(
    render_mask_dir: Path,
    mapping: dict,
    mask_dir: Path,
    num_steps: int,
    source_step_seconds: float,
    playback_fps: int,
    png_compress_level: int = 1,
    workers: int = 0,
    chunk_size: int = 0,
) -> Tuple[str, Dict[str, dict]]:
    """把 render_mask/*.exr 全部转成 mask/{frame_index:06d}.png。

    帧映射与 RGB 一致：annotation frame_index = step + 1 ↔ render 帧 round(step*step*fps)。
    返回 (status, per_frame)。status ∈ {"success","partial","failed"}。

    workers：0=自动（min(相机内帧数, max(1, cpu_count//2))），1=串行，>1=多进程
    并行（每帧一个任务，写入独立 PNG，天然无写冲突）。输出逐字节确定：
    per_frame 按 frame_index 升序组装，与并行与否无关。
    chunk_size：传给 ProcessPoolExecutor.map 的批大小（>0 时），控制每 worker 一批任务数。
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
    plan = _resolve_actor_plan(mapping)
    if workers == 0:
        workers = min(max(1, len(mapping_ann)), max(1, _cpu_count() // 2))

    if workers <= 1:
        per_frame: Dict[str, dict] = {}
        for frame_index, render_num in mapping_ann.items():
            counts = convert_frame(
                rendered[render_num], mapping, mask_dir / f"{frame_index:06d}.png",
                png_compress_level=png_compress_level, plan=plan,
            )
            per_frame[str(frame_index)] = {"render_frame": render_num, "pixel_counts": counts}
    else:
        tasks = [
            (str(rendered[render_num]), str(mask_dir / f"{frame_index:06d}.png"),
             mapping, plan, png_compress_level)
            for frame_index, render_num in mapping_ann.items()
        ]
        chunksize = chunk_size if chunk_size > 0 else 1
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_convert_frame_task, tasks, chunksize=chunksize))
        per_frame = {
            str(fi): {"render_frame": rn, "pixel_counts": counts}
            for (fi, rn), counts in zip(mapping_ann.items(), results)
        }
    total = len(per_frame)
    status = "success" if total == len(keep_indices) else "partial"
    return status, per_frame


def _cpu_count() -> int:
    """os.cpu_count 的防御包装（None 时返回 1）。"""
    import os
    return os.cpu_count() or 1


def _convert_frame_task(task: tuple) -> Dict[str, int]:
    """进程池 worker：转换单个 Cryptomatte EXR 帧 → mask PNG。

    task = (exr_path, out_png_path, mapping, plan, png_compress_level)。
    返回 {entity_id: 像素数}。模块级函数（可 pickle，Windows spawn 安全）。
    """
    exr_path, out_png_path, mapping, plan, png_compress_level = task
    return convert_frame(
        Path(exr_path), mapping, Path(out_png_path),
        png_compress_level=png_compress_level, plan=plan,
    )
