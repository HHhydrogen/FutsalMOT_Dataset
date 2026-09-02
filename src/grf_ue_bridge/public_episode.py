"""生成单 episode 的规范化公开跟踪、分割和姿态标注。"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from annotation_utils import entity_id_to_mask_id
from camera_projection import CameraExtrinsics, CameraIntrinsics, project_world_to_image
from dataset_export import write_json_atomic, write_text_atomic


def encode_coco_rle(mask: np.ndarray) -> dict:
    """按 COCO 的列优先顺序编码二值 mask。"""
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError("mask 必须是二维数组")
    flat = np.asarray(arr != 0, dtype=np.uint8).T.reshape(-1)
    runs: List[int] = []
    last = 0
    count = 0
    for value in flat:
        value = int(value)
        if value == last:
            count += 1
        else:
            runs.append(count)
            count = 1
            last = value
    runs.append(count)
    encoded = bytearray()
    for index, value in enumerate(runs):
        value -= runs[index - 2] if index > 1 else 0
        more = True
        while more:
            byte = value & 0x1F
            value >>= 5
            more = not ((value == 0 and not (byte & 0x10)) or
                        (value == -1 and (byte & 0x10)))
            if more:
                byte |= 0x20
            encoded.append(byte + 48)
    return {"size": [int(arr.shape[0]), int(arr.shape[1])], "counts": encoded.decode("ascii")}


def decode_coco_rle(rle: dict, height: int, width: int) -> np.ndarray:
    """解码 COCO 压缩 RLE，并校验尺寸与 run 总数。"""
    if not isinstance(rle, dict) or rle.get("size") != [int(height), int(width)]:
        raise ValueError("RLE 尺寸与目标尺寸不一致")
    counts = rle.get("counts")
    if not isinstance(counts, str):
        raise ValueError("RLE counts 必须是压缩字符串")
    runs: List[int] = []
    value = shift = 0
    for char in counts.encode("ascii"):
        byte = char - 48
        value |= (byte & 0x1F) << shift
        if byte & 0x20:
            shift += 5
            continue
        if byte & 0x10:
            value -= 1 << (shift + 5)
        if len(runs) > 1:
            value += runs[-2]
        if value < 0:
            raise ValueError("RLE run 长度不能为负数")
        runs.append(value)
        value = shift = 0
    if shift:
        raise ValueError("RLE counts 截断")
    if sum(runs) != height * width:
        raise ValueError("RLE run 总数与尺寸不一致")
    flat = np.zeros(height * width, dtype=np.uint8)
    cursor = 0
    for index, run in enumerate(runs):
        if index % 2:
            flat[cursor:cursor + run] = 1
        cursor += run
    return flat.reshape((width, height)).T


def public_track_id(entity_id: str) -> int:
    """返回公开的稳定 track ID。"""
    if entity_id == "BALL":
        return 100
    match = re.fullmatch(r"([LR])([0-4])", entity_id)
    if not match:
        raise ValueError("未知实体 ID: {!r}".format(entity_id))
    return int(match.group(2)) + (1 if match.group(1) == "L" else 6)


def _load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _load_camera(camera_dir: Path) -> Tuple[int, int, Optional[CameraIntrinsics], Optional[CameraExtrinsics]]:
    data = json.loads((camera_dir / "camera.json").read_text(encoding="utf-8"))
    intr = data.get("intrinsics", data)
    ext = data.get("extrinsics", {})
    width, height = int(intr["width"]), int(intr["height"])
    camera = CameraIntrinsics(width, height, float(intr["fx"]), float(intr["fy"]),
                              float(intr["cx"]), float(intr["cy"]))
    extrinsics = None
    if ext:
        extrinsics = CameraExtrinsics(tuple(ext["world_location_m"]), tuple(ext["forward"]),
                                       tuple(ext["right"]), tuple(ext["up"]))
    return width, height, camera, extrinsics


def _load_mask_for_frame(camera_dir: Path, frame_id: int, mapping: dict) -> np.ndarray:
    """从 Cryptomatte EXR 解出公开 mask；测试可替换此函数。"""
    from .cryptomatte import build_mask, load_cryptomatte
    candidates = sorted((camera_dir / "render_mask").glob("*.exr"))
    if not candidates:
        raise FileNotFoundError("缺少 render_mask Cryptomatte EXR")
    chosen = candidates[min(frame_id - 1, len(candidates) - 1)]
    manifest, ids = load_cryptomatte(chosen)
    return build_mask(manifest, ids, mapping)[0]


def _bbox(mask: np.ndarray, mask_id: int) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask == mask_id)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


def _pose_keypoints(obj: Optional[dict], camera, extrinsics) -> Optional[List[float]]:
    if obj is None or extrinsics is None:
        return None
    points = obj.get("keypoints_world")
    if not isinstance(points, list) or len(points) != 17:
        points = [None] * 17
    result: List[float] = []
    occluded = obj.get("occluded") or []
    for index, point in enumerate(points):
        uv = None
        if isinstance(point, (list, tuple)) and len(point) == 3 and all(v is not None for v in point):
            try:
                uv = project_world_to_image(tuple(float(v) for v in point), camera, extrinsics)
            except (TypeError, ValueError, ZeroDivisionError):
                uv = None
        if uv is None or not all(math.isfinite(float(v)) for v in uv):
            result.extend([0.0, 0.0, 0])
        else:
            visibility = 1 if index < len(occluded) and occluded[index] else 2
            result.extend([float(uv[0]), float(uv[1]), visibility])
    return result


def build_public_manifest(episode_id: str, sequences: list[dict], frame_count: int,
                          width: int, height: int) -> dict:
    """构建公开 episode manifest。"""
    return {
        "schema_version": "futsalmot_public_episode_v1",
        "episode_id": episode_id,
        "trajectory_id": episode_id,
        "sequences": sequences,
        "frame_count": int(frame_count),
        "dimensions": {"width": int(width), "height": int(height)},
        "modalities": ["rgb", "mot", "mots", "pose"],
        "public_classes": {"player": 1, "ball": 100},
        "track_policy": {"players": "L0..L4=1..5,R0..R4=6..10", "ball": 100},
    }


def _write_jpegs(camera_dir: Path, quality: int) -> None:
    from PIL import Image
    for source in sorted((camera_dir / "img1").glob("*")):
        if source.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        target = source.with_suffix(".jpg")
        if source.suffix.lower() != ".jpg":
            Image.open(source).convert("RGB").save(target, quality=int(quality), optimize=False)


def write_public_episode(episode_dir: Path, *, mapping: dict, sequence_configs: list[dict],
                         jpeg_quality: int = 95) -> dict:
    """从内部标注写出所有规范化公开文件。"""
    manifest_sequences = []
    episode_id = None
    max_frames = 0
    out_width = out_height = 0
    for config in sequence_configs:
        camera_dir = Path(config.get("camera_dir", config.get("path", config.get("directory"))))
        name = str(config.get("sequence_name", config.get("name", camera_dir.name)))
        width, height, camera, extrinsics = _load_camera(camera_dir)
        annotations = {int(row["frame_index"]): row for row in _load_jsonl(camera_dir / "annotations.jsonl")}
        pose_rows = _load_jsonl(camera_dir / "pose_keypoints.jsonl")
        pose_by_frame = {int(row["frame_index"]): row for row in pose_rows if row.get("kind") == "frame"}
        if annotations:
            episode_id = episode_id or next(iter(annotations.values())).get("episode_id")
        frame_ids = sorted(annotations)
        max_frames = max(max_frames, len(frame_ids))
        out_width, out_height = width, height
        mot_rows, mots_rows, pose_records = [], [], []
        for frame_id in frame_ids:
            mask = _load_mask_for_frame(camera_dir, frame_id, mapping)
            if np.asarray(mask).shape != (height, width):
                raise ValueError("mask 尺寸与 camera.json 不一致")
            pose_objects = {o.get("entity_id"): o for o in pose_by_frame.get(frame_id, {}).get("objects", [])}
            for entity_id in sorted(mapping, key=public_track_id):
                bbox = _bbox(mask, entity_id_to_mask_id(entity_id))
                if bbox is None:
                    continue
                track_id = public_track_id(entity_id)
                x, y, w, h = bbox
                class_id = 100 if entity_id == "BALL" else 1
                mot_rows.append(f"{frame_id},{track_id},{x},{y},{w},{h},1,{class_id},1.00")
                rle = encode_coco_rle(mask == entity_id_to_mask_id(entity_id))
                mots_rows.append("{} {} {} {} {} {}".format(frame_id, track_id, class_id, height, width,
                                                             json.dumps(rle, separators=(",", ":"))))
                keypoints = None if entity_id == "BALL" else _pose_keypoints(pose_objects.get(entity_id), camera, extrinsics)
                pose_records.append({"frame_id": frame_id, "track_id": track_id,
                                     "class": "ball" if entity_id == "BALL" else "player",
                                     "bbox": [x, y, w, h], "keypoints": keypoints})
        gt_dir = camera_dir / "gt"
        write_text_atomic(gt_dir / "gt.txt", "\n".join(mot_rows) + ("\n" if mot_rows else ""))
        write_json_atomic(gt_dir / "gt_pose.json", pose_records)
        write_text_atomic(gt_dir / "gt_mots.txt", "\n".join(mots_rows) + ("\n" if mots_rows else ""))
        fps = int(config.get("frame_rate", config.get("fps", 30)))
        seqinfo = "[Sequence]\nname={}\nimDir=img1\nframeRate={}\nseqLength={}\nimWidth={}\nimHeight={}\nimExt=.jpg\n".format(name, fps, len(frame_ids), width, height)
        write_text_atomic(camera_dir / "seqinfo.ini", seqinfo)
        _write_jpegs(camera_dir, jpeg_quality)
        manifest_sequences.append({"name": name, "frame_count": len(frame_ids), "width": width, "height": height})
    if episode_id is None:
        episode_id = episode_dir.name
    manifest = build_public_manifest(episode_id, manifest_sequences, max_frames, out_width, out_height)
    write_json_atomic(episode_dir / "episode_manifest.json", manifest)
    return manifest
