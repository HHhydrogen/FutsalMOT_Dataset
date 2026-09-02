"""验证公开 episode 输出目录。"""

from __future__ import annotations

import configparser
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from PIL import Image

SEQUENCE_NAME_RE = re.compile(r"^FutsalMOT_(?P<episode>.+)_C(?P<camera>\d{2})$")


@dataclass
class ValidationResult:
    """公开校验结果，同时可直接作为 CLI 退出码使用。"""

    ok: bool
    errors: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def exit_code(self) -> int:
        return 0 if self.ok else 1

    def __int__(self) -> int:
        return self.exit_code


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _number(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _strict_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, str) or not value or value[0] in "+-" and not value[1:].isdigit() or not value.lstrip("-").isdigit():
        raise ValueError("不是严格整数")
    return int(value)


def _json(path: Path, errors: List[str]) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"无法读取 JSON {path}: {exc}")
        return None


def _read_rle(value: Any, height: int, width: int) -> int:
    if not isinstance(value, dict) or value.get("size") != [height, width]:
        raise ValueError("RLE 尺寸不匹配")
    counts = value.get("counts")
    if not isinstance(counts, str):
        raise ValueError("RLE counts 不是字符串")
    runs: List[int] = []
    number = shift = 0
    for char in counts.encode("ascii"):
        token = char - 48
        if token < 0 or token > 63:
            raise ValueError("RLE 含非法字符")
        number |= (token & 31) << shift
        if token & 32:
            shift += 5
            continue
        if token & 16:
            number -= 1 << (shift + 5)
        if len(runs) > 1:
            number += runs[-2]
        if number < 0:
            raise ValueError("RLE run 长度不能为负数")
        runs.append(number)
        number = shift = 0
    if shift or sum(runs) != height * width:
        raise ValueError("RLE run 总数与声明尺寸不一致")
    return sum(runs[1::2])


def _seqinfo(path: Path, errors: List[str]) -> Dict[str, str]:
    try:
        parser = configparser.ConfigParser()
        parser.optionxform = str
        parser.read(path, encoding="utf-8")
        if not parser.has_section("Sequence"):
            errors.append(f"seqinfo.ini 缺少 [Sequence]: {path}")
            return {}
        return {key.lower(): value for key, value in parser.items("Sequence")}
    except (OSError, UnicodeDecodeError, configparser.Error, ValueError) as exc:
        errors.append(f"seqinfo.ini 读取失败: {path}: {exc}")
        return {}


def _validate_sequence(cam: Path, manifest_seq: dict, errors: List[str]) -> Tuple[Set[Tuple[int, int]], Dict[str, Any]]:
    name = cam.name
    width = manifest_seq.get("image_width")
    height = manifest_seq.get("image_height")
    valid_dimensions = all(
        isinstance(value, int) and not isinstance(value, bool) and value > 0
        for value in (width, height)
    )
    if not valid_dimensions:
        errors.append(f"{name}: sequence image dimensions 必须为正整数")
        width = height = 0
    match = SEQUENCE_NAME_RE.fullmatch(name)
    if not match:
        errors.append(f"{name}: 序列目录名不匹配规范")
    camera_id = manifest_seq.get("camera_id")
    if not isinstance(camera_id, str) or not re.fullmatch(r"C\d{2}", camera_id):
        errors.append(f"{name}: camera_id 必须为 C## 字符串")
    elif match and match.group("camera") != camera_id[1:]:
        errors.append(f"{name}: camera_id 与目录名不一致")
    stats: Dict[str, Any] = {"camera_id": name, "ok": True, "render_rgb": 0,
                             "render_mask_exr": 0, "img1_rgb": 0, "mask_png": 0,
                             "annotations_frames": 0, "det_txt": 0, "seg_txt": 0,
                             "gt_txt_lines": 0, "gt_mots_lines": 0, "gt_pose_records": 0,
                             "img1_missing": [], "img1_dup": [], "zero_byte": 0}
    frame_count = manifest_seq.get("frame_count")
    valid_frame_count = isinstance(frame_count, int) and not isinstance(frame_count, bool) and frame_count > 0
    if not valid_frame_count:
        errors.append(f"{name}: sequence frame_count 必须为正整数")
        frame_count = 0
    if manifest_seq.get("sequence_name") != name:
        errors.append(f"序列目录名 {name!r} 与 manifest sequence_name {manifest_seq.get('sequence_name')!r} 不一致")
    if manifest_seq.get("relative_path") != name:
        errors.append(f"{name}: relative_path 必须等于 sequence_name")
    if manifest_seq.get("modalities") != ["mot", "pose_tracking", "mots"]:
        errors.append(f"{name}: modalities 不匹配")
    seqinfo_path = cam / "seqinfo.ini"
    if not seqinfo_path.exists():
        errors.append(f"{name}: 缺少 seqinfo.ini")
        seq = {}
    else:
        seq = _seqinfo(seqinfo_path, errors)
    required_seq = {"name", "imdir", "framerate", "seqlength", "imwidth", "imheight", "imext"}
    missing = sorted(required_seq - set(seq))
    if missing:
        errors.append(f"{name}: seqinfo.ini 缺少字段 {missing}")
    for key, expected in (("name", name), ("imdir", "img1"), ("imext", ".jpg")):
        if key in seq and seq[key] != expected:
            errors.append(f"{name}: seqinfo.{key}={seq[key]!r}，期望 {expected!r}")
    for key, expected in (("seqlength", manifest_seq.get("frame_count")), ("imwidth", width), ("imheight", height)):
        if key in seq:
            try:
                if _strict_int(seq[key]) != expected:
                    errors.append(f"{name}: seqinfo.{key} 与 manifest 不一致")
            except (TypeError, ValueError):
                errors.append(f"{name}: seqinfo.{key} 不是整数")

    img_dir = cam / "img1"
    frames: Set[int] = set()
    if not img_dir.is_dir():
        errors.append(f"{name}: 缺少 img1/")
    else:
        for path in sorted(img_dir.iterdir()):
            if not path.is_file():
                continue
            if path.suffix != ".jpg" or len(path.stem) != 6 or not path.stem.isdigit():
                errors.append(f"{name}: img1 中存在非规范 JPG 文件: {path.name}")
                continue
            try:
                frame = _strict_int(path.stem)
            except ValueError:
                errors.append(f"{name}: JPG 文件名不是帧号: {path.name}")
                continue
            frames.add(frame)
            stats["img1_rgb"] += 1
            if path.stat().st_size == 0:
                errors.append(f"{name}: 零字节 JPG: {path.name}")
                stats["zero_byte"] += 1
            try:
                with Image.open(path) as image:
                    image.verify()
                    if image.size != (width, height):
                        errors.append(f"{name}: {path.name} 尺寸 {image.size} 不匹配")
            except (OSError, ValueError) as exc:
                errors.append(f"{name}: JPG 不可读 {path.name}: {exc}")
        expected_frames = set(range(1, frame_count + 1))
        if frames != expected_frames:
            errors.append(f"{name}: JPG 帧不连续或缺失（实际 {sorted(frames)}，期望 {sorted(expected_frames)}）")

    mot_path, mots_path, pose_path = cam / "gt" / "gt.txt", cam / "gt" / "gt_mots.txt", cam / "gt" / "gt_pose.json"
    mot_set = _validate_mot(mot_path, name, width, height, frame_count, errors)
    mots_set = _validate_mots(mots_path, name, width, height, frame_count, errors)
    pose_set = _validate_pose(pose_path, name, width, height, frame_count, errors)
    stats["gt_txt_lines"] = len(mot_set)
    stats["gt_mots_lines"] = len(mots_set)
    if pose_path.exists():
        try:
            records = json.loads(pose_path.read_text(encoding="utf-8"))
            stats["gt_pose_records"] = len(records) if isinstance(records, list) else 0
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            pass
    stats["annotations_frames"] = max((frame for frame, _ in mot_set), default=0)
    if not (mot_set == mots_set == pose_set):
        errors.append(f"{name}: MOT、MOTS、Pose 的 (frame_id, track_id) 集合不一致")
    return mot_set, stats


def _validate_mot(path: Path, name: str, width: int, height: int, frame_count: int, errors: List[str]) -> Set[Tuple[int, int]]:
    result: Set[Tuple[int, int]] = set()
    if not path.exists():
        errors.append(f"{name}: 缺少 gt/gt.txt")
        return result
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        errors.append(f"{name}: MOT 无法读取: {exc}")
        return result
    for line_no, line in enumerate(lines, 1):
        fields = line.split(",")
        if len(fields) != 9:
            errors.append(f"{name}: MOT 第 {line_no} 行字段数不是 9")
            continue
        if not all(_number(value.strip()) for value in fields[2:]) or any(not value.strip().lstrip("-").isdigit() for value in fields[:2] + [fields[7].strip()]):
            errors.append(f"{name}: MOT 第 {line_no} 行含非有限数字")
            continue
        frame, track = _strict_int(fields[0].strip()), _strict_int(fields[1].strip())
        x, y, w, h = map(float, fields[2:6])
        cls = _strict_int(fields[7].strip())
        expected_cls = 100 if track == 100 else 1
        if frame < 1 or frame > frame_count or track not in set(range(1, 11)) | {100} or cls != expected_cls or w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > width or y + h > height:
            errors.append(f"{name}: MOT 第 {line_no} 行 ID/class/bbox 非法")
        if (frame, track) in result:
            errors.append(f"{name}: MOT 第 {line_no} 行 frame/track 重复")
        result.add((frame, track))
    return result


def _validate_mots(path: Path, name: str, width_limit: int, height_limit: int, frame_count: int, errors: List[str]) -> Set[Tuple[int, int]]:
    result: Set[Tuple[int, int]] = set()
    if not path.exists():
        errors.append(f"{name}: 缺少 gt/gt_mots.txt")
        return result
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        errors.append(f"{name}: MOTS 无法读取: {exc}")
        return result
    for line_no, line in enumerate(lines, 1):
        fields = line.split()
        if len(fields) != 6:
            errors.append(f"{name}: MOTS 第 {line_no} 行字段数不是 6")
            continue
        try:
            frame, track, cls, height, width = (_strict_int(value) for value in fields[:5])
            if frame < 1 or frame > frame_count or height != height_limit or width != width_limit or height <= 0 or width <= 0:
                raise ValueError("帧号或尺寸非法")
            rle = json.loads(fields[5])
            area = _read_rle(rle, height, width)
            if frame < 1 or track not in set(range(1, 11)) | {100} or cls != (100 if track == 100 else 1) or area < 0:
                raise ValueError("ID/class/面积非法")
        except (ValueError, TypeError, json.JSONDecodeError, UnicodeError) as exc:
            errors.append(f"{name}: MOTS 第 {line_no} 行非法: {exc}")
            continue
        if (frame, track) in result:
            errors.append(f"{name}: MOTS 第 {line_no} 行 frame/track 重复")
        result.add((frame, track))
    return result


def _validate_pose(path: Path, name: str, width: int, height: int, frame_count: int, errors: List[str]) -> Set[Tuple[int, int]]:
    result: Set[Tuple[int, int]] = set()
    if not path.exists():
        errors.append(f"{name}: 缺少 gt/gt_pose.json")
        return result
    records = _json(path, errors)
    if not isinstance(records, list):
        errors.append(f"{name}: Pose 顶层不是数组")
        return result
    for index, record in enumerate(records, 1):
        if not isinstance(record, dict):
            errors.append(f"{name}: Pose 第 {index} 条不是对象")
            continue
        frame, track = record.get("frame_id"), record.get("track_id")
        valid_identity = isinstance(frame, int) and not isinstance(frame, bool) and 1 <= frame <= frame_count and isinstance(track, int) and not isinstance(track, bool) and track in set(range(1, 11)) | {100}
        if not valid_identity:
            errors.append(f"{name}: Pose 第 {index} 条 frame_id/track_id 非法")
        if valid_identity and (frame, track) in result:
            errors.append(f"{name}: Pose 第 {index} 条 frame/track 重复")
        if valid_identity:
            result.add((frame, track))
        is_ball = track == 100
        if record.get("class") != ("ball" if is_ball else "player"):
            errors.append(f"{name}: Pose 第 {index} 条 class 非法")
        points = record.get("keypoints")
        if is_ball:
            if points is not None:
                errors.append(f"{name}: Pose 第 {index} 条球的 keypoints 必须为 null")
            continue
        if not isinstance(points, list) or len(points) != 17:
            errors.append(f"{name}: Pose 第 {index} 条必须有 17 个关键点")
            continue
        for point in points:
            if not isinstance(point, list) or len(point) != 3 or not _finite(point[0]) or not _finite(point[1]) or isinstance(point[2], bool) or not isinstance(point[2], int) or point[2] not in (0, 1, 2):
                errors.append(f"{name}: Pose 第 {index} 条含非法关键点")
                break
            if not (0 <= point[0] <= width and 0 <= point[1] <= height):
                errors.append(f"{name}: Pose 第 {index} 条关键点坐标越界")
                break
    return result


def validate_public_episode(episode_dir: Path) -> ValidationResult:
    """验证公开 episode；返回 ok/errors/stats 和 CLI 兼容退出码。"""
    episode_dir = Path(episode_dir)
    errors: List[str] = []
    manifest_path = episode_dir / "episode_manifest.json"
    manifest = _json(manifest_path, errors) if manifest_path.exists() else None
    if not isinstance(manifest, dict):
        errors.append("缺少或无效 episode_manifest.json")
        return ValidationResult(False, errors, {"sequences": 0})
    required_root = {"schema_version", "episode_id", "trajectory_id", "sequences", "track_id_policy", "public_classes"}
    missing_root = sorted(required_root - set(manifest))
    if missing_root:
        errors.append(f"manifest 缺少根字段 {missing_root}")
    if manifest.get("schema_version") != 1 or isinstance(manifest.get("schema_version"), bool):
        errors.append("manifest schema_version 不匹配")
    if not isinstance(manifest.get("episode_id"), str) or not isinstance(manifest.get("trajectory_id"), str) or manifest.get("episode_id") != manifest.get("trajectory_id"):
        errors.append("manifest episode_id 与 trajectory_id 不一致")
    if manifest.get("public_classes") != ["player", "ball"]:
        errors.append("manifest public_classes 必须为 [player, ball]")
    expected_policy = {"players": "L0..L4=1..5,R0..R4=6..10", "ball": 100}
    if manifest.get("track_id_policy") != expected_policy:
        errors.append("manifest track_id_policy 不匹配")
    sequences = manifest.get("sequences")
    if not isinstance(sequences, list) or not sequences:
        errors.append("manifest sequences 为空或非法")
        sequences = []
    names = []
    all_sets = []
    for sequence in sequences:
        if not isinstance(sequence, dict) or not isinstance(sequence.get("sequence_name"), str):
            errors.append("manifest sequence 缺少 sequence_name")
            continue
        missing = sorted({
            "sequence_name", "camera_id", "relative_path", "frame_count",
            "image_width", "image_height", "modalities",
        } - set(sequence))
        if missing:
            errors.append(f"{sequence['sequence_name']}: manifest sequence 缺少字段 {missing}")
        names.append(sequence["sequence_name"])
        sequence_match = SEQUENCE_NAME_RE.fullmatch(sequence["sequence_name"])
        if sequence_match and sequence_match.group("episode") != manifest.get("episode_id"):
            errors.append(f"{sequence['sequence_name']}: 序列 episode_id 与 manifest 不一致")
        seq_count = sequence.get("frame_count")
        if isinstance(seq_count, bool) or not isinstance(seq_count, int) or seq_count <= 0:
            errors.append(f"{sequence['sequence_name']}: sequence frame_count 必须为正整数")
        cam = episode_dir / sequence.get("relative_path", "")
        if not cam.is_dir():
            errors.append(f"manifest sequence 目录不存在: {cam}")
            continue
        _, sequence_stats = _validate_sequence(cam, sequence, errors)
        all_sets.append(sequence_stats)
    if len(names) != len(set(names)):
        errors.append("manifest sequences 含重复 name")
    sequence_camera_ids = [sequence.get("camera_id") for sequence in sequences if isinstance(sequence, dict)]
    valid_camera_ids = [value for value in sequence_camera_ids if isinstance(value, str)]
    if len(valid_camera_ids) != len(set(valid_camera_ids)):
        errors.append("manifest sequences 含重复 camera_id")
    stats = {"sequences": len(sequences), "frames": sum(value.get("annotations_frames", 0) for value in all_sets),
             "sequence_names": names, "cameras": {value["camera_id"]: value for value in all_sets}}
    return ValidationResult(not errors, errors, stats)
