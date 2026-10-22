"""将已验收 episode 原子发布为按 seed 分组、共享静态信息的数据集。"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import shutil
import stat
import struct
import uuid
from .scene_lifecycle import retained_completion, admission_has_completed_scene
from pathlib import Path
from typing import Any, Mapping, NamedTuple

DATASET_SCHEMA = "futsalmot_seed_dataset"
GROUP_SCHEMA = "futsalmot_seed_group"
CLIP_SCHEMA = "futsalmot_seed_clip"
OBSERVATION_FIELDS = ("entity_id", "bbox_xyxy", "in_frame", "truncated", "visibility", "visible_pixel_count", "bbox_source", "segmentation")


class _SourceHash(NamedTuple):
    """仅当前发布调用内保存的内容摘要和文件身份，不写入数据 schema。"""
    sha256: str
    identity: tuple


def _file_identity(value: os.stat_result) -> tuple:
    """不使用会随硬链接计数改变的 ctime，只核对身份、大小和修改时间。"""
    if not stat.S_ISREG(value.st_mode) or not value.st_ino:
        raise ValueError("发布文件不是可验证身份的普通文件")
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def _require_identity(path: Path, expected: tuple) -> None:
    try:
        current = _file_identity(path.stat())
    except OSError as exc:
        raise ValueError("发布文件已缺失或发生漂移：{}".format(path)) from exc
    if current != expected:
        raise ValueError("发布文件身份、大小或 mtime 发生漂移：{}".format(path))


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON 顶层必须为对象：{}".format(path))
    return value


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _hash(path: Path) -> str:
    """完整读取文件，并拒绝读取期间发生的替换或内容漂移。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        before = _file_identity(os.fstat(stream.fileno()))
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        if _file_identity(os.fstat(stream.fileno())) != before:
            raise ValueError("哈希读取期间文件发生漂移：{}".format(path))
        _require_identity(path, before)
    return digest.hexdigest()


def _hash_source(path: Path) -> _SourceHash:
    """只信任本次真实读取前后均未变化的源文件摘要。"""
    identity = _file_identity(path.stat())
    digest = _hash(path)
    _require_identity(path, identity)
    return _SourceHash(digest, identity)


def _link_hashed_source(admitted: dict, source: Path, target: Path) -> str:
    """同 inode 且身份未漂移才复用源摘要；复制目标始终独立完整读取。"""
    record = admitted["_source_hashes"][source]
    _require_identity(source, record.identity)
    _link(source, target)
    _require_identity(source, record.identity)
    target_identity = _file_identity(target.stat())
    if os.path.samefile(source, target):
        if target_identity != record.identity:
            raise ValueError("硬链接目标身份发生漂移：{}".format(target))
        digest = record.sha256
    else:
        digest = _hash(target)
        if digest != record.sha256:
            raise ValueError("复制目标与本次源文件摘要不一致：{}".format(target))
    _require_identity(source, record.identity)
    _require_identity(target, target_identity)
    return digest


def _validate_source_hashes(records: Mapping[Path, _SourceHash]) -> None:
    """在整批哈希及移交结束时复查，防止较早完成的源在后续阶段漂移。"""
    for path, record in records.items():
        _require_identity(path, record.identity)


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp_" + uuid.uuid4().hex)
    with temporary.open("xb") as stream:
        stream.write(_json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def _shared_json(root: Path, relative: str, value: dict) -> str:
    destination = _safe_path(root, relative)
    if destination.exists():
        if _read(destination) != value:
            raise ValueError("共享资源存在内容冲突，拒绝覆盖：{}".format(relative))
    else:
        _atomic_json(destination, value)
    return relative


def _safe_path(root: Path, relative: str) -> Path:
    result = (root / relative).resolve()
    result.relative_to(root.resolve())
    return result


def _png_size(path: Path) -> tuple:
    with path.open("rb") as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError("不是有效 PNG header：{}".format(path))
    return struct.unpack(">II", header[16:24])


def _link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ValueError("拒绝覆盖已有帧：{}".format(destination))
    try:
        os.link(str(source), str(destination))
    except OSError:
        shutil.copy2(str(source), str(destination))


def _source_selection(episode: Path, entry: Mapping[str, Any], frame_count: int, trajectory_hash: str):
    """精简保存活跃裁剪证据；完整采集配方由 shared 单独存储。"""
    path = episode / "provenance/active_window.json"
    if not path.is_file():
        return None
    source = _read(path)
    if source.get("schema") != "futsalmot_active_episode" or source.get("version") != 1:
        raise ValueError("active_window schema/version 无效")
    if source.get("root_seed") != entry["root_seed"] or source.get("frame_count") != frame_count or source.get("frames_sha256") != trajectory_hash:
        raise ValueError("active_window 与当前 seed、帧数或轨迹指纹不一致")
    for key in ("source_frames_sha256", "source_meta_sha256", "policy_sha256"):
        if not isinstance(source.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", source[key]):
            raise ValueError("active_window 缺少有效 {}".format(key))
    window = source.get("source_window")
    if not isinstance(window, dict):
        raise ValueError("active_window 缺少 source_window")
    start, end = window.get("start_index"), window.get("end_index_exclusive")
    capture = source.get("source_capture_steps")
    if any(type(value) is not int for value in (start, end, capture)) or not (0 <= start < end <= capture) or end - start != frame_count or window.get("frame_count") != frame_count:
        raise ValueError("active_window 源窗口与完整采集帧数不一致")
    if entry.get("source_frame_range") is not None and entry["source_frame_range"] != [start, end]:
        raise ValueError("active_window 源窗口与发布请求不一致")
    steps = source.get("source_step_range")
    times = source.get("source_time_range_s")
    dt = source.get("dt_s")
    if not isinstance(steps, list) or len(steps) != 2 or any(type(value) is not int for value in steps) or steps[1] - steps[0] != frame_count:
        raise ValueError("active_window 原始 step 范围无效")
    if not isinstance(dt, (int, float)) or isinstance(dt, bool) or not math.isfinite(dt) or dt <= 0:
        raise ValueError("active_window dt 无效")
    if not isinstance(times, list) or len(times) != 2 or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in times) or not math.isclose(times[1] - times[0], frame_count * dt, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("active_window 原始时间范围无效")
    keys = ("source_episode_id", "source_frames_sha256", "source_meta_sha256", "source_capture_steps", "source_step_range", "source_time_range_s", "policy_sha256", "dt_s")
    normalized = {key: copy.deepcopy(source[key]) for key in keys if key in source}
    normalized.update(schema=source["schema"], version=source["version"])
    window_keys = ("start_index", "end_index_exclusive", "frame_count", "sample_duration_s", "trajectory_span_s", "team_active_coverage", "moving_outfield_ids", "local_motion", "players")
    normalized["source_window"] = {key: copy.deepcopy(window[key]) for key in window_keys if key in window}
    return normalized


def _full_audit_anchor(root: Path, reference: str) -> dict:
    """只读取小型基准证据；完整审计必须来自真实已发布clip。"""
    if not isinstance(reference, str) or not reference.startswith('groups/') or not reference.endswith('/clip.json'):
        raise ValueError('缺少合法完整审计基准引用')
    path = _safe_path(root, reference)
    clip = _read(path)
    admission = clip.get('admission') or {}
    required = ('pilot_ready', 'full_audit_passed', 'appearance_restored', 'environment_restored')
    if clip.get('schema') != CLIP_SCHEMA or any(admission.get(key) is not True for key in required) or admission.get('render_status') != 'success':
        raise ValueError('基准组未通过完整审计和恢复')
    digest = _hash(path)
    if _read(path.parent / 'checksums.json').get('clip.json') != digest:
        raise ValueError('完整审计基准描述已漂移')
    audit_sha = (admission.get('source_sha256') or {}).get('audit')
    if not isinstance(audit_sha, str) or not re.fullmatch('[0-9a-f]{64}', audit_sha):
        raise ValueError('完整审计基准缺少原始审计摘要')
    return {'clip_ref': reference, 'clip_sha256': digest, 'task_id': clip['task_id'], 'audit_sha256': audit_sha}


def _admit(entry: Mapping[str, Any], *, dataset_root=None, audit_policy=None) -> dict:
    """默认完整审计；显式授权可在完整基准之后仅做结构准入或像素抽样。"""
    episode = Path(entry["episode"]).resolve()
    seed = entry.get("root_seed")
    if type(seed) is not int or seed < 0:
        raise ValueError("root_seed 必须是非负整数")
    split = {"validation": "val", "synthetic_test": "test"}.get(entry.get("split"), entry.get("split"))
    if split not in {"train", "val", "test"}:
        raise ValueError("仅发布 train/val/test，gate 不能作为正式数据")
    basic = isinstance(audit_policy, Mapping) and audit_policy.get('mode') == 'basic_after_full' and dataset_root is not None
    files = {"ready": episode / "pilot_ready.json", "render": episode / "render_summary.json", "appearance_restore": episode / "provenance/appearance_application.json", "environment_restore": episode / "provenance/environment_application.json"}
    if not basic:
        files['audit'] = episode / 'audit/soak_audit_report.json'
    retained = retained_completion(episode)
    if retained is not None:
        files["scene_completion"] = episode / "provenance/scene_completion.json"
    reports = {key: _read(path) for key, path in files.items()}
    audit = reports.get("audit", {})
    validation = audit.get("validation") or {}
    if not basic and (audit.get("passed") is not True or audit.get("exit_code") != 0 or audit.get("errors")):
        raise ValueError("full audit 未通过")
    level = validation.get('level')
    anchor = None
    if basic or (level == 'quick' and isinstance(audit_policy, Mapping) and audit_policy.get('mode') == 'quick_after_full' and dataset_root is not None):
        anchor = _full_audit_anchor(Path(dataset_root), audit_policy.get('full_audit_anchor_ref'))
    elif level != 'full':
        raise ValueError("缺少 full annotation audit 证据")
    if not basic and (validation.get('passed') is not True or validation.get('exit_code') != 0):
        raise ValueError('annotation audit 未通过')
    if reports["render"].get("status") != "success":
        raise ValueError("MRQ 未成功结束")
    for key in ("appearance_restore", "environment_restore"):
        if reports[key].get("state") != ("applied" if retained else "restored") or reports[key].get("errors"):
            raise ValueError("外观或环境恢复未通过")
    ready = reports["ready"]
    if ready.get("status") != ('postprocess_complete' if basic else 'audit_passed') or ready.get("appearance_restored") is not (not bool(retained)) or ready.get("environment_restored") is not (not bool(retained)) or (retained and ready.get("scene_retained") is not True):
        raise ValueError("pilot_ready 未通过")
    for key in ("root_seed", "task_id"):
        if ready.get(key) != entry.get(key):
            raise ValueError("pilot_ready 与 entry 的 {} 不一致".format(key))
    ready_split = {"validation": "val", "synthetic_test": "test"}.get(ready.get("split"), ready.get("split"))
    if ready_split != split:
        raise ValueError("pilot_ready 的 split 与发布请求不同，不能把 gate 改作训练")
    if ready.get("source_frame_range") != entry.get("source_frame_range"):
        raise ValueError("pilot_ready 的源活动窗与发布请求不同")
    meta = _read(episode / "meta.json")
    if (meta.get("randomness") or {}).get("root_seed") != seed:
        raise ValueError("meta.root_seed 与分组 seed 不一致")
    frame_count = (meta.get("timing") or {}).get("num_steps")
    if type(frame_count) is not int or frame_count <= 0 or frame_count != entry.get("frames_per_camera"):
        raise ValueError("meta 与 entry 帧数不一致")
    cameras = sorted(path for path in episode.glob("CineCam_*") if path.is_dir())
    if len(cameras) != entry.get("camera_count") or not cameras:
        raise ValueError("相机数量不一致")
    resolution = entry.get("resolution")
    if resolution is not None:
        if not isinstance(resolution, list) or len(resolution) != 2 or any(type(value) is not int or value <= 0 for value in resolution):
            raise ValueError("声明分辨率必须是两个正整数")
        for camera in cameras:
            calibration = _read(camera / "camera.json")
            if [calibration.get("image_width"), calibration.get("image_height")] != resolution:
                raise ValueError("相机实际分辨率与发布声明不一致：{}".format(camera.name))
    count = frame_count * len(cameras)
    if ready.get("rgb_frames") != count or ready.get("mask_frames") != count:
        raise ValueError("pilot_ready 的 RGB/mask 数量不完整")
    for key in ("appearance_restore", "environment_restore"):
        if reports[key].get("mrq_pending") is True:
            raise ValueError("恢复报告仍有未结束 MRQ")
    appearance = _read(episode / "provenance/appearance.json")
    environment = _read(episode / "provenance/environment.json")
    profile = _read(episode / "provenance/ue-profile.json")
    catalog = profile.get("appearance")
    if not isinstance(catalog, dict) or _canonical_hash(catalog) != appearance.get("catalog_fingerprint"):
        raise ValueError("appearance catalog 指纹不匹配")
    # 两类恢复报告使用各自契约字段；环境字段不能回退成外观的通用字段。
    for manifest, restore, field in (
        (appearance, reports["appearance_restore"], "manifest_fingerprint"),
        (environment, reports["environment_restore"], "environment_manifest_fingerprint"),
    ):
        if restore.get(field) != manifest.get("fingerprint"):
            raise ValueError("恢复报告与 manifest 指纹不匹配")
    trajectory_hash = _hash(episode / "frames.jsonl")
    for manifest in (appearance, environment):
        if manifest.get("trajectory_sha256") != trajectory_hash:
            raise ValueError("manifest 对应的轨迹已漂移")
    source_selection = _source_selection(episode, entry, frame_count, trajectory_hash)
    summary = {"pilot_ready": True, "full_audit_passed": True, "render_status": "success", "appearance_restored": True, "environment_restored": True, "source_sha256": {key: _hash(path) for key, path in files.items()}}
    if retained:
        summary.update(scene_lifecycle="retain", scene_retained=True, appearance_restored=False, environment_restored=False, scene_completion_sha256=_hash(files["scene_completion"]))
    if anchor is not None:
        summary.update(full_audit_passed=False, validation_level='basic' if basic else 'quick', basic_checks_passed=True,
            full_audit_anchor=anchor)
        if basic:
            summary['annotation_audit_performed'] = False
    # 同 task 幂等重试也核对源图像和标注，不能只比较轨迹与成功报告。
    source_files = [episode / "meta.json", episode / "frames.jsonl"]
    if source_selection is not None:
        source_files.append(episode / "provenance/active_window.json")
    for camera in cameras:
        source_files.extend([camera / "camera.json", camera / "annotations.jsonl"])
        source_files.extend(sorted((camera / "img1").glob("*.png")))
        source_files.extend(sorted((camera / "mask").glob("*.png")))
    source_hashes = {path: _hash_source(path) for path in source_files}
    _validate_source_hashes(source_hashes)
    if source_hashes[episode / "frames.jsonl"].sha256 != trajectory_hash:
        raise ValueError("准入过程中轨迹内容发生漂移")
    summary["source_payload_sha256"] = _canonical_hash({path.relative_to(episode).as_posix(): record.sha256 for path, record in source_hashes.items()})
    return {"episode": episode, "seed": seed, "split": split, "meta": meta, "frames": frame_count, "cameras": cameras, "appearance": appearance, "environment": environment, "catalog": catalog, "trajectory_sha256": trajectory_hash, "source_selection": source_selection, "admission": summary, "_source_hashes": source_hashes}


def _group_payload(admitted: dict, catalog_ref: str) -> dict:
    meta, appearance, environment = admitted["meta"], admitted["appearance"], admitted["environment"]
    assignments = {entity["entity_id"]: entity for entity in appearance["entities"]}
    entities = []
    for source in meta.get("entities") or []:
        entity = {"entity_id": source["id"]}
        entity.update({key: value for key, value in source.items() if key != "id"})
        assignment = assignments.get(source["id"])
        if assignment:
            entity.update({key: assignment[key] for key in ("player_id", "skin_id", "kit_set_id", "kit_class")})
        entities.append(entity)
    timing = {key: value for key, value in meta["timing"].items() if key != "num_steps"}
    source = {key: value for key, value in meta.get("source", {}).items() if key != "seed"}
    randomness = {key: value for key, value in meta.get("randomness", {}).items() if key != "root_seed"}
    weather = {key: environment[key] for key in ("policy", "catalog_id", "catalog_fingerprint", "environment_seed", "namespace_seed", "preset", "parameters") if key in environment}
    return {"schema": GROUP_SCHEMA, "version": 1, "root_seed": admitted["seed"], "split": admitted["split"], "source": source, "randomness": randomness, "timing": timing, "field_ref": "dataset.json#/field", "entities": entities, "appearance": {"policy": appearance["policy"], "catalog_ref": catalog_ref}, "weather": weather}


def _stage_clip(admitted: dict, entry: Mapping[str, Any], destination: Path, root: Path) -> dict:
    episode, count = admitted["episode"], admitted["frames"]
    source_range = entry.get("source_frame_range")
    if source_range is not None and (not isinstance(source_range, list) or len(source_range) != 2 or any(type(x) is not int for x in source_range) or source_range[0] < 0 or source_range[1] - source_range[0] != count):
        raise ValueError("source_frame_range 必须是与 clip 帧数一致的 [start,end)")
    checksums = {}
    trajectory = destination / "trajectory.jsonl"
    checksums["trajectory.jsonl"] = _link_hashed_source(admitted, episode / "frames.jsonl", trajectory)
    if len(trajectory.read_text(encoding="utf-8").splitlines()) != count:
        raise ValueError("轨迹帧数与 meta 不一致")
    camera_names = []
    entity_ids = {entity["id"] for entity in admitted["meta"]["entities"]}
    for camera in admitted["cameras"]:
        if not re.fullmatch(r"CineCam_[A-Za-z0-9_]+", camera.name):
            raise ValueError("不合法 camera id")
        calibration = _read(camera / "camera.json")
        calibration_hash = _canonical_hash(calibration)
        calibration_ref = _shared_json(root, "shared/calibrations/{}_{}.json".format(camera.name, calibration_hash), calibration)
        width, height = calibration["image_width"], calibration["image_height"]
        view = destination / "cameras" / camera.name
        _atomic_json(view / "view.json", {"camera_id": camera.name, "calibration_ref": calibration_ref})
        rows = []
        for line in (camera / "annotations.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            source = json.loads(line)
            objects = []
            seen = set()
            for obj in source["objects"]:
                entity_id = obj.get("entity_id")
                if entity_id not in entity_ids or entity_id in seen:
                    raise ValueError("二维标注的 entity_id 缺失、未知或重复")
                seen.add(entity_id)
                objects.append({key: obj[key] for key in OBSERVATION_FIELDS if key in obj})
            rows.append({"frame_index": source["frame_index"], "objects": objects})
        if {row["frame_index"] for row in rows} != set(range(1, count + 1)) or len(rows) != count:
            raise ValueError("二维标注帧不完整或重复")
        annotation_path = view / "annotations.jsonl"
        annotation_path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n" for row in rows), encoding="utf-8")
        checksums[annotation_path.relative_to(destination).as_posix()] = _hash(annotation_path)
        checksums[(view / "view.json").relative_to(destination).as_posix()] = _hash(view / "view.json")
        for old, new in (("img1", "rgb"), ("mask", "mask")):
            frames = sorted((camera / old).glob("*.png"))
            if {path.stem for path in frames} != {"{:06d}".format(i) for i in range(1, count + 1)} or len(frames) != count:
                raise ValueError("RGB/mask 帧集合不完整")
            for source in frames:
                if _png_size(source) != (width, height):
                    raise ValueError("RGB/mask 分辨率与 camera 不一致")
                target = view / new / source.name
                checksums[target.relative_to(destination).as_posix()] = _link_hashed_source(admitted, source, target)
        camera_names.append(camera.name)
    _validate_source_hashes(admitted["_source_hashes"])
    clip = {"schema": CLIP_SCHEMA, "version": 1, "task_id": entry["task_id"], "frame_count": count, "source_frame_range": source_range, "source_selection": admitted["source_selection"], "cameras": camera_names, "trajectory_sha256": admitted["trajectory_sha256"], "appearance_manifest_fingerprint": admitted["appearance"]["fingerprint"], "environment_manifest_fingerprint": admitted["environment"]["fingerprint"], "admission": admitted["admission"]}
    _atomic_json(destination / "clip.json", clip)
    checksums["clip.json"] = _hash(destination / "clip.json")
    _atomic_json(destination / "checksums.json", checksums)
    return clip


def _index_groups(root: Path, index: dict) -> None:
    groups = []
    splits = {}
    for group_path in sorted((root / "groups").glob("seed_*/group.json")):
        group = _read(group_path)
        frames = clips = 0
        for reference in group["clips"]:
            clip = _read(_safe_path(group_path.parent, reference["path"]))
            clips += 1
            frames += clip["frame_count"] * len(clip["cameras"])
        groups.append({"root_seed": group["root_seed"], "split": group["split"], "path": group_path.relative_to(root).as_posix(), "clip_count": clips, "camera_frames": frames})
        stats = splits.setdefault(group["split"], {"groups": 0, "clips": 0, "camera_frames": 0})
        stats["groups"] += 1; stats["clips"] += clips; stats["camera_frames"] += frames
    index["groups"], index["splits"] = groups, splits
    index["counts"] = {"seed_groups": len(groups), "clips": sum(x["clip_count"] for x in groups), "rgb_frames": sum(x["camera_frames"] for x in groups), "mask_frames": sum(x["camera_frames"] for x in groups)}
    _atomic_json(root / "dataset.json", index)


def dataset_root_for_clip(clip_directory: Path) -> Path:
    """按索引定位数据集根，兼容按seed平铺及旧的多片段目录。"""
    directory = Path(clip_directory).resolve()
    for candidate in directory.parents:
        if (candidate / 'dataset.json').is_file() and (candidate / 'groups').is_dir():
            directory.relative_to(candidate / 'groups')
            return candidate
    raise ValueError('clip不属于可定位的规范数据集')


def verify_clip(clip_directory: Path, *, verification_level='full') -> dict:
    """默认重读全部SHA；metadata仅重读文本SHA及所有图像头，避免重复扫描图像。"""
    if verification_level not in ('full', 'metadata', 'structure'):
        raise ValueError('未知的发布验证级别')
    clip_directory = Path(clip_directory).resolve()
    checksums = _read(clip_directory / "checksums.json")
    for relative, expected in checksums.items():
        path = _safe_path(clip_directory, relative)
        media = '/rgb/' in relative or '/mask/' in relative
        if not path.is_file() or (not (verification_level in ('metadata', 'structure') and media) and _hash(path) != expected):
            raise ValueError("发布 clip 内容缺失或已变化：{}".format(relative))
    clip = _read(clip_directory / "clip.json")
    admission = clip.get("admission") or {}
    if clip.get("schema") != CLIP_SCHEMA or (admission.get("pilot_ready") is not True or not admission_has_completed_scene(admission)) or admission.get("render_status") != "success":
        raise ValueError("canonical clip 准入摘要无效")
    if admission.get('full_audit_passed') is not True:
        if admission.get('full_audit_passed') is not False or admission.get('validation_level') not in ('quick', 'basic') or admission.get('basic_checks_passed') is not True:
            raise ValueError('canonical clip 缺少真实完整/抽样准入证据')
        anchor = admission.get('full_audit_anchor')
        if not isinstance(anchor, dict) or _full_audit_anchor(dataset_root_for_clip(clip_directory), anchor.get('clip_ref')) != anchor:
            raise ValueError('canonical clip 完整审计基准不匹配')
    count, cameras = clip.get("frame_count"), clip.get("cameras")
    if type(count) is not int or count <= 0 or not isinstance(cameras, list) or not cameras or len(set(cameras)) != len(cameras):
        raise ValueError("canonical clip 帧数或相机列表无效")
    required = {"clip.json", "trajectory.jsonl"}
    for camera in cameras:
        if not isinstance(camera, str) or not re.fullmatch(r"CineCam_[A-Za-z0-9_]+", camera):
            raise ValueError("canonical camera id 无效")
        prefix = "cameras/{}/".format(camera)
        required.update(prefix + name for name in ("view.json", "annotations.jsonl"))
        required.update(prefix + "{}/{:06d}.png".format(kind, frame) for kind in ("rgb", "mask") for frame in range(1, count + 1))
    if set(checksums) != required or checksums["trajectory.jsonl"] != clip.get("trajectory_sha256"):
        raise ValueError("canonical checksums 未完整覆盖 clip 数据")
    if verification_level == 'metadata':
        for camera in cameras:
            view = clip_directory / 'cameras' / camera
            calibration = _read(_safe_path(dataset_root_for_clip(clip_directory), _read(view / 'view.json')['calibration_ref']))
            expected_size = (calibration['image_width'], calibration['image_height'])
            for kind in ('rgb', 'mask'):
                for frame in range(1, count+1):
                    path = view / kind / '{:06d}.png'.format(frame)
                    if path.stat().st_size <= 24 or _png_size(path) != expected_size:
                        raise ValueError('发布图像头或分辨率无效：{}'.format(path))
    return clip


def promote_episode(entry: dict, dataset_root: Path, *, audit_policy=None, layout=None) -> dict:
    """验收后发布一个 clip；同 seed 只能属于一个 split，禁止覆写不同内容。"""
    admitted = _admit(entry, dataset_root=dataset_root, audit_policy=audit_policy)
    root = Path(dataset_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".seed_dataset.lock"
    try:
        handle = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError("发布锁已存在，请先确认另一发布进程或中断状态") from exc
    os.close(handle)
    try:
        index_path = root / "dataset.json"
        if index_path.exists():
            index = _read(index_path)
            if index.get("schema") != DATASET_SCHEMA or index.get("version") != 1:
                raise ValueError("目标 dataset.json schema 不兼容，拒绝覆盖")
            if index.get("field") != admitted["meta"].get("field"):
                raise ValueError("同一数据集场地坐标定义不一致")
        else:
            index = {"schema": DATASET_SCHEMA, "version": 1, "layout": layout or 'seed_clips_v1', "field": admitted["meta"].get("field"), "annotation_semantics": {"bbox_xyxy": "pixel-tight 可见像素半开框，完全不可见时为 null", "world_trajectory": "每个 clip 的 trajectory.jsonl，仅保存一份", "entity_attributes": "group.json 中按 entity_id 查询", "camera_calibration": "shared/calibrations 中按 view.json 引用查询"}}
        flat = index.get('layout') == 'seed_flat_v1'
        if layout is not None and layout != index.get('layout', 'seed_clips_v1'):
            raise ValueError('发布layout与已有数据集不同')
        appearance = admitted["appearance"]
        catalog_ref = _shared_json(root, "shared/catalogs/{}.json".format(appearance["catalog_fingerprint"]), admitted["catalog"])
        payload = _group_payload(admitted, catalog_ref)
        group_path = root / "groups" / "seed_{:08d}".format(admitted["seed"])
        existing = None
        if (group_path / "group.json").exists():
            existing = _read(group_path / "group.json")
            if existing.get("split") != admitted["split"]:
                raise ValueError("同一 root_seed 不得跨 split")
            stable = {key: value for key, value in existing.items() if key != "clips"}
            if stable != payload:
                raise ValueError("同一 seed 的静态信息、外观或天气不一致，拒绝覆盖")
            for reference in existing["clips"]:
                old_path = _safe_path(group_path, reference["path"]).parent
                old = verify_clip(old_path)
                if old["task_id"] == entry["task_id"]:
                    if old["trajectory_sha256"] != admitted["trajectory_sha256"] or old["admission"] != admitted["admission"]:
                        raise ValueError("同 task 已发布不同内容，拒绝覆盖")
                    _index_groups(root, index)
                    return {"root_seed": admitted["seed"], "group": str(group_path), "clip": str(old_path), "dataset": str(index_path), "idempotent": True}
        group_path.parent.mkdir(parents=True, exist_ok=True)
        if flat and existing:
            raise ValueError('平铺seed只允许一个连续片段；新增片段需使用独立seed组或显式多片段版本')
        number = len(existing["clips"]) if existing else 0
        clip_id = "clip_{:03d}".format(number)
        if existing:
            staged = group_path / (".staging_" + uuid.uuid4().hex)
        else:
            staged_group = group_path.with_name(".staging_seed_" + uuid.uuid4().hex)
            staged = staged_group if flat else staged_group / "clips" / clip_id
        staged.mkdir(parents=True, exist_ok=False)
        _stage_clip(admitted, entry, staged, root)
        reference = {"clip_id": clip_id, "path": 'clip.json' if flat else "clips/{}/clip.json".format(clip_id)}
        if existing:
            target = group_path / "clips" / clip_id
            if target.exists():
                raise ValueError("clip 路径已存在，保留发布证据并停止")
            staged.rename(target)
            existing["clips"].append(reference)
            _atomic_json(group_path / "group.json", existing)
        else:
            payload["clips"] = [reference]
            _atomic_json(staged_group / "group.json", payload)
            if group_path.exists():
                raise ValueError("seed group 路径已存在，拒绝覆盖")
            staged_group.rename(group_path)
        _index_groups(root, index)
        return {"root_seed": admitted["seed"], "group": str(group_path), "clip": str(group_path if flat else group_path / "clips" / clip_id), "dataset": str(index_path), "idempotent": False}
    finally:
        lock.unlink()
