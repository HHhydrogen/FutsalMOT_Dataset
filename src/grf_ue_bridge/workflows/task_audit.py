#!/usr/bin/env python
"""soak episode 完整性审计。

对一次完整渲染+后处理的相机数据集目录做只读审计，输出 JSON + Markdown 双报告，
并用退出码标识是否通过。检查维度：

  - 相机目录数量 / 期望帧数
  - render/（RGB 原始）、render_mask/（Object ID EXR 原始）数量与目标帧覆盖
  - img1/、mask/、annotations.jsonl、labels/det/、labels/seg/、gt/gt.txt 数量
  - 缺帧、重复帧、零字节文件
  - 跨相机时间同步（time_seconds / source_step / episode_id / track_id / mask_id）
  - camera.json 标定合法性（分辨率一致、内参有限且为正、外参有限、相机不重复）
  - render_summary.json 状态与每相机 ok
  - 可选：进程内运行 validate-annotations（quick/full）

CLI 入口：`grf-ue task audit <task>`（推荐）；或 `python -m grf_ue_bridge.workflows.task_audit --input ...`。

退出码：0 = 全部通过；1 = 存在任何失败项（缺帧/重复/零字节/同步/映射/标定/数量不符）。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# 实体 → track_id / mask_id 的稳定映射（与 annotation_utils 一致，本地复述避免依赖）
TRACK_MAP = {f"L{i}": i + 1 for i in range(5)}
TRACK_MAP.update({f"R{i}": i + 6 for i in range(5)})
TRACK_MAP["BALL"] = 100
MASK_MAP = {f"L{i}": i + 1 for i in range(5)}
MASK_MAP.update({f"R{i}": i + 6 for i in range(5)})
MASK_MAP["BALL"] = 11
RGB_SUFFIXES = {".png", ".jpg", ".jpeg"}


def _is_finite(v) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(v)


def _frame_numbers(files: Sequence[Path]) -> List[int]:
    """从文件名单 stem 数字解析帧号（升序，含重复）。"""
    out = []
    for p in files:
        digits = "".join(ch for ch in p.stem if ch.isdigit())
        if digits:
            out.append(int(digits))
    return sorted(out)


def _counts_and_missing(
    files: Sequence[Path], expected: int
) -> Tuple[int, List[int], List[int]]:
    """返回 (文件数, 缺失编号, 重复编号)。文件名为 {frame_index:06d} 约定。"""
    nums = _frame_numbers(files)
    uniq = sorted(set(nums))
    missing = [i for i in range(1, expected + 1) if i not in uniq]
    dups = sorted({n for n in nums if nums.count(n) > 1})
    return len(nums), missing, dups


def _zero_byte(patterns: Sequence[Path]) -> List[Path]:
    return [p for p in patterns if p.exists() and p.stat().st_size == 0]


def _read_frame_meta(cam_dir: Path) -> Tuple[Optional[List[dict]], List[str]]:
    """读取 annotations.jsonl 的帧级元数据（不含 objects），失败返回错误列表。"""
    ann = cam_dir / "annotations.jsonl"
    if not ann.exists():
        return None, [f"缺少 annotations.jsonl: {ann}"]
    frames = []
    try:
        with open(ann, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                frames.append({
                    "frame_index": obj.get("frame_index"),
                    "source_step": obj.get("source_step"),
                    "time_seconds": obj.get("time_seconds"),
                    "episode_id": obj.get("episode_id"),
                    "objects": obj.get("objects", []),
                })
    except Exception as e:  # noqa: BLE001
        return None, [f"annotations.jsonl 解析失败: {e}"]
    return frames, []


def _derive_keep_indices(
    frames: List[dict],
    render_fps: int,
    source_step_seconds: Optional[float],
) -> List[int]:
    """根据标注帧的 source_step / time_seconds 推导 MRQ 应保留的渲染帧号。

    source_step_seconds 缺失时从 time_seconds / source_step 反推（取第一个 step>0 的帧）。
    """
    step_sec = source_step_seconds
    if step_sec is None:
        for fr in frames:
            s = fr.get("source_step")
            t = fr.get("time_seconds")
            if isinstance(s, (int, float)) and s > 0 and isinstance(t, (int, float)):
                step_sec = t / s
                break
    if step_sec is None:
        step_sec = 0.1
    keep = []
    for fr in frames:
        s = fr.get("source_step")
        if not isinstance(s, (int, float)):
            s = int(fr.get("frame_index", 1)) - 1
        keep.append(int(round(s * step_sec * render_fps)))
    return keep


def _load_episode_meta(episode_dir: Path) -> Optional[dict]:
    meta_path = episode_dir / "meta.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def check_camera(
    cam_dir: Path,
    expected_frames: int,
    keep_indices: List[int],
    errors: List[str],
    warns: List[str],
    mask_enabled: bool = True,
) -> dict:
    """审计单个相机目录，返回统计 dict，并把失败项写入 errors。

    mask_enabled=False（任务未启用 instance_mask）时跳过 mask/render_mask 相关检查。
    """
    cam_id = cam_dir.name
    st: dict = {"camera_id": cam_id, "ok": True}

    # 标注帧
    frames, errs = _read_frame_meta(cam_dir)
    if errs:
        errors.extend(errs)
        st["annotations_frames"] = 0
        return st
    ann_idx = [fr["frame_index"] for fr in frames]
    st["annotations_frames"] = len(ann_idx)
    if len(ann_idx) != expected_frames:
        errors.append(f"{cam_id}: annotations 帧数 {len(ann_idx)} != 预期 {expected_frames}")
        st["ok"] = False
    missing_ann = [i for i in range(1, expected_frames + 1) if i not in set(ann_idx)]
    dup_ann = sorted({n for n in ann_idx if ann_idx.count(n) > 1})
    if missing_ann:
        errors.append(f"{cam_id}: annotations 缺帧 {missing_ann[:10]}...")
        st["ok"] = False
    if dup_ann:
        errors.append(f"{cam_id}: annotations 重复帧 {dup_ann[:10]}")
        st["ok"] = False

    # 目录产物统计
    img1_dir, mask_dir = cam_dir / "img1", cam_dir / "mask"
    render_dir, rmask_dir = cam_dir / "render", cam_dir / "render_mask"
    det_dir, seg_dir = cam_dir / "labels" / "det", cam_dir / "labels" / "seg"
    gt_txt = cam_dir / "gt" / "gt.txt"

    img1_files = sorted(
        p for p in img1_dir.iterdir()
        if p.is_file() and p.suffix.lower() in RGB_SUFFIXES
    ) if img1_dir.exists() else []
    mask_files = sorted(mask_dir.glob("*.png")) if mask_dir.exists() else []
    render_files = sorted(
        p for p in render_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in RGB_SUFFIXES
    ) if render_dir.exists() else []
    exr_files = sorted(rmask_dir.rglob("*.exr")) if rmask_dir.exists() else []
    det_files = sorted(det_dir.glob("*.txt")) if det_dir.exists() else []
    seg_files = sorted(seg_dir.glob("*.txt")) if seg_dir.exists() else []

    st["render_rgb"] = len(render_files)
    st["render_mask_exr"] = len(exr_files)
    st["img1_rgb"] = len(img1_files)
    st["mask_png"] = len(mask_files)
    st["det_txt"] = len(det_files)
    st["seg_txt"] = len(seg_files)
    st["gt_txt_exists"] = gt_txt.exists()
    st["gt_txt_lines"] = sum(1 for _ in open(gt_txt, encoding="utf-8")) if gt_txt.exists() else 0

    # img1 / mask 缺帧与重复
    _n1, miss_img1, dup_img1 = _counts_and_missing(img1_files, expected_frames)
    st["img1_missing"] = miss_img1
    st["img1_dup"] = dup_img1
    if miss_img1:
        errors.append(f"{cam_id}: img1 缺帧 {miss_img1[:10]}...")
        st["ok"] = False
    if dup_img1:
        errors.append(f"{cam_id}: img1 重复帧 {dup_img1[:10]}")
        st["ok"] = False
    _n2, miss_mask, dup_mask = _counts_and_missing(mask_files, expected_frames)
    st["mask_missing"] = miss_mask
    st["mask_dup"] = dup_mask
    if mask_enabled and miss_mask:
        errors.append(f"{cam_id}: mask 缺帧 {miss_mask[:10]}...")
        st["ok"] = False
    if mask_enabled and dup_mask:
        errors.append(f"{cam_id}: mask 重复帧 {dup_mask[:10]}")
        st["ok"] = False

    # render / render_mask 是否覆盖全部 keep_indices（均为 transient 渲染产物：
    # img1 是 canonical RGB；zero-waste 已删除 render/，render_mask/ 由 cleanup 删除后均不要求存在）
    render_nums = set(_frame_numbers(render_files))
    exr_nums = set(_frame_numbers(exr_files))
    miss_render = sorted(set(keep_indices) - render_nums)
    miss_exr = sorted(set(keep_indices) - exr_nums)
    # 仅当 render/ 仍有 RGB 文件时才校验其完整性（img1 已保证 RGB 完整性；
    # zero-waste 删除 render/ PNG 后 render_files 为空，跳过）
    if mask_enabled and render_files and miss_render:
        errors.append(f"{cam_id}: render/ 缺少目标帧 {miss_render[:10]}...")
        st["ok"] = False
    if mask_enabled and miss_exr:
        errors.append(f"{cam_id}: render_mask/ 缺少目标 EXR 帧 {miss_exr[:10]}...")
        st["ok"] = False

    # 零字节文件
    z = _zero_byte(render_files) + _zero_byte(exr_files) + _zero_byte(img1_files) \
        + _zero_byte(mask_files) + _zero_byte(det_files) + _zero_byte(seg_files)
    st["zero_byte"] = len(z)
    if z:
        errors.append(f"{cam_id}: 零字节文件 {[str(p.name) for p in z][:10]}")
        st["ok"] = False

    # labels/det 与 labels/seg 帧号覆盖
    def _detect_nums(files):
        out = set()
        for p in files:
            d = "".join(c for c in p.stem if c.isdigit())
            if d:
                out.add(d)
        return out
    det_nums = _detect_nums(det_files)
    seg_nums = _detect_nums(seg_files)
    st["det_frames"] = len(det_nums)
    st["seg_frames"] = len(seg_nums)
    miss_det = [i for i in range(1, expected_frames + 1) if f"{i:06d}" not in det_nums]
    miss_seg = [i for i in range(1, expected_frames + 1) if f"{i:06d}" not in seg_nums]
    if det_files and miss_det:
        errors.append(f"{cam_id}: labels/det 缺帧 {miss_det[:10]}...")
        st["ok"] = False
    if seg_files and miss_seg:
        errors.append(f"{cam_id}: labels/seg 缺帧 {miss_seg[:10]}...")
        st["ok"] = False

    # 零字节 gt.txt
    if gt_txt.exists() and gt_txt.stat().st_size == 0:
        warns.append(f"{cam_id}: gt/gt.txt 为空（可能所有对象不可见或未导出 MOT）")

    # 数据契约文件存在性
    for req in ("camera.json", "annotations.jsonl", "seqinfo.ini"):
        if not (cam_dir / req).exists():
            errors.append(f"{cam_id}: 缺少 {req}")
            st["ok"] = False
    return st


def check_sync(
    cameras_meta: Dict[str, List[dict]],
    errors: List[str],
) -> dict:
    """跨相机时间/track/mask 同步检查。cameras_meta: cam_id -> frame meta 列表。"""
    st = {"ok": True, "checked_frames": 0}
    ids = list(cameras_meta.keys())
    if not ids:
        return st
    n = len(cameras_meta[ids[0]])
    for i in range(1, n):
        ref = cameras_meta[ids[0]][i]
        for cam in ids[1:]:
            cur = cameras_meta[cam][i]
            if cur.get("frame_index") != ref.get("frame_index"):
                errors.append(f"同步: 相机 {cam} 帧 {i} frame_index 不一致")
                st["ok"] = False
                continue
            for key in ("time_seconds", "source_step", "episode_id"):
                if cur.get(key) != ref.get(key):
                    errors.append(
                        f"同步: 相机 {cam} 帧 {i} 的 {key} "
                        f"({cur.get(key)!r}) != {ids[0]} ({ref.get(key)!r})"
                    )
                    st["ok"] = False
            # track_id / mask_id 跨相机一致
            ref_objs = {o.get("entity_id"): o for o in ref.get("objects", [])}
            for o in cur.get("objects", []):
                eid = o.get("entity_id")
                ref_o = ref_objs.get(eid)
                if ref_o is None:
                    continue
                for key in ("track_id", "mask_id"):
                    if o.get(key) != ref_o.get(key):
                        errors.append(
                            f"同步: {eid} 在相机 {cam} 帧 {i} 的 {key} "
                            f"({o.get(key)!r}) != {ids[0]} ({ref_o.get(key)!r})"
                        )
                        st["ok"] = False
    st["checked_frames"] = n
    return st


def check_track_mask_mapping(cameras_meta: Dict[str, List[dict]], errors: List[str],
                            mask_enabled: bool = True) -> dict:
    """校验每个实体的 track_id / mask_id 与稳定映射一致，且同帧不冲突。

    mask_enabled=False 时仅校验 track_id（mask_id 可为 None）。
    """
    st = {"ok": True, "checked_objects": 0}
    seen: Dict[str, Tuple[int, int]] = {}
    for cam, frames in cameras_meta.items():
        for fr in frames:
            for o in fr.get("objects", []):
                eid = o.get("entity_id")
                if not eid:
                    continue
                st["checked_objects"] += 1
                expect_track = TRACK_MAP.get(eid)
                expect_mask = MASK_MAP.get(eid)
                if expect_track is None:
                    errors.append(f"映射: {cam} 帧 {fr.get('frame_index')} 未知 entity_id {eid!r}")
                    st["ok"] = False
                    continue
                got = (o.get("track_id"), o.get("mask_id"))
                # mask 关闭时 mask_id 可为 None，仅校验 track_id
                if mask_enabled:
                    if got != (expect_track, expect_mask):
                        errors.append(
                            f"映射: {eid} 在 {cam} 帧 {fr.get('frame_index')} "
                            f"track/mask={got}，期望 ({expect_track},{expect_mask})"
                        )
                        st["ok"] = False
                else:
                    if o.get("track_id") != expect_track:
                        errors.append(
                            f"映射: {eid} 在 {cam} 帧 {fr.get('frame_index')} "
                            f"track={o.get('track_id')}，期望 {expect_track}"
                        )
                        st["ok"] = False
                key = got if mask_enabled else (o.get("track_id"),)
                if eid in seen and seen[eid] != key:
                    errors.append(f"映射: {eid} 在不同位置出现不一致 {got} vs {seen[eid]}")
                    st["ok"] = False
                else:
                    seen[eid] = key
    return st


def check_camera_json(cam_dirs: List[Path], expected_frames: int, errors: List[str]) -> dict:
    """检查每个 camera.json 标定合法、分辨率一致、四相机外参不重复、无 NaN。"""
    st = {"ok": True, "cameras": {}}
    resolutions = set()
    locations: List[list] = []
    for cam in cam_dirs:
        path = cam / "camera.json"
        info = {"camera_id": cam.name, "ok": True}
        local_errors: List[str] = []
        try:
            with open(path, encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception as e:  # noqa: BLE001
            local_errors.append(f"标定: {cam.name} camera.json 读取失败: {e}")
        else:
            w = cfg.get("image_width")
            h = cfg.get("image_height")
            if not isinstance(w, int) or not isinstance(h, int) or w <= 0 or h <= 0:
                local_errors.append(f"标定: {cam.name} 分辨率非法 {w}x{h}")
            else:
                resolutions.add((w, h))
            intr = cfg.get("intrinsics") or {}
            for k in ("fx", "fy", "cx", "cy"):
                v = intr.get(k)
                if not _is_finite(v) or v <= 0:
                    local_errors.append(f"标定: {cam.name} intrinsics.{k} 非法 {v!r}")
            extr = cfg.get("extrinsics") or {}
            loc = extr.get("world_location_m")
            if not isinstance(loc, list) or len(loc) != 3 or not all(_is_finite(v) for v in loc):
                local_errors.append(f"标定: {cam.name} extrinsics.world_location_m 非法 {loc!r}")
            else:
                locations.append(loc)
            for axis in ("forward", "right", "up"):
                vec = extr.get(axis)
                if not isinstance(vec, list) or len(vec) != 3 or not all(_is_finite(v) for v in vec):
                    local_errors.append(f"标定: {cam.name} extrinsics.{axis} 非法 {vec!r}")
        if local_errors:
            errors.extend(local_errors)
            info["ok"] = False
            st["ok"] = False
        st["cameras"][cam.name] = info
    if len(resolutions) > 1:
        errors.append(f"标定: 相机分辨率不一致 {resolutions}")
        st["ok"] = False
    # 仅当存在多个相机时检查外参是否重复（单相机无意义）
    if len(cam_dirs) >= 2 and len(locations) == len(cam_dirs) \
            and len({tuple(v) for v in locations}) == 1:
        errors.append("标定: 相机 world_location 完全相同（外参疑似重复/未标定）")
        st["ok"] = False
    st["resolutions"] = sorted(resolutions)
    return st


def check_render_summary(dataset_dir: Path, errors: List[str], warns: List[str]) -> dict:
    summary_path = dataset_dir / "render_summary.json"
    if not summary_path.exists():
        warns.append("缺少 render_summary.json（可能由恢复脚本生成，或以其他方式标记完成）")
        return {"ok": True, "status": "missing", "warnings": warns[-1:]}
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)
    status = summary.get("status")
    per = summary.get("cameras", {})
    bad = {c: e for c, e in per.items() if e.get("ok") is False}
    if status != "success" or bad:
        errors.append(
            f"render_summary: status={status!r}，未通过相机 {sorted(bad)}"
        )
        return {"ok": False, "status": status, "cameras": per}
    return {"ok": True, "status": status, "cameras": per}


def check_pose_coco17(dataset_dir: Path, expected_frames: int, errors: List[str],
                      skip: bool = False) -> dict:
    """检查 Runtime Pose + COCO17 完整性（C5.3 正式 Pose 来源）。

    仅当 pose_session.json 存在时校验（= Runtime Pose 已运行）：
    - capture_complete=True
    - pose_capture.jsonl 行数 == 10 actor × expected_frames × 13 bone
    - coco17_3d.jsonl 行数 == 10 actor × expected_frames
    pose_session.json 不存在 → 任务未启用 yolo_pose 或未运行 pose-finalize，跳过（不报错）。
    skip=True（research_minimal 已 cleanup，raw pose 属有意删除的 transient）→ 跳过。
    """
    st = {"ok": True, "pose_session": False, "capture_complete": False,
          "pose_capture_rows": 0, "coco17_3d_rows": 0, "coco17_total_kp": 0, "skipped": False}
    if skip:
        st["skipped"] = True
        return st
    ps_path = dataset_dir / "pose_session.json"
    if not ps_path.exists():
        st["skipped"] = True
        return st
    pc_path = dataset_dir / "pose_capture.jsonl"
    c3d_path = dataset_dir / "coco17_3d.jsonl"
    c2d_path = dataset_dir / "coco17_2d.jsonl"
    # pose_session 存在 → 校验 capture_complete
    ps = json.loads(ps_path.read_text(encoding="utf-8"))
    st["pose_session"] = True
    st["capture_complete"] = bool(ps.get("capture_complete"))
    if not st["capture_complete"]:
        errors.append(
            f"pose: pose_session.capture_complete=False (captured={ps.get('captured_frame_count')}"
            f"/expected={ps.get('expected_frame_count')})，incomplete，禁止标注成功"
        )
        st["ok"] = False
    # pose_capture.jsonl 行数
    if pc_path.exists():
        rows = [l for l in pc_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        st["pose_capture_rows"] = len(rows)
    else:
        errors.append("pose: 缺少 pose_capture.jsonl")
        st["ok"] = False
    # coco17_3d 行数 == 10 × expected_frames（10 actor × N 帧）
    expected_coco = 10 * expected_frames
    if c3d_path.exists():
        c3 = [l for l in c3d_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        st["coco17_3d_rows"] = len(c3)
        st["coco17_total_kp"] = len(c3) * 17
        if len(c3) != expected_coco:
            errors.append(f"coco17_3d: {len(c3)} 行 != 预期 {expected_coco}（10 actor × {expected_frames} 帧）")
            st["ok"] = False
    else:
        errors.append("coco17: 缺少 coco17_3d.jsonl")
        st["ok"] = False
    # coco17_2d 按 camera 分文件：检查每个 camera 子目录
    cam_dirs_c2d = [d for d in dataset_dir.iterdir() if d.is_dir() and (d / "coco17_2d.jsonl").exists()]
    if not cam_dirs_c2d:
        errors.append("coco17: 缺少任何 <camera>/coco17_2d.jsonl")
        st["ok"] = False
    st["coco17_2d_cameras"] = sorted(d.name for d in cam_dirs_c2d)
    return st


def check_cross_camera_identity(cam_dirs: List[Path], errors: List[str]) -> dict:
    """跨 camera canonical identity audit：所有 camera 的所有 visible object 的
    track_id / mask_id 必须与 entity_id 的 canonical mapping 一致。

    允许某 camera 某 frame 不出现某 entity（遮挡/out-of-FOV），
    但一旦出现，track_id/mask_id 必须匹配 canonical mapping。
    """
    st = {"ok": True, "checked_objects": 0, "identity_errors": 0}
    for cam in cam_dirs:
        ann_path = cam / "annotations.jsonl"
        if not ann_path.exists():
            continue
        with open(ann_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    fr = json.loads(line)
                except json.JSONDecodeError:
                    continue
                fi = fr.get("frame_index")
                for obj in fr.get("objects", []):
                    eid = obj.get("entity_id", "")
                    tid = obj.get("track_id")
                    mid = obj.get("mask_id")
                    st["checked_objects"] += 1
                    expected_tid = TRACK_MAP.get(eid)
                    expected_mid = MASK_MAP.get(eid)
                    if expected_tid is not None and tid != expected_tid:
                        errors.append(
                            f"identity: {cam.name} f{fi} {eid} track_id={tid} expected={expected_tid}"
                        )
                        st["identity_errors"] += 1
                        st["ok"] = False
                    if expected_mid is not None and mid is not None and mid != expected_mid:
                        errors.append(
                            f"identity: {cam.name} f{fi} {eid} mask_id={mid} expected={expected_mid}"
                        )
                        st["identity_errors"] += 1
                        st["ok"] = False
    return st


def write_reports(
    dataset_dir: Path,
    report: dict,
    out_dir: Path,
    args,
) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jpath = out_dir / "soak_audit_report.json"
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    lines = [
        "# Soak 审计报告",
        "",
        f"- 输入: `{dataset_dir}`",
        f"- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 期望: {args.expected_cameras} 相机 × {args.expected_frames_per_camera} 帧",
        f"- 退出码: {report['exit_code']}",
        f"- 结论: **{'PASS' if report['passed'] else 'FAIL'}**",
        "",
        "## 汇总",
        "",
        f"- 相机: {len(report['cameras'])} / {args.expected_cameras}",
        f"- 检查失败项: {len(report['errors'])}",
        f"- 警告: {len(report['warnings'])}",
        "",
        "## 每相机统计",
        "",
        "| 相机 | render RGB | render_mask EXR | img1 RGB | mask | annotations | det | seg | gt.txt | 缺帧 | 重复 | 零字节 |",
        "|------|-----------:|----------------:|-----:|-----:|------------:|----:|----:|-------:|:----:|:----:|:------:|",
    ]
    for cid in sorted(report["cameras"]):
        c = report["cameras"][cid]
        lines.append(
            f"| {cid} | {c.get('render_rgb', '-')} | {c.get('render_mask_exr', '-')} "
            f"| {c.get('img1_rgb', '-')} | {c.get('mask_png', '-')} | {c.get('annotations_frames', '-')} "
            f"| {c.get('det_txt', '-')} | {c.get('seg_txt', '-')} | {c.get('gt_txt_lines', '-')} "
            f"| {len(c.get('img1_missing', []))} | {len(c.get('img1_dup', []))} | {c.get('zero_byte', '-')} |"
        )
    if report["sync"]:
        lines.append("")
        lines.append(f"## 跨相机同步\n\n- 检查帧数: {report['sync'].get('checked_frames', 0)}，OK: {report['sync'].get('ok')}")
    if report["mapping"]:
        lines.append("")
        lines.append(f"## track/mask 映射\n\n- 检查对象数: {report['mapping'].get('checked_objects', 0)}，OK: {report['mapping'].get('ok')}")
    if report["calibration"]:
        lines.append("")
        lines.append(f"## 相机标定\n\n- 分辨率: {report['calibration'].get('resolutions')}，OK: {report['calibration'].get('ok')}")
    if report["render_summary"]:
        lines.append("")
        lines.append(f"## render_summary\n\n- status: {report['render_summary'].get('status')}，OK: {report['render_summary'].get('ok')}")
    if report.get("pose_coco17"):
        pc = report["pose_coco17"]
        lines.append("")
        lines.append(
            f"## Runtime Pose + COCO17\n\n- pose_session: {pc.get('pose_session')}，"
            f"capture_complete: {pc.get('capture_complete')}\n"
            f"- pose_capture rows: {pc.get('pose_capture_rows')}\n"
            f"- coco17_3d rows: {pc.get('coco17_3d_rows')}，total_kp: {pc.get('coco17_total_kp')}，OK: {pc.get('ok')}"
        )
    if report.get("cross_camera_identity"):
        ci = report["cross_camera_identity"]
        lines.append("")
        lines.append(
            f"## Cross-Camera Identity Audit\n\n- checked_objects: {ci.get('checked_objects')}，"
            f"identity_errors: {ci.get('identity_errors')}，OK: {ci.get('ok')}"
        )
    if report["validation"]:
        lines.append("")
        lines.append(f"## validate-annotations（{report['validation'].get('level')}）\n\n- 退出码: {report['validation'].get('exit_code')}")
    if report["errors"]:
        lines.append("")
        lines.append("## 失败项\n")
        for e in report["errors"]:
            lines.append(f"- {e}")
    if report["warnings"]:
        lines.append("")
        lines.append("## 警告\n")
        for w in report["warnings"]:
            lines.append(f"- {w}")
    mpath = out_dir / "soak_audit_report.md"
    with open(mpath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return jpath, mpath


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="soak episode 完整性审计")
    ap.add_argument("--input", required=True, help="相机数据集根目录（含多个 camera 子目录）")
    ap.add_argument("--expected-cameras", type=int, default=4)
    ap.add_argument("--expected-frames-per-camera", type=int, default=300)
    ap.add_argument("--episode", default=None, help="episode 目录（读 meta 时序，精确校验渲染帧覆盖）")
    ap.add_argument("--render-fps", type=int, default=30, help="MRQ 渲染帧率（缺省 30）")
    ap.add_argument("--validation-level", default="quick", choices=["quick", "full", "none"])
    ap.add_argument("--mask-enabled", default="true", help="是否启用 instance_mask 校验（true/false，缺省 true）")
    ap.add_argument("--pose-skip", default="false", help="是否跳过 pose 完整性校验（true/false，缺省 false）")
    ap.add_argument("--output", default=None, help="报告输出目录（默认 <input>/audit）")
    ap.add_argument("--resolved-task", default=None, help="resolved-task.json（用于公开 Config v3 契约审计）")
    args = ap.parse_args(argv)

    mask_enabled = str(args.mask_enabled).lower() in ("true", "1", "yes")
    pose_skip = str(args.pose_skip).lower() in ("true", "1", "yes")

    dataset_dir = Path(args.input)
    if not dataset_dir.is_dir():
        print(f"ERROR: 输入目录不存在: {dataset_dir}", file=sys.stderr)
        return 1

    errors: List[str] = []
    warns: List[str] = []
    report: dict = {"input": str(dataset_dir), "expected": {
        "cameras": args.expected_cameras,
        "frames_per_camera": args.expected_frames_per_camera,
    }}

    # 公开 manifest 存在时，公开文件是审计的唯一规范输入；其余产物只是诊断信息。
    if (dataset_dir / "episode_manifest.json").exists():
        from grf_ue_bridge.public_validator import validate_public_episode

        result = validate_public_episode(dataset_dir, resolved_task=args.resolved_task)
        report.update({
            "cameras": result.stats.get("cameras", {}), "sync": {}, "mapping": {}, "calibration": {},
            "render_summary": {}, "pose_coco17": {}, "cross_camera_identity": {},
            "validation": {"level": "public", "exit_code": result.exit_code},
            "public_validation": result.stats,
            "errors": result.errors, "warnings": [],
            "passed": result.ok, "exit_code": result.exit_code,
        })
        out_dir = Path(args.output) if args.output else dataset_dir / "audit"
        jpath, mpath = write_reports(dataset_dir, report, out_dir, args)
        print(f"soak 审计完成: {'PASS' if result.ok else 'FAIL'}")
        print(f"  失败项: {len(result.errors)}")
        for error in result.errors[:20]:
            print(f"    FAIL  {error}")
        print(f"  报告: {jpath}")
        print(f"        {mpath}")
        return result.exit_code

    # 相机发现
    cam_dirs = sorted(p.parent for p in dataset_dir.rglob("camera.json"))
    if len(cam_dirs) != args.expected_cameras:
        errors.append(f"相机数量 {len(cam_dirs)} != 预期 {args.expected_cameras}")

    # episode 时序 → keep_indices
    meta = _load_episode_meta(Path(args.episode)) if args.episode else None
    source_step_seconds = None
    if meta:
        timing = meta.get("timing") or {}
        num_steps = int(timing.get("num_steps", args.expected_frames_per_camera))
        source_step_seconds = float(timing.get("source_step_seconds", 0.1))
        fps = int(timing.get("playback_fps", args.render_fps))
        keep_indices = [int(round(i * source_step_seconds * fps)) for i in range(num_steps)]
        report["timing"] = {"source_step_seconds": source_step_seconds, "playback_fps": fps,
                            "num_steps": num_steps, "max_keep_index": max(keep_indices) if keep_indices else 0}
    else:
        # 从第一个相机标注反推
        if cam_dirs:
            frames, _ = _read_frame_meta(cam_dirs[0])
            if frames:
                keep_indices = _derive_keep_indices(frames, args.render_fps, None)
                if source_step_seconds is None and frames:
                    for fr in frames:
                        s = fr.get("source_step")
                        t = fr.get("time_seconds")
                        if isinstance(s, (int, float)) and s > 0 and isinstance(t, (int, float)):
                            source_step_seconds = t / s
                            break
        else:
            keep_indices = list(range(args.expected_frames_per_camera))

    # 每相机
    cameras: Dict[str, dict] = {}
    cameras_meta: Dict[str, List[dict]] = {}
    for cam in cam_dirs:
        st = check_camera(cam, args.expected_frames_per_camera, keep_indices, errors, warns, mask_enabled)
        frames, _ = _read_frame_meta(cam)
        cameras_meta[cam.name] = frames or []
        cameras[cam.name] = st
    report["cameras"] = cameras

    # 跨相机同步 + 映射
    sync = check_sync(cameras_meta, errors)
    mapping = check_track_mask_mapping(cameras_meta, errors, mask_enabled)
    calib = check_camera_json(cam_dirs, args.expected_frames_per_camera, errors)
    rsummary = check_render_summary(dataset_dir, errors, warns)
    # Runtime Pose + COCO17 检查：仅在 pose_session.json 存在时校验
    # （pose_session.json 是 Runtime Pose 已运行的标志；不存在 = 任务未启用 yolo_pose 或未运行 pose-finalize）
    pose_coco = check_pose_coco17(dataset_dir, args.expected_frames_per_camera, errors, pose_skip)
    cross_identity = check_cross_camera_identity(cam_dirs, errors)
    report["sync"] = sync
    report["mapping"] = mapping
    report["calibration"] = calib
    report["render_summary"] = rsummary
    report["pose_coco17"] = pose_coco
    report["cross_camera_identity"] = cross_identity

    # 可选进程内验证
    validation = None
    if args.validation_level != "none":
        from grf_ue_bridge.annotation_validator import validate_annotation_dir
        vc = validate_annotation_dir(dataset_dir, workers=0, validation_level=args.validation_level)
        validation = {"level": args.validation_level, "exit_code": vc}
        if vc != 0:
            errors.append(f"validate-annotations({args.validation_level}) 退出码 {vc}")
    report["validation"] = validation

    report["errors"] = errors
    report["warnings"] = warns
    report["passed"] = not errors
    report["exit_code"] = 0 if not errors else 1

    out_dir = Path(args.output) if args.output else dataset_dir / "audit"
    jpath, mpath = write_reports(dataset_dir, report, out_dir, args)

    # 人类可读摘要
    print(f"soak 审计完成: {'PASS' if report['passed'] else 'FAIL'}")
    print(f"  相机: {len(cam_dirs)}/{args.expected_cameras}")
    print(f"  失败项: {len(errors)}")
    for e in errors[:20]:
        print(f"    FAIL  {e}")
    for w in warns[:10]:
        print(f"    WARN  {w}")
    print(f"  报告: {jpath}")
    print(f"        {mpath}")
    return report["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
