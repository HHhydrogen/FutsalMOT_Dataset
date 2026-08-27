"""UE 侧自动渲染 RGB 帧（Movie Render Queue, MRQ）。

职责：
  1. 用 MRQ 渲染每个 Camera 对应的 Level Sequence，输出 PNG 到 <cam>/render/。
  2. 按帧同步契约从渲染结果里选出与 annotation 一一对应的帧，复制为
     <cam>/img1/{frame_index:06d}.png。

帧同步契约（与 annotation_exporter / README 一致）：
  - annotation frame_index（1 基）= 图片 img1/{frame_index:06d}.png。
  - annotation frame N ↔ Sequence 帧 round((N-1)*source_step_seconds*playback_fps)。
  - MRQ 以 Sequence 的 display rate（= playback_fps）渲染全范围，再按上面的映射取帧。

MRQ Python API 在不同 UE 版本命名/结构不同，本模块用多级 fallback + 清晰报错；
纯函数部分（帧选择、映射、复制）不依赖 unreal，可用 pytest 测试。

重要：渲染必须异步。提交 MRQ 后立即返回，后续阶段（复制 RGB / 写完成标记）由
Movie Render Queue 的 finished/error delegate 回调驱动，并有 slate post-tick
watchdog 兜底。delegate 回调须用与委托一致的显式签名（不能用 *args），否则
UE 5.8 绑定会报 "incorrect number of arguments" 而失败。PIE 渲染窗口需要编辑器
主线程持续 tick 才能推进，任何 time.sleep / Event.wait 同步阻塞都会卡死渲染。
"""

import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from annotation_utils import entity_id_to_mask_id  # noqa: E402
from dataset_export import ensure_dir, load_episode, load_mapping, write_json_atomic  # noqa: E402
from render_preset import (  # noqa: E402
    camera_post_process_overrides,
    mrq_aa_overrides,
    mrq_temporal_overrides,
    mrq_warmup_overrides,
    post_process_console_vars,
    resolve_preset,
)
from scene_apply import apply_preview_frame, find_all_actors, find_actor  # noqa: E402
from player_motion import gk_entity_ids_from_meta  # noqa: E402

# 模块级：当前活跃的异步渲染管线。脚本（main）返回后 MRQ delegate 仍持有它的
# 方法引用，但显式保留模块级引用可防止任何环境下对象被垃圾回收。
_ACTIVE_RENDER = None

# （临时调试）BurnIn Widget：仅利用 OnOutputFrameStarted 每输出帧回调，不合成到图。
BURN_IN_CLASS_PATH = "/Game/FutsalMOT/Blueprints/WBP_PoseMRQBurnInC4.WBP_PoseMRQBurnInC4_C"


# ── 纯函数：帧选择与映射 ────────────────────────────────────────────────

def select_rendered_frame_indices(
    num_steps: int,
    source_step_seconds: float,
    playback_fps: int,
) -> List[int]:
    """返回与 annotation 各帧对应的渲染帧序号（0 基）。

    annotation frame_index = i + 1（i 为 0 基 GRF step）对应的 Sequence 帧：
        round(i * source_step_seconds * playback_fps)
    """
    return [
        int(round(i * source_step_seconds * playback_fps))
        for i in range(num_steps)
    ]


def map_rendered_to_annotation(
    rendered_numbers: Sequence[int],
    keep_indices: Sequence[int],
) -> Dict[int, int]:
    """把渲染输出帧号映射到 annotation frame_index。

    rendered_numbers: render 目录中实际存在的渲染帧号列表（0 基，可为乱序）。
    keep_indices: select_rendered_frame_indices 的返回值。
    返回 {frame_index: rendered_number}，frame_index 从 1 开始。
    若某目标帧在渲染输出中缺失，跳过（不产生非法映射）。
    """
    available = set(rendered_numbers)
    mapping: Dict[int, int] = {}
    for i, num in enumerate(keep_indices):
        if num in available:
            mapping[i + 1] = num
    return mapping


def find_rendered_frame_numbers(render_dir: Path) -> Dict[int, Path]:
    """扫描 render 目录（含子目录），从 PNG 文件名里解析帧号。

    返回 {frame_number: path}。文件名须包含数字（MRQ 输出如 000000.png）。
    """
    result: Dict[int, Path] = {}
    for p in sorted(render_dir.rglob("*.png")):
        digits = "".join(ch for ch in p.stem if ch.isdigit())
        if digits:
            result[int(digits)] = p
    return result


def copy_rendered_frames(
    render_dir: Path,
    img1_dir: Path,
    keep_indices: Sequence[int],
) -> int:
    """把渲染帧中与 annotation 对齐的帧复制为 img1/{frame_index:06d}.png。

    返回复制的帧数。
    """
    rendered = find_rendered_frame_numbers(render_dir)
    mapping = map_rendered_to_annotation(sorted(rendered.keys()), keep_indices)
    ensure_dir(img1_dir)
    copied = 0
    for frame_index, num in mapping.items():
        dst = img1_dir / f"{frame_index:06d}.png"
        shutil.copy2(rendered[num], dst)
        copied += 1
    return copied


def find_mask_files(render_mask_dir: Path) -> Dict[int, Path]:
    """从 render_mask/ 中挑出 Instance-ID Mask 帧文件。返回 {frame_number: path}。

    支持两种 mask 源：
      - PNG 逐帧 mask（post_process_material 等）——与注解一致，本模块可直接复制；
      - Object ID Pass 的 Cryptomatte **multilayer EXR**（UE 5.8 实测可用）——
        文件名 {frame:06d}.exr，本模块只统计可用帧，真正转 mask/*.png 由
        P1 `grf-ue cryptomatte-to-mask` 完成。
    返回的 value 保留完整 path；调用方可用 .suffix 区分 EXR / PNG。
    正常情况下 mask job 只配置一个输出 → 每帧一个文件，直接解析。
    若与其它 pass 混出（文件名出现多前缀），优先取文件名含
    mask/depth/stencil/custom/object 前缀的组；否则取帧数最多的组兜底。
    """
    files = list(render_mask_dir.rglob("*.png")) + list(render_mask_dir.rglob("*.exr"))
    groups: Dict[str, Dict[int, Path]] = {}
    for p in files:
        digits = "".join(ch for ch in p.stem if ch.isdigit())
        if not digits:
            continue
        prefix = p.stem[: -len(digits)]
        frame = int(digits)
        groups.setdefault(prefix, {})[frame] = p
    if len(groups) == 1:
        return next(iter(groups.values()))
    for prefix, mapping in groups.items():
        if any(k in prefix.lower() for k in ("mask", "depth", "stencil", "custom", "object")):
            return mapping
    if groups:
        return max(groups.values(), key=len)
    return {}


def copy_mask_frames(
    render_mask_dir: Path,
    mask_dir: Path,
    keep_indices: Sequence[int],
) -> int:
    """把 render_mask/ 中与 annotation 对齐的 mask 帧落到 mask/，或统计其可用性。

    返回对齐帧数（frame_index ↔ rendered frame）：
      - PNG 源（post_process_material 等）→ 复制为 mask/{frame_index:06d}.png；
      - Object ID EXR 源（Cryptomatte）→ 无法在 UE/纯 Python 侧解码成 mask PNG，
        不复制，仅返回对齐帧数（真正转换由 P1 `grf-ue cryptomatte-to-mask` 完成）。

    与 img1/ 使用同一帧号（frame_index），保证 RGB 与 mask 一一对应。
    """
    rendered = find_mask_files(render_mask_dir)
    mapping = map_rendered_to_annotation(sorted(rendered.keys()), keep_indices)
    if not mapping:
        return 0
    srcs = [rendered[num] for num in mapping.values()]
    # Object ID EXR：仅统计可用帧，不复制（解码/转 PNG 在 P1 完成）
    if all(s.suffix.lower() == ".exr" for s in srcs):
        return len(mapping)
    ensure_dir(mask_dir)
    copied = 0
    for frame_index, num in mapping.items():
        src = rendered[num]
        if src.suffix.lower() != ".png":
            continue
        dst = mask_dir / f"{frame_index:06d}.png"
        shutil.copy2(src, dst)
        copied += 1
    return copied


def recover_render_to_img1(
    sequences_cfg,
    annotation_cfg: dict,
    episode_dir: Path,
    output_dir: Path,
):
    """从已渲染的 render/ 目录恢复 RGB 到 img1/（无需 MRQ / 无需 UE 渲染）。

    场景：上一次 MRQ 渲染已把 PNG 输出到各 camera 的 render/，但完成回调
    （finished delegate / watchdog）未触发，导致 img1/ 为空。用本函数从现有
    render/ 帧复制出 img1/ 并写 render_summary.json，避免重新渲染。

    纯 Python，不依赖 unreal，可在 P1（.venv）或 UE Python 中运行。

    Returns:
        (status, per_camera)。status ∈ {"success", "partial", "failed"}。
    """
    render_cfg = annotation_cfg.get("render_rgb") or {}
    meta, frames = load_episode(episode_dir)
    episode_id = meta.get("episode_id") or episode_dir.name
    source_step = float(meta["timing"].get("source_step_seconds", 0.1))
    frame_rate = int(render_cfg.get("frame_rate") or annotation_cfg.get("playback_fps") or 30)
    keep_indices = select_rendered_frame_indices(len(frames), source_step, frame_rate)

    per_camera = {}
    total_copied = 0
    total_mask_copied = 0
    expected = len(keep_indices)
    for seq_entry in (sequences_cfg or []):
        seq_name = seq_entry.get("name")
        cam_id = seq_entry.get("camera_actor") or seq_name
        if not seq_name:
            continue
        cam_out = Path(output_dir) / episode_id / cam_id
        render_dir, img1_dir = cam_out / "render", cam_out / "img1"
        if not render_dir.exists():
            print(f"  [SKIP] {cam_id}: 无 render/ 目录，跳过")
            continue
        copied = copy_rendered_frames(render_dir, img1_dir, keep_indices)
        total_copied += copied
        per_camera[cam_id] = {
            "sequence": seq_name,
            "img1_frames": copied,
            "expected_frames": expected,
            "ok": copied == expected,
        }
        if copied == 0:
            mark = "MISSING"
        elif copied == expected:
            mark = "OK"
        else:
            mark = "PARTIAL"
        print(f"  [{mark}] {cam_id}: img1/ 写入 {copied}/{expected} 帧")

        # Instance-ID Mask：从 render_mask/ 恢复（若存在）
        mask_render = cam_out / "render_mask"
        mask_dir = cam_out / "mask"
        if mask_render.exists():
            mask_copied = copy_mask_frames(mask_render, mask_dir, keep_indices)
            total_mask_copied += mask_copied
            mask_srcs = [p.suffix.lower() for p in find_mask_files(mask_render).values()]
            is_exr = bool(mask_srcs) and all(s == ".exr" for s in mask_srcs)
            per_camera[cam_id]["mask_frames"] = mask_copied
            per_camera[cam_id]["mask_source"] = "object_id_exr" if is_exr else "png"
            per_camera[cam_id]["ok"] = per_camera[cam_id]["ok"] and (mask_copied == expected)
            m_mark = "MISSING" if mask_copied == 0 else ("OK" if mask_copied == expected else "PARTIAL")
            m_label = "mask(EXR) 对齐" if is_exr else "mask/写入"
            print(f"  [{m_mark}] {cam_id}: {m_label} {mask_copied}/{expected} 帧")

    if not per_camera:
        print("WARNING: 没有任何可恢复的 camera render/ 目录")
        return "failed", per_camera

    if total_copied == 0:
        status = "failed"
    elif all(e["ok"] for e in per_camera.values()):
        status = "success"
    else:
        status = "partial"
    summary = {
        "episode_id": episode_id,
        "status": status,
        "reason": "从已有 render/ 目录恢复（MRQ 完成回调未触发）",
        "total_img1_frames": total_copied,
        "total_mask_frames": total_mask_copied,
        "cameras": per_camera,
        "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "note": "由 recover_render_to_img1 恢复写入。完整标注校验请在 P1 运行："
                "uv run grf-ue validate-annotations <output_dir>",
    }
    summary_path = Path(output_dir) / episode_id / "render_summary.json"
    write_json_atomic(summary_path, summary)
    print(f"\n=== 恢复完成: {status} ===")
    print(f"  render_summary.json -> {summary_path}")
    return status, per_camera


# ── UE 侧：MRQ 渲染 ─────────────────────────────────────────────────────

def _list_mrq_classes():
    """打印当前 unreal 模块里可用的 MRQ/渲染相关类，便于按 UE 版本适配。

    MRQ Python API 在不同 UE 版本命名不同（本工程为 UE 5.8）。当找不到期望
    的类时调用，把输出贴回即可定位正确的类名。
    """
    import unreal

    names = sorted(
        n for n in dir(unreal)
        if "MoviePipeline" in n
        or "RenderQueue" in n
        or "MovieRender" in n
        or "SequencerTools" in n
        or "SceneCapture" in n
    )
    print("  [MRQ 诊断] 相关 unreal 类:", names)
    return names


def _print_mrq_members(obj, label: str) -> None:
    """打印 MRQ 对象的可用成员名，便于适配不同 UE 版本的 API。"""
    names = sorted(
        n for n in dir(obj)
        if not n.startswith("__")
        and any(k in n.lower() for k in ("setting", "config", "shot", "render", "output", "add_", "set_", "get_"))
    )
    print(f"  [MRQ 诊断] {label} 相关成员:", names)


def _get_current_map_path() -> str:
    """获取当前关卡（world）的资源路径，供 MRQ job.map 使用。"""
    import unreal

    world = None
    for getter in (
        lambda: unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world(),
        lambda: unreal.EditorLevelLibrary.get_editor_world(),
        lambda: unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).get_editor_world(),
    ):
        try:
            world = getter()
            if world is not None:
                break
        except Exception:
            continue
    if world is None:
        return ""
    return world.get_path_name().split(".")[0]  # 去掉对象实例后缀


def _get_job_config(job):
    """取 job 的配置对象（MoviePipelinePrimaryConfig/MasterConfig）。

    不同 UE 版本获取方式不同，多级 fallback；取不到则新建。
    """
    import unreal

    for getter_name in ("get_configuration",):
        m = getattr(job, getter_name, None)
        if m is not None:
            try:
                cfg = m()
                if cfg is not None:
                    return cfg
            except Exception:
                pass
    for attr in ("configuration", "primary_config"):
        try:
            cfg = getattr(job, attr, None)
            if cfg is not None:
                return cfg
        except Exception:
            pass
    return unreal.MoviePipelinePrimaryConfig()


def _set_job_config(job, config) -> bool:
    """把配置挂到 job 上。不同 UE 版本 API 不同，多级 fallback。"""
    for method_name in ("set_configuration",):
        m = getattr(job, method_name, None)
        if m is not None:
            try:
                m(config)
                return True
            except Exception:
                pass
    for attr in ("configuration", "primary_config"):
        try:
            setattr(job, attr, config)
            return True
        except Exception:
            pass
    return False


def _mrq_get_queue(subsystem):
    """获取 MRQ 当前队列。"""
    try:
        return subsystem.get_queue()
    except Exception:
        return None


def _clear_queue(queue) -> None:
    """清空队列里的已有 job，避免残留。"""
    if queue is None:
        return
    for method_name in ("delete_all_jobs", "clear"):
        m = getattr(queue, method_name, None)
        if m is not None:
            try:
                m()
                return
            except Exception:
                continue
    try:
        queue.jobs.clear()
    except Exception:
        pass


def _clear_dir(path: Path) -> None:
    """清空一个目录的内容（含子目录），避免上一次渲染的残留帧混入。"""
    import shutil

    if not path.exists():
        return
    for p in path.iterdir():
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()


# ── CV GT deterministic preset 应用（render_preset.py 的 UE 侧执行）────────

def _set_config_overrides(obj, overrides: dict, label: str) -> None:
    """把 overrides（property → 值或 ("EnumClass","Member")）应用到对象。

    enum 值解析：先精确匹配，再大小写不敏感，再归一化（去下划线/前缀）——
    UE 各版本的枚举成员命名差异较大（如 TAA / TA_A、NONE / None / DOFM_None）。
    """
    import unreal

    for prop, val in overrides.items():
        if isinstance(val, tuple) and len(val) == 2 and isinstance(val[0], str):
            enum_name, member = val
            enum_cls = getattr(unreal, enum_name, None)
            target = None
            if enum_cls is not None:
                target = _resolve_enum_member(enum_cls, member)
            if target is None:
                print(
                    f"  WARNING: 无法解析枚举 {enum_name}.{member}"
                    f"（跳过 {label}.{prop}）"
                )
                continue
            val = target
        try:
            obj.set_editor_property(prop, val)
        except Exception as e:
            print(f"  WARNING: 设置 {label}.{prop} 失败: {e}")


def _resolve_enum_member(enum_cls, desired: str):
    """在枚举类里找与 desired 匹配的成员：精确 → 大小写不敏感 → 归一化。"""

    def _norm(s: str) -> str:
        return "".join(ch for ch in s if ch.isalnum()).lower()

    members = []
    for name in dir(enum_cls):
        if name.startswith("__"):
            continue
        v = getattr(enum_cls, name, None)
        if v is not None:
            members.append((name, v))
    for name, v in members:  # 1. 精确
        if name == desired:
            return v
    nd = desired.lower()
    for name, v in members:  # 2. 大小写不敏感
        if name.lower() == nd:
            return v
    nn = _norm(desired)
    for name, v in members:  # 3. 归一化（忽略大小写/下划线/前缀）
        if _norm(name) == nn:
            return v
    return None


def _find_or_add_overrides(config, cls_name: str, overrides: dict, label: str) -> None:
    """find_or_add 一个 MRQ 设置类并应用 overrides（类不存在/失败时告警跳过）。"""
    import unreal

    if not overrides:
        return
    cls = getattr(unreal, cls_name, None)
    if cls is None:
        print(f"  WARNING: 无 unreal.{cls_name}，跳过 cv_gt {label} 覆盖")
        return
    try:
        setting = config.find_or_add_setting_by_class(cls)
    except Exception as e:
        print(f"  WARNING: 添加 {cls_name} 失败: {e}")
        return
    _set_config_overrides(setting, overrides, label)


def _add_burn_in(config) -> None:
    """（临时调试）向 RGB job 添加 BurnInSetting：OnOutputFrameStarted 回调。

    BurnIn 不合成到最终图（b_composite_onto_final_image=False），Widget 本身透明，
    仅利用 MoviePipelineBurnInWidget 的 OnOutputFrameStarted 每输出帧回调。
    """
    import unreal

    cls = getattr(unreal, "MoviePipelineBurnInSetting", None)
    if cls is None:
        print("  WARNING: 无 MoviePipelineBurnInSetting，跳过 BurnIn")
        return
    try:
        setting = config.find_or_add_setting_by_class(cls)
    except Exception as e:
        print(f"  WARNING: 添加 MoviePipelineBurnInSetting 失败: {e}")
        return
    try:
        soft = unreal.SoftClassPath(BURN_IN_CLASS_PATH)
        setting.set_editor_property("burn_in_class", soft)
        setting.set_editor_property("composite_onto_final_image", False)
        print(f"  [BurnIn] 已启用 {BURN_IN_CLASS_PATH}（不合成到最终图）")
    except Exception as e:
        print(f"  WARNING: 设置 BurnIn 属性失败: {e}")


def _apply_mrq_preset(config, preset, cv_gt, is_mask: bool) -> None:
    """向 MRQ job 配置施加 cv_gt preset。

    - 时间确定性（temporal_accumulation=NONE、temporal_sample_count=1）：RGB 与 mask
      job 都应用，保证两路都表示单时刻（无 temporal motion integration）。
    - warm-up 契约（engine_warm_up_count=2 等）：RGB 与 mask job 相同，让 PIE/Camera Cut
      在输出帧 0 前完成初始化（修复首帧默认视角），不改变 frame 映射、不重开时间采样。
    - anti-aliasing（保留合理 AA）：仅 RGB job 应用（Object ID pass 不需要 TAA）。
    """
    aa_ov: dict = {}
    aa_ov.update(mrq_temporal_overrides(preset, cv_gt))
    aa_ov.update(mrq_warmup_overrides(preset, cv_gt))
    if not is_mask:
        aa_ov.update(mrq_aa_overrides(preset, cv_gt))
        _add_burn_in(config)
    _find_or_add_overrides(config, "MoviePipelineAntiAliasingSetting", aa_ov, "antialiasing")


def _add_job_console_variables(config, cvar_map: dict) -> None:
    """向 MRQ job 配置添加控制台变量（cvar），确定性关闭后处理效果。

    与相机 post_process 覆盖互补：即使某 UE 版本的相机后处理覆盖位不生效，
    cvar 仍强制关闭 motion blur / DOF / 色差，保证 RGB 与 mask 边界一致。
    """
    import unreal

    if not cvar_map:
        return
    cls = getattr(unreal, "MoviePipelineConsoleVariableSetting", None)
    if cls is None:
        print("  WARNING: 无 MoviePipelineConsoleVariableSetting，无法设置 cv_gt 后处理 cvar")
        return
    try:
        cvar = config.find_or_add_setting_by_class(cls)
    except Exception as e:
        print(f"  WARNING: 添加 MoviePipelineConsoleVariableSetting 失败: {e}")
        return
    entry_cls = getattr(unreal, "MoviePipelineConsoleVariableEntry", None)
    if entry_cls is None:
        print("  WARNING: 无 MoviePipelineConsoleVariableEntry，无法设置 cvar")
        return
    prop = "cvars"
    for name, value in cvar_map.items():
        entry = entry_cls()
        if not _set_console_entry(entry, name, float(value)):
            print(f"  WARNING: 无法设置 cvar 字段 {name}={value}")
            continue
        try:
            current = list(cvar.get_editor_property(prop) or [])
        except Exception:
            current = []
        if not any(name in str(x) for x in current):
            current.append(entry)
        try:
            cvar.set_editor_property(prop, current)
            print(f"  [cv_gt] cvar: {name}={value}")
        except Exception as e:
            print(f"  WARNING: 设置 cvar {name} 失败: {e}")


def _apply_cv_gt_camera_post_process(sequences_cfg, preset, cv_gt) -> None:
    """cv_gt 时把确定性后处理覆盖写到每个 Camera 的 CineCameraComponent。

    不依赖关卡手工设置的 Post Process Volume：脚本每次渲染都显式把
    post_process_settings（blend weight=1.0，覆盖关卡体积）写到相机并保存关卡，
    保证不同机器/关卡得到相同的 CV GT 语义。
    """
    import unreal

    overrides = camera_post_process_overrides(preset, cv_gt)
    if not overrides:
        return
    blend = overrides.get("post_process_blend_weight")
    pp_overrides = {k: v for k, v in overrides.items() if k != "post_process_blend_weight"}
    applied = 0
    for seq in (sequences_cfg or []):
        cam_id = seq.get("camera_actor") or seq.get("name")
        if not cam_id:
            continue
        actor = find_actor(cam_id)
        if actor is None:
            print(f"  WARNING: 找不到 Camera actor '{cam_id}'，无法应用 cv_gt 后处理")
            continue
        comps = actor.get_components_by_class(unreal.CineCameraComponent)
        if not comps:
            print(f"  WARNING: {cam_id} 无 CineCameraComponent，跳过 cv_gt 后处理")
            continue
        comp = comps[0]
        pp = None
        try:
            pp = comp.get_editor_property("post_process_settings")
        except Exception:
            pass
        if pp is None:
            try:
                pp = unreal.PostProcessSettings()
            except Exception as e:
                print(f"  WARNING: 无法创建 PostProcessSettings（{cam_id}）: {e}")
                continue
        _set_config_overrides(pp, pp_overrides, f"{cam_id}.post_process_settings")
        # 显式设置 bOverride_* 位：PostProcessSettings 只有置位的字段才参与混合，
        # 否则关卡 Post Process Volume 的 motion blur / DOF / 色差会照常生效
        for prop in pp_overrides:
            for flag in (f"b_override_{prop}", f"override_{prop}"):
                try:
                    pp.set_editor_property(flag, True)
                    break
                except Exception:
                    continue
        try:
            comp.set_editor_property("post_process_settings", pp)
        except Exception as e:
            print(f"  WARNING: 设置 {cam_id}.post_process_settings 失败: {e}")
        if blend is not None:
            try:
                comp.set_editor_property("post_process_blend_weight", float(blend))
            except Exception as e:
                print(f"  WARNING: 设置 {cam_id}.post_process_blend_weight 失败: {e}")
        applied += 1
    if applied:
        _save_current_level()
    print(f"  [cv_gt] 已为 {applied} 个 Camera 应用确定性后处理 preset（blend=1.0 覆盖关卡体积）")


def _build_mrq_job(
    queue,
    seq_asset_path: str,
    map_path: str,
    render_dir: Path,
    image_width: int,
    image_height: int,
    frame_rate: int,
    file_name_format: str,
    zero_pad: int,
    preset=None,
    cv_gt=None,
):
    """从队列分配并配置一个 MRQ job。返回 job，失败时抛错并附 API 说明。"""
    import unreal

    for cls_name in (
        "MoviePipelineExecutorJob",
        "MoviePipelineOutputSetting",
        "MoviePipelineImageSequenceOutput_PNG",
        "MoviePipelinePrimaryConfig",
    ):
        if getattr(unreal, cls_name, None) is None:
            _list_mrq_classes()
            raise RuntimeError(
                f"unreal.{cls_name} 不存在。已打印可用 MRQ 类，请把上面 "
                f"[MRQ 诊断] 输出贴回。"
            )

    # 从队列分配 job，保证它会被渲染队列执行
    try:
        job = queue.allocate_new_job(unreal.MoviePipelineExecutorJob)
    except Exception:
        job = unreal.MoviePipelineExecutorJob()
        try:
            queue.jobs.add(job)
        except Exception:
            pass
    job.sequence = unreal.SoftObjectPath(seq_asset_path)
    if map_path:
        job.map = unreal.SoftObjectPath(map_path)

    # UE 5.8 MRQ：配置用 find_or_add_setting_by_class 获取/创建设置并配置。
    # 注意：setup_basic_configuration() 依赖默认预设资产 /Temp/MovieRenderPipeline/
    # BasicConfigDefaults，本项目不存在，无法自动带上渲染 pass，需手动添加。
    config = _get_job_config(job)
    try:
        output = config.find_or_add_setting_by_class(unreal.MoviePipelineOutputSetting)
        output.output_directory = unreal.DirectoryPath(str(render_dir))
        output.file_name_format = file_name_format
        output.zero_pad_frame_numbers = zero_pad
        output.output_resolution = unreal.IntPoint(image_width, image_height)
        try:
            output.frame_rate = unreal.FrameRate(frame_rate, 1)
        except Exception:
            pass

        png = config.find_or_add_setting_by_class(
            unreal.MoviePipelineImageSequenceOutput_PNG
        )
        # PNG 输出无需额外配置

        # 渲染 pass（必须，否则 shot 为 0 passes 不输出任何帧）。
        # 标准 Deferred 渲染 pass；若该基类不可实例化，逐个尝试具体变体。
        pass_added = False
        pass_classes = (
            unreal.MoviePipelineDeferredPassBase,
            unreal.MoviePipelineDeferredPass_PathTracer,
            unreal.MoviePipelineDeferredPass_Unlit,
        )
        for idx, pass_cls in enumerate(pass_classes):
            try:
                config.find_or_add_setting_by_class(pass_cls)
                if idx > 0:
                    print(
                        f"  [MRQ] 使用备选渲染 pass: {pass_cls.__name__}"
                        f"（标准 MoviePipelineDeferredPassBase 不可用）"
                    )
                pass_added = True
                break
            except Exception:
                continue
        if not pass_added:
            raise RuntimeError(
                "无法添加 MRQ 渲染 pass（DeferredPassBase/PathTracer/Unlit 均失败）。"
                "请把 [MRQ 诊断] 输出贴回。"
            )

        # CV GT preset：RGB job 应用 anti-aliasing + 时间确定性（is_mask=False）
        _apply_mrq_preset(config, preset, cv_gt, is_mask=False)
        # CV GT preset：cvar 强制关闭 motion blur / DOF / 色差（后处理确定性兜底）
        _add_job_console_variables(config, post_process_console_vars(preset, cv_gt))

    except Exception as e:
        _print_mrq_members(config, "config")
        _print_mrq_members(job, "job")
        raise RuntimeError(
            f"无法用 find_or_add_setting_by_class 配置 MRQ 设置: {e}\n"
            f"已打印 config/job 相关成员，请把上面 [MRQ 诊断] 输出贴回。"
        )

    if not _set_job_config(job, config):
        _print_mrq_members(job, "job")
        raise RuntimeError(
            "无法把配置挂到 job（set_configuration 失败）。请把 [MRQ 诊断] 输出贴回。"
        )

    return job


def _ensure_custom_depth_stencil_enabled() -> None:
    """确保 Custom Depth-Stencil 已开启（含 stencil），供 Instance-ID Mask 使用。

    多级 fallback：
      1. 尝试设置 RendererSettings / RendererOverrideSettings 的 custom_depth_stencil_pass。
      2. 用 r.CustomDepth 控制台命令开启（3 = enable with stencil）。
      3. 仍失败则打印手动步骤。
    """
    import unreal

    for cls_name in ("RendererSettings", "RendererOverrideSettings"):
        cls = getattr(unreal, cls_name, None)
        if cls is None:
            continue
        try:
            settings = unreal.get_default_object(cls)
            enum = getattr(unreal, "ECustomDepthStencil", None)
            target = None
            if enum is not None:
                for name in ("ENABLED_WITH_STENCIL", "ENABLED_STENCIL", "ENABLED"):
                    v = getattr(enum, name, None)
                    if v is not None:
                        target = v
                        break
            if target is not None:
                settings.set_editor_property("custom_depth_stencil_pass", target)
                print("  [MRQ] Custom Depth-Stencil Pass = Enabled With Stencil")
                return
        except Exception as e:
            print(f"  WARNING: 自动开启 Custom Depth-Stencil 失败（{cls_name}）: {e}")
    try:
        unreal.SystemLibrary.execute_console_command(None, "r.CustomDepth 3")
        print("  [MRQ] 通过控制台命令开启 r.CustomDepth=3（Enable with Stencil）")
        return
    except Exception as e:
        print(f"  WARNING: 控制台命令开启 Custom Depth-Stencil 失败: {e}")
    print(
        "  WARNING: 请在 Project Settings → Rendering → Custom Depth-Stencil Pass"
        " 手动设为 'Enabled with Stencil'，否则 Instance-ID Mask 全为背景。"
    )


def _save_current_level() -> None:
    """保存当前关卡，把 actor 上运行时设置的 stencil 值写入磁盘。

    MRQ 的 PIE 渲染从磁盘加载/复制关卡；若不保存，运行时用 set_editor_property 设的
    stencil 值不会带入 PIE → stencil 缓冲为空 → Instance-ID Mask 全黑。保存后 PIE
    能读到持久化的 stencil 值（多级 fallback，兼容 World Partition / 旧版 API）。
    """
    import unreal

    attempts = (
        lambda: unreal.EditorLevelLibrary.save_current_level(),
        lambda: unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).save_current_level(),
        lambda: unreal.EditorLoadingAndSavingUtils.save_dirty_packages(True, True),
    )
    for fn in attempts:
        try:
            ok = bool(fn())
            print(f"  [MRQ] 已保存关卡（stencil 值持久化）: {ok}")
            return
        except Exception as e:
            print(f"  [MRQ] 保存关卡尝试失败: {e}")
    print("  WARNING: 无法保存当前关卡，stencil 值可能不带入 PIE")


def create_stencil_to_color_material(
    package_path: str = "/Game/FutsalMOT/Materials", force: bool = False
) -> Optional[str]:
    """创建 SceneTexture(PPI_CUSTOM_STENCIL) → EmissiveColor 的 post-process 材质。

    用于把每个实体的 Custom Depth Stencil 值（= mask_id，1..11）渲染为 mask 图：
    mask 像素值 == stencil 值，P1 侧可直接按 mask_id 解码（确定性、无 hash 歧义）。

    已确认 UE 5.8 的 SceneTextureId 枚举含 PPI_CUSTOM_STENCIL；本函数显式用它并读回验证。
    force=True 时总是删除重建（用于修复之前读错纹理的旧材质）。
    返回材质资产路径；创建失败返回 None（可手动建材质并填 post_process_material）。
    """
    import traceback

    import unreal

    asset_name = "M_StencilToID"
    asset_path = f"{package_path}/{asset_name}"
    if unreal.EditorAssetLibrary.does_asset_exist(asset_path) and not force:
        if _material_is_valid(asset_path):
            return asset_path  # 复用已有正确材质
        # 校验未通过（domain/SceneTexture 纹理不对），就地重建
    if not unreal.EditorAssetLibrary.does_directory_exist(package_path):
        unreal.EditorAssetLibrary.make_directory(package_path)
    # 存在时就地清空表达式重建（避免 delete+create_asset 触发"覆写现有Object"对话框）
    if unreal.EditorAssetLibrary.does_asset_exist(asset_path):
        mat = unreal.load_asset(asset_path)
        for ex in list(unreal.MaterialEditingLibrary.get_material_expressions(mat)):
            try:
                unreal.MaterialEditingLibrary.delete_material_expression(mat, ex)
            except Exception:
                pass
    else:
        factory = unreal.MaterialFactoryNew()
        mat = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
            asset_name, package_path, None, factory
        )
        if mat is None:
            print("  ERROR: create_asset 返回 None（MaterialFactoryNew 不可用？）")
            return None
    try:
        mat.set_editor_property("material_domain", unreal.MaterialDomain.MD_POST_PROCESS)
        stex = unreal.MaterialEditingLibrary.create_material_expression(
            mat, unreal.MaterialExpressionSceneTexture, -400.0, 0.0
        )
        tid = getattr(unreal.SceneTextureId, "PPI_CUSTOM_STENCIL", None)
        if tid is None:
            print("  WARNING: 未找到 SceneTextureId.PPI_CUSTOM_STENCIL")
        else:
            # 属性名是 scene_texture_id（不是 texture_id）——UE 5.8 实测
            stex.set_editor_property("scene_texture_id", tid)
        # CustomStencil.R / 255.0 → Emissive。直接输出会把 stencil 值(0~255)塞进
        # Emissive(0~1) 导致饱和，先除以 255 归一化。
        const255 = unreal.MaterialEditingLibrary.create_material_expression(
            mat, unreal.MaterialExpressionConstant, -200.0, -100.0
        )
        try:
            const255.set_editor_property("r", 255.0)
        except Exception:
            pass
        divide = unreal.MaterialEditingLibrary.create_material_expression(
            mat, unreal.MaterialExpressionDivide, -200.0, 0.0
        )
        unreal.MaterialEditingLibrary.connect_material_expressions(stex, "R", divide, "A")
        for out_name in ("", "Out", "output"):
            try:
                unreal.MaterialEditingLibrary.connect_material_expressions(const255, out_name, divide, "B")
                break
            except Exception:
                continue
        try:
            unreal.MaterialEditingLibrary.connect_material_property(
                divide, "Out", unreal.MaterialProperty.MP_EMISSIVE_COLOR
            )
        except Exception:
            unreal.MaterialEditingLibrary.connect_material_property(
                divide, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR
            )
        unreal.EditorAssetLibrary.save_loaded_asset(mat)
        return asset_path
    except Exception:
        print("  ERROR: 创建 stencil 材质失败：")
        traceback.print_exc()
        return None


def _material_is_valid(asset_path: str) -> bool:
    """校验 M_StencilToID 材质：PostProcess 域 + SceneTexture(PPI_CUSTOM_STENCIL) + Divide(÷255)。"""
    import unreal

    try:
        mat = unreal.load_asset(asset_path)
        if mat.get_editor_property("material_domain") != unreal.MaterialDomain.MD_POST_PROCESS:
            return False
        has_stencil_tex = False
        has_divide = False
        for ex in unreal.MaterialEditingLibrary.get_material_expressions(mat):
            tn = type(ex).__name__
            if "SceneTexture" in tn:
                val = ex.get_editor_property("scene_texture_id")
                if "PPI_CUSTOM_STENCIL" in str(val):
                    has_stencil_tex = True
            if "Divide" in tn:
                has_divide = True
        return has_stencil_tex and has_divide
    except Exception:
        return False


def _assign_custom_stencil(actors: dict) -> None:
    """为每个实体 actor 的 primitive 组件设置 Custom Depth Stencil 值 = mask_id。

    mask_id 由 annotation_utils.entity_id_to_mask_id 确定性映射（L0..L4→1..5、
    R0..R4→6..10、BALL→11）。CustomDepthStencilValue / bRenderCustomDepth 是
    EditAnywhere UPROPERTY，设置 + actor.modify() 后会随 PIE 实例化带入渲染。
    """
    import unreal

    if not actors:
        return
    assigned = 0
    verified = 0
    for entity_id, actor in actors.items():
        try:
            mask_id = entity_id_to_mask_id(entity_id)
        except (ValueError, TypeError):
            continue
        # 用具体组件类遍历（PrimitiveComponent 抽象基类在部分 UE 版本不可用于
        # get_components_by_class），确保不遗漏任何会写入 custom depth 的组件。
        comps = []
        for cls in (
            unreal.SkeletalMeshComponent,
            unreal.StaticMeshComponent,
            unreal.CapsuleComponent,
        ):
            try:
                comps.extend(actor.get_components_by_class(cls))
            except Exception:
                continue
        if not comps:
            continue
        for comp in comps:
            set_stencil = False
            try:
                comp.set_editor_property("render_custom_depth", True)
                set_stencil = True
            except Exception as e:
                print(f"    WARNING: {entity_id} 设置 render_custom_depth 失败: {e}")
            try:
                comp.set_editor_property("custom_depth_stencil_value", int(mask_id))
                set_stencil = True
            except Exception:
                m = getattr(comp, "set_custom_depth_stencil_value", None)
                if m is not None:
                    try:
                        m(int(mask_id))
                        set_stencil = True
                    except Exception:
                        pass
            if set_stencil:
                assigned += 1
                # 读回验证：确认 render_custom_depth / custom_depth_stencil_value 真正生效
                try:
                    rd = comp.get_editor_property("render_custom_depth")
                    sv = comp.get_editor_property("custom_depth_stencil_value")
                    if bool(rd) and int(sv) == int(mask_id):
                        verified += 1
                    else:
                        print(
                            f"    WARNING: {entity_id} 组件读回不一致"
                            f"（render_custom_depth={rd}, stencil={sv}, 期望 {mask_id}）"
                        )
                except Exception:
                    pass
        try:
            actor.modify()
        except Exception:
            pass
    print(
        f"  [MRQ] 已为 {assigned} 个组件设置 Custom Depth Stencil 值"
        f"（读回验证通过 {verified}）"
    )


def _build_mask_job(
    queue,
    seq_asset_path: str,
    map_path: str,
    mask_dir: Path,
    image_width: int,
    image_height: int,
    frame_rate: int,
    file_name_format: str,
    zero_pad: int,
    mask_source: str = "object_id_pass",
    post_process_material: str = None,
    preset=None,
    cv_gt=None,
):
    """构建渲染 Instance-ID Mask 的 MRQ job（输出到 mask_dir）。

    mask_source:
      "object_id_pass"（默认）—— 添加 MoviePipelineObjectIdRenderPass（UE 5.8 存在），
        配合每实体 Custom Depth Stencil 值（_assign_custom_stencil）输出 Instance-ID。
        最佳实践需设置 IdType=Actor 并开启 stencil 模式；不同 UE 版本属性名不同，
        这里做 best-effort 并打印成员供校准。
      "post_process_material" —— 用 MoviePipelineDeferredPassBase + 用户提供的
        stencil→颜色 材质（post_process_material 资产路径），输出为单张 mask 图。
    分辨率 / 帧率 / 帧范围与 RGB job 一致 → 帧号严格 1:1 对齐。

    注意：用 queue.allocate_new_job 加入队列（与 RGB job 一致——本 UE 5.8 的
    MoviePipelineQueue 无 jobs 属性，queue.jobs.add 会报错）；配置失败时用
    delete_job 移除，避免留下 0 pass 的孤儿 job 被 executor 渲染。
    """
    import unreal

    job = queue.allocate_new_job(unreal.MoviePipelineExecutorJob)
    try:
        _configure_mask_job(job, seq_asset_path, map_path, mask_dir, image_width,
                            image_height, frame_rate, file_name_format, zero_pad,
                            mask_source, post_process_material, preset, cv_gt)
    except Exception:
        _delete_queue_job(queue, job)
        raise
    return job


def _delete_queue_job(queue, job) -> None:
    """从 MRQ 队列移除一个 job（best-effort，不同 UE 版本方法名不同）。"""
    for m in ("delete_job", "remove_job"):
        fn = getattr(queue, m, None)
        if fn is None:
            continue
        try:
            fn(job)
            return
        except Exception:
            continue
    print(f"  WARNING: 无法从队列移除配置失败的 job（{job.get_name()}）")


def _configure_mask_job(
    job,
    seq_asset_path: str,
    map_path: str,
    mask_dir: Path,
    image_width: int,
    image_height: int,
    frame_rate: int,
    file_name_format: str,
    zero_pad: int,
    mask_source: str,
    post_process_material: str,
    preset=None,
    cv_gt=None,
) -> None:
    """配置单个 mask job（已入队）：输出设置 + 渲染 pass。配置失败时抛错。"""
    import unreal

    job.sequence = unreal.SoftObjectPath(seq_asset_path)
    if map_path:
        job.map = unreal.SoftObjectPath(map_path)

    config = _get_job_config(job)
    output = config.find_or_add_setting_by_class(unreal.MoviePipelineOutputSetting)
    output.output_directory = unreal.DirectoryPath(str(mask_dir))
    output.file_name_format = file_name_format
    output.zero_pad_frame_numbers = zero_pad
    output.output_resolution = unreal.IntPoint(image_width, image_height)
    try:
        output.frame_rate = unreal.FrameRate(frame_rate, 1)
    except Exception:
        pass
    # 输出格式：object_id_pass（Cryptomatte）用 multilayer EXR；post_process_material 用 PNG
    if mask_source == "object_id_pass":
        _add_exr_output(config)
    else:
        config.find_or_add_setting_by_class(unreal.MoviePipelineImageSequenceOutput_PNG)

    if mask_source == "post_process_material":
        # stencil 方案需要渲染期间开启 Custom Depth-Stencil（r.CustomDepth=3）
        _enable_custom_depth_in_job(config)
        if not post_process_material:
            raise RuntimeError(
                "post_process_material 未设置：请填 stencil→颜色 材质资产路径"
                "（或确认 create_stencil_to_color_material 自动创建成功）"
            )
        mat = unreal.load_asset(post_process_material)
        if mat is None:
            raise RuntimeError(
                f"post_process_material 资产不存在: {post_process_material}"
            )
        # DeferredPassBase：负责场景渲染，custom depth/stencil 在此写入。
        config.find_or_add_setting_by_class(unreal.MoviePipelineDeferredPassBase)
        # 独立 MoviePipelinePostProcessPass：直接输出材质结果（读取 custom stencil）。
        # 不再用 additional_post_process_materials（结构体数组）——它输出的
        # FinalImageM_StencilToID.* 实测全 0，怀疑其不在带 custom depth 的通道里渲染。
        pp_cls = getattr(unreal, "MoviePipelinePostProcessPass", None)
        if pp_cls is None:
            raise RuntimeError("MoviePipelinePostProcessPass 不存在")
        try:
            pp_pass = config.find_or_add_setting_by_class(pp_cls)
            pp_pass.set_editor_property("material", mat)
            pp_pass.set_editor_property("enabled", True)
            print(
                f"  [MRQ] mask pass: 独立 MoviePipelinePostProcessPass + {post_process_material}"
            )
        except Exception as e:
            raise RuntimeError(f"配置 MoviePipelinePostProcessPass 失败: {e}")
    else:
        # 默认 object_id_pass：MoviePipelineObjectIdRenderPass
        added = False
        for cls_name in ("MoviePipelineObjectIdRenderPass",):
            cls = getattr(unreal, cls_name, None)
            if cls is None:
                continue
            try:
                pass_setting = config.find_or_add_setting_by_class(cls)
                _configure_object_id_pass(pass_setting)
                added = True
                break
            except Exception as e:
                print(f"  [MRQ] 添加 {cls_name} 失败: {e}")
        if not added:
            _print_mrq_members(config, "config")
            _list_mrq_classes()
            raise RuntimeError(
                "无法添加 MoviePipelineObjectIdRenderPass 渲染 Instance-ID Mask。"
                "已打印可用 MRQ 类与 config 成员，请把 [MRQ 诊断] 输出贴回；或改用"
                " instance_mask.mask_source='post_process_material'（需提供 stencil→颜色"
                " 材质资产路径）。"
            )

    # CV GT preset：mask job 只应用时间确定性（is_mask=True），保证 ID 也是单时刻
    _apply_mrq_preset(config, preset, cv_gt, is_mask=True)

    if not _set_job_config(job, config):
        _print_mrq_members(job, "job")
        raise RuntimeError("无法把 mask job 配置挂到 job")


def _add_exr_output(config) -> None:
    """为 mask job 添加 multilayer EXR 输出（Object ID / Cryptomatte 需要）。"""
    import unreal

    added = False
    for cls_name in ("MoviePipelineImageSequenceOutput_EXR", "MoviePipelineImageSequenceOutput_OpenEXR"):
        cls = getattr(unreal, cls_name, None)
        if cls is None:
            continue
        try:
            exr = config.find_or_add_setting_by_class(cls)
            try:
                exr.set_editor_property("multilayer", True)
                print("  [MRQ] EXR 输出: multilayer=True")
            except Exception:
                pass
            added = True
            break
        except Exception as e:
            print(f"  [MRQ] 添加 {cls_name} 失败: {e}")
    if not added:
        raise RuntimeError("无法添加 EXR 输出（MoviePipelineImageSequenceOutput_EXR）")


def _enable_custom_depth_in_job(config) -> None:
    """在 MRQ job 配置中加入 r.CustomDepth=3（Enable with Stencil）。

    编辑器里的控制台命令不会自动带入 PIE 渲染（MRQ 渲染开始时按项目设置重置 cvar），
    必须把 cvar 写进 job 的 MoviePipelineConsoleVariableSetting，确保渲染期间
    custom depth/stencil 开启、各实体 stencil 值能被材质读到。
    """
    import unreal

    cls = getattr(unreal, "MoviePipelineConsoleVariableSetting", None)
    if cls is None:
        print("  WARNING: 无 MoviePipelineConsoleVariableSetting，无法在 job 内开启 Custom Depth-Stencil")
        return
    try:
        cvar = config.find_or_add_setting_by_class(cls)
    except Exception as e:
        print(f"  WARNING: 添加 MoviePipelineConsoleVariableSetting 失败: {e}")
        return
    entry_cls = getattr(unreal, "MoviePipelineConsoleVariableEntry", None)
    if entry_cls is None:
        print("  WARNING: 无 MoviePipelineConsoleVariableEntry，无法设置 cvar")
        return
    entry_struct = entry_cls()
    if not _set_console_entry(entry_struct, "r.CustomDepth", 3.0):
        print("  WARNING: 无法设置 MoviePipelineConsoleVariableEntry 字段，成员如下：")
        _print_mrq_members(entry_struct, "console_entry")
        return
    prop = "cvars"  # 由报错确认：属性名 cvars，元素类型 MoviePipelineConsoleVariableEntry
    try:
        current = list(cvar.get_editor_property(prop) or [])
    except Exception:
        current = []
    if not any("r.CustomDepth" in str(x) for x in current):
        current.append(entry_struct)
    try:
        cvar.set_editor_property(prop, current)
        print("  [MRQ] job 内设置 r.CustomDepth=3（Custom Depth-Stencil 开启）")
    except Exception as e:
        print(f"  WARNING: 设置 r.CustomDepth 3 失败: {e}")
        _print_mrq_members(entry_struct, "console_entry")


def _set_console_entry(entry_struct, name: str, value) -> bool:
    """best-effort 设置 cvar 结构体字段（name/value/is_enabled，字段名因版本而异）。"""
    ok = False
    for name_prop in ("name", "cvar_name", "console_variable_name"):
        try:
            entry_struct.set_editor_property(name_prop, name)
            ok = True
            break
        except Exception:
            continue
    for value_prop in ("value", "cvar_value"):
        try:
            entry_struct.set_editor_property(value_prop, float(value))
            ok = True
            break
        except Exception:
            try:
                entry_struct.set_editor_property(value_prop, str(value))
                ok = True
                break
            except Exception:
                continue
    for enabled_prop in ("is_enabled", "b_is_enabled", "enabled"):
        try:
            entry_struct.set_editor_property(enabled_prop, True)
            break
        except Exception:
            continue
    return ok


def _configure_object_id_pass(pass_setting) -> None:
    """配置 MoviePipelineObjectIdRenderPass：IdType=Actor（逐 actor 分组）。"""
    import unreal

    id_type_enum = getattr(unreal, "MoviePipelineObjectIdPassIdType", None)
    if id_type_enum is not None:
        for name in ("ACTOR", "ACTOR_WITH_HIERARCHY"):
            v = getattr(id_type_enum, name, None)
            if v is not None:
                try:
                    pass_setting.set_editor_property("id_type", v)
                    print(f"  [MRQ] ObjectId pass: id_type={name}")
                except Exception:
                    pass
                break


def _submit_render(subsystem, executor) -> bool:
    """提交渲染当前队列。不同 UE 版本子系统方法名可能不同。

    注意：UE 5.8 的 render_queue_with_executor_instance(executor) 只接受 1 个参数，
    渲染的是子系统当前队列（job 需先通过 queue.allocate_new_job 加入）。
    """
    for method_name in (
        "render_queue_with_executor_instance",
        "render_queue_with_executor",
        "render_queue_instance_with_executor",
        "render_queue_instance_with_executor_instance",
    ):
        m = getattr(subsystem, method_name, None)
        if m is None:
            continue
        try:
            m(executor)
            return True
        except Exception as e:
            print(f"  [MRQ] {method_name} 调用失败: {e}")
            continue
    _print_mrq_members(subsystem, "subsystem")
    _print_mrq_members(executor, "executor")
    return False


# ── MRQ 异步渲染状态机 ─────────────────────────────────────────────────

class _AsyncRenderPipeline:
    """MRQ 异步渲染状态机：提交后立即返回，由 finished/error delegate 驱动。

    阶段：
      start()    清空队列、把所有 Sequence 加入同一个 MRQ queue、绑定 delegate、
                 提交渲染并立即返回（不阻塞编辑器主线程）。
      finished   MRQ 整批渲染完成后触发：检查结果、复制 RGB 到 img1/、
                 写 render_summary.json 完成标记。
      errored    记录 MRQ 错误信息（最终成败由 finished 委托的 success 决定）。

    关键：绝不在本模块内用 time.sleep / Event.wait 阻塞主线程等待 MRQ 完成。
    MoviePipelinePIEExecutor 需要在编辑器主线程持续 tick 才能推进 PIE 窗口，
    任何同步阻塞都会让渲染卡死（上一版本 _render_and_wait 的根因）。
    """

    def __init__(self, jobs, episode_id, output_dir, keep_indices):
        self.jobs = jobs  # [{name, cam_id, seq_asset_path, cam_out, render_dir, img1_dir, job}]
        self.episode_id = episode_id
        self.output_dir = Path(output_dir)
        self.keep_indices = list(keep_indices)
        self.error_messages = []
        self.executor = None
        self.finished = False
        self._watch_handle = None  # slate post-tick 兜底检测的句柄
        self._watch_state = None

    def summary_path(self) -> Path:
        """渲染完成标记（render_summary.json）的路径。"""
        return self.output_dir / self.episode_id / "render_summary.json"

    # ── 生命周期 ──────────────────────────────────────────────────

    def start(self, subsystem, queue, executor, map_path,
              image_width, image_height, frame_rate,
              file_name_format, zero_pad) -> None:
        """清空队列、构建并加入全部 job、绑定 delegate、提交渲染后立即返回。"""
        _clear_queue(queue)

        prepared = []
        for info in self.jobs:
            try:
                if info.get("kind") == "mask":
                    job = _build_mask_job(
                        queue, info["seq_asset_path"], map_path, info["render_dir"],
                        image_width, image_height, frame_rate, file_name_format, zero_pad,
                        mask_source=info.get("mask_source", "custom_depth_pass"),
                        post_process_material=info.get("post_process_material"),
                        preset=info.get("preset"), cv_gt=info.get("cv_gt"),
                    )
                else:
                    job = _build_mrq_job(
                        queue, info["seq_asset_path"], map_path, info["render_dir"],
                        image_width, image_height, frame_rate, file_name_format, zero_pad,
                        preset=info.get("preset"), cv_gt=info.get("cv_gt"),
                    )
            except Exception as e:
                print(f"  ERROR: 构建 MRQ job 失败 {info['name']}: {e}")
                self.error_messages.append(f"{info['name']}: {e}")
                continue
            info["job"] = job
            prepared.append(info)
        if not prepared:
            self._finalize(failed=True, reason="没有任何可提交的 MRQ job")
            return
        self.jobs = prepared

        self.executor = executor

        # 绑定 delegate：MRQ 完成 / 出错时驱动后续阶段（而非同步等待）
        for delegate_name, cb in (
            ("on_executor_finished_delegate", self._on_finished),
            ("on_executor_errored_delegate", self._on_errored),
        ):
            delegate = getattr(executor, delegate_name, None)
            if delegate is None:
                print(f"  WARNING: executor 无 {delegate_name}，无法绑定回调")
                continue
            try:
                delegate.add_callable(cb)
                print(f"  已绑定 {delegate_name}")
            except Exception as e:
                print(f"  WARNING: 绑定 {delegate_name} 失败: {e}")

        if not _submit_render(subsystem, executor):
            self._finalize(failed=True,
                           reason="无法提交 MRQ 渲染（render_queue_with_executor_instance 等均失败）")
            return

        print("\n=== MRQ 渲染已异步提交（不阻塞编辑器主线程）===")
        print(f"  Sequence 数: {len(self.jobs)}，全部加入同一个 MRQ queue")
        for info in self.jobs:
            print(f"    - {info['name']} -> {info['img1_dir']}")
        print(f"  渲染完成后自动：复制 RGB -> img1/，写完成标记 {self.summary_path()}")
        print("  请保持编辑器运行；完成后控制台会打印结果汇总。")

        # 兜底：finished delegate 不触发的 UE 版本也能完成收尾（非阻塞轮询）
        self._start_completion_watchdog()

    # ── delegate 回调（游戏线程触发，可安全访问 unreal 与文件系统）──

    def _on_finished(self, executor, success):
        """MRQ 整批渲染结束。签名与 on_executor_finished_delegate 一致：(executor, success)。

        UE 5.8 的 delegate 绑定会检查回调签名：必须是显式参数，不能用 *args（变参
        计为 0 个形参，报 "Callable has the incorrect number of arguments
        (expected 2, got 0)"，导致绑定失败、回调永不触发）。
        """
        if self.finished:
            return
        try:
            print(f"  [MRQ] on_executor_finished_delegate 回调触发（success={success}）")
            if not success:
                self._finalize(failed=True, reason="MRQ finished 委托报告 success=False")
                return
            self._copy_and_finalize()
        except Exception as e:
            # 回调异常必须可见（UE 默认吞掉 delegate 异常）。watchdog / 手动入口
            # 仍可收尾，但打印 traceback 便于定位。
            print(f"  [MRQ 回调异常] _on_finished: {e}")
            traceback.print_exc()

    def _on_errored(self, executor, pipeline, is_fatal, errors):
        """MRQ 渲染出错（不中断最终收尾，成败由 finished 委托决定）。

        签名与 on_executor_errored_delegate 一致：
        (executor, pipeline, is_fatal, errors)。errors 为 MRQ 控制台输出条目列表。
        """
        try:
            try:
                entries = list(errors)
            except TypeError:
                entries = [errors]
            parts = [str(e) for e in entries if str(e)]
            if not parts:
                parts.append(str(errors))
            self.error_messages.append(" | ".join(parts))
            print(f"  [MRQ 错误] {' | '.join(parts)}")
        except Exception as e:
            print(f"  [MRQ 回调异常] _on_errored: {e}")
            traceback.print_exc()

    # ── 完成检测兜底（非阻塞 watchdog）──────────────────────────────

    def _start_completion_watchdog(self):
        """注册 slate post-tick 兜底：finished delegate 不触发时也能收尾。

        兜底：delegate 绑定须用显式签名，若绑定失败（UE 5.8 拒绝 *args 变参，
        报 "incorrect number of arguments"），finished 回调不触发，由 watchdog
        完成收尾。watchdog 每编辑帧 tick 一次，全部为非阻塞轮询（绝不 sleep），
        判定条件（满足其一即收尾）：
          1. executor 不再渲染 且 所有 camera 的目标帧已输出；
          2. is_rendering API 不可用 且 目标帧齐全 且 文件数 ~20s 无变化；
          3. 硬超时 30 分钟（按现有文件收尾）。
        与 delegate 幂等：_copy_and_finalize / _finalize 有 self.finished 守卫。
        """
        import unreal

        self._watch_state = {
            "last_total": -1,
            "stable_ticks": 0,
            "start": time.monotonic(),
            "seen_rendering": False,
        }

        def _on_tick(_delta):
            if self.finished:
                return False  # 已收尾，注销本回调
            st = self._watch_state
            rendering = self._executor_rendering()
            present = self._all_keep_frames_present()
            total = self._total_render_files()
            if total == st["last_total"]:
                st["stable_ticks"] += 1
            else:
                st["stable_ticks"] = 0
            st["last_total"] = total
            if rendering is True:
                st["seen_rendering"] = True
            elapsed = time.monotonic() - st["start"]

            if rendering is False and present:
                print("  [MRQ] watchdog：executor 不再渲染且目标帧齐全，开始收尾")
                self._copy_and_finalize()
                return False
            if rendering is None and present and st["stable_ticks"] >= 1200:
                print("  [MRQ] watchdog：文件数稳定兜底，开始收尾")
                self._copy_and_finalize()
                return False
            # 硬超时：仅当渲染已停止推进（文件数长时间无变化）才收尾。
            # 长序列 soak（900 帧×2 job×4 相机 ≈ 7200 帧）渲染可能超过 30 分钟，
            # 只要文件仍在增长就不应强制收尾，避免把进行中的渲染误判为 partial。
            if elapsed > 1800.0 and st["stable_ticks"] >= 600:
                print("  WARNING: MRQ 渲染超时且文件数长时间无变化，按当前文件收尾")
                self._copy_and_finalize()
                return False
            return True

        try:
            handle = unreal.register_slate_post_tick_callback(_on_tick)
        except Exception as e:
            print(f"  WARNING: 无法注册 slate post-tick 兜底检测: {e}（依赖 finished delegate）")
            self._watch_handle = None
            return
        self._watch_handle = handle
        print("  已注册非阻塞完成检测（watchdog，每编辑帧检查一次）")

    def _executor_rendering(self) -> Optional[bool]:
        """executor 是否正在渲染。is_rendering API 不可用时返回 None。"""
        m = getattr(self.executor, "is_rendering", None)
        if m is None:
            return None
        return bool(m())

    def _all_keep_frames_present(self) -> bool:
        """所有 camera 的目标渲染帧（keep_indices）是否已输出到 render/（mask 用 find_mask_files）。"""
        for info in self.jobs:
            if info.get("kind") == "mask":
                available = find_mask_files(info["render_dir"])
            else:
                available = find_rendered_frame_numbers(info["render_dir"])
            if not all(n in available for n in self.keep_indices):
                return False
        return True

    def _total_render_files(self) -> int:
        """所有 camera render/ 目录的 PNG + EXR 总数。"""
        return sum(
            len(list(info["render_dir"].rglob("*.png")))
            + len(list(info["render_dir"].rglob("*.exr")))
            for info in self.jobs
        )

    # ── 收尾：复制 RGB + 轻量校验 + 写完成标记 ─────────────────────

    def _copy_and_finalize(self):
        """finished 成功后：逐 camera 复制 RGB（img1/）与统计 Instance-ID Mask 对齐。"""
        if self.finished:
            return
        per_camera = {}
        total_copied = 0
        total_mask_copied = 0
        expected = len(self.keep_indices)
        for info in self.jobs:
            cam_id = info["cam_id"]
            render_dir, out_dir = info["render_dir"], info["img1_dir"]
            is_mask = info.get("kind") == "mask"
            if is_mask:
                copied = copy_mask_frames(render_dir, out_dir, self.keep_indices)
            else:
                copied = copy_rendered_frames(render_dir, out_dir, self.keep_indices)
            if is_mask:
                total_mask_copied += copied
            else:
                total_copied += copied
            entry = per_camera.setdefault(cam_id, {
                "sequence": info["name"],
                "expected_frames": expected,
                "ok": True,
            })
            if is_mask:
                entry["mask_frames"] = copied
                # Object ID EXR 源：记录来源；mask/*.png 由 P1 `cryptomatte-to-mask` 生成
                mask_srcs = [p.suffix.lower() for p in find_mask_files(render_dir).values()]
                is_exr = bool(mask_srcs) and all(s == ".exr" for s in mask_srcs)
                entry["mask_source"] = "object_id_exr" if is_exr else "png"
                label = "mask(EXR) 对齐" if is_exr else "mask/写入"
            else:
                entry["img1_frames"] = copied
                label = "img1/写入"
                ann_count = self._check_annotation_frame_count(info["cam_out"])
                if ann_count is not None:
                    entry["annotations_jsonl_frames"] = ann_count
                    entry["annotation_img1_match"] = ann_count == copied
            entry["ok"] = entry["ok"] and (copied == expected)
            mark = "MISSING" if copied == 0 else ("OK" if copied == expected else "PARTIAL")
            print(f"  [{mark}] {cam_id}: {label} {copied}/{expected} 帧")

        if total_copied == 0:
            status, reason = "failed", "渲染未产生任何可用的 RGB / mask 帧"
        elif all(e["ok"] for e in per_camera.values()):
            status, reason = "success", None
        else:
            status, reason = "partial", "部分 camera 的渲染帧数与预期不符（可能缺帧）"
        self._finalize(failed=(status == "failed"), status=status,
                       per_camera=per_camera, total_copied=total_copied,
                       total_mask_frames=total_mask_copied, reason=reason)

    def _check_annotation_frame_count(self, cam_out: Path) -> Optional[int]:
        """轻量校验：annotations.jsonl 行数 vs img1/ PNG 数。

        一致返回 None；不一致返回 annotations.jsonl 帧数；无标注文件返回 None。
        纯文件检查，不导入 P1 代码（保持 UE/P1 运行时隔离）。
        """
        ann_path = cam_out / "annotations.jsonl"
        img1_dir = cam_out / "img1"
        if not ann_path.exists() or not img1_dir.exists():
            return None
        try:
            ann_count = sum(1 for _ in open(ann_path, encoding="utf-8"))
        except OSError:
            return None
        img_count = len(list(img1_dir.glob("*.png")))
        return ann_count if ann_count != img_count else None

    def _finalize(self, failed, status=None, per_camera=None,
                  total_copied=0, total_mask_frames=0, reason=None):
        """写完成标记 render_summary.json，并释放模块级引用（只执行一次）。"""
        global _ACTIVE_RENDER
        if self.finished:
            return
        self.finished = True
        _ACTIVE_RENDER = None

        # 注销 watchdog（若有）
        if self._watch_handle is not None:
            try:
                import unreal
                unreal.unregister_slate_post_tick_callback(self._watch_handle)
            except Exception:
                pass
            self._watch_handle = None

        status = status or ("failed" if failed else "success")
        summary = {
            "episode_id": self.episode_id,
            "status": status,
            "reason": reason,
            "total_img1_frames": total_copied,
            "total_mask_frames": total_mask_frames,
            "cameras": per_camera or {},
            "mrq_errors": self.error_messages or None,
            "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "note": "MRQ 渲染完成回调写入。完整标注校验请在 P1 运行："
                    "uv run grf-ue validate-annotations <output_dir>",
        }
        write_json_atomic(self.summary_path(), summary)

        print(f"\n=== MRQ 渲染{'失败' if failed else '完成'}: {status} ===")
        print(f"  完成标记: {self.summary_path()}")
        if failed:
            print(f"  原因: {reason}")
        print(f"  后续步骤（P1）：uv run grf-ue validate-annotations {self.output_dir}")


def render_sequences(
    sequences_cfg,
    annotation_cfg: dict,
    sequence_package_path: str,
    episode_dir: Path,
    output_dir: Path,
    mapping_path: Path = None,
) -> None:
    """异步用 MRQ 渲染所有 Sequence 的 RGB 帧（+ Instance-ID Mask），提交后立即返回。

    sequences_cfg: ue profile 的 sequences 列表（每个含 name/camera_actor）。
    annotation_cfg: annotation_export 配置（含 render_rgb 段与 instance_mask 段）。
    mapping_path: actor 映射文件路径（instance_mask.enabled 时需要，用于给 actor 打 stencil）。

    渲染完成后（MRQ finished 回调，或 slate post-tick watchdog 兜底）自动：
      - 复制 RGB 帧到每个 camera 的 img1/（按帧同步契约取帧）。
      - instance_mask.enabled 时，额外渲染 mask 到 render_mask/ 并复制为 mask/（与 img1/ 同帧号）。
      - 写 <output_dir>/<episode_id>/render_summary.json 完成标记。
    本函数不阻塞编辑器主线程。
    """
    import unreal

    global _ACTIVE_RENDER
    if _ACTIVE_RENDER is not None:
        print("WARNING: 已有 MRQ 渲染正在进行，忽略本次请求（等待上一批完成）")
        return

    render_cfg = annotation_cfg.get("render_rgb") or {}
    if not render_cfg.get("enabled", False):
        print("render_rgb.enabled = false，跳过渲染")
        return

    # CV GT preset：缺省 cv_gt（deterministic），null/cinematic 为保留/不覆盖模式
    preset, cv_gt = resolve_preset(render_cfg)
    if preset:
        print(f"  [MRQ] render preset = {preset}")
    else:
        print("  [MRQ] render preset = null（不覆盖，保持关卡/相机现状）")

    mask_cfg = annotation_cfg.get("instance_mask") or {}
    mask_enabled = bool(mask_cfg.get("enabled", False))
    mask_source = mask_cfg.get("mask_source", "object_id_pass")
    post_process_material = mask_cfg.get("post_process_material")
    if mask_enabled:
        if mask_source == "post_process_material":
            # stencil 方案（fallback，实测本 5.8 不可用）需要 custom depth + actor stencil 值
            _ensure_custom_depth_stencil_enabled()
            if mapping_path is not None and Path(mapping_path).exists():
                mapping = load_mapping(Path(mapping_path))
                actors = find_all_actors(mapping)
                _assign_custom_stencil(actors)
                _save_current_level()
            else:
                print("  WARNING: instance_mask.enabled=true 但未提供有效 mapping_path，无法设置 stencil")
            if not post_process_material:
                post_process_material = create_stencil_to_color_material()
                if post_process_material:
                    print(f"  [MRQ] 自动创建 stencil→颜色 材质: {post_process_material}")
                else:
                    print("  WARNING: 自动创建 stencil 材质失败，请手动建材质并填 post_process_material")
        # object_id_pass（默认）：Object ID + Cryptomatte EXR，无需 custom stencil
    else:
        print("instance_mask.enabled = false，跳过 mask 渲染（仅 RGB）")

    image_width = int(
        render_cfg.get("output_resolution_x") or annotation_cfg.get("image_width", 1920)
    )
    image_height = int(
        render_cfg.get("output_resolution_y") or annotation_cfg.get("image_height", 1080)
    )
    frame_rate = int(
        render_cfg.get("frame_rate") or annotation_cfg.get("playback_fps") or 30
    )
    file_name_format = render_cfg.get("file_name_format", "{frame_number}")
    zero_pad = int(render_cfg.get("zero_pad_frame_numbers", 6))

    meta, frames = load_episode(episode_dir)
    # 首帧 spawn 状态烘焙：MRQ/PIE 中 possessable actor 的第 0 帧可能尚未被 Level
    # Sequence 接管，渲染成**关卡放置的默认位置**（与相机同一类问题，相机已用
    # _add_camera_transform_track 静态烘焙解决，球员没有）。先把 actor 设到第 0 帧
    # 并保存关卡，使 PIE spawn 状态 == Sequence 帧 0 值——无论接管是否滞后一帧，
    # 帧 0 都不会闪回默认位置（第 3/6 帧由 Sequence 接管，本就正确）。
    if frames and mapping_path is not None:
        try:
            mapping = load_mapping(Path(mapping_path))
            actors = find_all_actors(mapping)
            if actors:
                gk_ids = gk_entity_ids_from_meta(meta)
                apply_preview_frame(actors, frames[0], {}, gk_entity_ids=gk_ids)
                _save_current_level()
                print("  [MRQ] 首帧 spawn 状态已烘焙（actor 设到帧 0 并保存关卡）")
        except Exception as e:  # noqa: BLE001
            print(f"  WARNING: 首帧 spawn 状态烘焙失败（帧 0 可能渲染成关卡默认位置）: {e}")
    episode_id = meta.get("episode_id") or episode_dir.name
    source_step = float(meta["timing"].get("source_step_seconds", 0.1))
    keep_indices = select_rendered_frame_indices(len(frames), source_step, frame_rate)

    if not sequences_cfg:
        print("WARNING: sequences 为空，无可渲染的 Sequence")
        return

    # CV GT preset：先把确定性后处理写到每个 Camera（覆盖关卡 Post Process Volume）
    _apply_cv_gt_camera_post_process(sequences_cfg, preset, cv_gt)

    # 收集要渲染的 Sequence（资产必须已存在）
    jobs = []
    for seq_entry in sequences_cfg:
        seq_name = seq_entry.get("name")
        cam_id = seq_entry.get("camera_actor") or seq_name
        if not seq_name:
            continue
        seq_asset_path = f"{sequence_package_path}/{seq_name}"
        if not unreal.EditorAssetLibrary.does_asset_exist(seq_asset_path):
            print(f"  WARNING: Sequence 不存在，跳过: {seq_asset_path}（请先 --mode sequence 创建）")
            continue
        cam_out = Path(output_dir) / episode_id / cam_id
        render_dir = cam_out / "render"
        img1_dir = cam_out / "img1"
        ensure_dir(render_dir)
        _clear_dir(render_dir)  # 清空上一次渲染残留，避免旧帧混入
        jobs.append({
            "name": seq_name,
            "cam_id": cam_id,
            "seq_asset_path": seq_asset_path,
            "cam_out": cam_out,
            "render_dir": render_dir,
            "img1_dir": img1_dir,
            "kind": "rgb",
            "preset": preset,
            "cv_gt": cv_gt,
        })
        print(f"  待渲染 RGB: {seq_name} -> {img1_dir}")

        if mask_enabled:
            mask_render_dir = cam_out / "render_mask"
            mask_dir = cam_out / "mask"
            ensure_dir(mask_render_dir)
            _clear_dir(mask_render_dir)
            _clear_dir(mask_dir)  # 清空旧的 mask 拷贝，避免上一次的无效帧残留
            jobs.append({
                "name": seq_name + "_MASK",
                "cam_id": cam_id,
                "seq_asset_path": seq_asset_path,
                "cam_out": cam_out,
                "render_dir": mask_render_dir,
                "img1_dir": mask_dir,
                "kind": "mask",
                "mask_source": mask_source,
                "post_process_material": post_process_material,
                "preset": preset,
                "cv_gt": cv_gt,
            })
            print(f"  待渲染 MASK: {seq_name} -> {mask_dir}")

    if not jobs:
        print("WARNING: 没有任何可渲染的 Sequence")
        return

    map_path = _get_current_map_path()
    subsystem = _mrq_subsystem()
    queue = _mrq_get_queue(subsystem)
    if queue is None:
        _list_mrq_classes()
        raise RuntimeError(
            "无法获取 MRQ 队列（MoviePipelineQueueSubsystem.get_queue() 返回 None）。"
            "已打印 [MRQ 诊断]，请把输出贴回。"
        )
    executor = _mrq_executor()

    pipeline = _AsyncRenderPipeline(jobs, episode_id, output_dir, keep_indices)
    _ACTIVE_RENDER = pipeline  # 模块级引用：脚本返回后防止 pipeline 被 GC
    pipeline.start(
        subsystem, queue, executor, map_path,
        image_width, image_height, frame_rate,
        file_name_format, zero_pad,
    )


def force_finalize_render() -> bool:
    """手动触发当前活跃渲染管线的收尾（复制 img1/ + 写 render_summary.json）。

    极端情况下（finished delegate 与 watchdog 均未触发），在 UE 控制台执行：
        import render_episode
        render_episode.force_finalize_render()
    """
    global _ACTIVE_RENDER
    if _ACTIVE_RENDER is None:
        print("WARNING: 没有活跃的 MRQ 渲染管线（render_episode._ACTIVE_RENDER 为 None）")
        return False
    _ACTIVE_RENDER._copy_and_finalize()
    return True


def _mrq_subsystem():
    """获取 MRQ 子系统。

    UE 5.8 起类名由 MovieRenderQueueSubsystem 改为 MoviePipelineQueueSubsystem。
    """
    import unreal

    for cls_name in ("MoviePipelineQueueSubsystem", "MovieRenderQueueSubsystem"):
        cls = getattr(unreal, cls_name, None)
        if cls is not None:
            try:
                return unreal.get_editor_subsystem(cls)
            except Exception:
                continue
    _list_mrq_classes()
    raise RuntimeError(
        "找不到 MRQ 子系统（MoviePipelineQueueSubsystem / MovieRenderQueueSubsystem）。"
        "已打印可用 MRQ 类，请把上面 [MRQ 诊断] 输出贴回。"
    )


def _mrq_executor():
    """创建 MRQ 执行器（PIE 优先）。"""
    import unreal

    for cls_name in ("MoviePipelinePIEExecutor", "MoviePipelineEditorExecutor"):
        cls = getattr(unreal, cls_name, None)
        if cls is None:
            continue
        try:
            return cls()
        except Exception:
            continue
    _list_mrq_classes()
    raise RuntimeError(
        "无法创建 MRQ 执行器（MoviePipelinePIEExecutor / MoviePipelineEditorExecutor "
        "均不可用）。已打印可用 MRQ 类，请把上面 [MRQ 诊断] 输出贴回。"
    )
