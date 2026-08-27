"""CV GT deterministic render preset（纯配置，UE 侧与 pytest 共用，不依赖 unreal）。

正式 CV 数据集默认关闭会破坏「RGB 空间边界 == Instance-ID Mask 边界」的空间扩散效果，
并把时间采样固定为单时刻（无 shutter 时间积分），保证 frame-exact GT：

  关闭：motion_blur / depth_of_field / chromatic_aberration / lens_distortion
  保留：lighting / shadow / material / texture / 合理 anti-aliasing（默认 TAA）

preset 取值（render_rgb.preset）：
  "cv_gt"（默认）—— deterministic CV GT：显式关闭空间扩散 + 单时刻时间采样。
  "cinematic"     —— 保留模式（未实现）：放开画质效果，GT 语义随之变化。
  null            —— 不覆盖：完全保持关卡/相机的现状（legacy 行为）。

cv_gt 字段（render_rgb.cv_gt）：
  motion_blur        : bool  默认 false —— 关闭运动模糊（含 MRQ shutter 时间积分）
  depth_of_field     : bool  默认 false —— 关闭景深（后处理 DOF 方法置 off + 大 f-stop 兜底）
  chromatic_aberration: bool 默认 false —— 关闭色差
  lens_distortion    : bool  默认 false —— 关闭镜头畸变（UE 无内置后处理属性，
                                          畸变仅来自失真后处理材质，管线不添加即默认关闭）
  anti_aliasing      : str   默认 "taa" —— taa / tsr / none（保留合理 AA）
  temporal_sampling  : bool  默认 false —— 单时刻渲染（false=无 shutter/时间积分，与 mask 时间域一致）

enum 值在返回 dict 中以 ("EnumClassName", "MemberName") 标记，由 UE 侧 getattr 解析；
纯 Python 侧只保证契约（名称），不接触 unreal，因此本模块可被 pytest 独立测试。
"""

from typing import Dict, Optional, Tuple
import json
from pathlib import Path

PRESET_CV_GT = "cv_gt"
PRESET_CINEMATIC = "cinematic"
VALID_PRESETS = (PRESET_CV_GT, PRESET_CINEMATIC, None)

AA_TAA = "taa"
AA_TSR = "tsr"
AA_NONE = "none"
VALID_AA = (AA_TAA, AA_TSR, AA_NONE)

# cv_gt 各效果开关的默认值（正式 CV 数据集全部关闭空间扩散）
DEFAULT_CV_GT = {
    "motion_blur": False,
    "depth_of_field": False,
    "chromatic_aberration": False,
    "lens_distortion": False,
    "anti_aliasing": AA_TAA,
    "temporal_sampling": False,
}

_BOOL_KEYS = ("motion_blur", "depth_of_field", "chromatic_aberration",
              "lens_distortion", "temporal_sampling")


def resolve_preset(render_cfg: Optional[dict]) -> Tuple[str, dict]:
    """从 render_rgb 配置读取 (preset, cv_gt)。缺省 preset 为 "cv_gt"。"""
    render_cfg = render_cfg or {}
    preset = render_cfg.get("preset", PRESET_CV_GT)
    if preset not in VALID_PRESETS:
        raise ValueError(f"未知 render preset: {preset!r}（可选 {VALID_PRESETS}）")
    cv_gt = render_cfg.get("cv_gt") or {}
    return preset, cv_gt


def normalize_cv_gt(cv_gt: Optional[dict]) -> dict:
    """校验/默认化 cv_gt 配置。未知字段忽略，非法取值抛 ValueError。"""
    out = dict(DEFAULT_CV_GT)
    if cv_gt:
        for k in DEFAULT_CV_GT:
            if k in cv_gt:
                out[k] = cv_gt[k]
    if out["anti_aliasing"] not in VALID_AA:
        raise ValueError(f"未知 anti_aliasing 模式: {out['anti_aliasing']!r}（可选 {VALID_AA}）")
    for k in _BOOL_KEYS:
        if not isinstance(out[k], bool):
            raise ValueError(f"cv_gt.{k} 必须为 bool: {out[k]!r}")
    return out


def camera_post_process_overrides(preset, cv_gt) -> Dict[str, object]:
    """cv_gt → CineCameraComponent.post_process_settings 字段覆盖。

    返回 property 名 → 值（enum 值为 ("EnumClass", "MemberName")）。仅 cv_gt 生效；
    其它 preset 返回空 dict（不改相机后处理）。
    """
    if preset != PRESET_CV_GT:
        return {}
    cg = normalize_cv_gt(cv_gt)
    overrides: Dict[str, object] = {
        # blend weight=1.0：相机后处理完全覆盖关卡 Post Process Volume，不依赖手工设置
        "post_process_blend_weight": 1.0,
    }
    if not cg["motion_blur"]:
        overrides["motion_blur_amount"] = 0.0
    if not cg["depth_of_field"]:
        overrides["depth_of_field_method"] = ("EDepthOfFieldMethod", "DOFM_None")
        # 兜底：即使 DOF 方法枚举不可用，超大 f-stop 也使 DOF 深度极大、近似无景深
        overrides["depth_of_field_fstop"] = 32.0
    if not cg["chromatic_aberration"]:
        # UE 色差的 PostProcessSettings 属性名是 scene_fringe_intensity（Scene Color Fringe）
        overrides["scene_fringe_intensity"] = 0.0
    # lens_distortion：UE 无内置后处理属性；畸变仅来自失真后处理材质，管线不添加即关闭。
    return overrides


def mrq_temporal_overrides(preset, cv_gt) -> Dict[str, object]:
    """cv_gt → 时间采样确定性（RGB 与 mask 两个 job 都应用）。

    使 RGB 表示单个时刻（而非一段曝光时间），与 mask 的时间域一致：
      - MoviePipelineAntiAliasingSetting: temporal_accumulation_method=NONE、
        temporal_sample_count=1（每输出帧 = 单采样，无 temporal motion integration）。
    UE 5.8 实测：MoviePipelineCameraSetting 无 motion_blur 属性、shutter_timing 为枚举，
    故不设置它们；单采样（temporal NONE + sample_count=1）已足以消除时间域运动模糊。
    temporal_sampling=true 时返回空 dict（显式允许时间采样的未来模式）。
    """
    if preset != PRESET_CV_GT:
        return {}
    cg = normalize_cv_gt(cv_gt)
    if cg["temporal_sampling"]:
        return {}  # 未来模式：允许时间采样（会破坏 frame-exact GT，仅显式开启时生效）
    return {
        "temporal_accumulation_method": ("MoviePipelineTemporalAccumulationMethod", "NONE"),
        "temporal_sample_count": 1,
    }


def mrq_aa_overrides(preset, cv_gt) -> Dict[str, object]:
    """cv_gt → RGB job 的 anti-aliasing（保留合理 AA）。

    - anti_aliasing_method: taa / tsr / none
    （render_warm_up_count 等 warm-up 契约见 mrq_warmup_overrides，RGB/mask 相同）
    """
    if preset != PRESET_CV_GT:
        return {}
    cg = normalize_cv_gt(cv_gt)
    method = {
        AA_TAA: ("MoviePipelineAntiAliasingMethod", "TAA"),
        AA_TSR: ("MoviePipelineAntiAliasingMethod", "TSR"),
        AA_NONE: ("MoviePipelineAntiAliasingMethod", "None"),
    }[cg["anti_aliasing"]]
    return {
        "anti_aliasing_method": method,
    }


def mrq_warmup_overrides(preset, cv_gt) -> Dict[str, object]:
    """cv_gt → warm-up 契约（RGB 与 mask 两个 job 使用相同策略）。

    修复 MRQ 首帧视角错误：PIE/MRQ 启动后立即采集 frame 0，possessable camera /
    Camera Cut 尚未完成初始化 → 首帧用默认视角（世界原点）；下一个 engine tick 后正常。

    设置 engine_warm_up_count=2 让引擎在输出帧 0 前预热 2 tick（PIE / Sequencer /
    Camera Cut 完成初始化），同时保持 render_warm_up_count=0 / render_warm_up_frames=false /
    use_camera_cut_for_warm_up=false：
      - warm-up 帧**不输出**（不改变 Sequence 时间与 dataset frame 映射；frame 0 仍对应
        GRF step 0，不丢弃首帧、不做 frame shift）
      - 不引入时间采样（不重开 temporal accumulation / motion blur）
      - 不用 camera cut 做 warm-up（它正是未初始化的对象）
    temporal_sampling=true 时返回空 dict（未来模式，显式允许时间采样）。
    """
    if preset != PRESET_CV_GT:
        return {}
    cg = normalize_cv_gt(cv_gt)
    if cg["temporal_sampling"]:
        return {}  # 未来模式：允许时间采样（会破坏 frame-exact GT，仅显式开启时生效）
    return {
        "engine_warm_up_count": 2,
        "render_warm_up_count": 0,
        "render_warm_up_frames": False,
        "use_camera_cut_for_warm_up": False,
    }


def post_process_console_vars(preset, cv_gt) -> Dict[str, float]:
    """cv_gt → MRQ job 后处理控制台变量（cvar，强制关闭空间扩散）。

    与相机 post_process 覆盖互补：即使相机后处理覆盖位在特定 UE 版本不生效，
    cvar 也强制把 motion blur / DOF / 色差关闭，保证 RGB 与 mask 边界一致。
    """
    if preset != PRESET_CV_GT:
        return {}
    cg = normalize_cv_gt(cv_gt)
    cvars: Dict[str, float] = {}
    if not cg["motion_blur"]:
        cvars["r.MotionBlur.Amount"] = 0.0
    if not cg["depth_of_field"]:
        cvars["r.DepthOfFieldQuality"] = 0.0
    if not cg["chromatic_aberration"]:
        cvars["r.SceneColorFringeQuality"] = 0.0
    return cvars


# ── C5.1：输出分辨率唯一来源与一致性校验 ────────────────────────────────

def resolve_output_resolution(cfg: Optional[dict]) -> Tuple[int, int]:
    """从 annotation/render 配置解析最终输出分辨率（唯一来源）。

    C5.1 历史根因：camera calibration 曾按 annotation 的 image_width/height
    （1280×720）生成，而 MRQ 实际渲染 1920×1080，导致 Pose 整体错位。
    正式设计：camera calibration 与 MRQ render 必须消费**同一**分辨率。

    优先级：
      - render_rgb.output_resolution_x/y（MRQ 渲染分辨率，规范来源）
      - 兼容旧任务的 annotation image_width/height（legacy fallback）
    两者都存在且不一致 → 抛错（禁止两套分辨率）。
    两者都缺失 → 抛错（fail fast，绝不静默回退 1280×720）。

    Args:
        cfg: annotation_export 配置 dict（含 render_rgb 子块）。

    Returns:
        (width, height)。

    Raises:
        ValueError: 无法解析，或 render_rgb 与 image_width/height 冲突。
    """
    cfg = cfg or {}
    rr = cfg.get("render_rgb") or {}
    rw, rh = rr.get("output_resolution_x"), rr.get("output_resolution_y")
    aw, ah = cfg.get("image_width"), cfg.get("image_height")
    if rw is not None and rh is not None:
        w, h = int(rw), int(rh)
        if aw is not None and ah is not None and (int(aw) != w or int(ah) != h):
            raise ValueError(
                f"Camera calibration resolution mismatch: "
                f"render_rgb.output_resolution={w}x{h} vs "
                f"annotation image={aw}x{ah}"
            )
        return w, h
    if aw is not None and ah is not None:
        return int(aw), int(ah)
    raise ValueError(
        "无法解析输出分辨率：render_rgb.output_resolution 与 "
        "annotation image_width/height 均缺失（禁止静默回退 1280x720）"
    )


def png_dimensions(path) -> Tuple[int, int]:
    """读取 PNG IHDR 宽高（纯标准库，不依赖 PIL）。"""
    with open(path, "rb") as f:
        sig = f.read(8)
        if sig != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"非 PNG 文件: {path}")
        f.read(4)  # IHDR 块长度
        f.read(4)  # 'IHDR'
        width = int.from_bytes(f.read(4), "big")
        height = int.from_bytes(f.read(4), "big")
    return width, height


def validate_render_vs_calibration(render_dir, camera_json_path) -> None:
    """后处理阶段 fail-fast：实际渲染输出分辨率 == camera.json calibration 分辨率。

    C4 历史根因：calibration 1280×720 用于 1920×1080 渲染 → Pose 整体错位。
    若不一致立即抛错，禁止继续生成看似合法但实际错位的 Pose 数据。

    Args:
        render_dir: 渲染输出目录（含 FinalImage PNG）。
        camera_json_path: 该相机输出目录下的 camera.json。

    Raises:
        RuntimeError: 渲染分辨率与 calibration 分辨率不一致。
    """
    render_dir = Path(render_dir)
    if not render_dir.is_dir():
        return  # 无渲染输出可检查
    pngs = sorted(render_dir.rglob("*.png"))
    if not pngs:
        return
    w, h = png_dimensions(pngs[0])
    cam = json.loads(Path(camera_json_path).read_text(encoding="utf-8"))
    intr = cam.get("intrinsics") or {}
    cal_w = int(cam.get("image_width") or intr.get("width") or 0)
    cal_h = int(cam.get("image_height") or intr.get("height") or 0)
    if (w, h) != (cal_w, cal_h):
        raise RuntimeError(
            f"Camera calibration resolution mismatch: "
            f"rendered={w}x{h} calibration={cal_w}x{cal_h} "
            f"(camera.json 未按任务真实输出分辨率生成)"
        )
