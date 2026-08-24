"""Player Motion Layer：从轨迹数据推导运动参数（纯 Python，无 numpy/unreal）。

本模块是「GRF 轨迹 → 运动参数」的**唯一权威来源**，同时供：

- UE 侧（scene_apply / import_grf_episode）：把轨迹解释成 facing 朝向与运动参数；
- P1 侧（interpolate 下游 / pytest）：生成/校验运动状态。

不依赖 unreal / numpy / gfootball，UE Python 可直接 import，pytest 可直接测试。

核心原则（务必遵守）：
  位置 Ground Truth 由 GRF 轨迹决定（Actor 的 Location 轨道，厘米 = 米 ×100）。
  本模块只负责把「位置/速度」解释成「朝向、步态、运动状态」等人为可读参数，
  **绝不改写位置**——动画系统只能消费这些参数，不能反向决定 Actor 世界坐标。

单位约定（变量名即单位，避免 cm/m 混用）：
  位置   : position_m      （米）
  速度   : velocity_mps    （米/秒）
  速率   : speed_mps       （米/秒）
  时间步 : dt_s            （秒）
  朝向角 : yaw_deg         （度，-180 ~ 180）
  角速度 : yaw_rate_deg_s  （度/秒）
  加速度 : accel_mps2      （米/秒²）
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ── 运动枚举（字符串常量，便于序列化；不引入 Enum 依赖）──────────────────

class Gait:
    """步态（按速度分带）。"""

    IDLE = "idle"
    WALK = "walk"
    JOG = "jog"
    RUN = "run"
    SPRINT = "sprint"

    ALL = (IDLE, WALK, JOG, RUN, SPRINT)


class MotionState:
    """基础运动状态（含迟滞，避免相邻帧抖动）。"""

    IDLE = "idle"
    START = "start"
    LOCOMOTION = "locomotion"
    DECELERATE = "decelerate"
    STOP = "stop"
    PIVOT = "pivot"

    ALL = (IDLE, START, LOCOMOTION, DECELERATE, STOP, PIVOT)


class MotionAction:
    """足球动作状态（本轮只预留枚举，暂不消费）。"""

    NONE = "none"
    KICK = "kick"
    PASS = "pass"
    SHOT = "shot"
    RECEIVE = "receive"

    ALL = (NONE, KICK, PASS, SHOT, RECEIVE)


# ── Animation Selector 输出类别（第一版）──────────────────────────────────

class AnimationClass:
    """Animation Selector 输出的动画类别（供未来动画系统消费）。"""

    LOCOMOTION = "locomotion"      # 通用 locomotion（由 speed / BlendSpace 连续驱动）
    PIVOT_L = "pivot_l"            # 90° 左转（turn_rate>0，= UE Pivot90L +90°）
    PIVOT_R = "pivot_r"            # 90° 右转（turn_rate<0，= UE Pivot90R -90°）
    GK_SHUFFLE_L = "gk_shuffle_l"  # GK 向左横移（relative<0，资产命名镜像）
    GK_SHUFFLE_R = "gk_shuffle_r"  # GK 向右横移（relative>0，资产命名镜像）
    GK_BACKPEDAL = "gk_backpedal"  # GK 后退（|relative|>=135°）

    ALL = (LOCOMOTION, PIVOT_L, PIVOT_R,
           GK_SHUFFLE_L, GK_SHUFFLE_R, GK_BACKPEDAL)


# ── 运动参数配置（集中可调，禁止散落 magic number）───────────────────────

@dataclass(frozen=True)
class MotionConfig:
    """运动层阈值配置（全部为物理单位，与 FPS 无关）。"""

    # 步态分带（m/s）：<= idle_max → idle；<= walk_max → walk；以此类推；> run_max → sprint。
    idle_max_speed_mps: float = 0.3
    walk_max_speed_mps: float = 1.0
    jog_max_speed_mps: float = 2.0
    run_max_speed_mps: float = 3.5

    # Facing：速度低于该值保持上一帧朝向（m/s）。
    min_facing_speed_mps: float = 0.3
    # Facing：最大转向角速度（度/秒，硬上限）。
    max_yaw_rate_deg_s: float = 360.0
    # Facing：一阶滞后平滑时间常数（秒）；0 表示关闭平滑、仅限速。
    yaw_smoothing_time_s: float = 0.1

    # GK ball-aware facing：低于该速率（m/s）始终面向球（即使静止也缓慢转身）。
    gk_face_ball_max_speed_mps: float = 2.0
    # GK ball-aware facing：高于 max_speed 时，movement heading 与 ball heading
    # 夹角 <= 该值（度）才面向球；否则保持 movement heading（避免高速倒跑）。
    gk_face_ball_max_angle_deg: float = 90.0

    # Motion state 阈值
    start_accel_mps2: float = 1.2       # 加速度 > 该值且低速 → Start
    decel_accel_mps2: float = -1.2      # 加速度 < 该值且高速 → Decelerate
    stop_speed_mps: float = 0.4         # 速度降到该值以下且此前在运动 → Stop
    start_max_speed_mps: float = 1.5    # Start 判定时的最大当前速度
    decel_min_speed_mps: float = 1.0    # Decelerate 判定时的最小当前速度
    pivot_min_speed_mps: float = 2.0    # Pivot 判定的最小速度
    pivot_min_turn_rate_deg_s: float = 90.0  # Pivot 判定的最小角速度

    # Temporal Stabilization（秒，不依赖固定帧率）：raw_motion_state → animation_motion_state。
    # enter_confirm_s：候选状态需连续维持该时长才可进入 animation state（基础态即时确认）。
    state_enter_confirm_s: Dict[str, float] = field(default_factory=lambda: {
        MotionState.IDLE: 0.0,
        MotionState.START: 0.05,
        MotionState.LOCOMOTION: 0.0,
        MotionState.DECELERATE: 0.10,
        MotionState.STOP: 0.05,
        MotionState.PIVOT: 0.12,
    })
    # min_visible_s：进入后至少保持该时长（基础态 0，可随时被瞬态打断）。
    state_min_visible_s: Dict[str, float] = field(default_factory=lambda: {
        MotionState.IDLE: 0.0,
        MotionState.START: 0.20,
        MotionState.LOCOMOTION: 0.0,
        MotionState.DECELERATE: 0.15,
        MotionState.STOP: 0.20,
        MotionState.PIVOT: 0.18,
    })

    # Animation Selector 稳定化（秒，raw_animation_class → animation_class）。
    # 仅 GK 方向类（shuffle_l/r、backpedal）需要 enter_confirm / min_visible；
    # pivot_l/r 上游 animation_motion_state 已稳定，不加额外稳定；
    # locomotion 是安全兜底，不需确认。
    anim_class_enter_confirm_s: Dict[str, float] = field(default_factory=lambda: {
        AnimationClass.LOCOMOTION: 0.0,
        AnimationClass.PIVOT_L: 0.0,
        AnimationClass.PIVOT_R: 0.0,
        AnimationClass.GK_SHUFFLE_L: 0.10,
        AnimationClass.GK_SHUFFLE_R: 0.10,
        AnimationClass.GK_BACKPEDAL: 0.10,
    })
    anim_class_min_visible_s: Dict[str, float] = field(default_factory=lambda: {
        AnimationClass.LOCOMOTION: 0.0,
        AnimationClass.PIVOT_L: 0.0,
        AnimationClass.PIVOT_R: 0.0,
        AnimationClass.GK_SHUFFLE_L: 0.15,
        AnimationClass.GK_SHUFFLE_R: 0.15,
        AnimationClass.GK_BACKPEDAL: 0.15,
    })

    # 迟滞（避免阈值边界抖动）
    gait_hysteresis_mps: float = 0.15
    speed_hysteresis_mps: float = 0.15

    # 诊断上限（m/s）：超过即视为速度尖峰（疑似瞬移/脏数据）。
    max_speed_mps: float = 12.0
    max_accel_mps2: float = 15.0


DEFAULT_MOTION_CONFIG = MotionConfig()


# GK 实体 ID（与 exporter._build_entities 的 role=0 门将约定一致）。
GK_ENTITY_IDS = frozenset({"L0", "R0"})


def gk_entity_ids_from_meta(meta: Optional[dict]) -> frozenset:
    """从 meta.entities 的 is_goalkeeper / role 元数据推导 GK 实体 ID 集合。

    meta 形状：{ "entities": [{ "id": "L0", "is_goalkeeper": True, ... }, ...] }。
    优先使用 is_goalkeeper=True 的实体；当 meta 缺少 entities 或没有任何
    GK 标记时，回退到 GK_ENTITY_IDS（role=0 启发式，兼容旧数据）。

    不假设 L0/R0 一定是 GK——真实 episode 中 GK 可在任意 index。
    """
    entities = (meta or {}).get("entities") or []
    gk = frozenset(
        e.get("id") for e in entities if bool(e.get("is_goalkeeper")) and e.get("id")
    )
    return gk if gk else GK_ENTITY_IDS


# ── 角度工具 ─────────────────────────────────────────────────────────────

def normalize_angle_deg(angle_deg: float) -> float:
    """把任意角度归一到 (-180, 180]。"""
    a = math.fmod(angle_deg, 360.0)
    if a <= -180.0:
        a += 360.0
    elif a > 180.0:
        a -= 360.0
    if a == -0.0:
        a = 0.0
    return a


def shortest_angle_delta_deg(from_deg: float, to_deg: float) -> float:
    """返回从 from_deg 到 to_deg 的最短角度差（度，范围 (-180, 180]）。

    正确处理 wrap，例如 179° → -179° 返回 2°（而非 -358°）。
    """
    return normalize_angle_deg(float(to_deg) - float(from_deg))


def move_towards_angle_deg(from_deg: float, to_deg: float, max_delta_deg: float) -> float:
    """朝 to_deg 方向移动 from_deg 最多 max_delta_deg 度（限速转向）。"""
    delta = shortest_angle_delta_deg(from_deg, to_deg)
    if max_delta_deg <= 0.0:
        return normalize_angle_deg(from_deg)
    if delta > max_delta_deg:
        delta = max_delta_deg
    elif delta < -max_delta_deg:
        delta = -max_delta_deg
    return normalize_angle_deg(from_deg + delta)


def heading_from_velocity_deg(velocity_xy: Sequence[float]) -> Optional[float]:
    """由速度向量计算运动朝向角（度）。速度过小/非有限返回 None。"""
    vx = float(velocity_xy[0])
    vy = float(velocity_xy[1])
    if not (math.isfinite(vx) and math.isfinite(vy)):
        return None
    speed = math.hypot(vx, vy)
    if speed <= 0.0:
        return None
    return normalize_angle_deg(math.degrees(math.atan2(vy, vx)))


# ── Facing / Yaw ─────────────────────────────────────────────────────────

def compute_facing_yaw(
    velocity_xy: Sequence[float],
    previous_yaw_deg: float,
    dt_s: float,
    min_facing_speed_mps: float,
    max_yaw_rate_deg_s: float,
    yaw_smoothing_time_s: float,
    *,
    desired_yaw_deg: Optional[float] = None,
    allow_turn_when_slow: bool = False,
) -> float:
    """由速度向量计算平滑后的朝向角（度）。

    流程：velocity → desired_yaw = atan2(vy, vx) → 低速保持 previous_yaw →
    一阶滞后平滑 + 角速度硬限速 → 归一化。

    desired_yaw_deg / allow_turn_when_slow 用于 GK ball-aware facing：
    - desired_yaw_deg 不为 None 时，直接用该值作为目标朝向（如面向球），
      否则仍由 velocity 计算（普通球员 movement-facing，行为不变）。
    - allow_turn_when_slow=True 时，即使速度低于 min_facing_speed 也按
      desired_yaw_deg 缓慢转身（守门员静止时也面向球）。

    Args:
        velocity_xy: 速度向量 [vx, vy]，单位 m/s。
        previous_yaw_deg: 上一帧朝向（度）。
        dt_s: 时间步长（秒，真实 dt）。
        min_facing_speed_mps: 低于该速率（m/s）保持上一帧朝向。
        max_yaw_rate_deg_s: 最大转向角速度（度/秒，硬上限）。
        yaw_smoothing_time_s: 一阶滞后平滑时间常数（秒）；0 表示关闭平滑仅限速。
        desired_yaw_deg: 目标朝向（度）；None = 用 velocity 计算。
        allow_turn_when_slow: 低速时是否仍按 desired_yaw_deg 转身。

    Returns:
        归一化到 (-180, 180] 的朝向角（度）。所有输入异常（非有限/负 dt）均安全返回
        previous_yaw 的归一化值，绝不产生 NaN。
    """
    vx = float(velocity_xy[0]) if velocity_xy is not None else 0.0
    vy = float(velocity_xy[1]) if velocity_xy is not None else 0.0
    prev = normalize_angle_deg(float(previous_yaw_deg))

    if not (math.isfinite(vx) and math.isfinite(vy)):
        return prev
    speed = math.hypot(vx, vy)
    if dt_s <= 0.0 or not math.isfinite(dt_s):
        return prev
    if speed < float(min_facing_speed_mps):
        if desired_yaw_deg is not None and allow_turn_when_slow:
            desired = normalize_angle_deg(float(desired_yaw_deg))
        else:
            # 低速：保持上一帧朝向（静止人物不随机转头）
            return prev
    else:
        if desired_yaw_deg is not None:
            desired = normalize_angle_deg(float(desired_yaw_deg))
        else:
            desired = normalize_angle_deg(math.degrees(math.atan2(vy, vx)))
    delta = shortest_angle_delta_deg(prev, desired)

    # 一阶滞后平滑（指数形式，与 FPS 无关：经过 t 秒后剩余误差 = exp(-t/tau)）
    if yaw_smoothing_time_s > 0.0:
        alpha = 1.0 - math.exp(-dt_s / float(yaw_smoothing_time_s))
        delta *= alpha

    # 角速度硬限速（度/秒 → 本帧最大可转角度）
    if max_yaw_rate_deg_s > 0.0:
        max_step = float(max_yaw_rate_deg_s) * dt_s
        if delta > max_step:
            delta = max_step
        elif delta < -max_step:
            delta = -max_step

    return normalize_angle_deg(prev + delta)


# ── 步态 ─────────────────────────────────────────────────────────────────

def compute_gait(
    speed_mps: float,
    previous_gait: str,
    config: MotionConfig = DEFAULT_MOTION_CONFIG,
) -> str:
    """由速率分带推导步态（带迟滞，避免边界抖动）。

    Args:
        speed_mps: 当前速率（m/s）。
        previous_gait: 上一帧步态（用于迟滞）。
        config: 阈值配置。
    """
    speed = float(speed_mps)
    if not math.isfinite(speed):
        return Gait.IDLE

    hys = config.gait_hysteresis_mps
    # 迟滞：处于某个高档时，下降阈值降低 hys，避免在边界来回跳。
    if previous_gait == Gait.SPRINT:
        run_max = config.run_max_speed_mps - hys
    else:
        run_max = config.run_max_speed_mps
    if previous_gait == Gait.RUN:
        jog_max = config.jog_max_speed_mps - hys
    else:
        jog_max = config.jog_max_speed_mps
    if previous_gait == Gait.JOG:
        walk_max = config.walk_max_speed_mps - hys
    else:
        walk_max = config.walk_max_speed_mps
    if previous_gait == Gait.WALK:
        idle_max = config.idle_max_speed_mps - hys
    else:
        idle_max = config.idle_max_speed_mps

    if speed <= max(0.0, idle_max):
        return Gait.IDLE
    if speed <= walk_max:
        return Gait.WALK
    if speed <= jog_max:
        return Gait.JOG
    if speed <= run_max:
        return Gait.RUN
    return Gait.SPRINT


# ── 运动状态 ─────────────────────────────────────────────────────────────

def compute_motion_state(
    speed_mps: float,
    accel_mps2: float,
    turn_rate_deg_s: float,
    previous_speed_mps: float,
    previous_state: str,
    config: MotionConfig = DEFAULT_MOTION_CONFIG,
) -> str:
    """由速率/加速度/角速度推导基础运动状态（带迟滞，稳定）。

    优先级（自上而下）：
      speed 近 0 且此前静止 → Idle
      speed 近 0 且此前在运动 → Stop
      高速 + 强减速 → Decelerate
      低速 + 强加速 → Start
      速度足够 + 角速度足够 → Pivot
      否则 → Locomotion

    迟滞：阈值用 speed_hysteresis_mps 修正，避免相邻帧状态抖动。
    """
    speed = float(speed_mps)
    accel = float(accel_mps2)
    turn = abs(float(turn_rate_deg_s))
    prev_speed = float(previous_speed_mps)
    if not (math.isfinite(speed) and math.isfinite(accel) and math.isfinite(turn)):
        return MotionState.IDLE

    hys = config.speed_hysteresis_mps
    stop_hi = config.stop_speed_mps + hys
    stop_lo = config.stop_speed_mps - hys

    # 近静止
    if speed <= stop_lo:
        if previous_state in (MotionState.STOP, MotionState.LOCOMOTION,
                              MotionState.START, MotionState.DECELERATE,
                              MotionState.PIVOT) and prev_speed > stop_hi:
            return MotionState.STOP
        return MotionState.IDLE

    if speed < stop_hi and previous_state in (
        MotionState.STOP, MotionState.LOCOMOTION, MotionState.DECELERATE,
        MotionState.PIVOT, MotionState.START,
    ):
        # 减速接近停止
        return MotionState.STOP

    # 高速 + 强减速
    if speed >= config.decel_min_speed_mps and accel <= config.decel_accel_mps2:
        return MotionState.DECELERATE

    # 低速 + 强加速
    if speed <= config.start_max_speed_mps and accel >= config.start_accel_mps2:
        return MotionState.START

    # 速度足够 + 角速度足够 → Pivot（原地/行进中急转）
    if speed >= config.pivot_min_speed_mps and turn >= config.pivot_min_turn_rate_deg_s:
        return MotionState.PIVOT

    return MotionState.LOCOMOTION


# ── Animation Selector（第一版，输出 raw_animation_class）─────────────────

def select_animation_class(
    animation_motion_state: str,
    turn_rate_deg_s: Optional[float],
    relative_movement_heading_deg: Optional[float],
    is_goalkeeper: bool,
) -> str:
    """第一版 Animation Selector 原始规则（raw_animation_class）。

    优先级：
      1. animation_motion_state == pivot：
           turn_rate_deg_s > 0 → pivot_l（= UE Pivot90L +90°）
           turn_rate_deg_s < 0 → pivot_r（= UE Pivot90R -90°）
      2. GK 且 |relative_movement_heading_deg| >= 135 → gk_backpedal
      3. GK 且 45 <= |relative| < 135：
           relative > 0 → gk_shuffle_r
           relative < 0 → gk_shuffle_l
           （注意：资产命名镜像，此映射已用真实 velocity + UE 动画验证，勿反转）
      4. 其它 → locomotion（由 speed / BlendSpace 连续驱动）

    本函数只产出 raw_animation_class；时间稳定化由
    PlayerMotionTracker._stabilize_animation_class 负责。
    """
    if animation_motion_state == MotionState.PIVOT:
        return AnimationClass.PIVOT_L if (turn_rate_deg_s or 0.0) > 0 else AnimationClass.PIVOT_R
    if is_goalkeeper and relative_movement_heading_deg is not None:
        rel = float(relative_movement_heading_deg)
        a = abs(rel)
        if a >= 135.0:
            return AnimationClass.GK_BACKPEDAL
        if a >= 45.0:
            return AnimationClass.GK_SHUFFLE_R if rel > 0 else AnimationClass.GK_SHUFFLE_L
    return AnimationClass.LOCOMOTION


# ── 逐球员运动追踪器（流式，供 UE 侧逐帧调用；也可批量重放）──────────────

@dataclass
class PlayerMotionTracker:
    """单个球员的运动状态追踪器（确定性、无随机）。

    维护上一帧的 position / velocity / speed / facing / gait / state / time，
    供逐帧 update 计算当前帧运动参数。相同输入序列 ⇒ 相同输出序列。
    """

    config: MotionConfig = field(default_factory=lambda: DEFAULT_MOTION_CONFIG)

    previous_position_m: Optional[List[float]] = None
    previous_velocity_mps: Optional[List[float]] = None
    previous_speed_mps: float = 0.0
    previous_facing_yaw_deg: float = 0.0
    previous_gait: str = Gait.IDLE
    previous_state: str = MotionState.IDLE
    previous_time_s: Optional[float] = None

    # Temporal Stabilization（raw_motion_state → animation_motion_state）
    animation_motion_state: str = MotionState.IDLE
    animation_state_enter_time_s: Optional[float] = None
    candidate_state: Optional[str] = None
    candidate_since_s: Optional[float] = None

    # Animation Selector（raw_animation_class → animation_class）
    raw_animation_class: str = AnimationClass.LOCOMOTION
    animation_class: str = AnimationClass.LOCOMOTION
    animation_class_enter_time_s: Optional[float] = None
    anim_class_candidate_state: Optional[str] = None
    anim_class_candidate_since_s: Optional[float] = None

    def _stabilize(self, raw: str, time_s: float) -> str:
        """raw_motion_state → animation_motion_state 时间稳定化。

        规则（全部用 time_s 差分，不依赖固定帧率）：
          1. raw 改变时记录 candidate_state 与 candidate_since_s；
          2. candidate 连续达到对应 enter_confirm_s 才允许进入新 animation state
             （IDLE/LOCOMOTION 基础态即时确认）；
          3. 当前 animation state 未达到 min_visible_s 前，普通状态不得切走；
          4. PIVOT 达到确认时间后可优先打断任意状态（含 min_visible 内）；
          5. 无 cooldown。
        """
        cfg = self.config
        if raw != self.candidate_state:
            self.candidate_state = raw
            self.candidate_since_s = time_s
        cand = self.candidate_state
        cand_since = self.candidate_since_s

        def _is_basic(s: str) -> bool:
            return s in (MotionState.IDLE, MotionState.LOCOMOTION)

        cand_ready = _is_basic(cand) or (
            cand_since is not None
            and (time_s - cand_since)
            >= float(cfg.state_enter_confirm_s.get(cand, 0.0))
        )
        anim_enter = self.animation_state_enter_time_s
        anim_min_visible = float(
            cfg.state_min_visible_s.get(self.animation_motion_state, 0.0)
        )
        anim_ready = anim_enter is None or (time_s - anim_enter) >= anim_min_visible

        if cand_ready and cand != self.animation_motion_state:
            if _is_basic(cand):
                # 基础态候选：当前 anim 为基础态或已过 min_visible 才可接管
                if _is_basic(self.animation_motion_state) or anim_ready:
                    self.animation_motion_state = cand
                    self.animation_state_enter_time_s = time_s
            elif _is_basic(self.animation_motion_state) or anim_ready or cand == MotionState.PIVOT:
                # 瞬态候选：可打断基础态 / 当前 anim 已过 min_visible / PIVOT 优先
                self.animation_motion_state = cand
                self.animation_state_enter_time_s = time_s
        return self.animation_motion_state

    def _stabilize_animation_class(self, raw_anim: str, time_s: float) -> str:
        """raw_animation_class → animation_class 时间稳定化。

        仅对 GK 方向类（gk_shuffle_l/r、gk_backpedal）做 enter_confirm / min_visible：
          - enter_confirm：候选需连续维持该时长才进入；
          - min_visible：进入后至少保持，普通类不得切走；
          - pivot_l/r 最高优先级（无需确认，可打断 gk 类 min_visible 内）；
          - locomotion 是安全兜底，无需确认。
        全部用 time_s 差分，不依赖固定帧率。
        """
        cfg = self.config
        if raw_anim != self.anim_class_candidate_state:
            self.anim_class_candidate_state = raw_anim
            self.anim_class_candidate_since_s = time_s
        cand = self.anim_class_candidate_state
        cand_since = self.anim_class_candidate_since_s

        is_pivot = cand in (AnimationClass.PIVOT_L, AnimationClass.PIVOT_R)
        is_fallback = cand in (AnimationClass.LOCOMOTION, AnimationClass.PIVOT_L, AnimationClass.PIVOT_R)

        cand_ready = is_fallback or (
            cand_since is not None
            and (time_s - cand_since) >= float(cfg.anim_class_enter_confirm_s.get(cand, 0.0))
        )
        anim_enter = self.animation_class_enter_time_s
        anim_min_visible = float(
            cfg.anim_class_min_visible_s.get(self.animation_class, 0.0)
        )
        anim_ready = anim_enter is None or (time_s - anim_enter) >= anim_min_visible

        if cand_ready and cand != self.animation_class:
            if is_pivot:
                # pivot 最高优先级：可打断任何状态（含 gk 类 min_visible 内）
                self.animation_class = cand
                self.animation_class_enter_time_s = time_s
            elif cand == AnimationClass.LOCOMOTION:
                # 兜底：当前是兜底类或已过 min_visible 才接管
                if self.animation_class in (AnimationClass.LOCOMOTION,
                                            AnimationClass.PIVOT_L,
                                            AnimationClass.PIVOT_R) or anim_ready:
                    self.animation_class = cand
                    self.animation_class_enter_time_s = time_s
            elif is_fallback or anim_ready:
                # gk 类：可打断兜底类 / 当前已过 min_visible
                self.animation_class = cand
                self.animation_class_enter_time_s = time_s
        return self.animation_class

    def update(
        self,
        position_m: Sequence[float],
        velocity_mps: Optional[Sequence[float]],
        time_s: float,
        has_ball: bool = False,
        ball_position_m: Optional[Sequence[float]] = None,
        face_ball: bool = False,
        is_goalkeeper: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """处理一帧，返回该球员的运动参数 dict。

        Args:
            position_m: 当前帧位置 [x, y, z]（米）。
            velocity_mps: 当前帧速度 [vx, vy]（m/s）；None 时用相邻位置差分估算。
            time_s: 当前帧时间（秒）。
            has_ball: 是否持球（供未来 action 推断，本轮仅透出）。
            ball_position_m: 球的位置 [x, y, z]（米，最终 meter-space）。仅当
                face_ball=True 且不为 None 时启用 GK ball-aware facing。
            face_ball: 是否使用守门员 ball-aware facing 策略：
                速率 < gk_face_ball_max_speed_mps → 面向球；
                否则若 movement heading 与 ball heading 夹角
                <= gk_face_ball_max_angle_deg → 面向球；
                否则用 movement heading（避免高速倒跑）。
                普通球员（False）保持现有 movement-facing 不变。
            is_goalkeeper: 是否守门员（供 Animation Selector 判别 shuffle/backpedal）。
                None = 沿用 face_ball（UE 调用处 GK 必传 face_ball=True）。
        """
        cfg = self.config
        if is_goalkeeper is None:
            is_goalkeeper = bool(face_ball)
        gk_flag = bool(is_goalkeeper)
        px = float(position_m[0])
        py = float(position_m[1])
        pos = [px, py]

        # 速度：优先用帧提供的速度（GRF/Hermite 更准确），否则位置差分
        if velocity_mps is not None:
            vx = float(velocity_mps[0])
            vy = float(velocity_mps[1])
            if not (math.isfinite(vx) and math.isfinite(vy)):
                vx, vy = 0.0, 0.0
        elif self.previous_position_m is not None and self.previous_time_s is not None:
            dt = float(time_s) - float(self.previous_time_s)
            if dt > 0.0:
                vx = (px - self.previous_position_m[0]) / dt
                vy = (py - self.previous_position_m[1]) / dt
            else:
                vx, vy = 0.0, 0.0
        else:
            vx, vy = 0.0, 0.0
        vel = [vx, vy]

        speed = math.hypot(vx, vy)

        # 加速度（m/s²）
        if self.previous_time_s is not None:
            dt_accel = float(time_s) - float(self.previous_time_s)
        else:
            dt_accel = None
        if dt_accel is not None and dt_accel > 0.0 and self.previous_speed_mps is not None:
            accel = (speed - self.previous_speed_mps) / dt_accel
        else:
            accel = 0.0

        heading = heading_from_velocity_deg(vel)
        desired_facing = heading if heading is not None else self.previous_facing_yaw_deg

        # ── GK ball-aware facing ──────────────────────────────────────
        gk_mode = face_ball and ball_position_m is not None
        if gk_mode:
            bx = float(ball_position_m[0])
            by = float(ball_position_m[1])
            ball_heading = normalize_angle_deg(math.degrees(math.atan2(by - py, bx - px)))
            if speed < float(cfg.gk_face_ball_max_speed_mps):
                # 低速（含静止）：面向球（即使静止也缓慢转身）
                desired_facing = ball_heading
            elif heading is not None and (
                abs(shortest_angle_delta_deg(heading, ball_heading))
                <= float(cfg.gk_face_ball_max_angle_deg)
            ):
                # 高速且运动方向与球方向夹角不大：面向球
                desired_facing = ball_heading
            # else：高速且夹角 > 90°：保持 movement heading（避免高速倒跑）

        # 朝向：首帧无历史 → 直接用目标朝向初始化；否则平滑 + 限速转向
        if self.previous_time_s is None:
            facing = desired_facing
            turn_rate = 0.0
        else:
            facing_dt_s = float(time_s) - float(self.previous_time_s)
            facing = compute_facing_yaw(
                vel,
                self.previous_facing_yaw_deg,
                facing_dt_s if facing_dt_s > 0.0 else 0.0,
                cfg.min_facing_speed_mps,
                cfg.max_yaw_rate_deg_s,
                cfg.yaw_smoothing_time_s,
                desired_yaw_deg=desired_facing if gk_mode else None,
                allow_turn_when_slow=gk_mode,
            )
            if facing_dt_s > 0.0:
                turn_rate = shortest_angle_delta_deg(self.previous_facing_yaw_deg, facing) / facing_dt_s
            else:
                turn_rate = 0.0

        gait = compute_gait(speed, self.previous_gait, cfg)
        motion_state = compute_motion_state(
            speed, accel, turn_rate, self.previous_speed_mps, self.previous_state, cfg
        )
        animation_motion_state = self._stabilize(motion_state, float(time_s))

        # 相对运动方向：运动方向相对身体朝向的角度（= normalize(movement_heading - facing)），
        # 正值 = 逆时针（向左偏），负值 = 顺时针（向右偏），±180 = 后退。
        relative_movement_heading = None
        if heading is not None:
            relative_movement_heading = shortest_angle_delta_deg(facing, heading)

        # Animation Selector：raw_animation_class（纯规则）→ animation_class（时间稳定）
        raw_animation_class = select_animation_class(
            animation_motion_state,
            turn_rate,
            relative_movement_heading,
            gk_flag,
        )
        animation_class = self._stabilize_animation_class(raw_animation_class, float(time_s))

        # 更新内部状态
        self.previous_position_m = pos
        self.previous_velocity_mps = vel
        self.previous_speed_mps = speed
        self.previous_facing_yaw_deg = facing
        self.previous_gait = gait
        self.previous_state = motion_state
        self.previous_time_s = float(time_s)

        return {
            "speed_mps": round(speed, 6),
            "velocity_mps": [round(vx, 6), round(vy, 6)],
            "acceleration_mps2": round(accel, 6),
            "movement_heading_deg": round(heading, 6) if heading is not None else None,
            "desired_facing_deg": round(desired_facing, 6),
            "facing_deg": round(facing, 6),
            "relative_movement_heading_deg": (
                round(relative_movement_heading, 6)
                if relative_movement_heading is not None else None
            ),
            "turn_rate_deg_s": round(turn_rate, 6),
            "gait": gait,
            "motion_state": motion_state,
            "animation_motion_state": animation_motion_state,
            "raw_animation_class": raw_animation_class,
            "animation_class": animation_class,
            "has_ball": bool(has_ball),
            "action_state": MotionAction.NONE,
        }


# ── 批量工具 ─────────────────────────────────────────────────────────────

def compute_player_motion_sequence(
    frames: Sequence[Dict[str, Any]],
    config: MotionConfig = DEFAULT_MOTION_CONFIG,
    gk_entity_ids: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """批量计算每个球员在每一帧的运动参数（供 debug / sidecar 输出）。

    返回长度与 frames 相同的列表，每项为 {frame_index, step, time_seconds,
    players: {player_id: motion_params}}。
    确定性：相同 frames + 相同 config ⇒ 相同结果。

    gk_entity_ids: 守门员实体 ID 集合（None = 用 GK_ENTITY_IDS 回退）。
    """
    gk_ids = frozenset(gk_entity_ids) if gk_entity_ids is not None else GK_ENTITY_IDS
    trackers: Dict[str, PlayerMotionTracker] = {}
    result: List[Dict[str, Any]] = []
    for frame in frames:
        step = frame.get("step")
        time_s = float(frame.get("time_seconds", 0.0))
        players: Dict[str, Dict[str, Any]] = {}
        for p in frame.get("players", []):
            pid = p["id"]
            tracker = trackers.setdefault(pid, PlayerMotionTracker(config=config))
            players[pid] = tracker.update(
                p.get("position_m", [0.0, 0.0, 0.0]),
                p.get("velocity_mps"),
                time_s,
                is_goalkeeper=(pid in gk_ids),
            )
        result.append({
            "frame_index": (step + 1) if isinstance(step, int) else None,
            "step": step,
            "time_seconds": time_s,
            "players": players,
        })
    return result


def diagnose_trajectory(
    frames: Sequence[Dict[str, Any]],
    config: MotionConfig = DEFAULT_MOTION_CONFIG,
) -> List[Dict[str, Any]]:
    """轨迹诊断：检测位置断点 / 速度尖峰 / 角速度尖峰 / 加速度尖峰。

    返回 warning 列表（每项含 frame_index / player_id / kind / value / limit）。
    纯诊断，不修改任何数据；用于检查数据质量（如瞬移、脏速度）。
    """
    warnings: List[Dict[str, Any]] = []
    trackers: Dict[str, PlayerMotionTracker] = {}
    prev_frame_pos: Dict[str, List[float]] = {}
    prev_frame_time: Dict[str, float] = {}

    for frame in frames:
        step = frame.get("step")
        frame_index = (step + 1) if isinstance(step, int) else None
        time_s = float(frame.get("time_seconds", 0.0))
        for p in frame.get("players", []):
            pid = p["id"]
            pos = p.get("position_m", [0.0, 0.0, 0.0])
            vel = p.get("velocity_mps")

            # 位置断点（m/s，用差分速度近似，超过 max_speed 视为瞬移）
            if pid in prev_frame_pos and pid in prev_frame_time:
                dt = time_s - prev_frame_time[pid]
                if dt > 0.0:
                    dx = float(pos[0]) - prev_frame_pos[pid][0]
                    dy = float(pos[1]) - prev_frame_pos[pid][1]
                    inst_speed = math.hypot(dx, dy) / dt
                    if inst_speed > config.max_speed_mps:
                        warnings.append({
                            "frame_index": frame_index,
                            "player_id": pid,
                            "kind": "position_discontinuity",
                            "value": round(inst_speed, 3),
                            "limit": config.max_speed_mps,
                        })

            # 速度尖峰
            if vel is not None:
                spd = math.hypot(float(vel[0]), float(vel[1]))
                if spd > config.max_speed_mps:
                    warnings.append({
                        "frame_index": frame_index,
                        "player_id": pid,
                        "kind": "speed_spike",
                        "value": round(spd, 3),
                        "limit": config.max_speed_mps,
                    })

            # 加速度 / 角速度尖峰
            tracker = trackers.setdefault(pid, PlayerMotionTracker(config=config))
            params = tracker.update(pos, vel, time_s)
            if abs(params["acceleration_mps2"]) > config.max_accel_mps2:
                warnings.append({
                    "frame_index": frame_index,
                    "player_id": pid,
                    "kind": "acceleration_spike",
                    "value": params["acceleration_mps2"],
                    "limit": config.max_accel_mps2,
                })
            if abs(params["turn_rate_deg_s"]) > config.max_yaw_rate_deg_s:
                warnings.append({
                    "frame_index": frame_index,
                    "player_id": pid,
                    "kind": "yaw_rate_spike",
                    "value": params["turn_rate_deg_s"],
                    "limit": config.max_yaw_rate_deg_s,
                })

            prev_frame_pos[pid] = [float(pos[0]), float(pos[1])]
            prev_frame_time[pid] = time_s

    return warnings


def write_motion_debug_jsonl(
    path: Any,
    frames: Sequence[Dict[str, Any]],
    config: MotionConfig = DEFAULT_MOTION_CONFIG,
    gk_entity_ids: Optional[Iterable[str]] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """把逐球员运动参数写到 sidecar JSONL（调试用，不进正式 dataset）。

    写两个文件：
      <path>.motion.jsonl      逐帧逐球员运动参数（speed/velocity/facing/gait/state...）
      <path>.diagnostics.jsonl 轨迹诊断 warning（位置断点/速度尖峰/角速度尖峰/加速度尖峰）

    gk_entity_ids: 守门员实体 ID 集合（None = 用 GK_ENTITY_IDS 回退）。

    Returns:
        (motion_path, warnings)。确定性：相同输入 ⇒ 相同输出。
    """
    import json
    import os
    from pathlib import Path as _Path

    motion_path = _Path(str(path) + ".motion.jsonl")
    diag_path = _Path(str(path) + ".diagnostics.jsonl")

    seq = compute_player_motion_sequence(frames, config, gk_entity_ids=gk_entity_ids)
    warnings = diagnose_trajectory(frames, config)

    motion_path.parent.mkdir(parents=True, exist_ok=True)
    with open(motion_path, "w", encoding="utf-8") as f:
        for entry in seq:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    with open(diag_path, "w", encoding="utf-8") as f:
        for w in warnings:
            f.write(json.dumps(w, ensure_ascii=False) + "\n")
    return str(motion_path), warnings
