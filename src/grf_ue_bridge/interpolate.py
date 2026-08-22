"""GRF 10fps 轨迹 → 目标帧率（如 30fps）的 velocity-aware 插值（纯函数）。

GRF（gfootball）仿真固定 10fps（每步 0.1s），给球 + 10 球员的位置与方向
（每步位移）。Plan B 目标：30fps 标注（渲 900 标 900，1:1）。本模块把 10fps
轨迹插值到 30fps：

- 输出帧数 = 输入帧数 × factor。
- 输出下标 k 的源区间 i = k // factor，分数位置 frac = (k % factor) / factor。
- frac == 0（整数倍）→ 该帧 = 原 GRF 真值帧，位置**原样保留**（只更新 step/time），
  速度 = 原始 GRF 速度（或有限差分估算）。
- frac > 0 → velocity-aware 三次 Hermite 插值（position + 解析导数速度），
  并对 position 做**分段边界 clamp**（overshoot 防护）。

核心不变式：**原始 GRF sample 的 position 必须完全不变**。只允许修改
frac > 0 的补帧位置。

单位约定：
  位置   : position_m（米）
  速度   : velocity_mps（米/秒），Hermite 解析导数得到（不靠相邻帧差分），
           因此与 FPS 无关且无差分噪声。
  段时长 : h = factor × target_step_seconds（即 GRF 步长 0.1s）。

朝向（yaw）不在本模块——由 ue/player_motion 的 compute_facing_yaw 统一计算
（速度 → 朝向，低速滞回 + 平滑限速），在插值后的帧序列上自动一致。

本模块不依赖 unreal / pydantic / numpy，输入输出均为帧 dict（Frame JSON 形状），
可被 pytest 独立测试。
"""

from typing import Dict, List, Optional, Sequence

import math

# 速度/速率输出的极小阈值（m/s）：低于此值 heading 视为无定义（None）
_MIN_HEADING_SPEED_MPS = 1e-9


def _lerp(a: Sequence[float], b: Sequence[float], t: float) -> List[float]:
    """逐分量线性插值，保留 6 位小数（微米级，远低于像素精度）。"""
    return [round(ai + (bi - ai) * t, 6) for ai, bi in zip(a, b)]


def _finite_difference_tangent(positions: Sequence[Sequence[float]], i: int, h: float) -> List[float]:
    """对第 i 个样本做有限差分求切线（单位：位置量纲/秒）。

    边界用单侧差分，内部用中心差分。n==1 时返回零向量。
    """
    n = len(positions)
    dim = len(positions[0])
    if n == 1:
        return [0.0] * dim
    if i == 0:
        lo, hi = positions[0], positions[1]
        return [(hi[j] - lo[j]) / h for j in range(dim)]
    if i == n - 1:
        lo, hi = positions[n - 2], positions[n - 1]
        return [(hi[j] - lo[j]) / h for j in range(dim)]
    lo, hi = positions[i - 1], positions[i + 1]
    return [(hi[j] - lo[j]) / (2.0 * h) for j in range(dim)]


def _compute_tangents(
    positions: Sequence[Sequence[float]],
    velocities: Sequence[Optional[Sequence[float]]],
    h: float,
) -> List[List[float]]:
    """为每个样本计算 Hermite 切线（速度，m/s）。

    优先用帧自带的 velocity_mps（GRF/Hermite 更准确），缺失则有限差分估算。
    """
    tangents: List[List[float]] = []
    for i in range(len(positions)):
        v = velocities[i] if i < len(velocities) else None
        if v is not None:
            tangents.append([float(x) for x in v])
        else:
            tangents.append(_finite_difference_tangent(positions, i, h))
    return tangents


def _hermite_vector(
    p0: Sequence[float],
    p1: Sequence[float],
    v0: Sequence[float],
    v1: Sequence[float],
    h: float,
    u: float,
) -> tuple:
    """单段的 cubic Hermite 插值，返回 (position, velocity)。

    标准 Hermite 基（h 为段时长，v 为端点切线，量纲 = 位置/秒）：
        p(u)  = h00·p0 + h10·h·v0 + h01·p1 + h11·h·v1
        v(u)  = (h00'·p0 + h10'·h·v0 + h01'·p1 + h11'·h·v1) / h
    速度由解析导数给出（连续性好，不靠相邻帧差分）。
    """
    u2 = u * u
    u3 = u2 * u
    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = u3 - 2.0 * u2 + u
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = u3 - u2
    d00 = 6.0 * u2 - 6.0 * u
    d10 = 3.0 * u2 - 4.0 * u + 1.0
    d01 = -6.0 * u2 + 6.0 * u
    d11 = 3.0 * u2 - 2.0 * u

    pos: List[float] = []
    vel: List[float] = []
    for j in range(len(p0)):
        a, b = float(p0[j]), float(p1[j])
        c, d = float(v0[j]), float(v1[j])
        pos.append(h00 * a + h10 * h * c + h01 * b + h11 * h * d)
        vel.append((d00 * a + d10 * h * c + d01 * b + d11 * h * d) / h)
    return pos, vel


def _clamp_to_bounds(
    pos: Sequence[float],
    p0: Sequence[float],
    p1: Sequence[float],
) -> List[float]:
    """把位置 clamp 到分段 [min(p0,p1), max(p0,p1)]（overshoot 防护）。"""
    out: List[float] = []
    for j in range(len(pos)):
        lo = min(float(p0[j]), float(p1[j]))
        hi = max(float(p0[j]), float(p1[j]))
        out.append(max(lo, min(hi, pos[j])))
    return out


def _player_series_2d(
    frames: Sequence[Dict], player_ids: Sequence[str], key: str
) -> Dict[str, List]:
    """提取每个球员的 2 维（x, y）序列，缺失为 None。"""
    series: Dict[str, List] = {}
    for pid in player_ids:
        vals = []
        for f in frames:
            found = None
            for p in f["players"]:
                if p["id"] == pid:
                    found = p
                    break
            v = found.get(key) if found is not None else None
            if v is None:
                vals.append(None)
            else:
                vals.append([float(v[0]), float(v[1])])
        series[pid] = vals
    return series


def interpolate_frames(
    frames: Sequence[Dict],
    factor: int,
    target_step_seconds: float,
) -> List[Dict]:
    """把 10fps 帧序列 velocity-aware Hermite 插值到目标帧率。

    Args:
        frames: 帧 dict 列表，每帧含 step / time_seconds / score /
            ball{position_m, source_grf_position, velocity_mps?} /
            players[{id, position_m, velocity_mps?, speed_mps?,
            movement_heading_deg?, active?, has_ball?}]，
            以及可选的 ball_owned_team / ball_owned_player / game_mode。
        factor: 上采样倍数（10fps→30fps 为 3）。
        target_step_seconds: 目标帧率对应的步长（30fps 为 1/30）。
            段时长 h = factor × target_step_seconds（= GRF 步长 0.1s）。

    Returns:
        插值后的帧 dict 列表，长度 = len(frames) × factor；
        整数倍下标帧保留原 GRF 真值位置（速度 = 原始速度/有限差分），
        其余为 Hermite 插值（位置 + 解析导数速度），speed/heading 从速度重算。
    """
    n = len(frames)
    if n == 0:
        return []
    if factor <= 1:
        return [dict(f) for f in frames]

    h = float(factor) * float(target_step_seconds)

    # 球员 ID 顺序（保持源帧顺序）
    player_ids: List[str] = [p["id"] for p in frames[0]["players"]]

    # 球的位置/速度序列
    ball_positions = [f["ball"]["position_m"] for f in frames]
    ball_velocities = [f["ball"].get("velocity_mps") for f in frames]
    ball_tangents = _compute_tangents(ball_positions, ball_velocities, h)

    # 球员的位置/速度序列（球员水平运动，仅插值 [x, y]；z 恒为 0，不插值）
    player_positions = _player_series_2d(frames, player_ids, "position_m")
    player_velocities = _player_series_2d(frames, player_ids, "velocity_mps")
    player_tangents = {
        pid: _compute_tangents(player_positions[pid], player_velocities[pid], h)
        for pid in player_ids
    }

    def _round6(xs: Sequence[float]) -> List[float]:
        return [round(float(v), 6) for v in xs]

    out: List[Dict] = []
    total = n * factor
    for k in range(total):
        i = k // factor
        frac = (k % factor) / factor
        src = frames[i]

        new_frame = {
            "step": k,
            "time_seconds": round(k * target_step_seconds, 6),
            "score": list(src["score"]),
            "ball": {
                "position_m": list(src["ball"]["position_m"]),
                "source_grf_position": list(src["ball"].get("source_grf_position", [0.0, 0.0, 0.0])),
                "velocity_mps": list(ball_tangents[i]),
            },
            "players": [],
        }
        # 帧级所有权/游戏模式（与 score 一样，随源帧）
        for key in ("ball_owned_team", "ball_owned_player", "game_mode"):
            if key in src:
                new_frame[key] = src[key]

        if frac > 0.0 and i + 1 < n:
            # ── 补帧：Hermite + overshoot 防护 ──────────────────────────
            u = frac
            # 球（3 维位置，含 z）
            bpos, bvel = _hermite_vector(
                ball_positions[i], ball_positions[i + 1],
                ball_tangents[i], ball_tangents[i + 1], h, u,
            )
            bpos = _clamp_to_bounds(bpos, ball_positions[i], ball_positions[i + 1])
            new_frame["ball"]["position_m"] = _round6(bpos)
            new_frame["ball"]["velocity_mps"] = _round6(bvel)

            for pid in player_ids:
                ppos, pvel = _hermite_vector(
                    player_positions[pid][i], player_positions[pid][i + 1],
                    player_tangents[pid][i], player_tangents[pid][i + 1], h, u,
                )
                ppos = _clamp_to_bounds(
                    ppos, player_positions[pid][i], player_positions[pid][i + 1]
                )
                src_player = next(p for p in src["players"] if p["id"] == pid)
                new_frame["players"].append(_build_player_dict(
                    pid, _round6([ppos[0], ppos[1], 0.0]), _round6(pvel), src_player,
                ))
        else:
            # ── 真值帧（frac==0）或末尾 hold（i+1 越界）───────────────
            if frac > 0.0:
                # 末尾 hold：位置保持末帧真值，速度置 0（静止，不凭空续动）
                new_frame["ball"]["velocity_mps"] = [0.0, 0.0, 0.0]
            for pid in player_ids:
                src_player = next(p for p in src["players"] if p["id"] == pid)
                if frac > 0.0:
                    vel = [0.0, 0.0]
                else:
                    vel = list(player_tangents[pid][i])
                new_frame["players"].append(_build_player_dict(
                    pid, list(src_player["position_m"]), _round6(vel), src_player,
                ))

        out.append(new_frame)
    return out


def _build_player_dict(pid: str, position_m: List[float], velocity_mps: List[float],
                       src_player: Dict) -> Dict:
    """由插值后的位置/速度构建球员帧 dict（speed/heading 从速度重算）。

    position_m 为完整 3 维 [x, y, z]（米）：真值帧原样透传 z，补帧 z 恒为 0。
    """
    vx, vy = velocity_mps[0], velocity_mps[1]
    speed = (vx * vx + vy * vy) ** 0.5
    heading = None
    if speed > _MIN_HEADING_SPEED_MPS:
        heading = round(math.degrees(math.atan2(vy, vx)), 6)

    d: Dict = {
        "id": pid,
        "position_m": position_m,
        "velocity_mps": velocity_mps,
        "speed_mps": round(speed, 6),
        "movement_heading_deg": heading,
    }
    # 逐帧属性随源帧（active / has_ball 等）
    for key in ("active", "has_ball"):
        if key in src_player:
            d[key] = src_player[key]
    return d


# ── 时间轴缩放重采样（trajectory_time_scale > 1）───────────────────────

_GRF_SOURCE_STEP_SECONDS = 0.1  # GRF 固定 10fps


def resample_frames_time_scale(
    frames: Sequence[Dict],
    time_scale: float,
    target_fps: int,
    output_frames: int,
) -> List[Dict]:
    """按时间轴缩放重采样：source_time = dataset_time × time_scale。

    语义：dataset 每一帧 k（dataset_time = k / target_fps）对应 GRF 轨迹的
    source_time = k / target_fps × time_scale。位置用 velocity-aware cubic
    Hermite 在真实 GRF sample 之间做**时间型**重采样（与 interpolate_frames
    的定长 factor 插值不同，这里的源采样时刻不落在整数输出帧上）。

    - 当 source_time 恰好落在某个 GRF sample（0.1s 网格）上 → 该输出帧
      position 严格等于该 GRF 真值帧（轨迹连续通过原始 sample）。
    - 其余输出帧为两 sample 之间的 Hermite 补帧（带分段 clamp）。
    - dataset velocity = 位置对 dataset_time 的导数 = source 切线 × time_scale
      （source 切线取帧自带 velocity_mps，缺失时有限差分）。
    - score / ball_owned_team / ball_owned_player / game_mode 等离散状态用
      source_time 对应 sample 的 hold/nearest，不做数值插值。

    Args:
        frames: 10fps GRF 帧 dict 列表（长度 = 所需 GRF sample 数）。
        time_scale: 轨迹时间缩放（> 1 表示加快）。
        target_fps: dataset 输出帧率（30 等）。
        output_frames: dataset 输出帧数（900 等）。

    Returns:
        重采样后的帧 dict 列表，长度 = output_frames。
    """
    n = len(frames)
    if n == 0:
        return []
    h = _GRF_SOURCE_STEP_SECONDS  # 源段时长（GRF 步长 0.1s）
    dt = 1.0 / float(target_fps)

    player_ids: List[str] = [p["id"] for p in frames[0]["players"]]

    # 切线一律用源位置有限差分（而非 GRF direction 字段）：GRF direction 与真实
    # 位移有时偏差可达 ~3×，会导致 Hermite 切线过大 → clamp 压平位置路径，而
    # velocity 仍取解析导数，造成"速度 > 位置差分速度"。用位置差分保证
    # velocity == 重采样位置路径对 dataset_time 的导数（= source 位置导数 × scale）。
    ball_positions = [f["ball"]["position_m"] for f in frames]
    ball_tangents = _compute_tangents(ball_positions, [None] * n, h)

    player_positions = _player_series_2d(frames, player_ids, "position_m")
    player_tangents = {
        pid: _compute_tangents(player_positions[pid], [None] * n, h)
        for pid in player_ids
    }

    def _round6(xs):
        return [round(float(v), 6) for v in xs]

    out: List[Dict] = []
    for k in range(output_frames):
        s = float(k) * dt * float(time_scale)  # source_time
        i = int(math.floor(s / h))
        frac = (s - i * h) / h
        # 浮点容差：把几乎落在 sample 上的时间吸附为精确命中
        r = round(frac)
        if abs(frac - r) < 1e-9:
            if r == 1:
                i += 1
                frac = 0.0
            else:
                frac = 0.0
        i = max(0, min(i, n - 1))
        src = frames[i]

        new_frame = {
            "step": k,
            "time_seconds": round(k * dt, 6),
            "score": list(src["score"]),
            "ball": {
                "position_m": list(src["ball"]["position_m"]),
                "source_grf_position": list(
                    src["ball"].get("source_grf_position", [0.0, 0.0, 0.0])
                ),
                "velocity_mps": _round6([v * time_scale for v in ball_tangents[i]]),
            },
            "players": [],
        }
        # 离散状态：随 source_time 对应 sample 的 hold/nearest
        for key in ("ball_owned_team", "ball_owned_player", "game_mode"):
            if key in src:
                new_frame[key] = src[key]

        if frac > 1e-9 and i + 1 < n:
            # ── 两 sample 之间的 Hermite 补帧 ──────────────────────────
            u = frac
            bpos, bvel = _hermite_vector(
                ball_positions[i], ball_positions[i + 1],
                ball_tangents[i], ball_tangents[i + 1], h, u,
            )
            bpos = _clamp_to_bounds(bpos, ball_positions[i], ball_positions[i + 1])
            new_frame["ball"]["position_m"] = _round6(bpos)
            new_frame["ball"]["velocity_mps"] = _round6([v * time_scale for v in bvel])

            for pid in player_ids:
                ppos, pvel = _hermite_vector(
                    player_positions[pid][i], player_positions[pid][i + 1],
                    player_tangents[pid][i], player_tangents[pid][i + 1], h, u,
                )
                ppos = _clamp_to_bounds(
                    ppos, player_positions[pid][i], player_positions[pid][i + 1]
                )
                src_player = next(p for p in src["players"] if p["id"] == pid)
                new_frame["players"].append(_build_player_dict(
                    pid,
                    _round6([ppos[0], ppos[1], 0.0]),
                    _round6([v * time_scale for v in pvel]),
                    src_player,
                ))
        else:
            # ── 精确命中 GRF sample（frac==0）或末尾越界 hold ─────────
            if frac > 1e-9:
                # 末尾越界（异常保护）：位置保持末帧真值，速度置 0
                new_frame["ball"]["velocity_mps"] = [0.0, 0.0, 0.0]
            for pid in player_ids:
                src_player = next(p for p in src["players"] if p["id"] == pid)
                if frac > 1e-9:
                    vel = [0.0, 0.0]
                else:
                    vel = [v * time_scale for v in player_tangents[pid][i]]
                new_frame["players"].append(_build_player_dict(
                    pid, list(src_player["position_m"]), _round6(vel), src_player,
                ))

        out.append(new_frame)
    return out
