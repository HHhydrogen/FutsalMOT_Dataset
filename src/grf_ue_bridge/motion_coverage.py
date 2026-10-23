"""根轨迹运动及几何邻近覆盖统计，不推断肢体动作、接触或遮挡。"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from itertools import combinations


@dataclass(frozen=True)
class CoverageThresholds:
    """探索性描述阈值；不是生理学或真实动作的标注标准。"""
    active_speed_mps: float = 0.5
    gait_bounds_mps: tuple = (0.3, 1.0, 2.0, 3.5)
    accel_mps2: float = 1.2
    turn_rate_deg_s: float = 90.0
    min_turn_speed_mps: float = 0.5
    near_distances_m: tuple = (1.0, 2.0)
    min_interaction_duration_s: float = 0.5
    min_active_outfields: int = 4


def bouts(flags, durations, minimum_s=0.0):
    """连续真值区间的时间总量；假值必须分段，不跨帧拼接。"""
    if len(flags) != len(durations):
        raise ValueError("布尔区间与时间长度不一致")
    runs, current = [], 0.0
    for flag, dt in zip(flags, durations):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("区间时长必须是有限正数")
        if flag:
            current += dt
        elif current:
            runs.append(current)
            current = 0.0
    if current:
        runs.append(current)
    return {"total_s": sum(runs), "longest_s": max(runs, default=0.0),
            "episode_count": len(runs),
            "persistent_episode_count": sum(r + 1e-9 >= minimum_s for r in runs)}


def quantiles(values):
    """线性插值分位数；空支持返回 null，不伪造零值。"""
    values = sorted(values)
    def q(p):
        if not values:
            return None
        i = (len(values) - 1) * p
        lo, hi = math.floor(i), math.ceil(i)
        return values[lo] + (values[hi] - values[lo]) * (i - lo)
    return {"p05": q(.05), "p50": q(.5), "p95": q(.95), "max": max(values) if values else None}


def analyze_frames(frames, entities, thresholds=None):
    """位置差分定义速度；相邻速度中心差分定义速率变化及转向。"""
    cfg = thresholds or CoverageThresholds()
    # group.entities 同时声明 BALL；players 逐帧数组只包含球员。
    entities = [e for e in entities if e["entity_id"] != "BALL"]
    if len(frames) < 3:
        raise ValueError("至少三个连续样本才能统计加速度/转向")
    entity_ids = [e["entity_id"] for e in entities]
    if len(set(entity_ids)) != len(entity_ids):
        raise ValueError("实体重复")
    times = [float(f["time_seconds"]) for f in frames]
    if not all(math.isfinite(t) for t in times):
        raise ValueError("时间非有限数")
    dt = [b - a for a, b in zip(times, times[1:])]
    if any(v <= 0 for v in dt):
        raise ValueError("时间不严格递增")
    if any(b["step"] != a["step"] + 1 for a, b in zip(frames, frames[1:])):
        raise ValueError("step 不连续，禁止跨缺帧求导")
    players = []
    for frame in frames:
        ids = [p["id"] for p in frame["players"]]
        if len(ids) != len(set(ids)) or set(ids) != set(entity_ids):
            raise ValueError("每帧实体必须唯一且与group一致")
        row = {p["id"]: p for p in frame["players"]}
        if any(len(p["position_m"]) < 2 or not all(math.isfinite(float(x)) for x in p["position_m"])
               for p in row.values()):
            raise ValueError("位置不合法")
        players.append(row)
    acceleration_dt = [(dt[i - 1] + dt[i]) / 2 for i in range(1, len(dt))]
    span = times[-1] - times[0]
    active_flags, output = {}, {}
    speeds_all = []
    gait_total = {k: 0.0 for k in ("idle", "walk", "jog", "run", "sprint")}
    for identity in entity_ids:
        positions = [r[identity]["position_m"] for r in players]
        velocities = [[(b[k] - a[k]) / duration for k in (0, 1)]
                      for a, b, duration in zip(positions, positions[1:], dt)]
        speeds = [math.hypot(*v) for v in velocities]
        speeds_all.extend(speeds)
        accelerations = [(speeds[i] - speeds[i - 1]) / acceleration_dt[i - 1]
                         for i in range(1, len(speeds))]
        turn_rates = []
        valid_turn = []
        for i in range(1, len(velocities)):
            a, b = velocities[i - 1], velocities[i]
            delta = math.degrees(math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]))
            valid = min(speeds[i - 1], speeds[i]) >= cfg.min_turn_speed_mps
            valid_turn.append(valid)
            turn_rates.append(delta / acceleration_dt[i - 1] if valid else None)
        gait_seconds = {k: 0.0 for k in gait_total}
        for speed, duration in zip(speeds, dt):
            band = sum(speed > bound for bound in cfg.gait_bounds_mps)
            label = ("idle", "walk", "jog", "run", "sprint")[band]
            gait_seconds[label] += duration
            gait_total[label] += duration
        active = [s >= cfg.active_speed_mps for s in speeds]
        active_flags[identity] = active
        output[identity] = {
            "speed_mps": quantiles(speeds), "path_length_m": sum(s * d for s, d in zip(speeds, dt)),
            "gait_seconds": gait_seconds, "active": bouts(active, dt),
            "acceleration_mps2": quantiles(accelerations),
            "acceleration": bouts([a >= cfg.accel_mps2 for a in accelerations], acceleration_dt),
            "braking": bouts([a <= -cfg.accel_mps2 for a in accelerations], acceleration_dt),
            "turn_rate_abs_deg_s": quantiles([abs(v) for v in turn_rates if v is not None]),
            "sharp_turn": bouts([v is not None and abs(v) >= cfg.turn_rate_deg_s for v in turn_rates], acceleration_dt),
            "turn_valid_support_s": sum(d for v, d in zip(valid_turn, acceleration_dt) if v),
        }
    outfields = [e["entity_id"] for e in entities if not e.get("is_goalkeeper", False)]
    team_active = [sum(active_flags[p][i] for p in outfields) >= cfg.min_active_outfields for i in range(len(dt))]
    entity_team = {e["entity_id"]: e["team"] for e in entities}
    proximity = {}
    for threshold in cfg.near_distances_m:
        pairs = {}
        for a, b in combinations(entity_ids, 2):
            # 区间中点线性位置，避免把末帧重复算一个完整时间间隔。
            flags = []
            for i in range(len(dt)):
                pa = [(players[i][a]["position_m"][k] + players[i + 1][a]["position_m"][k]) / 2 for k in (0, 1)]
                pb = [(players[i][b]["position_m"][k] + players[i + 1][b]["position_m"][k]) / 2 for k in (0, 1)]
                flags.append(math.hypot(pa[0] - pb[0], pa[1] - pb[1]) <= threshold)
            pairs[a + ":" + b] = {"cross_team": entity_team[a] != entity_team[b],
                                   **bouts(flags, dt, cfg.min_interaction_duration_s)}
        proximity[str(threshold)] = {"pairs": pairs,
            "pair_seconds": sum(p["total_s"] for p in pairs.values()),
            "cross_team_pair_seconds": sum(p["total_s"] for p in pairs.values() if p["cross_team"]),
            "persistent_pair_episode_count": sum(p["persistent_episode_count"] for p in pairs.values()),
            "pair_time_denominator_s": len(pairs) * span}
    return {"frame_count": len(frames), "interval_count": len(dt), "trajectory_span_s": span,
            "time_interval_s": quantiles(dt), "derivative_support_s": sum(acceleration_dt),
            "entity_count": len(entity_ids), "player_seconds": len(entity_ids) * span,
            "pooled_interval_speed_mps": quantiles(speeds_all), "gait_seconds": gait_total,
            "team_active": bouts(team_active, dt), "team_active_ratio": sum(d for flag, d in zip(team_active, dt) if flag) / span,
            "players": output, "proximity": proximity,
            "thresholds": asdict(cfg)}
