"""GRF 10fps 位置轨迹 → 目标帧率（如 30fps）的线性插值（纯函数）。

GRF（gfootball）仿真固定 10fps（每步 0.1s），只给球 + 10 球员的位置。
Plan B 目标：30fps 标注（渲 900 标 900，1:1）。把 10fps 位置线性插值到 30fps：

- 输出帧数 = 输入帧数 × factor。
- 输出下标 k 的源区间 i = k // factor，分数位置 frac = (k % factor) / factor。
- frac == 0（整数倍）→ 该帧 = 原 GRF 真值帧，位置原样保留（只更新 step/time）。
- frac > 0 → 位置线性插值 `lerp(src, next, frac)`；next 越界时 clamp 到末帧（末尾保持）。

仅插值位置（ball x/y/z、player x/y），score 取源帧；step/time_seconds 重写为新 30fps 索引。
朝向（yaw）不在本模块——下游渲染与标注都用同一套 build_yaw（位置增量 + 低速滞回），
在插值后的帧序列上自动一致。

本模块不依赖 unreal / pydantic，输入输出均为帧 dict（Frame JSON 形状），可被 pytest 独立测试。
"""

from typing import Dict, List, Sequence


def _lerp(a: Sequence[float], b: Sequence[float], t: float) -> List[float]:
    """逐分量线性插值，保留 6 位小数（微米级，远低于像素精度）。"""
    return [round(ai + (bi - ai) * t, 6) for ai, bi in zip(a, b)]


def interpolate_frames(
    frames: Sequence[Dict],
    factor: int,
    target_step_seconds: float,
) -> List[Dict]:
    """把 10fps 帧序列线性插值到目标帧率。

    Args:
        frames: 帧 dict 列表，每帧含 step / time_seconds / score /
            ball{position_m, source_grf_position} / players[{id, position_m}]。
        factor: 上采样倍数（10fps→30fps 为 3）。
        target_step_seconds: 目标帧率对应的步长（30fps 为 1/30）。

    Returns:
        插值后的帧 dict 列表，长度 = len(frames) × factor；
        整数倍下标帧保留原 GRF 真值位置，其余为线性插值近似。
    """
    n = len(frames)
    if n == 0:
        return []
    if factor <= 1:
        return [dict(f) for f in frames]
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
            },
            "players": [{"id": p["id"], "position_m": list(p["position_m"])}
                        for p in src["players"]],
        }
        if frac > 0.0:
            nxt = frames[min(i + 1, n - 1)]
            new_frame["ball"]["position_m"] = _lerp(
                src["ball"]["position_m"], nxt["ball"]["position_m"], frac
            )
            for idx, p in enumerate(new_frame["players"]):
                p["position_m"] = _lerp(
                    src["players"][idx]["position_m"],
                    nxt["players"][idx]["position_m"],
                    frac,
                )
        out.append(new_frame)
    return out
