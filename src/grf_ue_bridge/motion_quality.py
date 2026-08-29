"""Motion Quality Audit（C6-P1.5）。

对一段 GRF trajectory（米制 frames，形如 frames.jsonl 的 players[].position_m）做运动质量审计，
用于：
  1. 诊断 GRF 后段静止问题（active_ratio / longest_stationary_streak）；
  2. 寻找连续 active-play window（供 deterministic crop，避免循环/人为移动）；
  3. 输出 JSON 报告（+ 可选诊断图）。

阈值是 QC 语义，不是 GRF Ground Truth semantic label：
  STATIONARY < 0.20 m/s
  ACTIVE     >= 0.50 m/s
外场球员：active_ratio >= 0.75，longest_stationary_streak <= 2.0s
门将：     longest_stationary_streak <= 5.0s
Team：     active_outfield_count >= 6 覆盖 >= 90% active-play 时间

env:
  MQ_FRAMES_JSONL  待审计的 frames.jsonl（可选；否则用 C5_POSE_TASK/resolved 派生）
  MQ_MIN_WINDOW_S  期望的最小连续 active-play 秒数（默认 60）
  MQ_OUT_JSON      报告输出路径（默认 <frames_dir>/motion_quality.json）

纯 Python + 标准库，可 import 可 CLI。
"""

import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

STATIONARY_MPS = 0.20
ACTIVE_MPS = 0.50
OUTFIELD_GK_RATIO_REQ = 0.75
OUTFIELD_STREAK_MAX_S = 2.0
GK_STREAK_MAX_S = 5.0
TEAM_ACTIVE_MIN = 6
TEAM_ACTIVE_COVERAGE_REQ = 0.90


def _gks(frames, gk_ids=None):
    """返回门将实体集合。优先用显式 gk_ids（来自 meta.entities.is_goalkeeper），
    否则从 frames 的 role/is_goalkeeper 推断。"""
    if gk_ids:
        return set(gk_ids)
    gks = set()
    for fr in frames:
        for p in fr.get("players", []):
            if p.get("is_goalkeeper") or p.get("role") == "goalkeeper":
                gks.add(p["id"])
    return gks


def analyze_frames(frames, dt_s=1.0 / 30.0, gk_ids=None):
    """审计一段帧序列，返回 metrics dict。

    frames: [{players:[{id, position_m:[x,y,z], ...}], ...}]（position_m 米）。
    dt_s: 相邻帧时间间隔（秒）。缺省 30fps。
    gk_ids: 门将实体集合（可选；来自 meta.entities.is_goalkeeper）。
    """
    ids = []
    for fr in frames:
        for p in fr.get("players", []):
            if p["id"] not in ids:
                ids.append(p["id"])
    gks = _gks(frames, gk_ids)
    n = len(frames)

    # 用位置差分计算每球员逐帧 speed（与 Hermite velocity 一致性更好）
    speeds = {pid: [0.0] * n for pid in ids}
    prev = {}
    for i, fr in enumerate(frames):
        for p in fr.get("players", []):
            pid = p["id"]
            pos = p["position_m"]
            if pid in prev:
                dx = pos[0] - prev[pid][0]
                dy = pos[1] - prev[pid][1]
                speeds[pid][i] = math.hypot(dx, dy) / dt_s
            prev[pid] = pos

    per_player = {}
    all_speeds = []
    for pid in ids:
        spd = speeds[pid]
        all_speeds.extend(spd)
        ss = sorted(spd)
        m = len(ss)
        active = sum(1 for v in spd if v >= ACTIVE_MPS) / m
        stationary = sum(1 for v in spd if v < STATIONARY_MPS) / m
        # longest stationary streak
        best = cur = 0
        for v in spd:
            if v < STATIONARY_MPS:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        per_player[pid] = {
            "mean_speed": round(sum(spd) / m, 3),
            "median_speed": round(ss[m // 2], 3),
            "max_speed": round(max(spd), 3),
            "active_ratio": round(active, 3),
            "stationary_ratio": round(stationary, 3),
            "longest_stationary_streak_s": round(best * dt_s, 2),
            "is_gk": pid in gks,
        }

    # 逐帧 active outfield count
    active_outfield_per_frame = [0] * n
    stationary_outfield_plateau = []  # (start, end, count) 全局低运动平台
    for i, fr in enumerate(frames):
        cnt = 0
        for p in fr.get("players", []):
            if p["id"] in gks:
                continue
            if speeds[p["id"]][i] >= ACTIVE_MPS:
                cnt += 1
        active_outfield_per_frame[i] = cnt

    # team active_outfield coverage（>=6 覆盖比例）
    frames_team_ok = sum(1 for c in active_outfield_per_frame if c >= TEAM_ACTIVE_MIN)
    team_coverage = frames_team_ok / n if n else 0

    # 全局低运动平台：连续帧 active_outfield < 6
    plateaus = []
    cur_start = None
    for i in range(n):
        if active_outfield_per_frame[i] < TEAM_ACTIVE_MIN:
            if cur_start is None:
                cur_start = i
        else:
            if cur_start is not None:
                plateaus.append((cur_start, i - 1, (i - cur_start) * dt_s))
                cur_start = None
    if cur_start is not None:
        plateaus.append((cur_start, n - 1, (n - cur_start) * dt_s))
    longest_plateau = max((d for _, _, d in plateaus), default=0)

    metrics = {
        "total_frames": n,
        "duration_s": round((n - 1) * dt_s, 2),
        "dt_s": dt_s,
        "players": per_player,
        "gks": sorted(gks),
        "outfields": sorted([pid for pid in ids if pid not in gks]),
        "global_mean_speed": round(sum(all_speeds) / len(all_speeds) if all_speeds else 0, 3),
        "team_active_outfield_coverage": round(team_coverage, 3),
        "longest_global_low_motion_plateau_s": round(longest_plateau, 2),
        "stationary_plateau_count": len(plateaus),
    }
    # 逐窗口（5s, stride 1s）统计
    metrics["windows"] = _sliding_windows(frames, speeds, ids, gks, dt_s)
    return metrics


def _sliding_windows(frames, speeds, ids, gks, dt_s, win_s=5.0, stride_s=1.0):
    n = len(frames)
    win = int(win_s / dt_s)
    stride = int(stride_s / dt_s)
    windows = []
    for start in range(0, max(1, n - win + 1), stride):
        end = min(start + win, n)
        speeds_in = []
        active_cnt = 0
        active_of = 0
        ball_disp = 0.0
        for i in range(start, end):
            for pid in ids:
                speeds_in.append(speeds[pid][i])
            # ball displacement
            if i < n - 1 and "ball" in frames[i] and "ball" in frames[i + 1]:
                b0 = frames[i]["ball"]["position_m"]
                b1 = frames[i + 1]["ball"]["position_m"]
                ball_disp += math.hypot(b1[0] - b0[0], b1[1] - b0[1])
        mean = sum(speeds_in) / len(speeds_in) if speeds_in else 0
        windows.append({
            "t_start_s": round(start * dt_s, 1),
            "mean_player_speed": round(mean, 3),
            "ball_displacement_m": round(ball_disp, 2),
        })
    return windows


def find_active_window(frames, min_duration_s, dt_s=1.0 / 30.0, gk_ids=None):
    """寻找连续 active-play window（返回 [start, end) 帧索引）。

    窗口内累计满足（允许个别静止点，符合真实比赛）：
      - 每个外场球员 active_ratio >= 0.75（stationary_ratio <= 0.25）
      - 门将 longest_stationary_streak <= 5s
      - active_outfield_count >= 6 覆盖 >= 90%
    返回 None 若找不到。
    """
    ids = []
    for fr in frames:
        for p in fr.get("players", []):
            if p["id"] not in ids:
                ids.append(p["id"])
    gks = _gks(frames, gk_ids)
    n = len(frames)
    win = int(min_duration_s / dt_s)
    if win > n:
        return None

    speeds = {pid: [0.0] * n for pid in ids}
    prev = {}
    for i, fr in enumerate(frames):
        for p in fr.get("players", []):
            pid = p["id"]
            pos = p["position_m"]
            if pid in prev:
                dx = pos[0] - prev[pid][0]
                dy = pos[1] - prev[pid][1]
                speeds[pid][i] = math.hypot(dx, dy) / dt_s
            prev[pid] = pos

    # 预计算每帧 active outfield count 前缀和
    active_pref = [0] * (n + 1)
    for i, fr in enumerate(frames):
        cnt = 0
        for p in fr.get("players", []):
            if p["id"] not in gks and speeds[p["id"]][i] >= ACTIVE_MPS:
                cnt += 1
        active_pref[i + 1] = active_pref[i] + cnt

    best = None
    for start in range(0, n - win + 1):
        end = start + win
        # 外场 active_ratio + 门将 streak
        ok = True
        for pid in ids:
            spd = speeds[pid][start:end]
            active = sum(1 for v in spd if v >= ACTIVE_MPS) / win
            if pid in gks:
                best_streak = cur = 0
                for v in spd:
                    if v < STATIONARY_MPS:
                        cur += 1
                        best_streak = max(best_streak, cur)
                    else:
                        cur = 0
                if best_streak * dt_s > GK_STREAK_MAX_S:
                    ok = False
                    break
            else:
                if active < OUTFIELD_GK_RATIO_REQ:
                    ok = False
                    break
        if not ok:
            continue
        team_ok_frames = 0
        for i in range(start, end):
            if active_pref[i + 1] - active_pref[i] >= TEAM_ACTIVE_MIN:
                team_ok_frames += 1
        if team_ok_frames / win >= TEAM_ACTIVE_COVERAGE_REQ:
            best = (start, end)
    return best


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Motion Quality Audit")
    ap.add_argument("--frames", help="frames.jsonl 路径")
    ap.add_argument("--min-window-s", type=float, default=60.0, help="期望连续 active window 秒数")
    ap.add_argument("--out", help="JSON 报告输出路径")
    args = ap.parse_args()

    frames_path = args.frames or os.environ.get("MQ_FRAMES_JSONL")
    if not frames_path:
        print("ERROR: 需要 --frames 或 MQ_FRAMES_JSONL")
        return 1
    frames = [json.loads(l) for l in open(frames_path, encoding="utf-8").readlines() if l.strip()]
    metrics = analyze_frames(frames)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    out = args.out or os.environ.get("MQ_OUT_JSON") or str(Path(frames_path).parent / "motion_quality.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
