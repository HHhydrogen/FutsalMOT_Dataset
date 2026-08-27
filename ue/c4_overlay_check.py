"""P1：C4 overlay 验收 sanity check + 最近距离/遮挡帧挑选。

读 coco17_2d_c4.jsonl + img1/ + overlay_coco17_c4/。
检查：
  1. 每帧恰好 10 actor，ID 严格 {L0..L4,R0..R4}
  2. 每 actor 17 keypoints，无 NaN/Inf
  3. 每 actor 投影点不全在同一位置
  4. 同一 actor nose(0) 与 ankle(15) 2D 距离明显非零（跨人体高度）
  5. L/R 两队空间分离（不混淆）
  6. InSocketName 修复验证：指定帧+actor 打印 head→nose、hand_l→9、hand_r→10、
     thigh_l→11、calf_l→13、foot_l→15 的 2D 投影，确认各不相同
  7. 全帧 actor 中心两两最近距离 → 挑出球员最接近（最可能遮挡）的 3 帧
"""

import json
import math
from pathlib import Path

EP = Path(r"G:\FutsalMOT_Dataset\episode_bp_frame_sync")
KP2D = EP / "coco17_2d_c4.jsonl"
OV = EP / "overlay_coco17_c4"
IMG = EP / "CineCam_01" / "img1"

EXPECT_ACTORS = [f"L{i}" for i in range(5)] + [f"R{i}" for i in range(5)]
# COCO17 索引：nose=0, wrist=9/10, hip=11/12, knee=13/14, ankle=15/16
PROBE = [("nose", 0), ("left_wrist", 9), ("right_wrist", 10),
         ("left_hip", 11), ("left_knee", 13), ("left_ankle", 15)]


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def main():
    rows = [json.loads(l) for l in KP2D.read_text(encoding="utf-8").splitlines()]
    by_root = {}
    for r in rows:
        by_root.setdefault(r["root"], []).append(r)

    print(f"总 actor×帧 记录: {len(rows)}")
    print(f"overlay 文件数: {len(list(OV.glob('*.png')))}")
    print(f"img1 文件数: {len(list(IMG.glob('*.png')))}")

    problems = []
    frame_centers = {}

    for root in sorted(by_root):
        frs = by_root[root]
        ids = {f["actor_id"] for f in frs}
        if ids != set(EXPECT_ACTORS):
            problems.append(f"frame {root}: actor 集合 {sorted(ids)} != 期望")
        centers = {}
        for fr in frs:
            aid = fr["actor_id"]
            kp = fr["keypoints_2d_px"]
            vis = fr["visible"]
            if len(kp) != 17:
                problems.append(f"frame {root} {aid}: keypoints {len(kp)} != 17")
                continue
            # NaN/Inf
            for p in kp:
                if p is not None and any(math.isnan(v) or math.isinf(v) for v in p):
                    problems.append(f"frame {root} {aid}: NaN/Inf keypoint")
            # 17 点是否全同
            pts = [p for p in kp if p]
            if pts:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                span = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
                if span < 1.0:
                    problems.append(f"frame {root} {aid}: 投影全集中 ({span:.3f}px)")
                centers[aid] = (sum(xs) / len(xs), sum(ys) / len(ys))
            # head(nose) vs ankle 距离
            if kp[0] and kp[15]:
                d = _dist(kp[0], kp[15])
                if d < 20.0:
                    problems.append(f"frame {root} {aid}: nose-ankle 距离过小 {d:.1f}px")
        if len(centers) == 10:
            frame_centers[root] = centers

    print(f"\n=== sanity 结果 ===")
    print(f"问题数: {len(problems)}")
    for p in problems[:30]:
        print(f"  PROBLEM: {p}")

    # L/R 分离：每帧 L 平均 x vs R 平均 x
    lr_diffs = []
    for root, centers in sorted(frame_centers.items()):
        lx = sum(centers[a][0] for a in EXPECT_ACTORS if a.startswith("L")) / 5
        rx = sum(centers[a][0] for a in EXPECT_ACTORS if a.startswith("R")) / 5
        lr_diffs.append((abs(lx - rx), root, lx, rx))
    min_diff = min(lr_diffs, key=lambda t: t[0])
    print(f"L/R 平均x最小差: {min_diff[0]:.1f}px @ frame {min_diff[1]} (Lx={min_diff[2]:.1f}, Rx={min_diff[3]:.1f})")
    print(f"L/R 平均x最大差: {max(lr_diffs, key=lambda t: t[0])[0]:.1f}px")

    # 最近距离帧（两两 actor 中心最小距离 → 球员最接近）
    frame_min_dists = []
    for root, centers in sorted(frame_centers.items()):
        dmin = min(_dist(centers[a], centers[b])
                   for i, a in enumerate(EXPECT_ACTORS)
                   for b in EXPECT_ACTORS[i + 1:])
        frame_min_dists.append((dmin, root))
    top_close = sorted(frame_min_dists)[:3]
    print(f"\n=== 球员最接近（最可能遮挡）的 3 帧 ===")
    for d, root in top_close:
        print(f"  frame {root}: 两两最小中心距离 {d:.1f}px")

    # InSocketName 修复验证：抽查帧 30, actor L2（或最接近帧之一）
    probe_root = 30
    probe_actor = "L2"
    fr = next((f for f in by_root[probe_root] if f["actor_id"] == probe_actor), None)
    print(f"\n=== InSocketName 2D 验证: frame {probe_root}, actor {probe_actor} ===")
    if fr:
        kp = fr["keypoints_2d_px"]
        for name, idx in PROBE:
            p = kp[idx]
            print(f"  {name:<12} idx={idx:<2} -> {p}")
        pts = [kp[idx] for _, idx in PROBE if kp[idx]]
        uniq = {(round(p[0], 1), round(p[1], 1)) for p in pts}
        print(f"  6 点去重后不同位置数: {len(uniq)} (期望 6)")
    else:
        print("  未找到该 actor")

    print(f"\n=== 关键帧 overlay 路径 ===")
    for root in [0, 15, 30, 45, 60, 75, 89] + [r for _, r in top_close]:
        p = OV / f"{root + 1:06d}.png"
        print(f"  frame {root:>3} -> {p}")


if __name__ == "__main__":
    main()