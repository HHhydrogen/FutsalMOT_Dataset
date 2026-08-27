"""P1：Phase 1 投影偏移量化 — pose_bbox vs actor_mask_bbox。

读 mask_bbox_c4.jsonl（真实像素弱 GT）+ coco17_2d_c4.jsonl（离线投影 pose）。
对关键帧 [0,15,30,45,60,75,89] 全部 10 actor 计算：
  dx = pose_center_x - mask_center_x
  dy = pose_center_y - mask_center_y
  pose_w/mask_w, pose_h/mask_h（缩放比）
并汇总判断偏移类型（平移/缩放/相机/其他）。
"""

import json
import math
from pathlib import Path

EP = Path(r"G:\FutsalMOT_Dataset\episode_bp_frame_sync")
CAM = EP / "CineCam_01"
MB = CAM / "mask_bbox_c4.jsonl"
KP = EP / "coco17_2d_c4.jsonl"
KEY_FRAMES = [0, 15, 30, 45, 60, 75, 89]
ACTORS = [f"L{i}" for i in range(5)] + [f"R{i}" for i in range(5)]


def center(bb):
    return ((bb["xmin"] + bb["xmax"]) / 2.0, (bb["ymin"] + bb["ymax"]) / 2.0)


def width(bb):
    return bb["xmax"] - bb["xmin"]


def height(bb):
    return bb["ymax"] - bb["ymin"]


def main():
    mb_rows = [json.loads(l) for l in MB.read_text(encoding="utf-8").splitlines()]
    kp_rows = [json.loads(l) for l in KP.read_text(encoding="utf-8").splitlines()]

    mask_by = {(r["root"], r["actor_id"]): r for r in mb_rows}
    pose_by = {}
    for r in kp_rows:
        pts = r["keypoints_2d_px"]
        vis = r["visible"]
        xs = [p[0] for p, v in zip(pts, vis) if v and p]
        ys = [p[1] for p, v in zip(pts, vis) if v and p]
        if xs and ys:
            pose_by[(r["root"], r["actor_id"])] = {
                "xmin": min(xs), "xmax": max(xs), "ymin": min(ys), "ymax": max(ys)}

    print(f"mask 记录: {len(mb_rows)}, pose 记录: {len(kp_rows)}")
    print(f"\n=== Phase 1 偏移量化（关键帧 × 全部 10 actor）===")
    print(f"{'frame':>5} {'actor':>5} {'dx':>8} {'dy':>8} {'pw/mw':>7} {'ph/mh':>7}")

    dxs, dys, wr, hr = [], [], [], []
    all_rows = []
    for root in KEY_FRAMES:
        for a in ACTORS:
            mb = mask_by.get((root, a))
            pb = pose_by.get((root, a))
            if not mb or not pb:
                continue
            pc, mc = center(pb), center(mb)
            dx = pc[0] - mc[0]
            dy = pc[1] - mc[1]
            w_ratio = width(pb) / width(mb)
            h_ratio = height(pb) / height(mb)
            dxs.append(dx)
            dys.append(dy)
            wr.append(w_ratio)
            hr.append(h_ratio)
            all_rows.append((root, a, dx, dy, w_ratio, h_ratio))
            print(f"{root:>5} {a:>5} {dx:>8.1f} {dy:>8.1f} {w_ratio:>7.3f} {h_ratio:>7.3f}")

    if not all_rows:
        print("  ERROR: 无数据可比较")
        return

    print("\n=== 汇总统计 ===")
    n = len(all_rows)
    mean_dx = sum(r[2] for r in all_rows) / n
    mean_dy = sum(r[3] for r in all_rows) / n
    mean_wr = sum(r[4] for r in all_rows) / n
    mean_hr = sum(r[5] for r in all_rows) / n
    madx = sum(abs(r[2]) for r in all_rows) / n
    mady = sum(abs(r[3]) for r in all_rows) / n
    print(f"mean dx = {mean_dx:.1f} px, mean dy = {mean_dy:.1f} px")
    print(f"mean |dx| = {madx:.1f} px, mean |dy| = {mady:.1f} px")
    print(f"mean pose_w/mask_w = {mean_wr:.3f}, mean pose_h/mask_h = {mean_hr:.3f}")

    # 跨帧稳定性：dx/dy 的 std（判断是固定平移还是逐帧变化）
    import statistics
    std_dx = statistics.pstdev([r[2] for r in all_rows])
    std_dy = statistics.pstdev([r[3] for r in all_rows])
    print(f"std dx = {std_dx:.1f} px, std dy = {std_dy:.1f} px（跨 actor/frame 波动）")

    # 缩放一致性：w_ratio / h_ratio 是否接近恒定且 <1（pose 偏小 = 焦距不足）
    print(f"w_ratio range: {min(r[4] for r in all_rows):.3f}..{max(r[4] for r in all_rows):.3f}")
    print(f"h_ratio range: {min(r[5] for r in all_rows):.3f}..{max(r[5] for r in all_rows):.3f}")

    # 推断：1280->1920 理论偏移（fx 808→1212, cx 639.5→959.5, cy 359.5→539.5）
    print("\n=== 理论推断（intrinsics 1280x720 用在 1920x1080）===")
    print("  u_correct = 959.5 + 1.5*(u_wrong - 639.5)")
    print("  v_correct = 539.5 + 1.5*(v_wrong - 359.5)")
    print("  → pose 相对 mask：中心向左上偏移 + 尺度 ×0.667")


if __name__ == "__main__":
    main()