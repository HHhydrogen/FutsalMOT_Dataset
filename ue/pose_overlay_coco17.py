"""从 coco17_2d.jsonl 直接生成 Pose overlay（无需 mask/labels_pose）。

读 <ep>/<cam>/img1/{frame:06d}.png + <ep>/<cam>/coco17_2d.jsonl
写 <ep>/<cam>/debug/pose/{frame:06d}.png（关键点 + COCO 骨架连线）

env C5_EPISODE_DIR, C5_COCO17_CAMERAS (逗号分隔)
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pose_bones import COCO_SKELETON_EDGES  # noqa: E402

def main():
    ep = Path(os.environ.get("C5_EPISODE_DIR", "."))
    cameras = [c.strip() for c in os.environ.get("C5_COCO17_CAMERAS", "CineCam_01").split(",") if c.strip()]
    out_total = 0
    for cam in cameras:
        c2d = ep / cam / "coco17_2d.jsonl"
        img_dir = ep / cam / "img1"
        out = ep / cam / "debug" / "pose"
        out.mkdir(parents=True, exist_ok=True)
        if not c2d.is_file():
            print(f"ERROR [{cam}]: 缺 {c2d}")
            continue
        from PIL import Image, ImageDraw
        rows = [json.loads(l) for l in c2d.read_text(encoding="utf-8").splitlines() if l.strip()]
        by_root = {}
        for r in rows:
            by_root.setdefault(r["root"], []).append(r)
        drawn = 0
        from pose_bones import COCO_KEYPOINT_NAMES
        idx = {n: i for i, n in enumerate(COCO_KEYPOINT_NAMES)}
        edges = [(idx[a], idx[b]) for a, b in COCO_SKELETON_EDGES if a in idx and b in idx]
        for root, actors in sorted(by_root.items()):
            fi = root + 1  # frame_index = step+1
            img_path = img_dir / f"{fi:06d}.png"
            if not img_path.exists():
                print(f"  [{cam}] skip frame {fi}: img1 not found")
                continue
            with Image.open(img_path) as im:
                draw = ImageDraw.Draw(im, "RGBA")
                for r in actors:
                    kps = r.get("keypoints_2d_px") or []
                    vis = r.get("visible") or []
                    if len(kps) != 17:
                        continue
                    pts = []
                    for i, p in enumerate(kps):
                        if p is None or not vis[i]:
                            pts.append(None)
                        else:
                            pts.append((p[0], p[1]))
                    for a, b in edges:
                        if pts[a] is not None and pts[b] is not None:
                            draw.line([pts[a], pts[b]], fill=(0, 255, 0, 200), width=3)
                    for i, p in enumerate(pts):
                        if p is None:
                            continue
                        color = (0, 255, 0, 255) if vis[i] else (255, 165, 0, 255)
                        x, y = p
                        draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=color)
                im.save(out / f"{fi:06d}.png")
            drawn += 1
            print(f"  [{cam}] drew frame {fi} -> {out / f'{fi:06d}.png'}")
        print(f"Pose overlay [{cam}] (coco17_2d): {drawn} 帧 -> {out}")
        out_total += drawn
    print(f"Pose overlay total: {out_total} 帧")
    return 0 if out_total else 1

if __name__ == "__main__":
    sys.exit(main())
