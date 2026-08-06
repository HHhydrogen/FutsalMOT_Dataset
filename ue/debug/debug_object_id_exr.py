"""诊断：检查 Object ID Pass 渲染出的 multilayer EXR 是否含 Cryptomatte 数据。

用法（P1，渲染完成后）：
    uv run python ue/debug/debug_object_id_exr.py <render_mask_dir>

打印：
  1. render_mask/ 下的 .exr 文件列表与大小。
  2. EXR 原始头字节里是否含 CryptoObject / cryptomatte / manifest 标记，并截取
     manifest 片段（"实体名:ID" 映射，确认每个 actor 都有独立 Cryptomatte ID）。
  3. cv2 读取该 EXR 的 shape/dtype（确认能读到数据）。

若 manifest 里出现 Player_L0 / Player_R4 / Ball_01 等实体名 → Cryptomatte 可用，
后续实现 P1 的 Cryptomatte 解析即可得到逐实体 mask。
"""

import sys
from pathlib import Path

import cv2


def main():
    if len(sys.argv) < 2:
        print("用法: uv run python ue/debug/debug_object_id_exr.py <render_mask_dir>")
        sys.exit(1)
    d = Path(sys.argv[1])
    exrs = sorted(d.glob("*.exr"))
    if not exrs:
        print(f"ERROR: {d} 下没有 .exr 文件（可能 ObjectId+EXR 渲染未生效）")
        # 列出目录内容帮助排查
        print("目录内容:", [p.name for p in sorted(d.iterdir())][:20])
        sys.exit(1)
    print(f"=== {d} 下 {len(exrs)} 个 EXR ===")
    for p in exrs[:3]:
        b = p.read_bytes()
        print(f"\n--- {p.name} ({len(b)} bytes) ---")
        for marker in (b"CryptoObject", b"cryptomatte", b"manifest", b"name"):
            idx = b.find(marker)
            print(f"  {marker.decode():14s} @ {idx}")
        # manifest 片段（实体名:浮点ID）
        m = b.find(b"manifest")
        if m != -1:
            seg = b[m : m + 400]
            print(f"  manifest 片段: {seg[:300]}")
        # cv2 读取像素
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            print("  cv2 读取: None（多层 EXR cv2 可能只读主层）")
        else:
            print(f"  cv2 读取: shape={img.shape} dtype={img.dtype}")


if __name__ == "__main__":
    main()
