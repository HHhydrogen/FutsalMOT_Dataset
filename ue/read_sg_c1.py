"""读取 SG_PoseCapture，验证 C1 OnOutputFrameStarted 采样。

用法（Unreal Editor Python Console）：
    py "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/ue/read_sg_c1.py"

验证：
  1. total_samples == 90（= RGB 帧数）
  2. capture_indices == 0..89，无重复、无缺失、无 warmup/trailing
  3. shot_frames 对应
  4. 最后几条记录（root, shot, game_time, hand_xyz）
"""

import sys
from pathlib import Path

LOG = []
LOG_PATH = Path(r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\read_sg_c1.log")
SLOT_NAME = "PoseCapture"


def _log(msg):
    print(msg)
    LOG.append(str(msg))


def _flush():
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH.write_text("\n".join(LOG), encoding="utf-8")
    except Exception as e:
        print(f"  写日志失败: {e}")


def _v3(v):
    return (round(v.x, 3), round(v.y, 3), round(v.z, 3))


def main():
    import unreal
    _log("======== 读取 SG_PoseCapture（C1 验证）========")

    sg = unreal.GameplayStatics.load_game_from_slot(SLOT_NAME, 0)
    _log(f"[1] load -> {sg}")
    if sg is None:
        _log("  ERROR: 空")
        _flush()
        return

    cap = list(sg.get_editor_property("capture_indices"))
    shot = list(sg.get_editor_property("shot_frames"))
    gt = list(sg.get_editor_property("game_times"))
    loc = list(sg.get_editor_property("world_locations"))
    total = sg.get_editor_property("total_samples")

    _log(f"[2] total_samples = {total}")
    _log(f"  capture_indices len = {len(cap)}")
    _log(f"  shot_frames len = {len(shot)}")
    _log(f"  game_times len = {len(gt)}")
    _log(f"  world_locations len = {len(loc)}")
    _log(f"  RGB 帧数 = 90")

    _log(f"\n[3] capture_indices 前10: {cap[:10]}")
    _log(f"  capture_indices 后10: {cap[-10:]}")
    _log(f"  shot_frames 前10: {shot[:10]}")
    _log(f"  shot_frames 后10: {shot[-10:]}")

    # 验证
    if cap:
        _log(f"\n[4] 验证:")
        _log(f"  first = {cap[0]}, last = {cap[-1]}")
        expected = list(range(len(cap)))
        match = (cap == expected)
        _log(f"  capture == 0..{len(cap)-1}: {match}")
        _log(f"  len == 90: {len(cap) == 90}")
        _log(f"  无重复: {len(set(cap)) == len(cap)}")
        _log(f"  shot == capture: {shot == cap if shot else 'N/A'}")

    # 最后 5 条
    _log(f"\n[5] 最后 5 条:")
    for i in range(max(0, len(cap)-5), len(cap)):
        _log(f"  root={cap[i]} shot={shot[i] if i < len(shot) else '?'} t={gt[i]:.6f} hand={_v3(loc[i]) if i < len(loc) else '?'}")

    _flush()


if __name__ == "__main__":
    main()
    print("\n脚本已执行，结果写入 read_sg_c1.log")