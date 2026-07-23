# FutsalMOT-RL UE Render Script
# 在 UE Python 控制台直接运行此脚本，不需要设置环境变量。
#
# 用法:
#   py "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/tools/legacy/rl_07b_ue_render_rl.py"

import sys
import os

# ── 硬编码 RL A3.3 路径 ─────────────────────────────────────
RL_A33_PATH = "D:/projects/FustalMOT_UEDataset/Saved/FutsalMOT_RL/ue_closed_loop/rl_rl_episode_random_0001_t1_p05_for_ue.json"
RL_SEQ_ID = "rl_episode_random_0001_t1_p05"

# ── 通过环境变量传递给 UE 脚本 ────────────────────────────────
os.environ["FUTSALMOT_CONFIG_PATH"] = RL_A33_PATH

print("=" * 60)
print("FutsalMOT-RL UE Render")
print("Config: {}".format(RL_A33_PATH))
print("Seq ID: {}".format(RL_SEQ_ID))
print("=" * 60)

# ── 验证配置文件存在 ─────────────────────────────────────────
if not os.path.isfile(RL_A33_PATH):
    print("[ERROR] RL A3.3 文件不存在: {}".format(RL_A33_PATH))
    print("请先运行: python rl_06_export_rl_a33.py")
    raise SystemExit(1)

# ── 运行 Preflight (只读检查) ────────────────────────────────
# Script is at tools/legacy/rl_07b_ue_render_rl.py, repo root is two levels up
CODE_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, CODE_DIR)

print("\n▶ 第 1 步: UE Preflight 检查...")
from futsalmot.scripts import ue_preflight
ue_preflight.main()

# ── 运行 Build Sequences ─────────────────────────────────────
print("\n▶ 第 2 步: 构建 Level Sequence 并导出标注...")
build_path = os.path.join(CODE_DIR, "futsalmot", "scripts", "ue_build_sequences.py")
source = open(build_path, encoding="utf-8").read()
globals_dict = {"__name__": "__main__", "__file__": build_path}
try:
    exec(source, globals_dict)
except SystemExit:
    pass
except Exception as e:
    try:
        import unreal
        unreal.log_error("[rl_07b] ue_build_sequences 失败: {}: {}".format(type(e).__name__, e))
    except Exception:
        pass
    print("[ERROR] ue_build_sequences 失败: {}: {}".format(type(e).__name__, e))
    raise

print("\n" + "=" * 60)
print("[OK] RL 渲染准备完成!")
print("")
print("接下来操作：")
print("1. 打开 Movie Render Queue")
print("2. 设置 Output Directory:")
print("   D:\\projects\\FustalMOT_UEDataset\\Saved\\FutsalMOT_RL\\ue_closed_loop\\images\\{}".format(RL_SEQ_ID))
print("3. File Name Format: {{frame_number}}")
print("4. Image Format: PNG")
print("5. Resolution: 1920 x 1080")
print("6. 点击 Render")
print("")
print("渲染完成后，在 Windows 终端运行:")
print("   python 03_check_labels.py --annotation \"D:\\projects\\FustalMOT_UEDataset\\Saved\\FutsalMOT\\annotations\\objects_bbox_2d_clean_{}.json\"".format(RL_SEQ_ID))
print("")
print("然后验证:")
print("   python rl_07_validate_ue_closed_loop.py --check --seq-id {}".format(RL_SEQ_ID))
print("=" * 60)
