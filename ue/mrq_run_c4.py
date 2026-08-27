"""C4 正式 90 帧渲染：LS_Cam_01 帧 0..89，BurnIn→5×CaptureOutputFrame→SaveGame。

复用 mrq_smoke_c4.py 已验证的 MRQ 作业构造。异步提交，完成后通过轮询 SaveGame
（total_samples==2340/slot）判定完成。RGB 输出到 .futsalmot/c4_render。

用法（Unreal Editor Python，FutsalMOTTools.run_python_file）：
    py ".../mrq_run_c4.py"
"""

from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from setup_c4_level import preflight_debug_cleanup  # noqa: E402
from render_preset import resolve_output_resolution  # noqa: E402

LOG = []
LOG_PATH = Path(r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\mrq_run_c4.log")
RENDER_DIR = Path(r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\c4_render")
SEQ_ASSET = "/Game/FutsalMOT/Sequences/LS_Cam_01.LS_Cam_01"
SEQ_LOAD = "/Game/FutsalMOT/Sequences/LS_Cam_01"
MAP = "/Game/FutsalMOT/Maps/L_FutsalCourt"
BURN_IN_CLASS = "/Game/FutsalMOT/Blueprints/WBP_PoseMRQBurnInC4.WBP_PoseMRQBurnInC4_C"
SLOTS = [f"PoseCaptureG{i}" for i in range(5)]
FRAME_START = 0
FRAME_END = 90  # 90 帧 (0..89)
# C5.1：MRQ 输出分辨率也来自 resolved task（render_rgb.output_resolution），
# 与 camera calibration 同一唯一来源，禁止渲染侧硬编码与配置脱节。
RESOLVED_TASK = Path(
    r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\runtime\bp_frame_sync_30f\resolved-task.json"
)
IMAGE_W = None
IMAGE_H = None


def _resolve_render_resolution():
    global IMAGE_W, IMAGE_H
    import json
    rt = json.loads(RESOLVED_TASK.read_text(encoding="utf-8"))
    ann_cfg = rt["ue_profile"].get("annotation_export") or {}
    IMAGE_W, IMAGE_H = resolve_output_resolution(ann_cfg)
    _log(f"  render resolution (resolved task) = {IMAGE_W}x{IMAGE_H}")


def _log(msg):
    print(msg)
    LOG.append(str(msg))


def _flush():
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH.write_text("\n".join(LOG), encoding="utf-8")
    except Exception as e:
        print(f"  写日志失败: {e}")


def main():
    import unreal
    _log("======== C4 正式 90 帧渲染 ========")

    # 0) Render Preflight：清理调试 actor（CAPTURE_DISPLAY / 'CAPTURE=' TextRender）
    ok, removed, remaining = preflight_debug_cleanup()
    _log(f"[preflight] removed={removed} remaining={remaining} ok={ok}")
    if not ok:
        _log("  ERROR: preflight 未通过（仍有调试 actor），禁止启动 MRQ")
        _flush()
        return

    # 0b) C5.1：从 resolved task 解析 MRQ 输出分辨率（与 camera calibration 同源）
    _resolve_render_resolution()
    if IMAGE_W is None or IMAGE_H is None:
        _log("  ERROR: 无法解析渲染分辨率，禁止启动 MRQ")
        _flush()
        return

    # 1) 清空 SaveGame slot
    for sl in SLOTS:
        try:
            unreal.GameplayStatics.delete_game_in_slot(sl, 0)
            _log(f"  清空 slot {sl}")
        except Exception as e:
            _log(f"  清空 slot {sl} ERR: {type(e).__name__} {e}")

    # 2) 清空输出目录
    import shutil
    if RENDER_DIR.exists():
        shutil.rmtree(RENDER_DIR, ignore_errors=True)
    RENDER_DIR.mkdir(parents=True, exist_ok=True)

    # 3) 设置 playback range 0..90
    seq = unreal.load_asset(SEQ_LOAD)
    if seq is None:
        _log("  ERROR: 序列加载失败")
        _flush()
        return
    seq.set_playback_start(FRAME_START)
    seq.set_playback_end(FRAME_END)
    _log(f"  playback range: {FRAME_START}..{FRAME_END}")

    # 4) 构建 MRQ job
    sub = None
    for cls_name in ("MoviePipelineQueueSubsystem", "MovieRenderQueueSubsystem"):
        cls = getattr(unreal, cls_name, None)
        if cls is not None:
            try:
                sub = unreal.get_editor_subsystem(cls)
                break
            except Exception:
                continue
    if sub is None:
        _log("  ERROR: 无 MRQ 子系统")
        _flush()
        return
    queue = sub.get_queue()
    try:
        queue.delete_all_jobs()
    except Exception:
        pass
    job = queue.allocate_new_job(unreal.MoviePipelineExecutorJob)
    job.sequence = unreal.SoftObjectPath(SEQ_ASSET)
    job.map = unreal.SoftObjectPath(MAP)

    try:
        config = job.get_configuration()
    except Exception:
        config = None
    if config is None:
        config = unreal.MoviePipelinePrimaryConfig()

    output = config.find_or_add_setting_by_class(unreal.MoviePipelineOutputSetting)
    output.output_directory = unreal.DirectoryPath(str(RENDER_DIR))
    output.file_name_format = "{frame_number}"
    output.zero_pad_frame_numbers = 4
    output.output_resolution = unreal.IntPoint(IMAGE_W, IMAGE_H)
    try:
        output.frame_rate = unreal.FrameRate(30, 1)
    except Exception:
        pass
    config.find_or_add_setting_by_class(unreal.MoviePipelineImageSequenceOutput_PNG)
    config.find_or_add_setting_by_class(unreal.MoviePipelineDeferredPassBase)
    try:
        burnin = config.find_or_add_setting_by_class(unreal.MoviePipelineBurnInSetting)
        burnin.set_editor_property("burn_in_class", unreal.SoftClassPath(BURN_IN_CLASS))
        burnin.set_editor_property("composite_onto_final_image", False)
        _log(f"  BurnIn: {BURN_IN_CLASS}（不合成）")
    except Exception as e:
        _log(f"  BurnIn ERR: {type(e).__name__} {e}")
    try:
        job.set_configuration(config)
    except Exception:
        try:
            job.configuration = config
        except Exception:
            _log("  set_configuration 失败")

    # 5) executor + 提交
    executor = None
    for cls_name in ("MoviePipelinePIEExecutor", "MoviePipelineEditorExecutor"):
        cls = getattr(unreal, cls_name, None)
        if cls is None:
            continue
        try:
            executor = cls()
            break
        except Exception:
            continue
    if executor is None:
        _log("  ERROR: 无 executor")
        _flush()
        return

    submitted = False
    for m_name in ("render_queue_with_executor_instance", "render_queue_with_executor"):
        m = getattr(sub, m_name, None)
        if m is None:
            continue
        try:
            m(executor)
            submitted = True
            _log(f"  已提交（{m_name}）")
            break
        except Exception as e:
            _log(f"  {m_name} ERR: {type(e).__name__} {e}")
    if not submitted:
        _log("  ERROR: 提交渲染失败")
    _log(f"  {IMAGE_W}x{IMAGE_H} x 90 帧，异步渲染中...")
    _flush()


if __name__ == "__main__":
    main()