"""C4 收尾：3 帧短测试 —— 确认 CAPTURE=TEST 已从场景移除后，Pose Capture 与 RootFrame 仍正常。

渲染 LS_Cam_01 帧 0..2（3 帧），BurnIn(OnOutputFrameStarted)→5×CaptureOutputFrame。
预期每 slot total=78（3×2×13），root 0..2。

用法（Unreal Editor Python，FutsalMOTTools.run_python_file）：
    py ".../mrq_shorttest_c4.py"
"""

from pathlib import Path

LOG = []
LOG_PATH = Path(r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\mrq_shorttest_c4.log")
RENDER_DIR = Path(r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\shorttest_render")
SEQ_ASSET = "/Game/FutsalMOT/Sequences/LS_Cam_01.LS_Cam_01"
SEQ_LOAD = "/Game/FutsalMOT/Sequences/LS_Cam_01"
MAP = "/Game/FutsalMOT/Maps/L_FutsalCourt"
BURN_IN_CLASS = "/Game/FutsalMOT/Blueprints/WBP_PoseMRQBurnInC4.WBP_PoseMRQBurnInC4_C"
SLOTS = [f"PoseCaptureG{i}" for i in range(5)]
FRAME_END = 3  # 3 帧
IMAGE_W = 1920
IMAGE_H = 1080


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
    _log("======== C4 收尾短测试：3 帧 RGB + Capture ========")

    for sl in SLOTS:
        try:
            unreal.GameplayStatics.delete_game_in_slot(sl, 0)
        except Exception:
            pass
    _log("  已清空 SaveGame slot")

    import shutil
    if RENDER_DIR.exists():
        shutil.rmtree(RENDER_DIR, ignore_errors=True)
    RENDER_DIR.mkdir(parents=True, exist_ok=True)

    seq = unreal.load_asset(SEQ_LOAD)
    if seq is None:
        _log("  ERROR: 序列加载失败")
        _flush()
        return
    seq.set_playback_start(0)
    seq.set_playback_end(FRAME_END)
    _log(f"  playback 0..{FRAME_END}")

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
    except Exception as e:
        _log(f"  BurnIn ERR: {type(e).__name__} {e}")
    try:
        job.set_configuration(config)
    except Exception:
        try:
            job.configuration = config
        except Exception:
            pass

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
        _log("  ERROR: 提交失败")
    _log("  异步渲染 3 帧中...")
    _flush()


if __name__ == "__main__":
    main()