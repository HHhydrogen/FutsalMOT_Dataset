"""C4 冒烟：3 帧 MRQ 渲染，验证 BurnIn(OnOutputFrameStarted) → 5×CaptureOutputFrame → SaveGame 链路。

只渲染 LS_Cam_01 的 0..2 帧（临时缩小 playback range，验证后由 check_smoke_c4.py 恢复），
160x90 极小分辨率，BurnIn 不合成。完成后 delegate 写 smoke_done.txt。

用法（Unreal Editor Python，FutsalMOTTools.run_python_file）：
    py ".../mrq_smoke_c4.py"
"""

from pathlib import Path

LOG = []
LOG_PATH = Path(r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\mrq_smoke_c4.log")
SMOKE_DIR = Path(r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\smoke_render")
DONE_FILE = SMOKE_DIR / "smoke_done.txt"
RANGE_FILE = SMOKE_DIR / "seq_range.txt"
SEQ_ASSET = "/Game/FutsalMOT/Sequences/LS_Cam_01.LS_Cam_01"
SEQ_LOAD = "/Game/FutsalMOT/Sequences/LS_Cam_01"
MAP = "/Game/FutsalMOT/Maps/L_FutsalCourt"
BURN_IN_CLASS = "/Game/FutsalMOT/Blueprints/WBP_PoseMRQBurnInC4.WBP_PoseMRQBurnInC4_C"
SLOTS = [f"PoseCaptureG{i}" for i in range(5)]
FRAME_START = 0
FRAME_END = 2  # 3 帧

_ACTIVE = {}


def _log(msg):
    print(msg)
    LOG.append(str(msg))


def _flush():
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH.write_text("\n".join(LOG), encoding="utf-8")
    except Exception as e:
        print(f"  写日志失败: {e}")


def _on_finished(executor, success):
    try:
        _log(f"MRQ finished delegate -> success={success}")
        DONE_FILE.write_text(f"success={success}", encoding="utf-8")
    except Exception as e:
        _log(f"_on_finished ERR: {e}")
        try:
            DONE_FILE.write_text(f"success=False\ncb_err={e}", encoding="utf-8")
        except Exception:
            pass


def main():
    import unreal
    _log("======== C4 冒烟：3 帧 MRQ（BurnIn→Capture→SaveGame）========")

    # 1) 清空 SaveGame slot
    for sl in SLOTS:
        try:
            unreal.GameplayStatics.delete_game_in_slot(sl, 0)
            _log(f"  清空 slot {sl}")
        except Exception as e:
            _log(f"  清空 slot {sl} ERR: {type(e).__name__} {e}")

    # 2) 清空输出目录
    import shutil
    if SMOKE_DIR.exists():
        shutil.rmtree(SMOKE_DIR, ignore_errors=True)
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)

    # 3) 缩小 playback range（记录原值，check 脚本恢复）
    seq = unreal.load_asset(SEQ_LOAD)
    if seq is None:
        _log("  ERROR: 序列加载失败")
        _flush()
        return
    try:
        orig_start = int(seq.get_playback_start())
        orig_end = int(seq.get_playback_end())
        _log(f"  原 playback range: {orig_start}..{orig_end}")
        RANGE_FILE.write_text(f"{orig_start} {orig_end}", encoding="utf-8")
        seq.set_playback_start(FRAME_START)
        seq.set_playback_end(FRAME_END)
        _log(f"  临时范围: {FRAME_START}..{FRAME_END}")
    except Exception as e:
        _log(f"  设置 range ERR: {type(e).__name__} {e}")

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
    _log(f"  job: {job}")

    config = None
    try:
        config = job.get_configuration()
    except Exception:
        config = None
    if config is None:
        config = unreal.MoviePipelinePrimaryConfig()

    output = config.find_or_add_setting_by_class(unreal.MoviePipelineOutputSetting)
    output.output_directory = unreal.DirectoryPath(str(SMOKE_DIR))
    output.file_name_format = "{frame_number}"
    output.zero_pad_frame_numbers = 4
    output.output_resolution = unreal.IntPoint(160, 90)
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

    # 5) executor + delegate + 提交（异步，立即返回）
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

    delegate = getattr(executor, "on_executor_finished_delegate", None)
    if delegate is not None:
        try:
            delegate.add_callable(_on_finished)
            _log("  已绑定 finished delegate")
        except Exception as e:
            _log(f"  绑定 delegate ERR: {type(e).__name__} {e}")

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
        DONE_FILE.write_text("submit_failed=True", encoding="utf-8")

    _log("  渲染已异步提交，等待 smoke_done.txt ...")
    _flush()


if __name__ == "__main__":
    main()