"""Phase A：诊断 C4 Recorder G0..G4 资产、GeneratedClass、CaptureOutputFrame 函数。

用法（Unreal Editor Python，FutsalMOTTools.run_python_file）：
    检查 5 个 BP 是否可加载、GeneratedClass 是否生成、
    CaptureOutputFrame 是否可通过 SoftObjectPath 解析（对应 AddCallFunctionNode 的解析路径）。
"""

from pathlib import Path

LOG = []
LOG_PATH = Path(r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\diag_c4_phase_a.log")


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
    _log("======== Phase A：C4 Recorder G0..G4 诊断 ========")

    for i in range(5):
        asset_path = f"/Game/FutsalMOT/Blueprints/BP_PoseRecorderC4_G{i}"
        bp = unreal.EditorAssetLibrary.load_asset(asset_path)
        _log(f"[G{i}] load_asset -> {bp}")
        if bp is None:
            continue
        gc = unreal.BlueprintEditorLibrary.generated_class(bp)
        _log(f"[G{i}] generated_class -> {gc}")
        if gc is None:
            continue
        _log(f"[G{i}] gc path: {gc.get_path_name()}")

        # 1) SoftObjectPath 解析 UFunction（AddCallFunctionNode 同路径）
        fn_path = f"{asset_path}.BP_PoseRecorderC4_G{i}_C:CaptureOutputFrame"
        try:
            fn = unreal.SoftObjectPath(fn_path).load_synchronous()
            _log(f"[G{i}] SoftObjectPath({fn_path}) -> {fn}")
        except Exception as e:
            _log(f"[G{i}] SoftObjectPath ERR: {type(e).__name__} {e}")

        # 2) 遍历类函数，找 CaptureOutputFrame（不经 Python 暴露的反射）
        try:
            found = []
            if gc is not None:
                it = gc
                funcs = []
                try:
                    funcs = list(gc.functions)
                except Exception:
                    funcs = []
                for f in funcs:
                    name = str(f.get_name())
                    if "capture" in name.lower():
                        found.append(name)
                _log(f"[G{i}] class functions: {[str(f.get_name()) for f in funcs]}")
        except Exception as e:
            _log(f"[G{i}] functions ERR: {type(e).__name__} {e}")

    # 现有 WBP 状态
    wbp = unreal.EditorAssetLibrary.load_asset("/Game/FutsalMOT/Blueprints/WBP_PoseMRQBurnInC4")
    _log(f"WBP_PoseMRQBurnInC4 -> {wbp}")

    _flush()


if __name__ == "__main__":
    main()