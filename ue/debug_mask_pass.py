"""调试探针：检查 MRQ Instance-ID Mask 渲染的 pass 与输出格式。

在 UE 编辑器 Python Console 运行：
    py "D:/.../code/ue/debug_mask_pass.py"

打印两件事：
  1. UE 环境：MRQ 中与 Instance-ID Mask 相关的类及其成员（MoviePipelineCustomDepthPass /
     MoviePipelineObjectIdRenderPass / MoviePipelineDeferredPassBase 等），用于确认
     用哪个 pass、如何配置（不同 UE 版本 API 不同）。
  2. 若 ue_import_config.json 的 annotation_export.output_dir 下已有 render_mask/ 或
     mask/ 输出，统计各通道取值分布，给出 mask_channel / id_scale 校准建议。

在普通 Python（P1 .venv）运行时跳过第 1 部分，只做第 2 部分的文件分析。
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

_CFG = Path(__file__).resolve().parent.parent / "ue_import_config.json"


def _probe_mrq_classes():
    """打印 MRQ 中与 Instance-ID Mask 相关的类与成员（仅在 UE 编辑器内）。"""
    try:
        import unreal
    except ImportError:
        print("[跳过] 非 UE 环境，无法探测 unreal 类")
        return
    print("=== MRQ 相关类（Instance-ID Mask）===")
    names = sorted(
        n for n in dir(unreal)
        if any(k in n for k in ("MoviePipeline", "MovieRender", "ObjectId",
                                "CustomDepth", "Stencil", "DeferredPass"))
    )
    print("类名:", names)
    for cls_name in (
        "MoviePipelineObjectIdRenderPass",
        "MoviePipelineObjectIdPassIdType",
        "MoviePipelinePostProcessPass",
        "MoviePipelineDeferredPassBase",
        "MoviePipelineDeferredPass_Unlit",
    ):
        cls = getattr(unreal, cls_name, None)
        if cls is None:
            print(f"\n{cls_name}: 不存在")
            continue
        if cls_name == "MoviePipelineObjectIdPassIdType":
            # 枚举：打印全部成员值
            values = sorted(n for n in dir(cls) if not n.startswith("_"))
            print(f"\n--- {cls_name} 枚举值 ---")
            print(values)
            continue
        interesting = sorted(
            n for n in dir(cls) if not n.startswith("_") and any(
                k in n.lower() for k in (
                    "stencil", "depth", "material", "setting", "id",
                    "color", "buffer", "pass", "enabled", "format",
                )
            )
        )
        print(f"\n--- {cls_name} 相关成员 ---")
        print(interesting[:60])


def _analyze_output():
    """扫描输出目录下的 render_mask/ 与 mask/，给出通道校准建议。

    instance_mask 依赖 numpy（UE Python 没有），故仅在 numpy 可用时分析。
    """
    try:
        import numpy  # noqa: F401
    except ImportError:
        print("\n[跳过] UE Python 无 numpy，mask 通道分析请在 P1（.venv）运行")
        return
    from instance_mask import analyze_mask_dir

    cfg = {}
    if _CFG.exists():
        raw = json.load(open(_CFG, encoding="utf-8"))
        cfg = {k: v for k, v in raw.items() if not k.startswith("comment_")}
    ann = cfg.get("annotation_export") or {}
    output_dir = Path(ann.get("output_dir") or "")
    if not output_dir.exists():
        print(f"\n输出目录不存在，跳过 mask 分析: {output_dir}")
        return
    print(f"\n=== 扫描输出目录: {output_dir} ===")
    found = False
    for cam_dir in sorted(output_dir.rglob("camera.json")):
        base = cam_dir.parent
        for sub in ("render_mask", "mask"):
            d = base / sub
            if d.exists():
                result = analyze_mask_dir(d)
                if result:
                    found = True
                    print(f"\n[{base.name} / {sub}] 采样 {result['sample_files']} 帧：")
                    for ch, vals in result["channel_unique_values"].items():
                        shown = vals[:30]
                        suffix = "..." if len(vals) > 30 else ""
                        print(f"  {ch}: {shown}{suffix}")
                    print(f"  建议: {result['note']}")
    if not found:
        print("未找到 render_mask/ 或 mask/ 输出（请先 --mode full 渲染）")


def main():
    _probe_mrq_classes()
    _try_create_material()  # 先尝试建材质（排查自动创建失败的关键）
    _inspect_material()      # 检查已建材质的 domain 与 SceneTextureId
    _try_cvar_struct()       # 测试 r.CustomDepth cvar 结构体字段是否可设
    _analyze_output()
    print("\n探针完成。请把上面 [MRQ 相关类] 的输出贴回以确认 pass 配置；")
    print("若 mask 已渲染，根据通道取值分布设置 instance_mask.mask_channel / id_scale。")


def _inspect_material():
    """检查已创建的 M_StencilToID 材质：domain 与 SceneTexture 的 texture_id。"""
    try:
        import unreal
    except ImportError:
        return
    path = "/Game/FutsalMOT/Materials/M_StencilToID"
    if not unreal.EditorAssetLibrary.does_asset_exist(path):
        print("\n[跳过] M_StencilToID 材质不存在")
        return
    mat = unreal.load_asset(path)
    print("\n=== 检查 M_StencilToID 材质 ===")
    try:
        print(f"  material_domain: {mat.get_editor_property('material_domain')}")
    except Exception as e:
        print(f"  读 material_domain 失败: {e}")
    try:
        exprs = unreal.MaterialEditingLibrary.get_material_expressions(mat)
        print(f"  表达式数: {len(exprs)}")
        for ex in exprs:
            tn = type(ex).__name__
            print(f"    {tn}")
            if "SceneTexture" in tn:
                for prop in ("scene_texture_id", "texture_id"):
                    try:
                        print(f"      {prop} = {ex.get_editor_property(prop)}")
                    except Exception:
                        pass
    except Exception as e:
        print(f"  读表达式失败: {e}")
    # SceneTextureId 枚举全部成员
    for enum_name in ("SceneTextureId", "EMaterialSceneTextureId", "SceneTextureIds"):
        enum = getattr(unreal, enum_name, None)
        if enum is not None:
            print(f"\n  {enum_name} 枚举成员: {sorted(n for n in dir(enum) if not n.startswith('_'))}")


def _try_cvar_struct():
    """测试 MoviePipelineConsoleVariableEntry 结构体字段是否可设置 r.CustomDepth。"""
    try:
        import unreal
    except ImportError:
        return
    print("\n=== 测试 r.CustomDepth cvar 结构体构造 ===")
    try:
        entry = unreal.MoviePipelineConsoleVariableEntry()
        from render_episode import _set_console_entry
        if _set_console_entry(entry, "r.CustomDepth", 2.0):
            print(f"  结构体设置成功: {entry}")
        else:
            print("  结构体字段设置失败，成员：")
            print(sorted(n for n in dir(entry) if not n.startswith("_")))
    except Exception as e:
        print(f"  构造失败: {e}")


def _try_create_material():
    """尝试创建 stencil→颜色 post-process 材质并打印结果（用于排查自动创建失败）。"""
    try:
        import importlib
        import render_episode
        importlib.reload(render_episode)  # UE Python 会话缓存旧版，先重载
        from render_episode import create_stencil_to_color_material
    except ImportError:
        print("\n[跳过] 无法 import render_episode 的材质创建函数")
        return
    print("\n=== 尝试创建 stencil→颜色 材质（force=True 强制重建）===")
    path = create_stencil_to_color_material(force=True)
    if path:
        print(f"材质路径: {path}（已可在 instance_mask.post_process_material 使用）")
    else:
        print("创建失败（上面应有完整 traceback）。请手动建材质并填 post_process_material。")


if __name__ == "__main__":
    main()
