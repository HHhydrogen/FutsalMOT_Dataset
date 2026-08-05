"""调试：打印球员 actor 的胶囊 / 网格 bounds，用于判断 bbox 数据源选择。

在 UE 编辑器 Python Console 中运行（需显式提供 episode 与 mapping，不再读根目录配置）：
    py "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/ue/debug_player_bounds.py" \
       --episode ".../outputs/episode_0001" --mapping ".../ue/actor_mapping.example.json"

把球员放到 episode 第 1 帧后，打印 L0/L1/BALL 的：
  - Capsule：radius / half_height / 世界中心 → 当前 bbox 数据源。
  - Mesh 组件本地 bounds（origin/extent）→ 判断改用 mesh bounds 是否更紧。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import unreal

from dataset_export import load_episode, load_mapping  # noqa: E402
from scene_apply import apply_preview_frame, find_all_actors  # noqa: E402


def _v3(v):
    return (round(float(v.x), 1), round(float(v.y), 1), round(float(v.z), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True, help="episode 目录（trajectory output）")
    ap.add_argument("--mapping", required=True, help="actor mapping JSON")
    args = ap.parse_args()
    episode_dir = Path(args.episode)
    mapping = load_mapping(Path(args.mapping))
    meta, frames = load_episode(episode_dir)

    actors = find_all_actors(mapping)
    prev_yaws, prev_positions = {}, {}
    apply_preview_frame(actors, frames[0], prev_yaws, prev_positions)  # 放到第 1 帧

    for entity_id in ("L0", "L1", "BALL"):
        actor = actors.get(entity_id)
        if not actor:
            print(f"{entity_id}: 未找到")
            continue
        label = actor.get_actor_label() or actor.get_name()
        print(f"\n=== {entity_id} ({label}) ===")

        caps = actor.get_components_by_class(unreal.CapsuleComponent)
        if caps:
            cap = caps[0]
            r = float(cap.get_scaled_capsule_radius())
            hh = float(cap.get_scaled_capsule_half_height())
            loc = cap.get_world_location()
            print(f"  Capsule: radius={r:.1f} half_height={hh:.1f} 世界中心={_v3(loc)}")
            print(f"    -> bbox half-extent (x,y,z)=({r:.1f}, {r:.1f}, {hh:.1f}) 全宽={2*r:.1f} 全高={2*hh:.1f}")
        else:
            print("  无 CapsuleComponent")

        for cls in (unreal.SkeletalMeshComponent, unreal.StaticMeshComponent):
            comps = actor.get_components_by_class(cls)
            if not comps:
                continue
            comp = comps[0]
            label = cls.__name__
            # 组件：直接读 bounds 相关编辑器属性
            for prop in ("bounds", "component_bounds", "local_bounds"):
                try:
                    v = comp.get_editor_property(prop)
                    print(f"  {label}: get_editor_property('{prop}') -> {v}")
                except Exception:
                    pass
            try:
                mesh = comp.skeletal_mesh if cls is unreal.SkeletalMeshComponent else comp.static_mesh
                if mesh:
                    print(f"  {label}: mesh={mesh.get_name()}")
                    # mesh 资产上所有含 bound 的成员
                    mnames = sorted(n for n in dir(mesh) if "bound" in n.lower())
                    print(f"  {label} mesh 含 bound 成员: {mnames}")
                    for prop in ("imported_bounds", "positive_bounds", "negative_bounds", "bounds", "extended_bounds", "imported_resource"):
                        try:
                            v = mesh.get_editor_property(prop)
                            print(f"  {label} mesh['{prop}'] -> {v}")
                        except Exception as e:
                            print(f"  {label} mesh['{prop}'] 失败: {type(e).__name__}: {e}")
                    m = getattr(mesh, "get_bounds", None)
                    if m is not None:
                        try:
                            print(f"  {label} mesh.get_bounds() -> {m()}")
                        except Exception as e:
                            print(f"  {label} mesh.get_bounds() 失败: {e}")
            except Exception as e:
                print(f"  {label}: 读 mesh 失败: {e}")

        try:
            res = actor.get_actor_bounds(False, False)
            print(f"  actor.get_actor_bounds(False,False) -> {res}")
        except Exception as e:
            print(f"  actor.get_actor_bounds 失败: {e}")


if __name__ == "__main__":
    main()
