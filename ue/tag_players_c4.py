"""C4：给 10 个 Player actor 打 Tag（独立脚本，不 build、不 delete、不保存关卡）。

Actor → Tag：
  Player_L0..L4 → PoseL0..PoseL4 → actor_id L0..L4
  Player_R0..R4 → PoseR0..PoseR4 → actor_id R0..R4

只在内存中设置 Tag（PIE 渲染可读），不保存关卡（避免 World Partition external actor 崩溃）。

用法（Unreal Editor Python Console）：
    py "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/ue/tag_players_c4.py"
"""

import sys
from pathlib import Path

LOG = []
LOG_PATH = Path(r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\tag_players_c4.log")

ACTORS = [
    {"tag": "PoseL0", "actor_id": "L0"},
    {"tag": "PoseL1", "actor_id": "L1"},
    {"tag": "PoseL2", "actor_id": "L2"},
    {"tag": "PoseL3", "actor_id": "L3"},
    {"tag": "PoseL4", "actor_id": "L4"},
    {"tag": "PoseR0", "actor_id": "R0"},
    {"tag": "PoseR1", "actor_id": "R1"},
    {"tag": "PoseR2", "actor_id": "R2"},
    {"tag": "PoseR3", "actor_id": "R3"},
    {"tag": "PoseR4", "actor_id": "R4"},
]
ACTOR_LABEL = {a["actor_id"]: f"Player_{a['actor_id']}" for a in ACTORS}


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
    _log("======== 给 Player 打 Tag（C4）========")

    ed_sub = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    found = {}
    for a in ed_sub.get_all_level_actors():
        try:
            label = a.get_actor_label()
            if label in ACTOR_LABEL.values():
                found[label] = a
        except Exception:
            pass
    _log(f"找到 Player actor: {sorted(found.keys())}")

    ok = 0
    for tgt in ACTORS:
        label = ACTOR_LABEL[tgt["actor_id"]]
        a = found.get(label)
        if a is None:
            _log(f"  {label} 未找到！")
            continue
        try:
            tags = list(a.get_editor_property("tags") or [])
            if tgt["tag"] not in tags:
                tags.append(tgt["tag"])
                a.set_editor_property("tags", tags)
                _log(f"  {label} -> Tag {tgt['tag']}")
            else:
                _log(f"  {label} 已有 Tag {tgt['tag']}")
            ok += 1
        except Exception as e:
            _log(f"  {label} ERR: {type(e).__name__} {e}")

    _log(f"\n完成 {ok}/10 个 actor 打 Tag（内存中，未保存关卡）")
    _flush()


if __name__ == "__main__":
    main()
    print("\n脚本已执行，结果写入 tag_players_c4.log")