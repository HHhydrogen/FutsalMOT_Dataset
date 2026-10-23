"""读取冻结生产组与根轨迹，输出只读描述统计；不读取RGB、mask或真实测试集。"""

import argparse
import hashlib
import json
from pathlib import Path

from grf_ue_bridge.motion_coverage import CoverageThresholds, analyze_frames


def scoped_reference(base, reference, boundary):
    """解析manifest引用并拒绝越界，绝不通过媒体junction递归扫描。"""
    result = (base / reference).resolve()
    try:
        result.relative_to(boundary.resolve())
    except ValueError:
        raise ValueError("manifest引用超出数据集边界: " + str(result))
    return result


def produce(dataset_root, production_plan, output, project_root):
    root = Path(dataset_root).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("拒绝覆盖已有统计attempt")
    try:
        output.relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError("输出不能写入正式数据根")
    records = []
    def read_json(path):
        data = path.read_bytes()
        records.append({"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
        return json.loads(data)
    index = read_json(root / "dataset.json")
    plan = read_json(Path(production_plan).resolve())
    expected = {e["root_seed"]: e for e in plan["entries"]}
    if len(expected) != len(plan["entries"]) or set(expected) != {g["root_seed"] for g in index["groups"]}:
        raise ValueError("生产冻结seed清单与正式索引不一致")
    clips = []
    for indexed in index["groups"]:
        group_path = scoped_reference(root, indexed["path"], root)
        group = read_json(group_path)
        seed = group["root_seed"]
        if indexed["root_seed"] != seed or expected[seed]["split"] != group["split"]:
            raise ValueError("seed/split不一致")
        for clip_ref in group["clips"]:
            clip_path = scoped_reference(group_path.parent, clip_ref["path"], root)
            clip = read_json(clip_path)
            trajectory_path = clip_path.parent / "trajectory.jsonl"
            data = trajectory_path.read_bytes()
            sha = hashlib.sha256(data).hexdigest()
            records.append({"path": str(trajectory_path), "sha256": sha, "bytes": len(data)})
            if sha != clip["trajectory_sha256"]:
                raise ValueError("轨迹与冻结clip摘要不符")
            frames = [json.loads(row) for row in data.splitlines() if row.strip()]
            if len(frames) != clip["frame_count"] or len(frames) != expected[seed]["frames_per_camera"]:
                raise ValueError("冻结帧数不符")
            if clip["source_frame_range"] != expected[seed]["source_frame_range"]:
                raise ValueError("连续窗口与冻结清单不一致")
            metrics = analyze_frames(frames, group["entities"])
            clips.append({"root_seed": seed, "split": group["split"],
                          "weather": group.get("weather", {}).get("preset"),
                          "clip_id": clip_ref["clip_id"], "clip_path": str(clip_path),
                          "trajectory_path": str(trajectory_path), "trajectory_sha256": sha,
                          "camera_count_not_a_multiplier": len(clip["cameras"]), "metrics": metrics})
    project = Path(project_root).resolve()
    source_evidence = []
    for rel, needles in (("Content/FutsalMOT/code/ue/player_motion.py", ("暂不消费", '"action_state": MotionAction.NONE', "class AnimationClass")),
                         ("Source/FutsalMOT/Private/FutsalPlayerAnimInstance.cpp", ("ComputeAutoMotion", "UpdateMotionOutputs"))):
        path = project / rel
        raw = path.read_bytes()
        lines = raw.decode("utf-8-sig").splitlines()
        source_evidence.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
            "selected_lines": [{"line": i + 1, "text": line.strip()} for i, line in enumerate(lines)
                               if any(n in line for n in needles)]})
    animation_dir = project / "Content/FutsalMOT/Characters/FutsalPlayerBase/Animation"
    assets = sorted(str(p.relative_to(project)) for p in animation_dir.rglob("*.uasset"))
    span = sum(c["metrics"]["trajectory_span_s"] for c in clips)
    gait = {key: sum(c["metrics"]["gait_seconds"][key] for c in clips) for key in clips[0]["metrics"]["gait_seconds"]}
    report = {"schema": "futsalmot_motion_coverage", "version": 1,
        "parameters": {"dataset_root": str(root), "production_plan": str(Path(production_plan).resolve()),
                       "project_root": str(project), "thresholds": clips[0]["metrics"]["thresholds"]},
        "inputs": records, "clips": clips, "summary": {"seed_count": len(expected), "clip_count": len(clips),
            "unique_trajectory_frames": sum(c["metrics"]["frame_count"] for c in clips),
            "unique_trajectory_span_s": span, "player_seconds": sum(c["metrics"]["player_seconds"] for c in clips),
            "gait_seconds": gait,
            "team_active_ratio": sum(c["metrics"]["team_active"]["total_s"] for c in clips) / span,
            "acceleration_player_seconds": sum(p["acceleration"]["total_s"] for c in clips for p in c["metrics"]["players"].values()),
            "braking_player_seconds": sum(p["braking"]["total_s"] for c in clips for p in c["metrics"]["players"].values()),
            "sharp_turn_player_seconds": sum(p["sharp_turn"]["total_s"] for c in clips for p in c["metrics"]["players"].values()),
            "near_cross_team_pair_seconds": {str(d): sum(c["metrics"]["proximity"][str(d)]["cross_team_pair_seconds"] for c in clips) for d in (1.0, 2.0)}},
        "static_animation_evidence": {"code": source_evidence, "asset_name_inventory": assets,
            "scope": "静态代码及文件名，不证明运行时绑定、加载class、片段实际触发或姿态真实"},
        "limitations": ["位置差分统计根轨迹，不证明骨骼关节动作、传球、射门、盘带、铲抢或碰撞。",
            "近距离仅为世界XY几何邻近代理，不等于人体遮挡、接触或已标注交互事件。",
            "四机位共享同一轨迹，只计一次运动时间，不能把32视角当32独立比赛。",
            "总时长为末采样减首采样(N-1区间)；不把N/FPS样本支持当作已观察运动跨度。",
            "阈值是预先固定的描述规则，不是人工语义动作标注；速度来自位置差分而非直接相信导出speed字段。",
            "当前正式v1没有Pose输出，此统计不能作为Pose准确率或新Pose数据集已完成的证据。"]}
    output.mkdir(parents=True)
    (output / "motion_coverage.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    rows = ["# 根轨迹运动与邻近覆盖", "", "四相机不重复计时；不等同于骨骼或足球语义动作覆盖。", "",
        "|seed|帧数|运动跨度s|团队活跃比例|最长连续团队活跃s|≤1m跨队pair-seconds|", "|---|---:|---:|---:|---:|---:|"]
    for c in clips:
        m = c["metrics"]
        rows.append(f"|{c['root_seed']}|{m['frame_count']}|{m['trajectory_span_s']:.1f}|{m['team_active_ratio']:.3f}|{m['team_active']['longest_s']:.1f}|{m['proximity']['1.0']['cross_team_pair_seconds']:.1f}|")
    rows += ["", "摘要：", "", "```json", json.dumps(report["summary"], ensure_ascii=False, indent=2), "```", "", "限制：", ""]
    rows += ["- " + line for line in report["limitations"]]
    (output / "motion_coverage.md").write_text("\n".join(rows), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--production-plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--project-root", required=True)
    args = parser.parse_args()
    result = produce(args.dataset_root, args.production_plan, args.output, args.project_root)
    print(json.dumps(result["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
