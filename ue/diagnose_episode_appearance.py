"""只读检查 episode appearance 的 UE 地图、球员和资产前置条件。"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))


_FORMAL_MAP = "/Game/FutsalMOT/Maps/L_FutsalCourt"
_ENTITY_IDS = ("L0", "L1", "L2", "L3", "L4", "R0", "R1", "R2", "R3", "R4")
_ASSET_ROOT = "/Game/FutsalMOT/Characters/MPFB"
_ASSET_SPECS = tuple(
    [
        (
            "formal_map",
            "L_FutsalCourt",
            _FORMAL_MAP,
            "World",
        ),
    ]
    + [
        (
            "mesh",
            "Player_{:03d}".format(player_id),
            "{}/Players/Player_{:03d}/{}MPFB_Player_{:03d}".format(
                _ASSET_ROOT,
                player_id,
                "" if player_id == 3 else "Mesh/",
                player_id,
            ),
            "SkeletalMesh",
        )
        for player_id in range(3, 10)
    ]
    + [
        (
            "skin_texture",
            "skin-{:02d}".format(index),
            "{}/Textures/Skins/Male/{}".format(_ASSET_ROOT, name),
            "Texture2D",
        )
        for index, name in enumerate(
            (
                "T_MPFB_Skin_Old_Caucasian_Light",
                "T_MPFB_Skin_Old_African_Dark",
                "T_MPFB_Skin_Young_Asian_Light",
                "T_MPFB_Skin_MiddleAge_African_Dark",
                "T_MPFB_Skin_MiddleAge_Caucasian_Light",
                "T_MPFB_Skin_Old_Asian_Light",
                "T_MPFB_Skin_Young_Caucasian_Light",
                "T_MPFB_Skin_MiddleAge_Asian_Light",
                "T_MPFB_Skin_Young_African_Dark",
            ),
            1,
        )
    ]
    + [
        (
            "jersey_material",
            "Set_{:02d}".format(set_id),
            "{}/Materials/M_MPFB_Jersey_Set_{:02d}".format(_ASSET_ROOT, set_id),
            "Material",
        )
        for set_id in range(1, 6)
    ]
    + [
        (
            "shorts_material",
            "Set_{:02d}".format(set_id),
            "{}/Materials/M_MPFB_Shorts_Set_{:02d}".format(_ASSET_ROOT, set_id),
            "Material",
        )
        for set_id in range(1, 6)
    ]
    + [
        (
            "shared_asset",
            "SKEL_MPFB_Base",
            "{}/Core/Skeleton/SKEL_MPFB_Base".format(_ASSET_ROOT),
            "Skeleton",
        ),
        (
            "shared_asset",
            "ABP_MPFB_RuntimeRetarget",
            "{}/Core/Retarget/ABP_MPFB_RuntimeRetarget".format(_ASSET_ROOT),
            "AnimBlueprint",
        ),
        (
            "shared_asset",
            "M_MPFB_Body",
            "{}/Materials/M_MPFB_Body".format(_ASSET_ROOT),
            "Material",
        ),
    ]
)


def build_diagnostic_report(
    current_level_path: Any,
    mapping: Any,
    actors: Mapping[str, Mapping[str, Any]],
    dirty_package_paths: Sequence[str],
    asset_checks: Sequence[Mapping[str, Any]],
    errors: Sequence[str] = (),
) -> dict:
    """从只读采集结果生成可在普通 Python 测试的门禁报告。"""
    mapping = mapping if isinstance(mapping, Mapping) else {}
    actors = actors if isinstance(actors, Mapping) else {}
    dirty_paths = sorted(set(path for path in dirty_package_paths if isinstance(path, str)))
    dirty_set = set(dirty_paths)
    formal_map_dirty = _FORMAL_MAP in dirty_set
    roster_actors = []
    missing = []
    duplicate_labels = []
    duplicate_actor_paths = []
    duplicate_actor_packages = []
    dirty_actor_packages = []
    for entity_id in _ENTITY_IDS:
        expected_label = mapping.get(entity_id)
        record = actors.get(entity_id) or {}
        actor_label = record.get("actor_label")
        match_count = record.get("match_count", 1 if actor_label is not None else 0)
        resolved = (
            isinstance(expected_label, str)
            and match_count == 1
            and actor_label == expected_label
            and isinstance(record.get("actor_path"), str)
            and isinstance(record.get("package_path"), str)
        )
        if not resolved:
            missing.append(entity_id)
        if match_count > 1:
            duplicate_labels.append(entity_id)
        package_path = record.get("package_path")
        if isinstance(package_path, str) and package_path in dirty_set:
            dirty_actor_packages.append(package_path)
        roster_actors.append(
            {
                "entity_id": entity_id,
                "expected_label": expected_label,
                "actor_label": actor_label,
                "actor_path": record.get("actor_path"),
                "package_path": package_path,
                "package_dirty": isinstance(package_path, str) and package_path in dirty_set,
                "match_count": match_count,
                "resolved": resolved,
            }
        )

    checked_assets = [dict(item) for item in asset_checks]
    for item in checked_assets:
        item["package_dirty"] = item.get("path") in dirty_set
        item["class_match"] = (
            item.get("exists") is True
            and item.get("actual_class") == item.get("expected_class")
        )
    dirty_asset_packages = sorted(
        {
            item["path"]
            for item in checked_assets
            if item.get("package_dirty") and isinstance(item.get("path"), str)
        }
    )
    resolved_paths = [
        item["actor_path"] for item in roster_actors if item["resolved"]
    ]
    resolved_packages = [
        item["package_path"] for item in roster_actors if item["resolved"]
    ]
    duplicate_actor_paths = sorted(
        path for path in set(resolved_paths) if resolved_paths.count(path) > 1
    )
    duplicate_actor_packages = sorted(
        path for path in set(resolved_packages) if resolved_packages.count(path) > 1
    )
    formal_map_match = current_level_path == _FORMAL_MAP
    roster_ok = (
        not missing
        and not duplicate_labels
        and not duplicate_actor_paths
        and not duplicate_actor_packages
        and len(roster_actors) == len(_ENTITY_IDS)
    )
    key_asset_checks_ok = bool(checked_assets) and all(
        item.get("class_match") is True and not item.get("error")
        for item in checked_assets
    )
    collected_errors = list(errors)
    if not formal_map_match:
        collected_errors.append(
            "当前 Level 不匹配正式地图：expected={} actual={}".format(
                _FORMAL_MAP, current_level_path
            )
        )
    if missing:
        collected_errors.append("十人 roster 缺失或不精确：{}".format(", ".join(missing)))
    if duplicate_labels:
        collected_errors.append("十人 roster 标签重复：{}".format(", ".join(duplicate_labels)))
    if duplicate_actor_paths:
        collected_errors.append("十人 roster Actor path 重复：{}".format(", ".join(duplicate_actor_paths)))
    if duplicate_actor_packages:
        collected_errors.append("十人 roster package path 重复：{}".format(", ".join(duplicate_actor_packages)))
    if dirty_actor_packages:
        collected_errors.append("球员 External Actor package 已 dirty")
    if formal_map_dirty:
        collected_errors.append("正式地图 package 已 dirty")
    if not key_asset_checks_ok:
        collected_errors.append("关键外观资产存在性或 class 检查未通过")
    if dirty_asset_packages:
        collected_errors.append("关键外观资产 package 已 dirty")

    return {
        "schema": "futsalmot_episode_appearance_diagnostic",
        "version": 1,
        "current_level": current_level_path,
        "formal_map": _FORMAL_MAP,
        "formal_map_match": formal_map_match,
        "roster": {
            "expected_count": len(_ENTITY_IDS),
            "found_count": sum(item["resolved"] for item in roster_actors),
            "ok": roster_ok,
            "missing_entity_ids": missing,
            "duplicate_entity_ids": duplicate_labels,
            "duplicate_actor_paths": duplicate_actor_paths,
            "duplicate_actor_package_paths": duplicate_actor_packages,
            "actors": roster_actors,
        },
        "dirty_package_paths": dirty_paths,
        "formal_map_package_dirty": formal_map_dirty,
        "dirty_actor_package_paths": sorted(set(dirty_actor_packages)),
        "dirty_key_asset_package_paths": dirty_asset_packages,
        "key_asset_checks_ok": key_asset_checks_ok,
        "asset_checks": checked_assets,
        "errors": list(dict.fromkeys(collected_errors)),
        "task7_preflight_can_start": (
            formal_map_match
            and not formal_map_dirty
            and roster_ok
            and not dirty_actor_packages
            and key_asset_checks_ok
            and not dirty_asset_packages
            and not errors
        ),
    }


def _load_mapping(mapping_path: Optional[str] = None) -> dict:
    path = Path(
        mapping_path
        or os.environ.get("FUTSALMOT_ACTOR_MAPPING")
        or Path(__file__).with_name("actor_mapping.example.json")
    )
    with path.open("r", encoding="utf-8") as stream:
        mapping = json.load(stream)
    return mapping


def _asset_checks(unreal_module: Any) -> list:
    checks = []
    for category, asset_id, path, expected_class in _ASSET_SPECS:
        check = {
            "category": category,
            "asset_id": asset_id,
            "path": path,
            "expected_class": expected_class,
            "exists": None,
            "actual_class": None,
        }
        try:
            library = unreal_module.EditorAssetLibrary
            asset_data = library.find_asset_data(path)
            is_valid = getattr(asset_data, "is_valid", None)
            if asset_data is None or not callable(is_valid) or not is_valid():
                check["exists"] = False
                check["error"] = "find_asset_data 返回无效 AssetData"
            else:
                class_path = asset_data.asset_class_path
                asset_name = class_path.asset_name
                check["exists"] = True
                check["actual_class"] = str(asset_name)
        except Exception as exc:
            check["error"] = "{}: {}".format(type(exc).__name__, exc)
        checks.append(check)
    return checks


def collect_diagnostic(unreal_module: Any, mapping: Mapping[str, str]) -> dict:
    """只从当前 Editor 状态采集地图、Actor、dirty package 和资产元数据。"""
    from episode_appearance import (
        _all_level_actors,
        _current_level_path,
        _dirty_package_paths,
        _object_path,
    )

    errors = []
    if not isinstance(mapping, Mapping):
        errors.append("actor mapping 必须是 JSON 对象")
        mapping = {}
    try:
        current_level_path = _current_level_path(unreal_module)
    except Exception as exc:
        current_level_path = None
        errors.append("当前 Level 查询失败: {}: {}".format(type(exc).__name__, exc))

    actors_by_label = {}
    try:
        for actor in _all_level_actors(unreal_module):
            try:
                label = actor.get_actor_label()
                actors_by_label.setdefault(label, []).append(actor)
            except Exception:
                continue
    except Exception as exc:
        errors.append("当前 Level Actor 查询失败: {}: {}".format(type(exc).__name__, exc))

    actor_records = {}
    for entity_id in _ENTITY_IDS:
        expected_label = mapping.get(entity_id)
        matches = actors_by_label.get(expected_label, []) if isinstance(expected_label, str) else []
        record = {"match_count": len(matches)}
        if len(matches) == 1:
            actor = matches[0]
            try:
                record["actor_label"] = actor.get_actor_label()
            except Exception as exc:
                record["actor_label_error"] = "{}: {}".format(type(exc).__name__, exc)
            try:
                record["actor_path"] = actor.get_path_name()
            except Exception as exc:
                record["actor_path_error"] = "{}: {}".format(type(exc).__name__, exc)
            try:
                record["package_path"] = _object_path(actor.get_outermost())
            except Exception as exc:
                record["package_path_error"] = "{}: {}".format(type(exc).__name__, exc)
        actor_records[entity_id] = record

    try:
        dirty_paths = sorted(_dirty_package_paths(unreal_module))
    except Exception as exc:
        dirty_paths = []
        errors.append("dirty package 查询失败: {}: {}".format(type(exc).__name__, exc))
    try:
        checks = _asset_checks(unreal_module)
    except Exception as exc:
        checks = []
        errors.append("关键资产检查初始化失败: {}: {}".format(type(exc).__name__, exc))
    report = build_diagnostic_report(
        current_level_path, mapping, actor_records, dirty_paths, checks, errors
    )
    return report


def main(mapping_path: Optional[str] = None) -> dict:
    """在 Unreal Editor Python 环境采集并输出 JSON 诊断结果。"""
    import unreal

    try:
        mapping = _load_mapping(mapping_path)
    except Exception as exc:
        report = build_diagnostic_report(
            None, {}, {}, [], [], ["actor mapping 读取失败: {}: {}".format(type(exc).__name__, exc)]
        )
    else:
        report = collect_diagnostic(unreal, mapping)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return report


if __name__ == "__main__":
    main()
