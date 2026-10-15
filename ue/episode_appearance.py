"""UE episode Appearance 的纯数据契约和报告状态机。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


APPEARANCE_SCHEMA = "futsalmot_episode_appearance"
APPEARANCE_VERSION = 1
APPEARANCE_POLICY = "futsalmot_appearance_v1"
APPLICATION_REPORT_SCHEMA = "futsalmot_appearance_application"
APPLICATION_REPORT_VERSION = 1
ENTITY_IDS = ("L0", "L1", "L2", "L3", "L4", "R0", "R1", "R2", "R3", "R4")
KIT_CLASSES = (
    "left_outfield",
    "right_outfield",
    "left_goalkeeper",
    "right_goalkeeper",
)
REPORT_STATES = (
    "not_started",
    "preflight_failed",
    "applied",
    "rendering",
    "restored",
    "restore_failed",
)
_ALLOWED_TRANSITIONS = {
    "not_started": {"preflight_failed", "applied", "restored", "restore_failed"},
    "preflight_failed": set(),
    "applied": {"rendering", "restored", "restore_failed"},
    "rendering": {"restored", "restore_failed"},
    "restored": set(),
    "restore_failed": {"restored"},
}
_MODEL_IDS = set(range(3, 10))
_KIT_IDS = {"01", "02", "03", "04", "05"}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REPORT_FIELDS = {
    "schema",
    "version",
    "state",
    "manifest_fingerprint",
    "trajectory_sha256",
    "episode_id",
    "catalog_id",
    "catalog_fingerprint",
    "transaction_id",
    "actors",
    "mrq_jobs",
    "mrq_pending",
    "render_result",
    "errors",
    "warnings",
}
_ENTITY_ASSET_FIELDS = (
    "mesh",
    "anim_class",
    "skeleton",
    "skin_texture",
    "jersey_material",
    "shorts_material",
)
_TERMINAL_RENDER_STATUSES = {
    "success",
    "failed",
    "partial",
    "cancelled",
    "canceled",
    "error",
}
_TERMINAL_MRQ_STATUSES = {"success", "failed", "cancelled", "canceled", "skipped"}
_FORMAL_MAP = "/Game/FutsalMOT/Maps/L_FutsalCourt"
_VISUAL_COMPONENT = "MPFBVisual"
_BODY_MATERIAL = "/Game/FutsalMOT/Characters/MPFB/Materials/M_MPFB_Body"
_BODY_PARAMETER = "SkinTexture"
_SHARED_SKELETON = "/Game/FutsalMOT/Characters/MPFB/Core/Skeleton/SKEL_MPFB_Base"
_SHARED_ANIM_BLUEPRINT = "/Game/FutsalMOT/Characters/MPFB/Core/Retarget/ABP_MPFB_RuntimeRetarget"
_SNAPSHOT_SCHEMA = "futsalmot_episode_appearance_rollback"
_SNAPSHOT_VERSION = 3
_RESTORE_VERIFIED_TOKEN = object()
_SNAPSHOT_FIELDS = (
    "entity_id",
    "actor_label",
    "actor_path",
    "package_path",
    "component_name",
    "skeletal_mesh",
    "anim_class",
    "attach_parent",
    "actor_transform",
    "relative_transform",
    "override_materials",
    "leader_pose_component",
    "visibility",
    "package_dirty",
)
_SNAPSHOT_REQUIRED = frozenset(_SNAPSHOT_FIELDS)
_SHORTS_SLOT_NAMES = {
    "Human_futsalmot_shorts_male_v2_005",
    "Human_futsalmot_shorts_male_final",
}
_HAIR_SLOT_NAMES = {
    "Human_afro01",
    "Human_short01",
    "Human_short02",
    "Human_short03",
    "Human_short04",
    "Human_littleright_bobcut_hair",
    "Human_cortu_short_messy_hair",
}


def _canonical_json(value: Any) -> bytes:
    """生成与 P1 manifest 相同的 canonical JSON 字节。"""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _manifest_fingerprint(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("fingerprint", None)
    return _fingerprint(payload)


def _is_asset_path(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith("/Game/"):
        return False
    if value == "/Game/" or "\\" in value or "." in value:
        return False
    return all(part not in ("", ".", "..") for part in value.split("/Game/", 1)[1].split("/"))


def _rank_candidate(namespace: str, namespace_seed: int, entity_id: str, candidate_id: Any):
    candidate_text = str(candidate_id)
    payload = (
        "{}:rank:{}:{}:{}:{}".format(
            APPEARANCE_POLICY, namespace, namespace_seed, entity_id, candidate_text
        )
    ).encode("utf-8")
    return hashlib.sha256(payload).digest(), candidate_text


def _select_candidate(candidates: list, namespace: str, namespace_seed: int, entity_id: str, key: str):
    return min(
        candidates,
        key=lambda candidate: _rank_candidate(
            namespace, namespace_seed, entity_id, candidate[key]
        ),
    )


def _manifest_errors(manifest: Any, catalog: Any) -> list:
    errors = []
    if not isinstance(manifest, Mapping):
        return ["manifest 必须是 JSON 对象"]
    if not isinstance(catalog, Mapping):
        return ["catalog 必须是 JSON 对象"]

    if manifest.get("schema") != APPEARANCE_SCHEMA:
        errors.append("manifest schema 无效")
    if type(manifest.get("version")) is not int or manifest.get("version") != APPEARANCE_VERSION:
        errors.append("manifest version 无效")
    if manifest.get("policy") != APPEARANCE_POLICY:
        errors.append("manifest policy 无效")
    for field in ("root_seed", "ue_visual_seed"):
        value = manifest.get(field)
        if type(value) is not int:
            errors.append("manifest {} 必须是非 bool 整数".format(field))
    visual_seed = manifest.get("ue_visual_seed")
    expected_seeds = None
    if type(visual_seed) is int and 0 <= visual_seed <= 0x7FFFFFFF:
        expected_seeds = {}
        for namespace in ("model", "skin", "kit"):
            digest = hashlib.sha256(
                "{}:{}:{}".format(APPEARANCE_POLICY, visual_seed, namespace).encode("utf-8")
            ).digest()
            expected_seeds[namespace] = int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
        if manifest.get("namespace_seeds") != expected_seeds:
            errors.append("manifest namespace_seeds 与 ue_visual_seed 不一致")
    elif type(visual_seed) is int:
        errors.append("manifest ue_visual_seed 超出有效范围")

    for field in ("fingerprint", "catalog_fingerprint", "trajectory_sha256"):
        value = manifest.get(field)
        if not isinstance(value, str) or not _SHA256_RE.match(value):
            errors.append("manifest {} 必须是小写 SHA-256".format(field))
    if isinstance(manifest.get("fingerprint"), str) and _SHA256_RE.match(manifest["fingerprint"]):
        try:
            if manifest["fingerprint"] != _manifest_fingerprint(manifest):
                errors.append("manifest fingerprint 无效")
        except (TypeError, ValueError):
            errors.append("manifest fingerprint 无法计算")

    if not isinstance(catalog.get("catalog_id"), str) or not catalog.get("catalog_id"):
        errors.append("catalog_id 缺失或无效")
    if catalog.get("mode") != "seeded":
        errors.append("catalog mode 必须为 seeded")
    if catalog.get("policy") != APPEARANCE_POLICY:
        errors.append("catalog policy 无效")
    if not isinstance(catalog.get("body_texture_parameter"), str) or not catalog.get(
        "body_texture_parameter", ""
    ).strip():
        errors.append("catalog body_texture_parameter 缺失")

    models = catalog.get("models")
    model_by_id = {}
    if not isinstance(models, list) or len(models) != 7:
        errors.append("catalog models 必须包含 7 个候选")
    else:
        for index, model in enumerate(models):
            if not isinstance(model, Mapping):
                errors.append("catalog models[{}] 必须是对象".format(index))
                continue
            player_id = model.get("player_id")
            if type(player_id) is not int or player_id not in _MODEL_IDS:
                errors.append("catalog models[{}].player_id 无效".format(index))
            elif player_id in model_by_id:
                errors.append("catalog models 存在重复 player_id: {}".format(player_id))
            else:
                model_by_id[player_id] = model
            for field in ("mesh", "anim_class", "skeleton"):
                if not _is_asset_path(model.get(field)):
                    errors.append("catalog models[{}].{} 必须是 /Game/ 资产路径".format(index, field))
        if set(model_by_id) != _MODEL_IDS:
            errors.append("catalog models 必须完整覆盖 player_id 3-9")

    skins = catalog.get("skins")
    skin_by_id = {}
    if not isinstance(skins, list) or len(skins) != 9:
        errors.append("catalog skins 必须包含 9 个候选")
    else:
        for index, skin in enumerate(skins):
            if not isinstance(skin, Mapping):
                errors.append("catalog skins[{}] 必须是对象".format(index))
                continue
            skin_id = skin.get("skin_id")
            if not isinstance(skin_id, str) or not skin_id:
                errors.append("catalog skins[{}].skin_id 无效".format(index))
            elif skin_id in skin_by_id:
                errors.append("catalog skins 存在重复 skin_id: {}".format(skin_id))
            else:
                skin_by_id[skin_id] = skin
            if not _is_asset_path(skin.get("texture")):
                errors.append("catalog skins[{}].texture 必须是 /Game/ 资产路径".format(index))

    kits = catalog.get("kits")
    kit_by_id = {}
    if not isinstance(kits, list) or len(kits) != 5:
        errors.append("catalog kits 必须包含 5 个候选")
    else:
        for index, kit in enumerate(kits):
            if not isinstance(kit, Mapping):
                errors.append("catalog kits[{}] 必须是对象".format(index))
                continue
            set_id = kit.get("set_id")
            if not isinstance(set_id, str) or set_id not in _KIT_IDS:
                errors.append("catalog kits[{}].set_id 无效".format(index))
            elif set_id in kit_by_id:
                errors.append("catalog kits 存在重复 set_id: {}".format(set_id))
            else:
                kit_by_id[set_id] = kit
            for field in ("jersey_material", "shorts_material"):
                if not _is_asset_path(kit.get(field)):
                    errors.append("catalog kits[{}].{} 必须是 /Game/ 资产路径".format(index, field))
        if set(kit_by_id) != _KIT_IDS:
            errors.append("catalog kits 必须完整覆盖 set_id 01-05")

    if isinstance(catalog.get("catalog_id"), str):
        try:
            if manifest.get("catalog_id") != catalog.get("catalog_id"):
                errors.append("manifest catalog_id 与 catalog 不一致")
            if manifest.get("catalog_fingerprint") != _fingerprint(catalog):
                errors.append("catalog_fingerprint 与当前 catalog 不一致")
        except (TypeError, ValueError):
            errors.append("catalog fingerprint 无法计算")

    if manifest.get("body_texture_parameter") != catalog.get("body_texture_parameter"):
        errors.append("manifest body_texture_parameter 与 catalog 不一致")

    kit_classes = manifest.get("kit_classes")
    if not isinstance(kit_classes, Mapping) or set(kit_classes) != set(KIT_CLASSES):
        errors.append("manifest kit_classes 必须完整覆盖四种 kit class")
    else:
        kit_values = [kit_classes.get(name) for name in KIT_CLASSES]
        valid_kit_values = [value for value in kit_values if isinstance(value, str)]
        if (
            len(valid_kit_values) != len(KIT_CLASSES)
            or len(set(valid_kit_values)) != len(KIT_CLASSES)
            or any(value not in kit_by_id for value in valid_kit_values)
        ):
            errors.append("manifest kit_classes 必须引用四个不同的有效 Set ID")

    entities = manifest.get("entities")
    entity_by_id = {}
    if not isinstance(entities, list) or len(entities) != len(ENTITY_IDS):
        errors.append("manifest entities 必须恰好包含 10 名球员")
    else:
        for index, entity in enumerate(entities):
            if not isinstance(entity, Mapping):
                errors.append("manifest entities[{}] 必须是对象".format(index))
                continue
            entity_id = entity.get("entity_id")
            if entity_id not in ENTITY_IDS:
                errors.append("manifest entities[{}].entity_id 非法".format(index))
                continue
            if entity_id in entity_by_id:
                errors.append("manifest entities 存在重复 entity_id: {}".format(entity_id))
                continue
            entity_by_id[entity_id] = entity

            team = "left" if entity_id.startswith("L") else "right"
            if entity.get("team") != team:
                errors.append("{} team 与 entity_id 不一致".format(entity_id))
            if type(entity.get("is_goalkeeper")) is not bool:
                errors.append("{}.is_goalkeeper 必须是 bool".format(entity_id))
            elif isinstance(kit_classes, Mapping):
                expected_class = "{}_{}".format(
                    team,
                    "goalkeeper" if entity["is_goalkeeper"] else "outfield",
                )
                if entity.get("kit_class") != expected_class:
                    errors.append("{} kit_class 与 goalkeeper/team 不一致".format(entity_id))
                if entity.get("kit_set_id") != kit_classes.get(expected_class):
                    errors.append("{} kit_set_id 与 kit_classes 不一致".format(entity_id))

            player_id = entity.get("player_id")
            model = model_by_id.get(player_id) if type(player_id) is int else None
            if model is None:
                errors.append("{} 引用了无效 player_id".format(entity_id))
            else:
                for field in ("mesh", "anim_class", "skeleton"):
                    if entity.get(field) != model.get(field):
                        errors.append("{} {} 与 catalog model 不一致".format(entity_id, field))
            skin_id = entity.get("skin_id")
            skin = skin_by_id.get(skin_id) if isinstance(skin_id, str) else None
            if skin is None or entity.get("skin_texture") != skin.get("texture"):
                errors.append("{} skin assignment 与 catalog 不一致".format(entity_id))
            kit_set_id = entity.get("kit_set_id")
            kit = kit_by_id.get(kit_set_id) if isinstance(kit_set_id, str) else None
            if kit is None:
                errors.append("{} 引用了无效 kit_set_id".format(entity_id))
            else:
                for field in ("jersey_material", "shorts_material"):
                    if entity.get(field) != kit.get(field):
                        errors.append("{} {} 与 catalog kit 不一致".format(entity_id, field))

            for field in _ENTITY_ASSET_FIELDS:
                if not _is_asset_path(entity.get(field)):
                    errors.append("{} {} 必须是 /Game/ 资产路径".format(entity_id, field))
        if set(entity_by_id) != set(ENTITY_IDS):
            errors.append("manifest entities 必须完整覆盖 L0-L4 和 R0-R4")

    if set(entity_by_id) == set(ENTITY_IDS):
        for team in ("left", "right"):
            keepers = [
                entity for entity in entity_by_id.values()
                if entity.get("team") == team and entity.get("is_goalkeeper") is True
            ]
            if len(keepers) != 1:
                errors.append("{} 队必须恰好有一名 goalkeeper".format(team))

    if expected_seeds is not None and len(model_by_id) == 7 and len(skin_by_id) == 9 and len(kit_by_id) == 5:
        ranked_kits = sorted(
            kits,
            key=lambda kit: _rank_candidate(
                "kit", expected_seeds["kit"], "kit", kit["set_id"]
            ),
        )[:4]
        expected_kit_classes = {
            kit_class: kit["set_id"]
            for kit_class, kit in zip(KIT_CLASSES, ranked_kits)
        }
        if kit_classes != expected_kit_classes:
            errors.append("manifest kit_classes 与 ue_visual_seed 的 deterministic ranking 不一致")

        if set(entity_by_id) == set(ENTITY_IDS):
            for entity_id in ENTITY_IDS:
                entity = entity_by_id[entity_id]
                team = "left" if entity_id.startswith("L") else "right"
                goalkeeper = entity.get("is_goalkeeper")
                if type(goalkeeper) is not bool:
                    continue
                kit_class = "{}_{}".format(
                    team, "goalkeeper" if goalkeeper else "outfield"
                )
                expected_model = _select_candidate(
                    models, "model", expected_seeds["model"], entity_id, "player_id"
                )
                expected_skin = _select_candidate(
                    skins, "skin", expected_seeds["skin"], entity_id, "skin_id"
                )
                expected_kit_id = expected_kit_classes[kit_class]
                if entity.get("player_id") != expected_model["player_id"]:
                    errors.append("{} player assignment 与 ue_visual_seed 的 deterministic ranking 不一致".format(entity_id))
                if entity.get("skin_id") != expected_skin["skin_id"]:
                    errors.append("{} skin assignment 与 ue_visual_seed 的 deterministic ranking 不一致".format(entity_id))
                if entity.get("kit_set_id") != expected_kit_id:
                    errors.append("{} kit assignment 与 ue_visual_seed 的 deterministic mapping 不一致".format(entity_id))
    return list(dict.fromkeys(errors))


def _meta_errors(meta: Any, manifest: Mapping[str, Any]) -> list:
    if not isinstance(meta, Mapping):
        return ["meta 必须是 JSON 对象"]
    errors = []
    if meta.get("schema") != "grf_ue_episode" or type(meta.get("version")) is not int or meta.get("version") != 1:
        errors.append("meta schema/version 无效")
    if meta.get("episode_id") != manifest.get("episode_id"):
        errors.append("manifest episode_id 与 meta 不一致")
    randomness = meta.get("randomness")
    if not isinstance(randomness, Mapping):
        errors.append("meta.randomness 缺失或无效")
    else:
        ue_visual_seed = randomness.get("ue_visual_seed")
        if type(ue_visual_seed) is not int:
            errors.append("meta.randomness.ue_visual_seed 必须是非 bool 整数")
        elif ue_visual_seed != manifest.get("ue_visual_seed"):
            errors.append("manifest ue_visual_seed 与 meta 不一致")
        root_seed = randomness.get("root_seed")
        if root_seed is None:
            source = meta.get("source")
            root_seed = source.get("seed") if isinstance(source, Mapping) else None
        if type(root_seed) is not int:
            errors.append("meta root_seed 必须是非 bool 整数")
        elif root_seed != manifest.get("root_seed"):
            errors.append("manifest root_seed 与 meta 不一致")

    raw_entities = meta.get("entities")
    if not isinstance(raw_entities, list):
        errors.append("meta.entities 必须是列表")
        return errors
    by_id = {}
    for index, entity in enumerate(raw_entities):
        if not isinstance(entity, Mapping):
            errors.append("meta.entities[{}] 必须是对象".format(index))
            continue
        entity_id = entity.get("id")
        if entity_id == "BALL":
            continue
        if not isinstance(entity_id, str):
            errors.append("meta.entities[{}].id 必须是字符串".format(index))
            continue
        if entity_id in by_id:
            errors.append("meta.entities 存在重复 entity id: {}".format(entity_id))
        else:
            by_id[entity_id] = entity
    if set(by_id) != set(ENTITY_IDS):
        errors.append("meta.entities 必须完整覆盖 L0-L4 和 R0-R4")
    for entity in manifest.get("entities", []):
        if not isinstance(entity, Mapping):
            continue
        meta_entity = by_id.get(entity.get("entity_id"))
        if meta_entity is None:
            continue
        if meta_entity.get("team") != entity.get("team"):
            errors.append("{} team 与 meta.entities 不一致".format(entity["entity_id"]))
        if meta_entity.get("is_goalkeeper") is not entity.get("is_goalkeeper"):
            errors.append("{} goalkeeper metadata 与 meta.entities 不一致".format(entity["entity_id"]))
    for team in ("left", "right"):
        keepers = [
            item for item in by_id.values()
            if item.get("team") == team and item.get("is_goalkeeper") is True
        ]
        if len(keepers) != 1:
            errors.append("meta.entities 中 {} 队必须恰好有一名 goalkeeper".format(team))
    return list(dict.fromkeys(errors))


def validate_appearance_inputs(
    manifest: dict,
    catalog: dict,
    meta: dict,
    frames_path: Any,
) -> list:
    """在 P2 侧独立验证 manifest、catalog、meta 和轨迹原始字节。"""
    errors = _manifest_errors(manifest, catalog)
    if not isinstance(manifest, Mapping):
        return errors
    errors.extend(_meta_errors(meta, manifest))
    try:
        path = Path(frames_path)
        if not path.is_file():
            errors.append("frames.jsonl 不存在或不是文件")
        else:
            actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if manifest.get("trajectory_sha256") != actual_hash:
                errors.append("trajectory_sha256 与当前 frames.jsonl 不一致")
    except (OSError, TypeError, ValueError) as exc:
        errors.append("frames.jsonl 无法读取: {}".format(exc))
    return list(dict.fromkeys(errors))


def _validate_mapping(mapping: Any) -> list:
    if not isinstance(mapping, Mapping):
        return ["actor mapping 必须是对象"]
    errors = []
    labels = []
    for entity_id in ENTITY_IDS:
        label = mapping.get(entity_id)
        if not isinstance(label, str) or not label.strip():
            errors.append("actor mapping 缺少有效实体 {}".format(entity_id))
        elif "\\" in label or re.match(r"^[A-Za-z]:[/\\]", label):
            errors.append("actor mapping {} 不能包含本机文件路径".format(entity_id))
        else:
            labels.append(label.strip().casefold())
    if len(labels) != len(set(labels)):
        errors.append("actor mapping 的球员标签必须唯一")
    return errors


def _preflight_contract(manifest: dict, catalog: dict, mapping: dict) -> dict:
    """只读检查 manifest/catalog 与十名球员的 mapping 契约。"""
    errors = _manifest_errors(manifest, catalog)
    errors.extend(_validate_mapping(mapping))
    mapped = []
    if isinstance(mapping, Mapping):
        for entity_id in ENTITY_IDS:
            label = mapping.get(entity_id)
            if isinstance(label, str) and label.strip():
                mapped.append({"entity_id": entity_id, "actor_label": label.strip()})
    return {
        "ok": not errors,
        "operation": "preflight",
        "errors": list(dict.fromkeys(errors)),
        "warnings": [],
        "actors": mapped,
    }


def preflight_episode_appearance(manifest: dict, catalog: dict, mapping: dict) -> dict:
    """执行纯契约及当前 UE Editor 十人 roster 的只读 preflight。"""
    result = _preflight_contract(manifest, catalog, mapping)
    if not result["ok"]:
        return result
    try:
        ue = _load_unreal()
        ue_result = _ue_preflight(ue, manifest, catalog, mapping)
    except Exception as exc:
        ue_result = {
            "ok": False,
            "errors": ["UE preflight 失败: {}: {}".format(type(exc).__name__, exc)],
            "actors": {},
        }
    result["errors"].extend(ue_result["errors"])
    result["ok"] = not result["errors"]
    if ue_result.get("actors"):
        result["actors"] = [
            {
                "entity_id": entity_id,
                "actor_label": actor.get_actor_label(),
                "actor_path": actor.get_path_name(),
            }
            for entity_id, actor in sorted(ue_result["actors"].items())
        ]
    return result


def validate_local_recovery_evidence(
    snapshot: Mapping[str, Any],
    report: Mapping[str, Any],
    manifest: Mapping[str, Any],
    catalog: Mapping[str, Any],
) -> list:
    """验证本机 snapshot/report 是否属于当前 episode 事务。"""
    errors = _validate_rollback_payload(snapshot)
    if not isinstance(snapshot, Mapping):
        return errors
    try:
        _validate_report(report)
    except (TypeError, ValueError, KeyError, OverflowError, RecursionError) as exc:
        errors.append("application report 无效: {}".format(exc))
    if not isinstance(report, Mapping):
        return errors
    if report.get("state") not in {"applied", "rendering", "restore_failed", "restored"}:
        errors.append("application report state 不属于可恢复证据")
    if snapshot.get("manifest_fingerprint") != manifest.get("fingerprint"):
        errors.append("rollback manifest_fingerprint 与当前 manifest 不一致")
    expected_report_fields = {
        "manifest_fingerprint": manifest.get("fingerprint"),
        "catalog_fingerprint": manifest.get("catalog_fingerprint"),
        "trajectory_sha256": manifest.get("trajectory_sha256"),
    }
    for field, expected in expected_report_fields.items():
        if report.get(field) != expected:
            errors.append("application report {} 与当前 manifest 不一致".format(field))
    for field in ("episode_id", "catalog_id"):
        if report.get(field) != manifest.get(field):
            errors.append("application report {} 与当前 manifest 不一致".format(field))
    if report.get("transaction_id") != snapshot.get("transaction_id"):
        errors.append("rollback transaction_id 与 application report 不一致")
    if report.get("catalog_fingerprint") != _fingerprint(catalog):
        errors.append("application report catalog_fingerprint 与当前 catalog 不一致")
    actors = report.get("actors")
    entities = {
        entity.get("entity_id"): entity
        for entity in manifest.get("entities", [])
        if isinstance(entity, Mapping)
    }
    if isinstance(actors, list) and set(entities) == set(ENTITY_IDS):
        snapshots = {
            item.get("entity_id"): item
            for item in snapshot.get("actors", [])
            if isinstance(item, Mapping)
        }
        for actor in actors:
            if not isinstance(actor, Mapping):
                continue
            entity_id = actor.get("entity_id")
            entity = entities.get(entity_id)
            if entity is None:
                continue
            if report.get("state") in {"applied", "rendering"} and actor.get("apply_status") != "applied":
                errors.append("{} unresolved recovery report apply_status 不是 applied".format(entity_id))
            if report.get("state") == "restored" and actor.get("restore_status") != "restored":
                errors.append("{} restored recovery report restore_status 不是 restored".format(entity_id))
            if actor.get("actor_label") != snapshots.get(entity_id, {}).get("actor_label"):
                errors.append("{} actor_label 与 rollback snapshot 不一致".format(entity_id))
            assets = actor.get("assets")
            slots = actor.get("slots")
            if not isinstance(assets, Mapping):
                errors.append("{} application report assets 缺失".format(entity_id))
            else:
                for field in _ENTITY_ASSET_FIELDS:
                    if assets.get(field) != entity.get(field):
                        errors.append("{} assets.{} 与 manifest 不一致".format(entity_id, field))
            expected_slots = {
                "body_texture_parameter": manifest.get("body_texture_parameter"),
                "Body": entity.get("skin_texture"),
                "Jersey": entity.get("jersey_material"),
                "Shorts": entity.get("shorts_material"),
            }
            if not isinstance(slots, Mapping):
                errors.append("{} application report slots 缺失".format(entity_id))
            else:
                for slot, expected in expected_slots.items():
                    if slots.get(slot) != expected:
                        errors.append("{} slots.{} 与 manifest 不一致".format(entity_id, slot))
    return list(dict.fromkeys(errors))


def preflight_episode_appearance_recovery(
    snapshot: Mapping[str, Any],
    mapping: Mapping[str, Any],
    verify_restored: bool = False,
) -> dict:
    """恢复前只读确认正式地图、十名映射 Actor 和 snapshot identity。"""
    errors = _validate_actor_snapshots(snapshot.get("actors"))
    if errors:
        return {"ok": False, "errors": errors, "actors": []}
    try:
        ue = _load_unreal()
        errors.extend(
            _appearance_safety_gate(
                _current_level_path(ue),
                [item.get("entity_id") for item in snapshot["actors"]],
            )
        )
        actors, actor_errors = _actor_matches(ue, mapping)
        errors.extend(actor_errors)
        if set(actors) == set(ENTITY_IDS):
            snapshot_by_id = {item["entity_id"]: item for item in snapshot["actors"]}
            dirty_paths = _dirty_package_paths(ue) if verify_restored else set()
            for entity_id, actor in actors.items():
                expected = snapshot_by_id[entity_id]
                if actor.get_actor_label() != expected.get("actor_label"):
                    errors.append("{} recovery Actor label 与 snapshot 不一致".format(entity_id))
                if actor.get_path_name() != expected.get("actor_path"):
                    errors.append("{} recovery Actor identity 与 snapshot 不一致".format(entity_id))
                if _actor_package_path(actor) != expected.get("package_path"):
                    errors.append("{} recovery package identity 与 snapshot 不一致".format(entity_id))
                if verify_restored:
                    try:
                        component = _component_for_actor(ue, actor)
                        current = _component_state(
                            actor, component, expected.get("package_dirty")
                        )
                        current["entity_id"] = entity_id
                        if not _snapshot_matches_state(expected, current):
                            errors.append("{} live baseline 与 restored snapshot 不一致".format(entity_id))
                        if (expected["package_path"] in dirty_paths) != expected["package_dirty"]:
                            errors.append("{} External Actor dirty state 与 snapshot 不一致".format(entity_id))
                    except Exception as exc:
                        errors.append("{} restored baseline 查询失败: {}".format(entity_id, exc))
    except Exception as exc:
        errors.append("recovery UE preflight 失败: {}: {}".format(type(exc).__name__, exc))
    return {
        "ok": not errors,
        "errors": list(dict.fromkeys(errors)),
        "actors": [
            {"entity_id": entity_id, "actor_label": actor.get_actor_label(), "actor_path": actor.get_path_name()}
            for entity_id, actor in actors.items()
        ] if "actors" in locals() else [],
    }


def _build_actor_snapshot(state: Mapping[str, Any]) -> dict:
    """复制 UE 外观回滚所需字段，保留 override_materials 中的 None。"""
    missing = sorted(_SNAPSHOT_REQUIRED - set(state))
    if missing:
        raise ValueError("actor 快照缺少字段: {}".format(", ".join(missing)))
    snapshot = {field: deepcopy(state[field]) for field in _SNAPSHOT_FIELDS}
    if not isinstance(snapshot["override_materials"], list):
        raise ValueError("override_materials 必须是原生槽位数组")
    if type(snapshot["package_dirty"]) is not bool:
        raise ValueError("package_dirty 必须是 bool")
    return snapshot


def _validate_actor_snapshots(snapshots: Any) -> list:
    """验证十人快照完整，避免缺项时进入任何 UE mutation。"""
    if not isinstance(snapshots, list) or len(snapshots) != len(ENTITY_IDS):
        return ["actor snapshots 必须恰好包含十名球员"]
    errors = []
    seen = set()
    seen_paths = set()
    for index, snapshot in enumerate(snapshots):
        if not isinstance(snapshot, Mapping):
            errors.append("actor snapshots[{}] 必须是对象".format(index))
            continue
        entity_id = snapshot.get("entity_id")
        if entity_id not in ENTITY_IDS or entity_id in seen:
            errors.append("actor snapshots entity 集合非法或重复: {}".format(entity_id))
        else:
            seen.add(entity_id)
        actor_path = snapshot.get("actor_path")
        if (
            not isinstance(actor_path, str)
            or not actor_path.startswith("/Game/")
            or actor_path.startswith("/Game/__ExternalActors__/")
        ):
            errors.append("{} actor_path 必须是 UE Actor object path".format(entity_id))
        elif actor_path in seen_paths:
            errors.append("actor_path 重复: {}".format(actor_path))
        else:
            seen_paths.add(actor_path)
        missing = sorted(_SNAPSHOT_REQUIRED - set(snapshot))
        errors.extend(
            "actor snapshots[{}] 缺少 {}".format(index, field) for field in missing
        )
        if missing:
            continue
        package_path = snapshot.get("package_path")
        if (
            not isinstance(package_path, str)
            or not package_path.startswith("/Game/__ExternalActors__/")
            or package_path == actor_path
        ):
            errors.append("{} package_path 必须是独立的 External Actor package path".format(entity_id))
        if not isinstance(snapshot.get("override_materials"), list):
            errors.append("{} override_materials 必须是数组".format(entity_id))
        if not isinstance(snapshot.get("relative_transform"), Mapping):
            errors.append("{} relative_transform 必须是对象".format(entity_id))
        else:
            errors.extend(
                "{} {}".format(entity_id, error)
                for error in _transform_payload_errors(
                    snapshot["relative_transform"], "relative_transform"
                )
            )
        if not isinstance(snapshot.get("actor_transform"), Mapping):
            errors.append("{} actor_transform 必须是对象".format(entity_id))
        else:
            errors.extend(
                "{} {}".format(entity_id, error)
                for error in _transform_payload_errors(
                    snapshot["actor_transform"], "actor_transform"
                )
            )
        if type(snapshot.get("package_dirty")) is not bool:
            errors.append("{} package_dirty 必须是 bool".format(entity_id))
        elif snapshot["package_dirty"]:
            errors.append("{} baseline package_dirty 必须为 false".format(entity_id))
        if not isinstance(snapshot.get("actor_label"), str) or not snapshot.get("actor_label"):
            errors.append("{} actor_label 缺失".format(entity_id))
        if snapshot.get("component_name") != _VISUAL_COMPONENT:
            errors.append("{} component_name 必须是 {}".format(entity_id, _VISUAL_COMPONENT))
        if not isinstance(snapshot.get("skeletal_mesh"), str) or not snapshot.get("skeletal_mesh"):
            errors.append("{} skeletal_mesh 缺失".format(entity_id))
        if not isinstance(snapshot.get("anim_class"), str) or not snapshot.get("anim_class"):
            errors.append("{} anim_class 缺失".format(entity_id))
        if not isinstance(snapshot.get("relative_transform"), Mapping):
            errors.append("{} relative_transform 必须是对象".format(entity_id))
        if not isinstance(snapshot.get("visibility"), Mapping):
            errors.append("{} visibility 必须是对象".format(entity_id))
        else:
            for field in ("visible", "render_in_main_pass", "render_in_depth_pass"):
                if type(snapshot["visibility"].get(field)) is not bool:
                    errors.append("{} visibility.{} 缺失或非 bool".format(entity_id, field))
    if seen != set(ENTITY_IDS):
        errors.append("actor snapshots 必须完整覆盖 L0-L4 和 R0-R4")
    return list(dict.fromkeys(errors))


def _snapshot_matches_state(snapshot: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    """比较原始组件状态；不会把有效材质误当成 override 数组。"""
    return all(
        field in current and current[field] == snapshot[field]
        for field in _SNAPSHOT_FIELDS
    )


def _transform_payload_errors(transform: Mapping[str, Any], field_name: str) -> list:
    errors = []
    for field in ("location", "rotation", "scale"):
        values = transform.get(field)
        if (
            not isinstance(values, list)
            or len(values) != 3
            or any(
                type(value) not in (int, float)
                or not math.isfinite(float(value))
                for value in values
            )
        ):
            errors.append("{}.{} 无效".format(field_name, field))
    return errors


def _snapshot_transaction_id(
    manifest_fingerprint: str,
    snapshots: Sequence[Mapping[str, Any]],
    asset_packages: Sequence[Mapping[str, Any]] = (),
) -> str:
    return _fingerprint({
        "manifest_fingerprint": manifest_fingerprint,
        "actors": list(snapshots),
        "asset_packages": list(asset_packages),
    })


def _validate_asset_package_baseline(asset_packages: Any) -> list:
    if not isinstance(asset_packages, list):
        return ["asset_packages 必须是数组"]
    errors = []
    if not asset_packages:
        errors.append("asset_packages baseline 不能为空")
    seen = set()
    for index, record in enumerate(asset_packages):
        if not isinstance(record, Mapping):
            errors.append("asset_packages[{}] 必须是对象".format(index))
            continue
        path = record.get("package_path")
        if (
            not isinstance(path, str)
            or not path.startswith("/Game/")
            or path.startswith("/Game/__ExternalActors__/")
            or "." in path
        ):
            errors.append("asset_packages[{}].package_path 无效".format(index))
        elif path in seen:
            errors.append("asset_packages duplicate package_path: {}".format(path))
        else:
            seen.add(path)
        if type(record.get("package_dirty")) is not bool:
            errors.append("asset_packages[{}].package_dirty 必须是 bool".format(index))
        elif record["package_dirty"]:
            errors.append("asset package baseline dirty，拒绝丢弃用户修改: {}".format(path))
    return list(dict.fromkeys(errors))


def _asset_package_reload_candidates(
    baseline: Sequence[Mapping[str, Any]], current_dirty_paths: Sequence[str]
) -> set:
    dirty_now = set(current_dirty_paths)
    return {
        record["package_path"]
        for record in baseline
        if record.get("package_dirty") is False
        and record.get("package_path") in dirty_now
    }


def _selected_appearance_asset_paths(manifest: Mapping[str, Any]) -> set:
    paths = {
        entity[field]
        for entity in manifest.get("entities", [])
        if isinstance(entity, Mapping)
        for field in _ENTITY_ASSET_FIELDS
        if isinstance(entity.get(field), str)
    }
    paths.add(_BODY_MATERIAL)
    return paths


def _asset_outermost_package_path(asset: Any) -> str:
    outermost = asset.get_outermost()
    path = _object_path(outermost)
    if (
        not path
        or not path.startswith("/Game/")
        or path.startswith("/Game/__ExternalActors__/")
        or "." in path
    ):
        raise RuntimeError("asset outermost package path 无效: {}".format(path))
    return path


def _reloadable_asset_package_path(reference_path: str, asset: Any) -> Optional[str]:
    """仅返回可由 reload_packages 丢弃的普通 /Game 内容资产 package。"""
    if not callable(getattr(asset, "get_outermost", None)):
        return None
    outermost_path = _object_path(asset.get_outermost())
    if not outermost_path or outermost_path.startswith("/Game/__ExternalActors__/"):
        return None
    if not outermost_path.startswith("/Game/") or "." in outermost_path:
        raise RuntimeError(
            "reloadable asset package path 无效: reference={} outermost={}".format(
                reference_path, outermost_path
            )
        )
    return outermost_path


def _capture_asset_package_baseline(
    unreal_module: Any,
    loaded_assets: Mapping[str, Any],
    baseline_dirty_paths: Optional[set] = None,
) -> list:
    current_dirty_paths = _dirty_content_package_paths(unreal_module)
    dirty_paths = (
        set(baseline_dirty_paths)
        if baseline_dirty_paths is not None
        else current_dirty_paths
    )
    packages = {}
    for reference_path, asset in loaded_assets.items():
        package_path = _reloadable_asset_package_path(reference_path, asset)
        if package_path is None:
            continue
        if package_path in current_dirty_paths and package_path not in dirty_paths:
            raise RuntimeError(
                "asset package became dirty before actor mutation: {}".format(package_path)
            )
        packages[package_path] = {
            "package_path": package_path,
            "package_dirty": package_path in dirty_paths,
        }
    baseline = [packages[path] for path in sorted(packages)]
    errors = _validate_asset_package_baseline(baseline)
    if errors:
        raise RuntimeError("asset package preflight 失败: {}".format("; ".join(errors)))
    return baseline


def _select_loaded_assets(
    paths: Sequence[str], loaded: Mapping[str, Any]
) -> dict:
    selected = {}
    for path in paths:
        asset = loaded.get(path)
        if asset is None:
            raise RuntimeError("selected package asset was not preloaded: {}".format(path))
        selected[path] = asset
    return selected


def _merge_asset_package_baselines(*baselines: Sequence[Mapping[str, Any]]) -> list:
    merged = {}
    for baseline in baselines:
        for record in baseline:
            path = record["package_path"]
            previous = merged.get(path)
            if previous is not None and previous["package_dirty"] != record["package_dirty"]:
                raise ValueError("asset package baseline 状态冲突: {}".format(path))
            merged[path] = {
                "package_path": path,
                "package_dirty": record["package_dirty"],
            }
    result = [merged[path] for path in sorted(merged)]
    errors = _validate_asset_package_baseline(result)
    if errors:
        raise ValueError("asset package baseline 无效: {}".format("; ".join(errors)))
    return result


def _reload_asset_packages(unreal_module: Any, package_paths: Sequence[str]) -> list:
    paths = sorted(set(package_paths))
    if not paths:
        return []
    loading = unreal_module.EditorLoadingAndSavingUtils
    reload_packages = getattr(loading, "reload_packages", None)
    interaction_mode = getattr(unreal_module, "ReloadPackagesInteractionMode", None)
    assume_positive = getattr(interaction_mode, "ASSUME_POSITIVE", None)
    if not callable(reload_packages) or assume_positive is None:
        raise RuntimeError("UE 缺少已验证的非交互 reload_packages ASSUME_POSITIVE 能力")
    packages = []
    for path in paths:
        asset = unreal_module.load_asset(path)
        if asset is None:
            raise RuntimeError("dirty asset package 无法加载资产引用: {}".format(path))
        package = asset.get_outermost()
        if _object_path(package) != path:
            raise RuntimeError("reload asset package path readback 不一致: {}".format(path))
        packages.append(package)
    result = reload_packages(packages, assume_positive)
    if not isinstance(result, (tuple, list)) or not result or result[0] is not True:
        raise RuntimeError("非交互 asset package reload 失败: {}".format(result))
    dirty_after = _dirty_package_paths(unreal_module)
    still_dirty = sorted(set(paths) & dirty_after)
    if still_dirty:
        raise RuntimeError("asset package reload 后仍 dirty: {}".format(", ".join(still_dirty)))
    return paths


def _build_rollback_payload(
    manifest_fingerprint: str,
    snapshots: Sequence[Mapping[str, Any]],
    asset_packages: Sequence[Mapping[str, Any]],
) -> dict:
    errors = _validate_actor_snapshots(list(snapshots))
    errors.extend(_validate_asset_package_baseline(list(asset_packages)))
    if errors:
        raise ValueError("rollback snapshot 无效: {}".format("; ".join(errors)))
    return {
        "schema": _SNAPSHOT_SCHEMA,
        "version": _SNAPSHOT_VERSION,
        "transaction_id": _snapshot_transaction_id(
            manifest_fingerprint, snapshots, asset_packages
        ),
        "manifest_fingerprint": manifest_fingerprint,
        "map": _FORMAL_MAP,
        "actors": deepcopy(list(snapshots)),
        "asset_packages": deepcopy(list(asset_packages)),
    }


def _validate_rollback_payload(payload: Any) -> list:
    if not isinstance(payload, Mapping):
        return ["rollback snapshot 必须是对象"]
    errors = []
    if payload.get("schema") != _SNAPSHOT_SCHEMA:
        errors.append("rollback snapshot schema 无效")
    if type(payload.get("version")) is not int or payload.get("version") != _SNAPSHOT_VERSION:
        errors.append("rollback snapshot version 无效")
    if payload.get("map") != _FORMAL_MAP:
        errors.append("rollback snapshot map 无效")
    snapshots = payload.get("actors")
    errors.extend(_validate_actor_snapshots(snapshots))
    asset_packages = payload.get("asset_packages")
    errors.extend(_validate_asset_package_baseline(asset_packages))
    manifest_fingerprint = payload.get("manifest_fingerprint")
    if not isinstance(manifest_fingerprint, str) or not _SHA256_RE.match(manifest_fingerprint):
        errors.append("rollback manifest_fingerprint 无效")
    transaction_id = payload.get("transaction_id")
    if not isinstance(transaction_id, str) or not _SHA256_RE.match(transaction_id):
        errors.append("rollback transaction_id 无效")
    elif isinstance(snapshots, list) and isinstance(asset_packages, list):
        expected = _snapshot_transaction_id(
            manifest_fingerprint, snapshots, asset_packages
        )
        if transaction_id != expected:
            errors.append("rollback snapshot transaction_id 校验失败")
    return list(dict.fromkeys(errors))


def _snapshot_asset_paths(snapshots: Sequence[Mapping[str, Any]]) -> set:
    paths = set()
    for snapshot in snapshots:
        for field in ("skeletal_mesh", "anim_class", "attach_parent", "leader_pose_component"):
            value = snapshot.get(field)
            if isinstance(value, str) and value:
                paths.add(value)
        paths.update(
            value for value in snapshot.get("override_materials", [])
            if isinstance(value, str) and value
        )
    return paths


def _resolve_semantic_slots(slot_names: Sequence[str]) -> dict:
    """按 Task 1 实测 slot 名称解析七类正式材质语义。"""
    names = list(slot_names)
    targets = {
        "Body": {"Human_body"},
        "Hair": _HAIR_SLOT_NAMES,
        "Eyebrows": {"Human_eyebrow001"},
        "Jersey": {"材质"},
        "Shorts": _SHORTS_SLOT_NAMES,
        "Eyes": {"Human_low-poly"},
        "Shoes": {"Human_shoes06"},
    }
    known_names = set().union(*targets.values())
    unknown = [name for name in names if name not in known_names]
    if unknown:
        raise ValueError("unknown semantic material slot names: {}".format(unknown))
    for semantic, accepted_names in targets.items():
        if sum(1 for name in names if name in accepted_names) != 1:
            raise ValueError(
                "{} semantic material slot 必须精确匹配一次".format(semantic)
            )
    result = {}
    for semantic, accepted_names in targets.items():
        matches = [index for index, name in enumerate(names) if name in accepted_names]
        if len(matches) != 1:
            raise ValueError(
                "{} semantic material slot 必须精确匹配一次，实际 {}".format(
                    semantic, len(matches)
                )
            )
        result[semantic] = matches[0]
    return result


def _remap_non_target_overrides(
    source_slot_names: Sequence[str],
    source_overrides: Sequence[Any],
    target_slot_names: Sequence[str],
) -> list:
    """按语义槽位迁移非随机化 override，目标槽位与空位保留为 None。"""
    source_names = list(source_slot_names)
    overrides = list(source_overrides)
    target_names = list(target_slot_names)
    if len(overrides) > len(source_names):
        raise ValueError("override_materials 长于源 Mesh material slots")
    if len(set(source_names)) != len(source_names):
        raise ValueError("源 Mesh material slot name 必须唯一")
    target_semantics = _resolve_semantic_slots(target_names)
    preserved_semantics = {"Hair", "Eyebrows", "Eyes", "Shoes"}
    source_semantics = _resolve_semantic_slots(source_names)
    semantic_by_source_index = {
        index: semantic for semantic, index in source_semantics.items()
    }
    target_by_semantic = {
        semantic: index
        for semantic, index in target_semantics.items()
        if semantic in preserved_semantics
    }
    result = [None] * len(target_names)
    for source_index, override in enumerate(overrides):
        semantic = semantic_by_source_index.get(source_index)
        if semantic not in preserved_semantics:
            continue
        target_index = target_by_semantic.get(semantic)
        if target_index is not None:
            result[target_index] = override
    return result


def _non_target_override_errors(
    slot_names: Sequence[str], expected: Sequence[Optional[str]], actual: Sequence[Optional[str]]
) -> list:
    if len(expected) != len(slot_names) or len(actual) != len(slot_names):
        return ["override_materials length 与 semantic slots 不一致"]
    semantic_slots = _resolve_semantic_slots(slot_names)
    errors = []
    for semantic in ("Hair", "Eyebrows", "Eyes", "Shoes"):
        index = semantic_slots[semantic]
        if expected[index] != actual[index]:
            errors.append("{} non-target material override readback 不一致".format(semantic))
    return errors


def _object_path(value: Any) -> Optional[str]:
    if value is None:
        return None
    getter = getattr(value, "get_path_name", None)
    return str(getter()) if callable(getter) else str(value)


def _package_path(value: Any) -> Optional[str]:
    path = _object_path(value)
    return path.split(".", 1)[0] if path else None


def _actor_package_path(actor: Any) -> str:
    outermost = actor.get_outermost()
    package_path = _object_path(outermost)
    if not package_path or not package_path.startswith("/Game/__ExternalActors__/"):
        raise RuntimeError("Actor External Actor package path 无效: {}".format(package_path))
    return package_path


def _anim_blueprint_asset_path(value: Any) -> Optional[str]:
    path = _object_path(value)
    if path is None:
        return None
    package = path.split(".", 1)[0]
    if package.endswith("_C"):
        package = package[:-2]
    return package


def _vector_state(value: Any) -> list:
    return [float(value.x), float(value.y), float(value.z)]


def _rotator_state(value: Any) -> list:
    return [float(value.pitch), float(value.yaw), float(value.roll)]


def _actor_transform_api_errors(actor: Any) -> list:
    required_methods = (
        "get_actor_location",
        "get_actor_rotation",
        "get_actor_scale3d",
        "set_actor_location_and_rotation",
        "set_actor_scale3d",
    )
    return [
        "Actor 缺少世界变换能力 {}".format(name)
        for name in required_methods
        if not callable(getattr(actor, name, None))
    ]


def _actor_transform_state(actor: Any) -> dict:
    errors = _actor_transform_api_errors(actor)
    if errors:
        raise RuntimeError("; ".join(errors))
    return {
        "location": _vector_state(actor.get_actor_location()),
        "rotation": _rotator_state(actor.get_actor_rotation()),
        "scale": _vector_state(actor.get_actor_scale3d()),
    }


def _current_level_path(unreal_module: Any) -> Optional[str]:
    world = unreal_module.EditorLevelLibrary.get_editor_world()
    return _package_path(world) if world is not None else None


def _appearance_safety_gate(current_level_path: Any, roster_entity_ids: Any) -> list:
    """检查外观事务必须满足的正式地图和十人 roster 门禁。"""
    errors = []
    if current_level_path != _FORMAL_MAP:
        errors.append(
            "外观事务已 fail-closed：当前 Level 必须是正式地图 {}，实际为 {}".format(
                _FORMAL_MAP, current_level_path
            )
        )
    if isinstance(roster_entity_ids, (str, bytes)) or roster_entity_ids is None:
        return errors + ["外观事务已 fail-closed：必须验证完整的十人 roster"]
    try:
        roster = list(roster_entity_ids)
    except TypeError:
        return errors + ["外观事务已 fail-closed：roster 必须是实体 ID 集合"]
    seen = set()
    duplicates = set()
    unexpected = set()
    for entity_id in roster:
        if not isinstance(entity_id, str) or entity_id not in ENTITY_IDS:
            unexpected.add(str(entity_id))
            continue
        if entity_id in seen:
            duplicates.add(entity_id)
        seen.add(entity_id)
    missing = set(ENTITY_IDS) - seen
    if missing:
        errors.append(
            "外观事务已 fail-closed：十人 roster 缺少 {}".format(
                ", ".join(sorted(missing))
            )
        )
    if duplicates:
        errors.append(
            "外观事务已 fail-closed：十人 roster 存在重复实体 {}".format(
                ", ".join(sorted(str(item) for item in duplicates))
            )
        )
    if unexpected:
        errors.append(
            "外观事务已 fail-closed：十人 roster 含意外实体 {}".format(
                ", ".join(sorted(unexpected))
            )
        )
    if len(roster) != len(ENTITY_IDS) and not missing:
        errors.append(
            "外观事务已 fail-closed：roster 必须恰好包含十名球员，实际 {}".format(
                len(roster)
            )
        )
    return errors


def _all_level_actors(unreal_module: Any) -> list:
    subsystem = unreal_module.get_editor_subsystem(unreal_module.EditorActorSubsystem)
    return list(subsystem.get_all_level_actors())


def _actor_matches(unreal_module: Any, mapping: Mapping[str, str]) -> tuple:
    actors = _all_level_actors(unreal_module)
    found = {}
    errors = []
    for entity_id in ENTITY_IDS:
        expected_label = mapping[entity_id].strip()
        matches = []
        for actor in actors:
            try:
                if actor.get_actor_label() == expected_label:
                    matches.append(actor)
            except Exception:
                continue
        if len(matches) != 1:
            errors.append(
                "Actor label {!r} 必须精确匹配一个 Actor，实际 {}".format(
                    expected_label, len(matches)
                )
            )
        else:
            found[entity_id] = matches[0]
    return found, errors


def _verify_pre_mutation_roster(
    unreal_module: Any,
    mapping: Mapping[str, str],
    expected_actors: Optional[Mapping[str, Any]] = None,
) -> tuple:
    """在外观写入前重新确认正式地图和映射 Actor 身份。"""
    errors = []
    try:
        current_level_path = _current_level_path(unreal_module)
    except Exception as exc:
        current_level_path = None
        errors.append("当前 Level 查询失败: {}: {}".format(type(exc).__name__, exc))
    actors, actor_errors = _actor_matches(unreal_module, mapping)
    errors.extend(actor_errors)
    errors.extend(_appearance_safety_gate(current_level_path, actors.keys()))
    if expected_actors is not None and set(actors) == set(ENTITY_IDS):
        for entity_id in ENTITY_IDS:
            expected_path = _object_path(expected_actors[entity_id])
            actual_path = _object_path(actors[entity_id])
            if actual_path != expected_path:
                errors.append(
                    "{} 外观写入前 Actor identity 与 preflight 不一致".format(entity_id)
                )
    return actors, list(dict.fromkeys(errors))


def _dirty_package_paths(unreal_module: Any) -> set:
    loading = unreal_module.EditorLoadingAndSavingUtils
    packages = []
    for method_name in ("get_dirty_map_packages", "get_dirty_content_packages"):
        method = getattr(loading, method_name, None)
        if not callable(method):
            raise RuntimeError("UE 缺少已验证 dirty package 查询: {}".format(method_name))
        packages.extend(method())
    return {
        path
        for path in (_package_path(package) for package in packages)
        if path
    }


def _dirty_content_package_paths(unreal_module: Any) -> set:
    method = getattr(
        unreal_module.EditorLoadingAndSavingUtils,
        "get_dirty_content_packages",
        None,
    )
    if not callable(method):
        raise RuntimeError("UE 缺少已验证 dirty content package 查询")
    return {
        path for path in (_package_path(package) for package in method()) if path
    }


def _asset_class_name(asset: Any) -> Optional[str]:
    if asset is None:
        return None
    getter = getattr(asset, "get_class", None)
    if not callable(getter):
        return None
    asset_class = getter()
    return asset_class.get_name() if asset_class is not None else None


def _baseline_component_errors(mesh: Any, anim_class: Any) -> list:
    errors = []
    if _asset_class_name(mesh) != "SkeletalMesh":
        errors.append("baseline visual asset must be SkeletalMesh")
    if _asset_class_name(anim_class) != "AnimBlueprintGeneratedClass":
        errors.append("baseline AnimBP must be AnimBlueprintGeneratedClass")
    return errors


def _load_asset(unreal_module: Any, path: str, expected_class: str) -> Any:
    asset = unreal_module.load_asset(path)
    actual_class = _asset_class_name(asset)
    if asset is None or actual_class != expected_class:
        raise RuntimeError(
            "资产 class 不匹配: {} expected={} actual={}".format(
                path, expected_class, actual_class
            )
        )
    return asset


def _material_slot_names(mesh: Any) -> list:
    materials = list(mesh.get_editor_property("materials"))
    return [str(item.get_editor_property("material_slot_name")) for item in materials]


def _semantic_slot_data(mesh: Any) -> dict:
    return _resolve_semantic_slots(_material_slot_names(mesh))


def _validate_candidate_compatibility(
    candidate: Mapping[str, Any], mesh: Any, anim_blueprint: Any
) -> list:
    errors = []
    if candidate.get("skeleton") != _SHARED_SKELETON:
        errors.append("candidate 必须使用 shared Skeleton {}".format(_SHARED_SKELETON))
    if candidate.get("anim_class") != _SHARED_ANIM_BLUEPRINT:
        errors.append("candidate 必须使用 shared AnimBP {}".format(_SHARED_ANIM_BLUEPRINT))
    mesh_skeleton = getattr(mesh, "skeleton", None)
    if _package_path(mesh_skeleton) != _SHARED_SKELETON:
        errors.append("candidate mesh Skeleton 与 shared Skeleton 不兼容")
    anim_skeleton = anim_blueprint.get_editor_property("target_skeleton")
    if _package_path(anim_skeleton) != _SHARED_SKELETON:
        errors.append("candidate AnimBP Target Skeleton 与 shared Skeleton 不兼容")
    try:
        _semantic_slot_data(mesh)
    except (AttributeError, TypeError, ValueError) as exc:
        errors.append("candidate semantic material slots 无效: {}".format(exc))
    return errors


def _baseline_material_layout_errors(mesh: Any, override_materials: Sequence[Any]) -> list:
    try:
        slot_names = _material_slot_names(mesh)
        _resolve_semantic_slots(slot_names)
    except (AttributeError, TypeError, ValueError) as exc:
        return ["baseline semantic material slots 无效: {}".format(exc)]
    if not isinstance(override_materials, (list, tuple)):
        return ["baseline override_materials 必须是数组"]
    if len(override_materials) > len(slot_names):
        return ["baseline override_materials 长于原生 material slots"]
    return []


def _component_state(actor: Any, component: Any, package_dirty: bool) -> dict:
    mesh = component.get_skeletal_mesh_asset()
    anim_class = component.get_editor_property("anim_class")
    parent = component.get_attach_parent()
    leader = component.get_editor_property("leader_pose_component")
    return _build_actor_snapshot(
        {
            "entity_id": "",
            "actor_label": actor.get_actor_label(),
            "actor_path": actor.get_path_name(),
            "package_path": _actor_package_path(actor),
            "component_name": component.get_name(),
            "skeletal_mesh": _object_path(mesh),
            "anim_class": _object_path(anim_class),
            "attach_parent": _object_path(parent),
            "actor_transform": _actor_transform_state(actor),
            "relative_transform": {
                "location": _vector_state(component.get_editor_property("relative_location")),
                "rotation": _rotator_state(component.get_editor_property("relative_rotation")),
                "scale": _vector_state(component.get_editor_property("relative_scale3d")),
            },
            "override_materials": [
                _object_path(material)
                for material in component.get_editor_property("override_materials")
            ],
            "leader_pose_component": _object_path(leader),
            "visibility": {
                "visible": bool(component.get_editor_property("visible")),
                "render_in_main_pass": bool(component.get_editor_property("render_in_main_pass")),
                "render_in_depth_pass": bool(component.get_editor_property("render_in_depth_pass")),
            },
            "package_dirty": package_dirty,
        }
    )


def _component_for_actor(unreal_module: Any, actor: Any) -> Any:
    components = actor.get_components_by_class(unreal_module.SkeletalMeshComponent)
    matches = [item for item in components if item.get_name() == _VISUAL_COMPONENT]
    if len(matches) != 1:
        raise RuntimeError(
            "{} 的 {} 组件必须恰好一个，实际 {}".format(
                actor.get_actor_label(), _VISUAL_COMPONENT, len(matches)
            )
        )
    return matches[0]


def _capture_snapshot(unreal_module: Any, entity_id: str, actor: Any, dirty_paths: set) -> dict:
    component = _component_for_actor(unreal_module, actor)
    actor_path = actor.get_path_name()
    package_path = _actor_package_path(actor)
    state = _component_state(actor, component, package_path in dirty_paths)
    state["entity_id"] = entity_id
    if state["package_dirty"]:
        raise RuntimeError("{} External Actor package 已 dirty".format(entity_id))
    if not state["skeletal_mesh"] or not state["anim_class"]:
        raise RuntimeError("{} 缺少基线 Skeletal Mesh 或 AnimBP".format(entity_id))
    for material_path in state["override_materials"]:
        if material_path is not None and not material_path.startswith("/Game/"):
            raise RuntimeError("{} 存在不能持久回滚的 transient material override".format(entity_id))
    return _build_actor_snapshot(state)


def _preload_assignments(
    unreal_module: Any,
    manifest: Mapping[str, Any],
    catalog: Mapping[str, Any],
) -> dict:
    loaded = {}
    classes = {
        "mesh": "SkeletalMesh",
        "anim_class": "AnimBlueprint",
        "skeleton": "Skeleton",
        "skin_texture": "Texture2D",
        "jersey_material": "Material",
        "shorts_material": "Material",
    }
    referenced = {}
    for field, class_name in classes.items():
        for item in catalog.get({
            "mesh": "models",
            "anim_class": "models",
            "skeleton": "models",
            "skin_texture": "skins",
            "jersey_material": "kits",
            "shorts_material": "kits",
        }[field], []):
            catalog_field = {
                "mesh": "mesh",
                "anim_class": "anim_class",
                "skeleton": "skeleton",
                "skin_texture": "texture",
                "jersey_material": "jersey_material",
                "shorts_material": "shorts_material",
            }[field]
            if isinstance(item, Mapping) and isinstance(item.get(catalog_field), str):
                path = item[catalog_field]
                previous_class = referenced.setdefault(path, class_name)
                if previous_class != class_name:
                    raise RuntimeError(
                        "同一资产路径声明了冲突 class: {} {} / {}".format(
                            path, previous_class, class_name
                        )
                    )
    referenced[_BODY_MATERIAL] = "Material"
    for entity in manifest["entities"]:
        for field, class_name in classes.items():
            path = entity[field]
            previous_class = referenced.setdefault(path, class_name)
            if previous_class != class_name:
                raise RuntimeError(
                    "同一资产路径声明了冲突 class: {} {} / {}".format(
                        path, previous_class, class_name
                    )
                )
    for path, class_name in referenced.items():
        if path not in loaded:
            loaded[path] = _load_asset(unreal_module, path, class_name)
    if _BODY_MATERIAL not in loaded:
        loaded[_BODY_MATERIAL] = _load_asset(unreal_module, _BODY_MATERIAL, "Material")

    skeleton_path = _SHARED_SKELETON
    anim_path = _SHARED_ANIM_BLUEPRINT
    loaded_skeleton = _load_asset(unreal_module, skeleton_path, "Skeleton")
    loaded_anim = _load_asset(unreal_module, anim_path, "AnimBlueprint")
    loaded[skeleton_path] = loaded_skeleton
    loaded[anim_path] = loaded_anim
    generated_class = unreal_module.BlueprintEditorLibrary.generated_class(loaded_anim)
    if generated_class is None:
        raise RuntimeError("共享 AnimBP GeneratedClass 加载失败: {}".format(anim_path))
    loaded["{}.__generated_class__".format(anim_path)] = generated_class

    for candidate in catalog["models"]:
        compatibility_errors = _validate_candidate_compatibility(
            candidate,
            loaded[candidate["mesh"]],
            loaded[candidate["anim_class"]],
        )
        if compatibility_errors:
            raise RuntimeError(
                "Player_{:03d} compatibility 失败: {}".format(
                    candidate["player_id"], "; ".join(compatibility_errors)
                )
            )
    return loaded


def _preload_baseline_restore_assets(
    unreal_module: Any, snapshots: Sequence[Mapping[str, Any]]
) -> dict:
    loaded = _load_snapshot_assets(unreal_module, snapshots)
    for path in _snapshot_asset_paths(snapshots):
        if path in loaded:
            continue
        if path.endswith("_C") or "." in path:
            asset = unreal_module.load_object(None, path)
        else:
            asset = unreal_module.load_asset(path)
        if asset is None:
            raise RuntimeError("baseline package preflight asset load failed: {}".format(path))
        loaded[path] = asset
    return loaded


def _asset_package_paths_from_loaded(loaded: Mapping[str, Any]) -> set:
    return {
        _asset_outermost_package_path(asset)
        for path, asset in loaded.items()
        if isinstance(path, str)
        and not path.endswith(".__generated_class__")
        and asset is not None
        and callable(getattr(asset, "get_outermost", None))
    }


def _ue_preflight(unreal_module: Any, manifest: Mapping[str, Any], catalog: Mapping[str, Any], mapping: Mapping[str, Any]) -> dict:
    errors = []
    try:
        current_level_path = _current_level_path(unreal_module)
    except Exception as exc:
        current_level_path = None
        errors.append("当前 Level 查询失败: {}: {}".format(type(exc).__name__, exc))
    if catalog.get("body_texture_parameter") != _BODY_PARAMETER:
        errors.append("Body texture parameter 必须是已验证的 {}".format(_BODY_PARAMETER))
    editor_assets = unreal_module.get_editor_subsystem(unreal_module.EditorAssetSubsystem)
    if not callable(getattr(editor_assets, "set_dirty_flag", None)):
        errors.append("UE EditorAssetSubsystem 缺少 set_dirty_flag，无法保证 package clean restore")
    if not callable(getattr(unreal_module, "load_class", None)):
        errors.append("UE 缺少 load_class，无法预载恢复所需的 AnimBP class")
    for method_name in ("set_skeletal_mesh_asset", "create_dynamic_material_instance", "set_material"):
        if not callable(getattr(unreal_module.SkeletalMeshComponent, method_name, None)):
            errors.append("SkeletalMeshComponent 缺少能力 {}".format(method_name))
    actors = {}
    dirty_paths = set()
    initial_asset_dirty_paths = set()
    try:
        dirty_paths = _dirty_package_paths(unreal_module)
        initial_asset_dirty_paths = _dirty_content_package_paths(unreal_module)
        actors, actor_errors = _actor_matches(unreal_module, mapping)
        errors.extend(actor_errors)
        errors.extend(_appearance_safety_gate(current_level_path, actors.keys()))
    except Exception as exc:
        errors.append("UE actor/package discovery 失败: {}: {}".format(type(exc).__name__, exc))
    if set(actors) == set(ENTITY_IDS):
        for entity_id in ENTITY_IDS:
            try:
                actor = actors[entity_id]
                errors.extend(
                    "{} {}".format(entity_id, issue)
                    for issue in _actor_transform_api_errors(actor)
                )
                package_path = _actor_package_path(actor)
                if package_path in dirty_paths:
                    errors.append("{} External Actor package 已 dirty".format(entity_id))
                component = _component_for_actor(unreal_module, actor)
                current_mesh = component.get_skeletal_mesh_asset()
                current_anim = component.get_editor_property("anim_class")
                if current_mesh is None or current_anim is None:
                    errors.append("{} MPFBVisual 缺少基线 Mesh/AnimBP".format(entity_id))
                else:
                    errors.extend(
                        "{} {}".format(entity_id, issue)
                        for issue in _baseline_component_errors(current_mesh, current_anim)
                    )
                    if _package_path(getattr(current_mesh, "skeleton", None)) != _SHARED_SKELETON:
                        errors.append("{} MPFBVisual 基线 Skeleton 不兼容".format(entity_id))
                    if _anim_blueprint_asset_path(current_anim) != _SHARED_ANIM_BLUEPRINT:
                        errors.append("{} MPFBVisual 基线 AnimBP 不是共享 AnimBP".format(entity_id))
                    errors.extend(
                        "{} {}".format(entity_id, issue)
                        for issue in _baseline_material_layout_errors(
                            current_mesh,
                            list(component.get_editor_property("override_materials")),
                        )
                    )
            except Exception as exc:
                errors.append(
                    "{} UE baseline preflight 失败: {}: {}".format(
                        entity_id, type(exc).__name__, exc
                    )
                )

    loaded = None
    baseline_loaded = None
    asset_packages = None
    if not errors:
        try:
            loaded = _preload_assignments(unreal_module, manifest, catalog)
            if set(actors) == set(ENTITY_IDS):
                baseline_snapshots = _snapshot_roster(unreal_module, actors, dirty_paths)
                baseline_loaded = _preload_baseline_restore_assets(
                    unreal_module, baseline_snapshots
                )
                asset_packages = _merge_asset_package_baselines(
                    _capture_asset_package_baseline(
                        unreal_module,
                        _select_loaded_assets(
                            _selected_appearance_asset_paths(manifest), loaded
                        ),
                        initial_asset_dirty_paths,
                    ),
                    _capture_asset_package_baseline(
                        unreal_module, baseline_loaded, initial_asset_dirty_paths
                    ),
                )
        except Exception as exc:
            errors.append("Appearance asset/package preflight 失败: {}: {}".format(type(exc).__name__, exc))
    return {
        "ok": not errors,
        "errors": list(dict.fromkeys(errors)),
        "actors": actors if "actors" in locals() else {},
        "dirty_paths": dirty_paths if "dirty_paths" in locals() else set(),
        "loaded": loaded,
        "baseline_loaded": baseline_loaded,
        "asset_packages": asset_packages,
        "initial_asset_dirty_paths": initial_asset_dirty_paths,
    }


def _write_local_json(path: Any, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, str(target))
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return target


def _load_snapshot_assets(unreal_module: Any, snapshots: Sequence[Mapping[str, Any]]) -> dict:
    assets = {}
    for snapshot in snapshots:
        mesh_path = snapshot["skeletal_mesh"]
        mesh_asset_path = _package_path(mesh_path)
        if mesh_path not in assets:
            assets[mesh_path] = unreal_module.load_asset(mesh_asset_path)
        if assets[mesh_path] is None:
            raise RuntimeError("rollback Mesh 加载失败: {}".format(mesh_path))
        anim_path = snapshot["anim_class"]
        if anim_path not in assets:
            assets[anim_path] = unreal_module.load_class(None, anim_path)
        if assets[anim_path] is None:
            raise RuntimeError("rollback AnimBP class 加载失败: {}".format(anim_path))
        for material_path in snapshot["override_materials"]:
            if material_path is not None and material_path not in assets:
                assets[material_path] = unreal_module.load_asset(_package_path(material_path))
                if assets[material_path] is None:
                    raise RuntimeError("rollback material override 加载失败: {}".format(material_path))
    return assets


def _preload_snapshot_paths(unreal_module: Any, snapshots: Sequence[Mapping[str, Any]]) -> dict:
    assets = _load_snapshot_assets(unreal_module, snapshots)
    object_paths = set()
    for actor in _all_level_actors(unreal_module):
        object_paths.add(actor.get_path_name())
        for component in actor.get_components_by_class(unreal_module.ActorComponent):
            object_paths.add(component.get_path_name())
    for snapshot in snapshots:
        for field in ("attach_parent", "leader_pose_component"):
            path = snapshot[field]
            if path is not None and path not in object_paths:
                raise RuntimeError(
                    "baseline restore component reference not in loaded map: {}".format(path)
                )
    return assets


def _set_component_transform(unreal_module: Any, component: Any, transform: Mapping[str, Any]) -> None:
    component.set_editor_property("relative_location", unreal_module.Vector(*transform["location"]))
    component.set_editor_property(
        "relative_rotation",
        unreal_module.Rotator(
            pitch=transform["rotation"][0],
            yaw=transform["rotation"][1],
            roll=transform["rotation"][2],
        ),
    )
    component.set_editor_property("relative_scale3d", unreal_module.Vector(*transform["scale"]))


def _restore_component(
    unreal_module: Any,
    actor: Any,
    snapshot: Mapping[str, Any],
    assets: Mapping[str, Any],
    object_by_path: Mapping[str, Any],
) -> None:
    _set_actor_transform(unreal_module, actor, snapshot["actor_transform"])
    component = _component_for_actor(unreal_module, actor)
    component.set_skeletal_mesh_asset(assets[snapshot["skeletal_mesh"]])
    component.set_editor_property("anim_class", assets[snapshot["anim_class"]])

    parent_path = snapshot["attach_parent"]
    current_parent = component.get_attach_parent()
    current_parent_path = _object_path(current_parent)
    if current_parent_path != parent_path:
        if parent_path is None:
            detach = getattr(component, "detach_from_component", None)
            if not callable(detach):
                raise RuntimeError("UE 缺少 detach_from_component，无法恢复 attachment")
            detach(unreal_module.DetachmentRule.KEEP_RELATIVE, True)
        else:
            parent = object_by_path.get(parent_path)
            if parent is None:
                raise RuntimeError("rollback attachment parent 加载失败: {}".format(parent_path))
            component.attach_to_component(
                parent,
                unreal_module.AttachmentRule.KEEP_RELATIVE,
                unreal_module.AttachmentRule.KEEP_RELATIVE,
                unreal_module.AttachmentRule.KEEP_RELATIVE,
                True,
            )
    _set_component_transform(unreal_module, component, snapshot["relative_transform"])

    overrides = [assets[path] if path is not None else None for path in snapshot["override_materials"]]
    component.set_editor_property("override_materials", overrides)
    leader_path = snapshot["leader_pose_component"]
    if leader_path is None:
        leader = None
    else:
        leader = object_by_path.get(leader_path)
        if leader is None:
            raise RuntimeError("rollback leader-pose component 加载失败: {}".format(leader_path))
    component.set_editor_property("leader_pose_component", leader)
    for field, value in snapshot["visibility"].items():
        component.set_editor_property(field, value)


def _set_actor_transform(
    unreal_module: Any, actor: Any, transform: Mapping[str, Any]
) -> None:
    actor.set_actor_location_and_rotation(
        unreal_module.Vector(*transform["location"]),
        unreal_module.Rotator(
            pitch=transform["rotation"][0],
            yaw=transform["rotation"][1],
            roll=transform["rotation"][2],
        ),
        False,
        False,
    )
    actor.set_actor_scale3d(unreal_module.Vector(*transform["scale"]))


def _set_package_dirty(unreal_module: Any, actor: Any, dirty: bool) -> None:
    editor_assets = unreal_module.get_editor_subsystem(unreal_module.EditorAssetSubsystem)
    if editor_assets.set_dirty_flag(actor, dirty) is False:
        raise RuntimeError("无法恢复 External Actor package dirty flag")


def _override_paths(component: Any) -> list:
    return [
        _object_path(material)
        for material in component.get_editor_property("override_materials")
    ]


def _dirty_reset_candidates(
    entity_ids: Sequence[str],
    component_restored: Mapping[str, bool],
    readback_verified: Mapping[str, bool],
) -> set:
    """只允许组件恢复和精确 readback 均成功的 Actor 清理 dirty flag。"""
    return {
        entity_id
        for entity_id in entity_ids
        if component_restored.get(entity_id) is True
        and readback_verified.get(entity_id) is True
    }


def _snapshot_roster(unreal_module: Any, actors: Mapping[str, Any], dirty_paths: set) -> list:
    snapshots = [
        _capture_snapshot(unreal_module, entity_id, actors[entity_id], dirty_paths)
        for entity_id in ENTITY_IDS
    ]
    errors = _validate_actor_snapshots(snapshots)
    if errors:
        raise RuntimeError("十人 baseline snapshot 不完整: {}".format("; ".join(errors)))
    return snapshots


def _restore_roster(
    unreal_module: Any,
    snapshots: Sequence[Mapping[str, Any]],
    asset_package_baseline: Sequence[Mapping[str, Any]],
) -> list:
    errors = []
    errors.extend(
        _appearance_safety_gate(
            _current_level_path(unreal_module),
            [snapshot.get("entity_id") for snapshot in snapshots],
        )
    )
    if errors:
        return errors
    all_actors = _all_level_actors(unreal_module)
    actors_by_path = {actor.get_path_name(): actor for actor in all_actors}
    object_by_path = dict(actors_by_path)
    for actor in all_actors:
        try:
            for component in actor.get_components_by_class(unreal_module.ActorComponent):
                object_by_path[component.get_path_name()] = component
        except Exception:
            continue
    resolved = {}
    for snapshot in snapshots:
        actor = actors_by_path.get(snapshot["actor_path"])
        if actor is None or actor.get_actor_label() != snapshot["actor_label"]:
            errors.append("{} restore actor identity 无法精确解析".format(snapshot["entity_id"]))
        else:
            resolved[snapshot["entity_id"]] = actor
    if errors:
        return errors
    try:
        assets = _load_snapshot_assets(unreal_module, snapshots)
    except Exception as exc:
        errors.append("restore assets pre-load 失败: {}: {}".format(type(exc).__name__, exc))
        return errors

    component_restored = {snapshot["entity_id"]: False for snapshot in snapshots}
    readback_verified = {snapshot["entity_id"]: False for snapshot in snapshots}
    for snapshot in snapshots:
        actor = resolved.get(snapshot["entity_id"])
        if actor is None:
            continue
        try:
            _restore_component(unreal_module, actor, snapshot, assets, object_by_path)
            component_restored[snapshot["entity_id"]] = True
        except Exception as exc:
            errors.append("{} component restore 失败: {}: {}".format(
                snapshot["entity_id"], type(exc).__name__, exc
            ))

    for snapshot in snapshots:
        actor = resolved.get(snapshot["entity_id"])
        if actor is None:
            continue
        try:
            component = _component_for_actor(unreal_module, actor)
            current = _component_state(actor, component, snapshot["package_dirty"])
            current["entity_id"] = snapshot["entity_id"]
            if not _snapshot_matches_state(snapshot, current):
                errors.append("{} restore readback 与 snapshot 不一致".format(snapshot["entity_id"]))
            else:
                readback_verified[snapshot["entity_id"]] = True
        except Exception as exc:
            errors.append("{} restore readback 失败: {}: {}".format(
                snapshot["entity_id"], type(exc).__name__, exc
            ))

    # Mesh restore may dirty shared material dependencies. Only baseline-clean
    # touched content packages are eligible for the verified noninteractive discard.
    package_baseline = list(asset_package_baseline)
    baseline_errors = _validate_asset_package_baseline(package_baseline)
    if baseline_errors:
        errors.extend("asset package baseline invalid: {}".format(item) for item in baseline_errors)
    else:
        try:
            dirty_assets = _dirty_package_paths(unreal_module)
            candidates = _asset_package_reload_candidates(package_baseline, dirty_assets)
            _reload_asset_packages(unreal_module, candidates)
        except Exception as exc:
            errors.append("asset package discard/reload 失败: {}: {}".format(type(exc).__name__, exc))

        # Reload can invalidate component-side references, so prove the complete
        # snapshot again after discard before changing External Actor dirty state.
        for snapshot in snapshots:
            actor = resolved.get(snapshot["entity_id"])
            if actor is None:
                readback_verified[snapshot["entity_id"]] = False
                continue
            try:
                component = _component_for_actor(unreal_module, actor)
                current = _component_state(actor, component, snapshot["package_dirty"])
                current["entity_id"] = snapshot["entity_id"]
                component_ok = _snapshot_matches_state(snapshot, current)
                readback_verified[snapshot["entity_id"]] = (
                    component_restored[snapshot["entity_id"]] and component_ok
                )
                if not component_ok:
                    errors.append(
                        "{} post-discard restore readback 与 snapshot 不一致".format(
                            snapshot["entity_id"]
                        )
                    )
            except Exception as exc:
                readback_verified[snapshot["entity_id"]] = False
                errors.append("{} post-discard readback 失败: {}: {}".format(
                    snapshot["entity_id"], type(exc).__name__, exc
                ))

        try:
            dirty_assets = _dirty_package_paths(unreal_module)
            for record in package_baseline:
                if (record["package_path"] in dirty_assets) != record["package_dirty"]:
                    errors.append(
                        "asset package dirty state 与 baseline 不一致: {}".format(
                            record["package_path"]
                        )
                    )
        except Exception as exc:
            errors.append("asset package dirty 最终验证失败: {}: {}".format(type(exc).__name__, exc))

    dirty_paths = _dirty_package_paths(unreal_module)
    dirty_reset_candidates = _dirty_reset_candidates(
        ENTITY_IDS, component_restored, readback_verified
    )
    for snapshot in snapshots:
        actor = resolved.get(snapshot["entity_id"])
        if actor is None:
            continue
        entity_id = snapshot["entity_id"]
        if entity_id not in dirty_reset_candidates:
            errors.append(
                "{} package dirty flag 保留：component restore/readback 未全部验证成功".format(
                    entity_id
                )
            )
            continue
        try:
            package_path = snapshot["package_path"]
            dirty_now = package_path in dirty_paths
            if dirty_now != snapshot["package_dirty"]:
                _set_package_dirty(unreal_module, actor, snapshot["package_dirty"])
        except Exception as exc:
            errors.append("{} package dirty restore 失败: {}: {}".format(
                snapshot["entity_id"], type(exc).__name__, exc
            ))
    try:
        dirty_paths = _dirty_package_paths(unreal_module)
        for snapshot in snapshots:
            package_path = snapshot["package_path"]
            if (package_path in dirty_paths) != snapshot["package_dirty"]:
                errors.append("{} package dirty readback 与 baseline 不一致".format(snapshot["entity_id"]))
    except Exception as exc:
        errors.append("package dirty 最终验证失败: {}: {}".format(type(exc).__name__, exc))
    return errors


def _attempt_restore_roster(
    unreal_module: Any,
    snapshots: Sequence[Mapping[str, Any]],
    asset_package_baseline: Sequence[Mapping[str, Any]],
) -> list:
    try:
        return _restore_roster(unreal_module, snapshots, asset_package_baseline)
    except Exception as exc:
        return ["restore 执行异常: {}: {}".format(type(exc).__name__, exc)]


def _apply_entity(unreal_module: Any, actor: Any, entity: Mapping[str, Any], loaded: Mapping[str, Any]) -> dict:
    component = _component_for_actor(unreal_module, actor)
    source_mesh = component.get_skeletal_mesh_asset()
    source_slot_names = _material_slot_names(source_mesh)
    source_overrides = list(component.get_editor_property("override_materials"))
    mesh = loaded[entity["mesh"]]
    generated_class = loaded["{}.__generated_class__".format(entity["anim_class"])]
    target_slot_names = _material_slot_names(mesh)
    slot_map = _semantic_slot_data(mesh)
    retained_overrides = _remap_non_target_overrides(
        source_slot_names, source_overrides, target_slot_names
    )

    component.set_skeletal_mesh_asset(mesh)
    if _package_path(component.get_skeletal_mesh_asset()) != entity["mesh"]:
        raise RuntimeError("Skeletal Mesh assignment readback 不一致")
    component.set_editor_property("anim_class", generated_class)
    if _anim_blueprint_asset_path(component.get_editor_property("anim_class")) != entity["anim_class"]:
        raise RuntimeError("AnimBP assignment readback 不一致")
    component.set_editor_property("override_materials", retained_overrides)
    if len(_override_paths(component)) != len(target_slot_names):
        raise RuntimeError("非目标 material override 数组长度 readback 不一致")
    remap_errors = _non_target_override_errors(
        target_slot_names,
        retained_overrides,
        _override_paths(component),
    )
    if remap_errors:
        raise RuntimeError("; ".join(remap_errors))
    body_mid = component.create_dynamic_material_instance(
        slot_map["Body"], loaded[_BODY_MATERIAL]
    )
    if body_mid is None:
        raise RuntimeError("Body transient MID 创建失败")
    if component.get_material(slot_map["Body"]) is not body_mid:
        raise RuntimeError("Body MID slot assignment readback 不一致")
    body_mid.set_texture_parameter_value(_BODY_PARAMETER, loaded[entity["skin_texture"]])
    texture_getter = getattr(body_mid, "get_texture_parameter_value", None)
    if not callable(texture_getter):
        raise RuntimeError("UE MID 缺少 get_texture_parameter_value readback")
    if _package_path(texture_getter(_BODY_PARAMETER)) != entity["skin_texture"]:
        raise RuntimeError("SkinTexture assignment readback 不一致")
    component.set_material(slot_map["Jersey"], loaded[entity["jersey_material"]])
    if _package_path(component.get_material(slot_map["Jersey"])) != entity["jersey_material"]:
        raise RuntimeError("Jersey material assignment readback 不一致")
    component.set_material(slot_map["Shorts"], loaded[entity["shorts_material"]])
    if _package_path(component.get_material(slot_map["Shorts"])) != entity["shorts_material"]:
        raise RuntimeError("Shorts material assignment readback 不一致")
    actual_overrides = _override_paths(component)
    remap_errors = _non_target_override_errors(
        target_slot_names, retained_overrides, actual_overrides
    )
    if remap_errors:
        raise RuntimeError("; ".join(remap_errors))

    effective = {
        semantic: component.get_material(index)
        for semantic, index in slot_map.items()
    }
    if _object_path(effective["Body"]) != _object_path(body_mid):
        raise RuntimeError("Body MID slot readback 不一致")
    for semantic, field in (("Jersey", "jersey_material"), ("Shorts", "shorts_material")):
        if _package_path(effective[semantic]) != entity[field]:
            raise RuntimeError("{} material assignment readback 不一致".format(semantic))
    return {
        "apply_status": "applied",
        "restore_status": "not_started",
        "slots": {
            "body_texture_parameter": _BODY_PARAMETER,
            "Body": entity["skin_texture"],
            "Jersey": entity["jersey_material"],
            "Shorts": entity["shorts_material"],
        },
    }


def create_application_report(manifest: Mapping[str, Any], mapping: Mapping[str, Any]) -> dict:
    """建立不含本机路径的 application report 初始状态。"""
    mapping_errors = _validate_mapping(mapping)
    if mapping_errors:
        raise ValueError("; ".join(mapping_errors))
    entities = {
        entity["entity_id"]: entity
        for entity in manifest.get("entities", [])
        if isinstance(entity, Mapping) and entity.get("entity_id") in ENTITY_IDS
    }
    report = {
        "schema": APPLICATION_REPORT_SCHEMA,
        "version": APPLICATION_REPORT_VERSION,
        "state": "not_started",
        "manifest_fingerprint": manifest.get("fingerprint"),
        "trajectory_sha256": manifest.get("trajectory_sha256"),
        "episode_id": manifest.get("episode_id"),
        "catalog_id": manifest.get("catalog_id"),
        "catalog_fingerprint": manifest.get("catalog_fingerprint"),
        "actors": [
            {
                "entity_id": entity_id,
                "actor_label": mapping[entity_id].strip(),
                "apply_status": "not_started",
                "restore_status": "not_started",
                "assets": {
                    field: entities[entity_id].get(field)
                    for field in _ENTITY_ASSET_FIELDS
                    if entity_id in entities
                },
                "slots": {
                    "body_texture_parameter": manifest.get("body_texture_parameter"),
                    "Body": entities[entity_id].get("skin_texture"),
                    "Jersey": entities[entity_id].get("jersey_material"),
                    "Shorts": entities[entity_id].get("shorts_material"),
                } if entity_id in entities else {},
            }
            for entity_id in ENTITY_IDS
        ],
        "mrq_jobs": [],
        "mrq_pending": False,
        "errors": [],
        "warnings": [],
    }
    _validate_report(report)
    return report


def _validate_report(report: Any) -> None:
    if not isinstance(report, Mapping):
        raise ValueError("application report 必须是对象")
    if report.get("schema") != APPLICATION_REPORT_SCHEMA:
        raise ValueError("application report schema 无效")
    if type(report.get("version")) is not int or report.get("version") != APPLICATION_REPORT_VERSION:
        raise ValueError("application report version 无效")
    if report.get("state") not in REPORT_STATES:
        raise ValueError("application report state 无效")
    for field in ("manifest_fingerprint", "trajectory_sha256", "catalog_fingerprint"):
        value = report.get(field)
        if not isinstance(value, str) or not _SHA256_RE.match(value):
            raise ValueError("application report {} 无效".format(field))
    if "transaction_id" in report and (
        not isinstance(report["transaction_id"], str)
        or not _SHA256_RE.match(report["transaction_id"])
    ):
        raise ValueError("application report transaction_id 无效")
    actors = report.get("actors")
    if not isinstance(actors, list) or len(actors) != len(ENTITY_IDS):
        raise ValueError("application report actors 必须恰好包含十名球员")
    ids = [actor.get("entity_id") for actor in actors if isinstance(actor, Mapping)]
    if (
        len(ids) != len(ENTITY_IDS)
        or any(not isinstance(entity_id, str) for entity_id in ids)
        or set(ids) != set(ENTITY_IDS)
    ):
        raise ValueError("application report actor entity 集合无效")
    if not isinstance(report.get("mrq_jobs"), list):
        raise ValueError("application report mrq_jobs 必须是列表")
    if type(report.get("mrq_pending")) is not bool:
        raise ValueError("application report mrq_pending 必须是 bool")
    for field in ("errors", "warnings"):
        if not isinstance(report.get(field), list) or any(not isinstance(item, str) for item in report[field]):
            raise ValueError("application report {} 必须是字符串列表".format(field))
    if report["state"] == "restored":
        if any(
            not isinstance(actor, Mapping) or actor.get("restore_status") != "restored"
            for actor in actors
        ):
            raise ValueError("restored 状态要求十名 actor 的 restore_status 均为 restored")
        if report["mrq_pending"]:
            raise ValueError("restored 状态要求 mrq_pending 为 false")
        if any(not _is_terminal_job(job) for job in report["mrq_jobs"]):
            raise ValueError("restored 状态要求每个 MRQ job 均为 terminal")
    if "render_result" in report:
        _validate_render_result(report["render_result"])


def _is_terminal_job(job: Any) -> bool:
    if not isinstance(job, Mapping):
        return False
    return job.get("status") in _TERMINAL_MRQ_STATUSES


def _is_json_value(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return value == value and value not in (float("inf"), float("-inf"))
    if isinstance(value, list):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, Mapping):
        return all(
            isinstance(key, str) and _is_json_value(item)
            for key, item in value.items()
        )
    return False


def _validate_render_result(render_result: Any) -> None:
    if not isinstance(render_result, Mapping):
        raise ValueError("render_result 必须是 JSON 对象")
    if not _is_json_value(render_result):
        raise ValueError("render_result 必须只包含 JSON 类型")
    status = render_result.get("status")
    if not isinstance(status, str) or status not in _TERMINAL_RENDER_STATUSES:
        raise ValueError("render_result.status 必须是合法终态")
    try:
        json.dumps(render_result, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("render_result 必须是 portable JSON 对象") from exc
    _reject_local_paths(render_result)


def transition_application_report(
    report: Mapping[str, Any],
    state: str,
    *,
    actor_updates: Optional[Mapping[str, Mapping[str, Any]]] = None,
    mrq_jobs: Optional[Sequence[Mapping[str, Any]]] = None,
    mrq_pending: Optional[bool] = None,
    render_result: Optional[Mapping[str, Any]] = None,
    error: Optional[str] = None,
    warning: Optional[str] = None,
    _restore_verified: Any = None,
) -> dict:
    """按允许的状态边更新 report，并拒绝不完整的 restored 状态。"""
    _validate_report(report)
    current = report["state"]
    if state not in _ALLOWED_TRANSITIONS[current] or (
        current == "restore_failed"
        and state == "restored"
        and _restore_verified is not _RESTORE_VERIFIED_TOKEN
    ):
        raise ValueError("application report transition 不允许: {} -> {}".format(current, state))
    updated = deepcopy(dict(report))
    updated["state"] = state

    actor_updates = actor_updates or {}
    if not isinstance(actor_updates, Mapping):
        raise ValueError("actor_updates 必须是对象")
    actors_by_id = {actor["entity_id"]: actor for actor in updated["actors"]}
    for entity_id, changes in actor_updates.items():
        if entity_id not in actors_by_id or not isinstance(changes, Mapping):
            raise ValueError("actor_updates 含未知实体或非法内容")
        actors_by_id[entity_id].update(deepcopy(dict(changes)))
    if mrq_jobs is not None:
        updated["mrq_jobs"] = deepcopy(list(mrq_jobs))
    if render_result is not None:
        updated["render_result"] = deepcopy(dict(render_result))
    if mrq_pending is not None:
        if type(mrq_pending) is not bool:
            raise ValueError("mrq_pending 必须是 bool")
        updated["mrq_pending"] = mrq_pending
    if error:
        updated["errors"].append(str(error))
    if warning:
        updated["warnings"].append(str(warning))

    if state == "applied":
        if any(actor.get("apply_status") != "applied" for actor in updated["actors"]):
            raise ValueError("applied 状态要求十名 actor 均已应用")
    if state == "restored":
        if any(actor.get("restore_status") != "restored" for actor in updated["actors"]):
            raise ValueError("restored 状态要求每个 actor 都已恢复")
        if updated["mrq_pending"]:
            raise ValueError("MRQ job 仍在 pending，不能标记 restored")
        if any(not _is_terminal_job(job) for job in updated["mrq_jobs"]):
            raise ValueError("MRQ job 尚未全部进入 terminal 状态，不能标记 restored")
    _validate_report(updated)
    return updated


def serialize_application_report(report: Mapping[str, Any]) -> str:
    """序列化便携报告，并剔除任何机器本地快照/报告路径。"""
    _validate_report(report)
    portable = {key: deepcopy(report[key]) for key in _REPORT_FIELDS if key in report}
    for actor in portable["actors"]:
        actor.pop("rollback_path", None)
        actor.pop("report_path", None)
        if "baseline_assets_preloaded" in actor:
            del actor["baseline_assets_preloaded"]
    payload = json.dumps(portable, ensure_ascii=False, sort_keys=True, indent=2)
    _reject_local_paths(portable)
    return payload + "\n"


def _reject_local_paths(value: Any, key: str = "") -> None:
    if isinstance(value, Mapping):
        for child_key, child in value.items():
            lowered = str(child_key).lower()
            if lowered in {"rollback_path", "report_path", "snapshot_path", "local_path"}:
                raise ValueError("application report 不得包含本机文件路径")
            _reject_local_paths(child, str(child_key))
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_local_paths(child, key)
    elif isinstance(value, str):
        if "path" in key.lower() and value and not value.startswith("/Game/"):
            raise ValueError("application report 不得包含本机文件路径")
        portable_text = re.sub(r"/Game/[A-Za-z0-9_./-]+", "UE_ASSET", value)
        if re.search(r"[A-Za-z]:[/\\]|\\\\|(?<![A-Za-z0-9_])/[^/\s][^\s,;\"']*", portable_text):
            raise ValueError("application report 不得包含本机文件路径")


def write_application_report(path: Any, report: Mapping[str, Any]) -> Path:
    """在目标目录内原子写入 application report。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = serialize_application_report(report).encode("utf-8")
    descriptor, temp_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, str(target))
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise
    return target


def read_application_report(path: Any) -> dict:
    """读取并验证本机 application report。"""
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    _validate_report(report)
    return report


def apply_episode_appearance(
    manifest: dict,
    catalog: dict,
    mapping: dict,
    rollback_path: str,
    report_path: str,
) -> dict:
    """预检后以本地快照事务性应用十名球员外观，不保存 External Actor。"""
    result = _preflight_contract(manifest, catalog, mapping)
    report = None
    if isinstance(manifest, Mapping) and not _validate_mapping(mapping):
        try:
            report = create_application_report(manifest, mapping)
        except (TypeError, ValueError, KeyError):
            report = None
    if report is None:
        return {
            "ok": False,
            "operation": "apply",
            "errors": result["errors"] or ["无法建立 application report"],
            "warnings": [],
            "actors": result["actors"],
            "transaction_id": None,
            "state": "preflight_failed",
            "rollback_snapshot_written": False,
        }

    try:
        if Path(rollback_path).exists():
            return {
                "ok": False,
                "operation": "apply",
                "errors": ["拒绝覆盖已存在的首次 rollback 快照"],
                "warnings": [],
                "actors": result["actors"],
                "transaction_id": None,
                "state": "preflight_failed",
                "rollback_snapshot_written": False,
            }
        if Path(report_path).exists():
            return {
                "ok": False,
                "operation": "apply",
                "errors": ["拒绝覆盖已存在的 application report"],
                "warnings": [],
                "actors": result["actors"],
                "transaction_id": None,
                "state": "preflight_failed",
                "rollback_snapshot_written": False,
            }
    except (OSError, RuntimeError) as exc:
        report = transition_application_report(
            report, "preflight_failed", error=str(exc)
        )
        write_application_report(report_path, report)
        return {
            "ok": False,
            "operation": "apply",
            "errors": [str(exc)],
            "warnings": [],
            "actors": result["actors"],
            "transaction_id": None,
            "state": report["state"],
            "rollback_snapshot_written": False,
        }

    ue = None
    snapshots = None
    snapshot_written = False
    mutation_started = False
    restore_attempted = False
    try:
        if result["errors"]:
            report = transition_application_report(
                report,
                "preflight_failed",
                error="; ".join(result["errors"]),
            )
            write_application_report(report_path, report)
            return {
                "ok": False,
                "operation": "apply",
                "errors": result["errors"],
                "warnings": [],
                "actors": result["actors"],
                "transaction_id": None,
                "state": report["state"],
                "rollback_snapshot_written": False,
            }

        ue = _load_unreal()
        ue_result = _ue_preflight(ue, manifest, catalog, mapping)
        if not ue_result["ok"]:
            report = transition_application_report(
                report,
                "preflight_failed",
                error="; ".join(ue_result["errors"]),
            )
            write_application_report(report_path, report)
            return {
                "ok": False,
                "operation": "apply",
                "errors": ue_result["errors"],
                "warnings": [],
                "actors": result["actors"],
                "transaction_id": None,
                "state": report["state"],
                "rollback_snapshot_written": False,
            }

        snapshots = _snapshot_roster(ue, ue_result["actors"], ue_result["dirty_paths"])
        baseline_assets = _preload_snapshot_paths(ue, snapshots)
        asset_packages = _merge_asset_package_baselines(
            ue_result["asset_packages"],
            _capture_asset_package_baseline(ue, baseline_assets),
        )
        _, safety_errors = _verify_pre_mutation_roster(
            ue, mapping, ue_result["actors"]
        )
        if safety_errors:
            raise RuntimeError("外观写入前安全门禁失败: {}".format("; ".join(safety_errors)))
        rollback_payload = _build_rollback_payload(
            manifest["fingerprint"], snapshots, asset_packages
        )
        transaction_id = rollback_payload["transaction_id"]
        _write_local_json(rollback_path, rollback_payload)
        snapshot_written = True
        report["transaction_id"] = transaction_id
        _validate_report(report)
        write_application_report(report_path, report)

        entities = {entity["entity_id"]: entity for entity in manifest["entities"]}
        actor_updates = {}
        try:
            mutation_started = True
            for entity_id in ENTITY_IDS:
                actor_updates[entity_id] = _apply_entity(
                    ue,
                    ue_result["actors"][entity_id],
                    entities[entity_id],
                    ue_result["loaded"],
                )
        except Exception as apply_exc:
            restore_attempted = True
            rollback_errors = _attempt_restore_roster(ue, snapshots, asset_packages)
            if rollback_errors:
                report = transition_application_report(
                    report,
                    "restore_failed",
                    error="apply 失败: {}。rollback 失败: {}".format(
                        apply_exc, "; ".join(rollback_errors)
                    ),
                    actor_updates={
                        entity_id: {"restore_status": "failed"}
                        for entity_id in ENTITY_IDS
                    },
                )
            else:
                report = transition_application_report(
                    report,
                    "restored",
                    error="apply 失败，十人 baseline 已恢复: {}".format(apply_exc),
                    actor_updates={
                        entity_id: {
                            "apply_status": "failed",
                            "restore_status": "restored",
                        }
                        for entity_id in ENTITY_IDS
                    },
                    mrq_pending=False,
                )
            write_application_report(report_path, report)
            return {
                "ok": False,
                "operation": "apply",
                "errors": ["外观应用失败: {}".format(apply_exc)] + rollback_errors,
                "warnings": [],
                "actors": result["actors"],
                "transaction_id": transaction_id,
                "state": report["state"],
                "rollback_snapshot_written": snapshot_written,
            }

        report = transition_application_report(report, "applied", actor_updates=actor_updates)
        write_application_report(report_path, report)
        return {
            "ok": True,
            "operation": "apply",
            "errors": [],
            "warnings": [],
            "actors": result["actors"],
            "transaction_id": transaction_id,
            "state": report["state"],
            "rollback_snapshot_written": snapshot_written,
        }
    except Exception as exc:
        errors = ["UE 外观事务失败: {}: {}".format(type(exc).__name__, exc)]
        if snapshots is not None and ue is not None and mutation_started and not restore_attempted:
            restore_attempted = True
            rollback_errors = _attempt_restore_roster(
                ue, snapshots, rollback_payload["asset_packages"]
            )
            errors.extend(rollback_errors)
            try:
                if rollback_errors:
                    failed = transition_application_report(
                        report,
                        "restore_failed",
                        error="; ".join(errors),
                        actor_updates={
                            entity_id: {"restore_status": "failed"}
                            for entity_id in ENTITY_IDS
                        },
                    )
                else:
                    failed = transition_application_report(
                        report,
                        "restored",
                        error="事务异常，baseline 已恢复",
                        actor_updates={
                            entity_id: {"restore_status": "restored"}
                            for entity_id in ENTITY_IDS
                        },
                        mrq_pending=False,
                    )
                write_application_report(report_path, failed)
                report = failed
            except Exception as report_exc:
                errors.append("无法记录事务回滚结果: {}".format(report_exc))
        elif report is not None and report["state"] == "not_started":
            report = transition_application_report(
                report, "preflight_failed", error="; ".join(errors)
            )
            try:
                write_application_report(report_path, report)
            except Exception as report_exc:
                errors.append("无法写入 preflight_failed report: {}".format(report_exc))
        return {
            "ok": False,
            "operation": "apply",
            "errors": errors,
            "warnings": [],
            "actors": result["actors"],
            "transaction_id": None,
            "state": report["state"] if report else "preflight_failed",
            "rollback_snapshot_written": snapshot_written,
        }


def restore_episode_appearance(
    rollback_path: str,
    report_path: str,
    final_status: str,
    render_result: Optional[Mapping[str, Any]] = None,
) -> dict:
    """按首次快照恢复十名球员，验证组件状态并保留 package clean state。"""
    errors = []
    try:
        report = read_application_report(report_path)
    except (OSError, ValueError, TypeError) as exc:
        return {
            "ok": False,
            "operation": "restore",
            "errors": ["application report 无法读取: {}".format(exc)],
            "warnings": [],
            "state": "restore_failed",
        }
    retrying_failed_restore = report["state"] == "restore_failed"
    if report["state"] in ("preflight_failed", "restored"):
        errors.append("当前 report 状态不允许执行 restore: {}".format(report["state"]))
    elif final_status not in ("success", "failed", "cancelled", "canceled", "no_render"):
        errors.append("final_status 不是已结束的处理/MRQ 状态")
    else:
        if render_result is not None:
            try:
                _validate_render_result(render_result)
                report["render_result"] = deepcopy(dict(render_result))
                report["mrq_jobs"] = deepcopy(list(render_result.get("jobs") or []))
            except (TypeError, ValueError) as exc:
                errors.append("terminal render_result 无效: {}".format(exc))
        if not retrying_failed_restore or not errors:
            try:
                if render_result is not None and not errors:
                    write_application_report(report_path, report)
                snapshot_payload = json.loads(Path(rollback_path).read_text(encoding="utf-8"))
                if (
                    not isinstance(snapshot_payload, Mapping)
                    or snapshot_payload.get("schema") != _SNAPSHOT_SCHEMA
                    or type(snapshot_payload.get("version")) is not int
                    or snapshot_payload.get("version") != _SNAPSHOT_VERSION
                    or snapshot_payload.get("map") != _FORMAL_MAP
                    or not isinstance(snapshot_payload.get("actors"), list)
                    or not isinstance(snapshot_payload.get("asset_packages"), list)
                ):
                    raise ValueError("rollback snapshot schema/map 无效")
                snapshots = snapshot_payload["actors"]
                snapshot_errors = _validate_rollback_payload(snapshot_payload)
                if snapshot_errors:
                    raise ValueError("; ".join(snapshot_errors))
                if snapshot_payload.get("transaction_id") != report.get("transaction_id"):
                    raise ValueError("rollback transaction_id 与 application report 不一致")
                manifest_fingerprint = snapshot_payload.get("manifest_fingerprint")
                if not isinstance(manifest_fingerprint, str) or not _SHA256_RE.match(manifest_fingerprint):
                    raise ValueError("rollback manifest_fingerprint 无效")
                if manifest_fingerprint != report.get("manifest_fingerprint"):
                    raise ValueError("rollback manifest_fingerprint 与 application report 不一致")
                ue = _load_unreal()
                safety_errors = _appearance_safety_gate(
                    _current_level_path(ue),
                    [snapshot.get("entity_id") for snapshot in snapshots],
                )
                if safety_errors:
                    raise RuntimeError("; ".join(safety_errors))
                restore_errors = _restore_roster(
                    ue, snapshots, snapshot_payload["asset_packages"]
                )
                if restore_errors:
                    errors.extend(restore_errors)
                else:
                    updates = {
                        entity_id: {"restore_status": "restored"}
                        for entity_id in ENTITY_IDS
                    }
                    restored = transition_application_report(
                        report,
                        "restored",
                        actor_updates=updates,
                        mrq_pending=False,
                        _restore_verified=(
                            _RESTORE_VERIFIED_TOKEN if retrying_failed_restore else None
                        ),
                    )
                    restored["transaction_id"] = snapshot_payload["transaction_id"]
                    write_application_report(report_path, restored)
                    return {
                        "ok": True,
                        "operation": "restore",
                        "errors": [],
                        "warnings": [],
                        "state": restored["state"],
                    }
            except Exception as exc:
                errors.append("UE restore 失败: {}: {}".format(type(exc).__name__, exc))
    if retrying_failed_restore:
        try:
            if render_result is not None:
                try:
                    _validate_render_result(render_result)
                    report["render_result"] = deepcopy(dict(render_result))
                    report["mrq_jobs"] = deepcopy(list(render_result.get("jobs") or []))
                except (TypeError, ValueError):
                    pass
            report["errors"].extend(str(error) for error in errors)
            if final_status in ("success", "failed", "cancelled", "canceled", "no_render"):
                report["mrq_pending"] = False
            write_application_report(report_path, report)
        except (OSError, ValueError) as exc:
            errors.append("无法写入 restore_failed retry report: {}".format(exc))
    elif report["state"] in ("not_started", "applied", "rendering"):
        try:
            if render_result is not None:
                try:
                    _validate_render_result(render_result)
                    report["render_result"] = deepcopy(dict(render_result))
                    report["mrq_jobs"] = deepcopy(list(render_result.get("jobs") or []))
                except (TypeError, ValueError):
                    pass
            failed = transition_application_report(
                report,
                "restore_failed",
                error="; ".join(errors),
                actor_updates={
                    entity_id: {"restore_status": "failed"}
                    for entity_id in ENTITY_IDS
                },
                mrq_pending=False if final_status in ("success", "failed", "cancelled", "canceled", "no_render") else None,
            )
            write_application_report(report_path, failed)
            report = failed
        except (OSError, ValueError) as exc:
            errors.append("无法写入 restore_failed report: {}".format(exc))
    return {
        "ok": False,
        "operation": "restore",
        "errors": errors,
        "warnings": [],
        "state": report["state"],
    }


def _load_unreal() -> Any:
    """仅在 UE adapter 调用时加载 Unreal Python 模块。"""
    import unreal

    return unreal
