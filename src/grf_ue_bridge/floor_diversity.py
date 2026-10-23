"""地板外观的独立确定性因子；不导入 UE，不改变运动或天气随机流。"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from typing import Any, Dict, Optional

POLICY = "futsalmot_floor_v1"
NAMESPACE = "futsalmot:floor:v1"
# 颜色为线性 RGB；贴图仅调制 ±4% 明度，不直接乘灰色而造成过暗。
CATALOG = (
    {"id": "sport_blue_v1", "base_color_linear": [0.075, 0.19, 0.31]},
    {"id": "sport_teal_v1", "base_color_linear": [0.075, 0.25, 0.22]},
    {"id": "sport_terracotta_v1", "base_color_linear": [0.31, 0.115, 0.075]},
    {"id": "sport_gray_v1", "base_color_linear": [0.24, 0.25, 0.27]},
)


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def _sha(value: str, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{field} 必须为小写 SHA256")
    return value


def select_floor_id(seed: int) -> str:
    """独立哈希选择；可供 UE 预览使用，不伪造生产轨迹摘要。"""
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 0x7fffffff:
        raise ValueError("seed 必须是 0 至 2147483647 的整数")
    rank = _hash({"namespace": NAMESPACE, "seed": seed, "catalog_sha256": _hash(CATALOG)})
    return CATALOG[int(rank, 16) % len(CATALOG)]["id"]


def build_floor_manifest(
    seed: int, *, trajectory_sha256: str, frozen_context_sha256: str,
    texture_sha256: str, floor_id: Optional[str] = None,
) -> Dict[str, Any]:
    """构造可复现、可配对的地板记录；context 应冻结相机、动画、天气及外观。"""
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 0x7fffffff:
        raise ValueError("seed 必须是 0 至 2147483647 的整数")
    for value, field in ((trajectory_sha256, "trajectory_sha256"),
                         (frozen_context_sha256, "frozen_context_sha256"),
                         (texture_sha256, "texture_sha256")):
        _sha(value, field)
    catalog_hash = _hash(CATALOG)
    rank = _hash({"namespace": NAMESPACE, "seed": seed, "catalog_sha256": catalog_hash})
    selected_id = floor_id if floor_id is not None else select_floor_id(seed)
    matches = [item for item in CATALOG if item["id"] == selected_id]
    if len(matches) != 1:
        raise ValueError(f"未知地板 ID: {selected_id}")
    record = {
        "schema": "futsalmot_episode_floor", "version": 1, "policy": POLICY,
        "seed": seed, "selection_namespace": NAMESPACE, "selection_sha256": rank,
        "selection_mode": "paired_override" if floor_id is not None else "seeded",
        "catalog_sha256": catalog_hash, "texture_sha256": texture_sha256,
        "trajectory_sha256": trajectory_sha256,
        "frozen_context_sha256": frozen_context_sha256,
        "pairing_key": _hash({"seed": seed, "trajectory": trajectory_sha256,
                              "context": frozen_context_sha256}),
        "variant": deepcopy(matches[0]),
        "material_parameters": {"roughness": 0.72, "metallic": 0.0,
                                "texture_modulation_amplitude": 0.04,
                                "tile_size_m": 2.0},
        "application_status": "planned_not_applied",
    }
    record["fingerprint"] = _hash(record)
    return record


def paired_floor_manifests(seed: int, **kwargs: Any) -> list:
    """同一个运动和视觉上下文，只枚举地板；同一配对组必须归同一 split。"""
    return [build_floor_manifest(seed, floor_id=item["id"], **kwargs) for item in CATALOG]
