"""只读校验诊断候选包；不导入UE、不读取媒体或升级候选状态。"""

import argparse
import hashlib
import json
import math
from pathlib import Path


HEAD_NAMES = {"nose", "left_eye", "right_eye", "left_ear", "right_ear"}


def checked_bytes(root, relative, expected_sha256):
    """拒绝越界、联接和替换文件；SHA按原始字节计算。"""
    path = root / relative
    if path.is_symlink() or root not in path.resolve().parents:
        raise ValueError("包内引用越界或为链接: " + relative)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("文件SHA不同: " + relative)
    return raw


def verify(root, source_root=None):
    """仅验证七个候选及小型JSON收据；source_root可额外比较原manifest字节。"""
    root = Path(root).resolve()
    registry = json.loads((root / "registry.json").read_text(encoding="utf-8"))
    if registry["production_ready"] is not False or [e["player_id"] for e in registry["entries"]] != list(range(3, 10)):
        raise ValueError("候选scope或七角色集合错误")
    results = []
    for entry in registry["entries"]:
        raw = checked_bytes(root, entry["candidate_path"], entry["candidate_sha256"])
        candidate = json.loads(raw)
        if candidate["ue_identity"] != entry["ue_identity"] or candidate["status"] != entry["status"]:
            raise ValueError("实际mesh身份或候选状态不同")
        identity = candidate["ue_identity"]
        role = entry["player_id"]
        stem = "Player_%03d" % role
        asset = "/Game/FutsalMOT/Characters/MPFB/Players/" + stem + ("/" if role == 3 else "/Mesh/") + "MPFB_" + stem + ".MPFB_" + stem
        if identity["asset_path"] != asset or identity["signature_scheme"] != "ue_lod_le_u32_f32_v1" or int(identity["lod"]) != 0:
            raise ValueError("非精确生产资产/LOD0签名")
        if any(candidate.get(key) is not True for key in ("not_animated", "not_MRQ_verified", "precision_not_measured")):
            raise ValueError("原始静态/未验标志被提升")
        if set(candidate["anchors"]) != HEAD_NAMES:
            raise ValueError("不是五个头部锚点")
        for name, anchor in candidate["anchors"].items():
            indices, weights = anchor["triangle_vertices"], anchor["barycentric"]
            if len(indices) != 3 or len(set(indices)) != 3 or any(type(i) is not int or not 0 <= i < int(identity["vertex_count"]) for i in indices):
                raise ValueError("实际mesh索引无效")
            if len(weights) != 3 or any(type(w) not in (int, float) or not math.isfinite(w) or not 0 <= w <= 1 for w in weights) or abs(sum(weights)-1) > 1e-9:
                raise ValueError("重心无效")
            expected = "recalibrated_on_ue_retriangulated_polygon" if name.endswith("ear") else "source_surface_triangle_geometry_matched"
            if anchor["method"] != expected:
                raise ValueError("原锚点方法发生变化")
        if source_root is not None and raw != (Path(source_root) / Path(entry["candidate_path"]).name).read_bytes():
            raise ValueError("原manifest与包内副本字节不同")
        for evidence in entry["bundled_evidence"]:
            correspondence = json.loads(checked_bytes(root, evidence["path"], evidence["sha256"]))
            if correspondence["ue_identity"] != identity or any(correspondence["anchors"]["0"][ear]["status"] != "REJECTED" for ear in ("left_ear", "right_ear")):
                raise ValueError("严格匹配失败历史或mesh身份被改写")
            if evidence.get("manifest_reference_field"):
                if candidate[evidence["manifest_reference_field"]] != evidence["original_reference"] or candidate["strict_correspondence_sha256"] != evidence["sha256"]:
                    raise ValueError("原严格匹配失败收据引用不同")
        results.append({"player_id": role, "candidate_sha256": entry["candidate_sha256"], "mesh_identity_preserved": True})
    source_registry = registry["source_registry_snapshot"]
    original = json.loads(checked_bytes(root, source_registry["path"], source_registry["sha256"]))
    for current, previous in zip(registry["entries"], original["entries"]):
        if any(current[key] != previous[key] for key in ("player_id", "candidate_sha256", "ue_identity", "status")):
            raise ValueError("原registry内容不一致")
    return {"schema": "futsalmot_diagnostic_calibration_package_check_v1", "verified_candidates": results,
            "exact_source_bytes_compared": source_root is not None, "production_ready": False,
            "external_assets_or_review_pixels_verified": False, "errors": []}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, help="可选：原Saved/ResearchDiagnostics目录，只读比较七manifest")
    args = parser.parse_args()
    print(json.dumps(verify(Path(__file__).parent, args.source_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
