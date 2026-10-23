"""按原始字节打包七个静态头部候选；不处理图像、UE或Blender。"""

import argparse
import hashlib
import json
from pathlib import Path


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def store_exact(path, raw):
    """只新建或确认已有字节完全相同；拒绝覆盖不同内容。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError("拒绝覆盖不同文件: " + str(path))
    else:
        path.write_bytes(raw)


def package(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    original_registry_path = source / "production_pose17_head_candidate_registry.json"
    original_raw = original_registry_path.read_bytes()
    original = json.loads(original_raw)
    if original["production_ready"] is not False or [e["player_id"] for e in original["entries"]] != list(range(3, 10)):
        raise ValueError("原registry不是七角色诊断候选")
    registry = {"schema": "futsalmot_head_diagnostic_calibration_package_v1", "production_ready": False,
                "path_base": "registry.json所在目录", "status_policy": "原manifest字节和静态候选标志不变；本包不升级验收",
                "source_registry_snapshot": {"path": "evidence/source_registry_snapshot.json", "sha256": digest(original_raw)},
                "entries": []}
    store_exact(output / registry["source_registry_snapshot"]["path"], original_raw)
    for old in original["entries"]:
        role = old["player_id"]
        name = "mpfb_player%03d_ue_surface_head_candidates.json" % role
        raw = (source / name).read_bytes()
        if digest(raw) != old["candidate_sha256"]:
            raise ValueError("原registry SHA不匹配: " + name)
        candidate = json.loads(raw)
        if candidate["ue_identity"] != old["ue_identity"] or candidate["status"] != old["status"]:
            raise ValueError("原registry身份不同")
        store_exact(output / "manifests" / name, raw)
        receipt_name = "mpfb_player%03d_source_fbx_ue_correspondence.json" % role
        receipt_raw = (source / receipt_name).read_bytes()
        evidence = {"path": "evidence/" + receipt_name, "sha256": digest(receipt_raw),
                    "role": "严格源锚点匹配收据，保留双耳严格匹配失败，不替代UE表面重标定候选"}
        if "strict_correspondence_receipt" in candidate:
            if digest(receipt_raw) != candidate["strict_correspondence_sha256"]:
                raise ValueError("原manifest严格收据SHA不同")
            evidence.update(manifest_reference_field="strict_correspondence_receipt",
                            original_reference=candidate["strict_correspondence_receipt"])
        store_exact(output / evidence["path"], receipt_raw)
        # 外部资源只查存在性，不读取媒体/Blend字节，也不将其SHA声称为本次已验证。
        dependencies = [{"field": "source_blend", "reference": candidate["source_blend"],
                         "kind": "外部Blender源资产，重建依赖；不在包内", "exists_at_packaging": Path(candidate["source_blend"]).is_file()}]
        correspondence = json.loads(receipt_raw)
        dependencies.append({"field": "bundled_evidence.fbx_path", "reference": correspondence["fbx_path"],
                             "expected_sha256": correspondence["fbx_sha256"],
                             "kind": "外部FBX重建依赖；SHA来自历史收据，本次仅查存在性，不重读FBX",
                             "exists_at_packaging": Path(correspondence["fbx_path"]).is_file()})
        for image, sha in candidate.get("reviewed_images_sha256", {}).items():
            dependencies.append({"field": "reviewed_images_sha256." + image, "reference": image,
                                 "expected_sha256": sha, "kind": "历史静态目视证据，未打包；本次不读像素/不重算SHA",
                                 "lookup_base": "原Saved/ResearchDiagnostics", "exists_at_packaging": (source / image).is_file()})
        registry["entries"].append({key: old[key] for key in ("player_id", "candidate_sha256", "ue_identity", "status")})
        registry["entries"][-1].update(candidate_path="manifests/" + name,
            original_candidate_path=old["candidate_path"], bundled_evidence=[evidence], external_dependencies=dependencies)
    raw = (json.dumps(registry, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    store_exact(output / "registry.json", raw)
    return {"candidates_copied": 7, "strict_receipts_copied": 7, "production_ready": False,
            "registry": str(output / "registry.json"), "external_missing": [d for e in registry["entries"] for d in e["external_dependencies"] if not d["exists_at_packaging"]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.source_root, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
