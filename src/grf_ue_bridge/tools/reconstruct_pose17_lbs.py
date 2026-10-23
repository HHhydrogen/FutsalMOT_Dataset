"""只读旧native及显式静态skin JSON，写新目录的严格SHA候选报告。"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from grf_ue_bridge.pose17_offline_lbs_v2 import (
    ARITHMETIC, candidate_ref_to_local, decode_reference_geometry, decode_skin_snapshot, reconstruct_surface,
)
from grf_ue_bridge.tools.inspect_pose17_smoke import check_gpu_lbs_sample, sampling_context


def read_json(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def inspect(native_paths, skin_paths, output, snapshot_field=None, reference_paths=(), surface_frame_indices=None):
    """不读取媒体、不得覆盖已有目录；包装层只由显式字段选择。"""
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("输出目录已存在，拒绝覆盖")
    if surface_frame_indices is not None and (not surface_frame_indices or len(set(surface_frame_indices)) != len(surface_frame_indices)
        or any(type(frame) is not int or frame < 0 for frame in surface_frame_indices)):
        raise ValueError("表面子集帧须为显式不重复非负整数")
    skins, skin_sources = {}, {}
    for path in skin_paths:
        payload, sha = read_json(path)
        if snapshot_field is not None:
            payload = payload[snapshot_field]
        decoded = decode_skin_snapshot(payload)
        asset = decoded["identity"]["asset_path"]
        if asset in skins:
            raise ValueError("静态skin资产重复，不自动选择其一")
        skins[asset] = decoded
        skin_sources[asset] = {"path": str(Path(path).resolve()), "file_sha256": sha,
                               "weights_sha256": decoded["weights_sha256"], "inverse_matrices_sha256": decoded["inverse_matrices_sha256"]}
    geometries = {}
    for path in reference_paths:
        payload, sha = read_json(path)
        asset = payload["identity"]["asset_path"]
        if asset not in skins or asset in geometries:
            raise ValueError("参考几何资产缺skin或重复")
        positions, _ = decode_reference_geometry(payload, skins[asset])
        geometries[asset] = (positions, {"path": str(Path(path).resolve()), "file_sha256": sha})
    ue_dir = Path(__file__).resolve().parents[3]/"ue"
    if str(ue_dir) not in sys.path:
        sys.path.insert(0, str(ue_dir))
    from pose17_native_receipt import normalize_native_receipt
    report = {"schema": "futsalmot_offline_ref_to_local_candidate_v1", "arithmetic": ARITHMETIC,
              "skin_sources": skin_sources, "native_sources": [], "rows": [], "errors": [],
              "visibility_state": "unknown", "render_surface_equivalence_verified": False,
              "surface_frame_indices": surface_frame_indices, "surface_dtype": "float32", "surface_coordinate_space": "mesh_component", "surface_unit": "cm",
              "media_read": False, "surface_files_require_exact_native_matrix_sha": True}
    output.mkdir(parents=True)
    pending_surfaces = []
    for source_index, path in enumerate(native_paths):
        path = Path(path).resolve()
        try:
            raw, sha = read_json(path)
            receipt = normalize_native_receipt(raw)
            plan, _ = read_json(path.parent/"plan.json")
            context = sampling_context(plan, receipt)
            if context["path"] != "gpu_render_cpu_lbs":
                raise ValueError("离线工具仅处理已显式GPU/LBS native分支")
            if surface_frame_indices is not None and not set(surface_frame_indices).issubset(range(plan["frame_count"])):
                raise ValueError("请求的表面子集帧不在native冻结帧窗")
            entities = {target["entity_id"] for target in plan["targets"]}
            expected = {(frame, entity) for frame in range(plan["frame_count"]) for entity in entities}
            if receipt.get("setup_error") != "" or receipt.get("record_limit_reached") is not False or len(receipt["records"]) != len(expected) or {(r["output_frame"], r["entity_id"]) for r in receipt["records"]} != expected:
                raise ValueError("native错误/截断或帧实体集合不完整")
            report["native_sources"].append({"path": str(path), "file_sha256": sha})
            for row in receipt["records"]:
                check_gpu_lbs_sample(row, context)
                skin = skins[row["mesh_identity"]["asset_path"]]
                evidence, _ = candidate_ref_to_local(row, skin)
                evidence.update(entity_id=row["entity_id"], camera_id=row["camera_name"],
                    output_frame=row["output_frame"], source_frame=row["sequence_source_frame"], native_ref=str(path),
                    mesh_identity=skin["identity"], component_surface_file=None)
                if evidence["ref_to_local_sha256_matched"] and skin["identity"]["asset_path"] in geometries and (surface_frame_indices is None or row["output_frame"] in surface_frame_indices):
                    filename = "source%02d_%s_%06d_%s_component_surface.npz" % (source_index, row["camera_name"], row["output_frame"], row["entity_id"])
                    pending_surfaces.append((row, skin, evidence, filename))
                report["rows"].append(evidence)
        except Exception as error:
            report["errors"].append(str(path)+": "+repr(error))
    report["matched_rows"] = sum(row["ref_to_local_sha256_matched"] for row in report["rows"])
    report["mismatched_rows"] = len(report["rows"])-report["matched_rows"]
    report["all_requested_native_matrix_sha_matched"] = bool(report["rows"]) and not report["errors"] and report["mismatched_rows"] == 0
    # 整次请求全部SHA通过后才写表面；任何一个不匹配都只有诊断报告。
    if report["all_requested_native_matrix_sha_matched"]:
        for row, skin, evidence, filename in pending_surfaces:
            positions, geometry_ref = geometries[skin["identity"]["asset_path"]]
            _, deformed = reconstruct_surface(row, skin, positions)
            np.savez(output/filename, positions_cm=deformed,
                mesh_identity_json=np.asarray(json.dumps(skin["identity"],sort_keys=True)),
                coordinate_space=np.asarray("mesh_component"), unit=np.asarray("cm"),
                visual_world_transform_json=np.asarray(json.dumps(row["visual_world_transform_ue_cm"],sort_keys=True)),
                native_ref_to_local_sha256=np.asarray(row["ref_to_local_matrices_sha256"]),
                skin_weight_sha256=np.asarray(row["skin_weight_data_sha256"]),
                entity_id=np.asarray(row["entity_id"]),camera_id=np.asarray(row["camera_name"]),
                output_frame=np.asarray(row["output_frame"],dtype=np.int64),source_frame=np.asarray(row["sequence_source_frame"],dtype=np.int64))
            evidence.update(component_surface_file=filename, reference_geometry_ref=geometry_ref,
                            reconstructed_component_vertex_count=len(deformed))
    (output/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-json", type=Path, action="append", required=True)
    parser.add_argument("--skin-json", type=Path, action="append", required=True)
    parser.add_argument("--skin-snapshot-field", help="若JSON有包装层，显式传其直接字段名；不自动猜snapshot/data")
    parser.add_argument("--reference-json", type=Path, action="append", default=[], help="可选，仅严格矩阵SHA匹配后写component-space表面NPZ")
    parser.add_argument("--surface-frame-index", type=int, action="append", help="可选：只输出这些相对帧表面，全部native帧仍做SHA门禁")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = inspect(args.native_json, args.skin_json, args.output, args.skin_snapshot_field, args.reference_json, args.surface_frame_index)
    print(json.dumps({key: report[key] for key in ("matched_rows", "mismatched_rows", "errors", "all_requested_native_matrix_sha_matched")}, ensure_ascii=False))
    return 0 if report["all_requested_native_matrix_sha_matched"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
