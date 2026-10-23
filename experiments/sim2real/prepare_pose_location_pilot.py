"""仅消费冻结四seed位置试产文本证据，生成研究候选JSONL与配对采样表。"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

CODE = Path(__file__).resolve().parents[2]
PROJECT = CODE.parents[2]
sys.path[:0] = [str(CODE / "src"), str(CODE / "ue")]
from grf_ue_bridge.pose17_export import project_landmarks
from grf_ue_bridge.pose_location_supervision_v2 import synthetic_person_location, collate_location_contract
from pose17_native_receipt import normalize_native_receipt
from pose17_override_receipt import validate_joint_bundle
from pose17_location_pilot_contract import PILOT_WINDOWS, validate_location_pilot_contract
from pose17_calibration_assets import JOINT_VISUAL_IDS
import prepare_coco_pose_subset as common

DEFAULT_OUTPUT = Path("D:/projects/FutsalMOT_Research/pose/location_pilot_4seed_a01")
REAL_ROOT = Path("D:/projects/FutsalMOT_Research/pose/coco2017_location_a01")
CANDIDATE_PROTOCOL = CODE / "experiments/sim2real/pose_location_coco_candidate_protocol.json"
PROTOCOL_SHA = "b2946b58b8e2b10510f185815018e698cee25b705433590494ae3913f7d15c33"
SOURCE_CANDIDATE_SHA = "21cdf06aa1de3b009b13855a6870fb16f57800102611a2ce7e52d7e5c7c9a672"
TRAIN_SEEDS = [1042, 1044, 1048]
TRAINING_SEEDS = [20261024, 20261025, 20261026]
CAMERAS = ["CineCam_%02d" % i for i in range(1, 5)]
SALT = "futsalmot:location_pilot:synthetic_balanced_selection:v1"
SCHEDULE_SALT = "futsalmot:location_pilot:paired_update_schedule:v1"
MEDIA_POLICY = "export_mask_decode_and_text_evidence_no_media_rehash"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def saved(path):
    path = common.plain(path)
    root = PROJECT / "Saved"
    require(root in path.parents and path != root, "只读当前项目Saved文本来源")
    return path


class TextEvidence:
    """每份输入文本一次读取并计算摘要；不接受媒体后缀或链接。"""
    def __init__(self):
        self.files = {}

    def _bytes(self, path, kind):
        path = saved(path)
        require(path.suffix in {".json", ".jsonl"}, "不读取媒体像素或媒体SHA")
        require(str(path) not in self.files, "同一文本应在一次适配内只读一次: " + str(path))
        raw = path.read_bytes()
        self.files[str(path)] = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "kind": kind}
        return raw

    def json(self, path, kind):
        return json.loads(self._bytes(path, kind))

    def jsonl(self, path, kind):
        return [json.loads(line) for line in self._bytes(path, kind).splitlines() if line.strip()]


def source_flags(value):
    return all(value.get(key) is expected for key, expected in
               (("diagnostic_only", True), ("not_training", True), ("production_accepted", False), ("MRQ_sync_verified", False)))


def percentile_summary(values):
    values = sorted(values)
    if not values:
        return None
    def quantile(q):
        index = (len(values) - 1) * q
        low, high = math.floor(index), math.ceil(index)
        return values[low] + (values[high] - values[low]) * (index - low)
    return {"min": values[0], "p05": quantile(.05), "p25": quantile(.25), "median": quantile(.5),
            "p75": quantile(.75), "p95": quantile(.95), "max": values[-1]}


def image_object_target(obj, world, native, view, seed, camera, source_frame):
    """全部真实world保留；area0不捏造bbox，存在bbox的全缺点人仍保留。"""
    entity = obj["entity_id"]
    track = int(entity[1]) + (1 if entity[0] == "L" else 6)
    require(obj["track_id"] == obj["mask_id"] == world["track_id"] == track, "实体track/mask身份变化")
    require(world["world_m"] == native["world_m"] and world["mesh_identity"] == native["mesh_identity"]
            and world["world_location_valid"] == [True] * 17, "world或mesh与原生记录不一致")
    projected = project_landmarks([{"world_m": point, "source_method": native["sampling_method"]} for point in native["world_m"]], view)
    valid = [point.get("uv_px") is not None and point["visibility_state"] == "unknown" for point in projected]
    require(len(obj["location_valid"]) == 17 and all(type(flag) is bool for flag in obj["location_valid"])
            and obj["location_valid"] == valid and obj["uv_px"] == [point.get("uv_px") for point in projected]
            and obj["visibility_state"] == [point["visibility_state"] for point in projected]
            and obj["camera_forward_depth_m"] == [point.get("camera_forward_depth_m") for point in projected]
            and obj["visibility_method"] == "not_measured", "位置mask/投影/depth/unknown与world及camera文本不一致")
    bbox, area = obj["bbox_visible_xywh"], obj["visible_mask_area"]
    if area == 0:
        require(type(area) is int and bbox is None, "area0不能发明可见bbox")
        return None
    width, height = view["intrinsics"]["width"], view["intrinsics"]["height"]
    require(type(area) is int and area > 0 and isinstance(bbox, list) and len(bbox) == 4
            and all(type(number) in (int, float) and math.isfinite(number) for number in bbox), "可见bbox/area非法")
    x, y, w, h = bbox
    require(w > 0 and h > 0 and x >= 0 and y >= 0 and x + w <= width and y + h <= height and area <= w * h,
            "可见bbox超界或空，不能补框裁框")
    target = synthetic_person_location({"id": f"seed_{seed:08d}:{camera}:source{source_frame:06d}:{entity}", "bbox": bbox,
                                        "xy_px": [point if point is not None else [0., 0.] for point in obj["uv_px"]],
                                        "location_valid": obj["location_valid"], "visibility_state": obj["visibility_state"]}, width, height)
    target.update(entity_id=entity, source_native_record_ref=world["native_record_ref"])
    target["source_visible_mask_area"] = area
    return target


def summarize_frames(frames):
    """仅统计文本位置/框分布；画内位置有效性不解释成可见性。"""
    states = Counter()
    edges = Counter({"lt2px": 0, "2_to_10px": 0, "10_to_50px": 0, "ge50px": 0})
    missing, partial, people, valid_total, outside_bbox, world_count, area_zero = 0, 0, 0, 0, 0, 0, 0
    heights, small_heights, small_widths, small_areas, valid_counts = [], [], [], [], []
    is_synthetic = all(frame.get("source") == "synthetic_location_pilot_candidate" for frame in frames)
    for frame in frames:
        width, height = frame["width"], frame["height"]
        area_zero += len(frame.get("nonvisible_entities_retained_in_world", []))
        world_count += len(frame["instances"]) + len(frame.get("nonvisible_entities_retained_in_world", []))
        for instance in frame["instances"]:
            count = sum(instance["location_valid"])
            people += 1
            missing += count == 0
            partial += 0 < count < 17
            valid_total += count
            valid_counts.append(count)
            states.update(instance["visibility_state"])
            x, y, w, h = instance["bbox_xywh_px"]
            heights.append(h)
            small_heights.append(h * min(640 / width, 640 / height))
            small_widths.append(w * min(640 / width, 640 / height))
            small_areas.append(w * h * min(640 / width, 640 / height) ** 2)
            for (u, v), ok in zip(instance["xy_normalized"], instance["location_valid"]):
                if not ok:
                    continue
                u, v = u * width, v * height
                outside_bbox += not (x <= u < x + w and y <= v < y + h)
                distance = min(u, v, width - u, height - v)
                edges["lt2px" if distance < 2 else "2_to_10px" if distance < 10 else "10_to_50px" if distance < 50 else "ge50px"] += 1
    return {"images": len(frames), "world_entity_records": world_count if is_synthetic else None, "positive_visible_bbox_people": people,
            "area0_people_retained_in_source_world": area_zero if is_synthetic else None,
            "area0_fraction_all_world_entities": area_zero / world_count if is_synthetic and world_count else None,
            "positive_bbox_all_missing_location_people": missing, "all_missing_fraction_positive_bbox": missing / people if people else 0,
            "partial_location_people": partial, "valid_location_points": valid_total, "source_visibility_counts_positive_bbox": dict(states),
            "valid_points_outside_visible_bbox": outside_bbox, "outside_visible_bbox_fraction_valid_points": outside_bbox / valid_total if valid_total else 0,
            "image_boundary_distance_of_valid_points_native_pixels": dict(edges), "boundary_definition": "min(u,v,width-u,height-v)_half_open_image",
            "bbox_pixel_height": percentile_summary(heights), "bbox_height_after_640_letterbox_geometry": percentile_summary(small_heights),
            "bbox_width_after_640_letterbox_geometry": percentile_summary(small_widths),
            "bbox_area_after_640_letterbox_geometry": percentile_summary(small_areas),
            "valid_location_count_per_positive_bbox_person": percentile_summary(valid_counts),
            "bbox_semantics": "visible_mask_tight_half_open_xywh" if is_synthetic else "official_COCO_bbox_clipped_to_image_by_location_bridge",
            "distribution_used_for_selection": False}


def balanced_selection(frames, count=1000):
    """先seed后camera平衡，配额余数及各cell均用独立SHA排序。"""
    require(type(count) is int and 0 < count <= len(frames), "选择数量无效")
    require(len({row["frame_key"] for row in frames}) == len(frames), "源frame_key重复")
    require({row["root_seed"] for row in frames} == set(TRAIN_SEEDS), "只选三个训练seed，不混1102")
    def rank(kind, value):
        return common.canonical_hash({"salt": SALT, "kind": kind, "value": value})
    seed_order = sorted(TRAIN_SEEDS, key=lambda seed: (rank("seed_quota", seed), seed))
    seed_quota = {seed: count // 3 + (index < count % 3) for index, seed in enumerate(seed_order)}
    selected = []
    quotas = {}
    for seed in TRAIN_SEEDS:
        camera_order = sorted(CAMERAS, key=lambda camera: (rank("camera_quota", [seed, camera]), camera))
        per_camera = {camera: seed_quota[seed] // 4 + (index < seed_quota[seed] % 4) for index, camera in enumerate(camera_order)}
        quotas[str(seed)] = per_camera
        for camera in CAMERAS:
            pool = [(index, row) for index, row in enumerate(frames) if row["root_seed"] == seed and row["camera_id"] == camera]
            require(len(pool) == 120, "每个cell必须完整120帧，不补邻帧")
            pool.sort(key=lambda value: (rank("source_frame", value[1]["frame_key"]), value[1]["frame_key"]))
            selected.extend(index for index, _ in pool[:per_camera[camera]])
    selected.sort(key=lambda index: (rank("selected_order", frames[index]["frame_key"]), frames[index]["frame_key"]))
    require(len(selected) == len(set(selected)) == count, "选样不足或重复")
    return selected, quotas


def paired_schedule(real_ids, selected_indices, selected_keys, seed):
    """同seed两臂共用三真实槽；每个real恰3次shared加1次R额外槽。"""
    require(len(real_ids) == len(set(real_ids)) == 1000 and len(selected_indices) == len(set(selected_indices)) == 1000
            and len(selected_keys) == len(set(selected_keys)) == 1000 and type(seed) is int and seed in TRAINING_SEEDS,
            "只接收冻结1000R/1000S及三个训练seed")
    def order(slot, values):
        return sorted(values, key=lambda index: (common.canonical_hash({"salt": SCHEDULE_SALT, "seed": seed, "slot": slot,
                                                                       "key": real_ids[index] if slot.startswith("real") else selected_keys[index]}), index))
    real_perms = [order("real_slot_%d" % slot, list(range(1000))) for slot in range(4)]
    synthetic_perm = order("synthetic", list(range(1000)))
    return [{"schema": "futsalmot_pose_location_paired_schedule_row_v1", "update_index": index,
             "shared_real_indices": [real_perms[slot][index] for slot in range(3)],
             "extra_real_index": real_perms[3][index], "synthetic_index": selected_indices[synthetic_perm[index]]} for index in range(1000)]


def adapt_group(entry, output, expected_source):
    seed = entry["seed"]
    require(type(seed) is int and seed in PILOT_WINDOWS, "只接收冻结四seed")
    start, end, split = PILOT_WINDOWS[seed]
    attempt_name = f"location_pilot_s{seed:08d}_a01_gpu_lbs"
    release = saved(entry["export"])
    require(entry["attempt"] == attempt_name and release == PROJECT / "Saved/Pose17Development" / ("diagnostic_" + attempt_name + "_export_a01")
            and entry["rgb_frames"] == entry["mask_frames"] == 480 and entry["native_pose_records"] == 4800 and entry["not_training"] is True,
            "拒绝旧40图或任意导出路径冒充本批")
    reader = TextEvidence()
    dataset = reader.json(release / "dataset.json", "dataset")
    require(dataset.get("schema") == "futsalmot_pose17_diagnostic_release_v1" and dataset.get("status") == "diagnostic_export_complete_not_training"
            and source_flags(dataset) and dataset.get("rgb_frames") == dataset.get("mask_frames") == dataset.get("exr_decode_count") == 480
            and dataset.get("native_pose_records") == 4800 and dataset.get("visibility_upgrade_performed") is False, "源导出终态不完整或诊断边界改变")
    group_dir = saved(release / Path(dataset["group_ref"]).parent)
    require(group_dir == release / "groups" / f"seed_{seed:08d}", "group引用越界")
    group = reader.json(group_dir / "group.json", "group")
    require(source_flags(group) and group.get("seed") == seed and group.get("world_storage") == "per_camera_native"
            and group.get("cross_camera_world_shared") is False and group.get("world_values_modified") is False and group.get("rgb_mode") == "copy",
            "组world存储/原值/拷贝RGB或源边界不符")
    provenance = group_dir / "provenance"
    plan = reader.json(provenance / "plan.json", "plan")
    state = reader.json(provenance / "state.json", "state")
    require(validate_location_pilot_contract(plan) is True and plan["source_motion"] == expected_source["proposed_runtime_plan"]["source_motion"]
            and plan["attempt"] == attempt_name and group["source_motion"] == plan["source_motion"] and group["location_pilot"] == plan["location_pilot"]
            and group["source_scope"] == plan["source_scope"] and {row["entity_id"]: row["calibration_player_id"] for row in group["roster"]} == JOINT_VISUAL_IDS,
            "源窗口/roster/显式location契约不符")
    require(state["plan_sha256"] == reader.files[str(provenance / "plan.json")]["sha256"] and state.get("session_released") is True
            and state.get("status") == "rendered_requires_RGB_mask_pose_alignment_acceptance" and state.get("executor_success") is True and not state.get("errors"),
            "原生会话没有完成并释放")
    alignment = reader.json(provenance / "alignment_report.json", "alignment")
    acceptance = reader.json(provenance / "native_acceptance_report.json", "native_acceptance")
    require(alignment.get("structural_pairing_passed") is True and not alignment.get("errors") and alignment.get("MRQ_sync_verified") is False
            and acceptance.get("acceptance_passed") is True and acceptance.get("component_transaction_passed") is True
            and not acceptance.get("errors") and acceptance.get("MRQ_sync_verified") is False, "配对/native文本证据未通过")
    attempt = PROJECT / "Saved/Pose17Development/smoke" / attempt_name
    require(Path(alignment["attempt"]) == Path(acceptance["attempt"]) == attempt, "配对/native来源不是本attempt")
    before = reader.json(provenance / "preflight.json", "editor_before")
    after = reader.json(provenance / "editor_after.json", "editor_after")
    manifests, hashes, loaded = {}, {}, {}
    for entity, relative in group["landmark_manifest_refs"].items():
        path = saved(group_dir / relative)
        if str(path) not in loaded:
            loaded[str(path)] = reader.json(path, "landmark")
        manifests[entity] = loaded[str(path)]
        hashes[entity] = reader.files[str(path)]["sha256"]
        require(path.stem == hashes[entity], "冻结landmark文件名与SHA不符")
    native = {camera: normalize_native_receipt(reader.json(provenance / (camera + "_native.json"), "native")) for camera in CAMERAS}
    checked = validate_joint_bundle(plan, state, before, after, manifests, hashes, native)
    require(checked.get("acceptance_passed") is True and not checked.get("errors"), "原生事务/身份文本验证失败: " + repr(checked.get("errors", [])[:3]))
    timeline = reader.jsonl(group_dir / group["timeline_ref"], "timeline")
    require(len(timeline) == 120 and all(row["frame_index"] == index and row["source_frame_index"] == row["sequence_frame_index"] == start + index
                                       and row["source_time_seconds"] == (start + index) / 10 for index, row in enumerate(timeline)), "连续来源时间线改变")
    require(set(group["cameras"]) == set(alignment["cameras"]) == set(CAMERAS), "四相机不完整")
    frames = []
    for camera in CAMERAS:
        camera_dir = group_dir / "cameras" / camera
        camera_ref = group["cameras"][camera]
        require(camera_ref == {"view_ref": f"cameras/{camera}/view.json", "pose3d_world_ref": f"cameras/{camera}/pose3d_world.jsonl",
                               "pose2d_ref": f"cameras/{camera}/pose2d.jsonl"}, "相机文件引用变化")
        view = reader.json(group_dir / camera_ref["view_ref"], "view")
        require([view["intrinsics"]["width"], view["intrinsics"]["height"]] == [3840, 2160], "本批只接收原生4K")
        world_rows = reader.jsonl(group_dir / camera_ref["pose3d_world_ref"], "world")
        image_rows = reader.jsonl(group_dir / camera_ref["pose2d_ref"], "pose2d")
        require(len(world_rows) == len(image_rows) == 120, "源帧数不足或多余，不补帧")
        native_index = {(row["output_frame"], row["entity_id"]): row for row in native[camera]["records"]}
        pairs = alignment["cameras"][camera]["frames"]
        paired_index = {(row["output_frame"], row["entity_id"]): row for row in pairs}
        require(len(paired_index) == len(pairs) == len(native_index) == 1200, "native/配对实体帧重复或不完整")
        for index, (world, image) in enumerate(zip(world_rows, image_rows)):
            source_frame = start + index
            require(world["frame_index"] == image["frame_index"] == image["world_frame_index"] == index
                    and world["source_frame_index"] == world["sequence_frame_index"] == source_frame, "图像/world/源帧不同步")
            world_objects = {row["entity_id"]: row for row in world["objects"]}
            require(len(world_objects) == len(world["objects"]) == len(image["objects"]) == 10
                    and set(world_objects) == {row["entity_id"] for row in image["objects"]} == set(JOINT_VISUAL_IDS), "十实体遗漏/重复")
            instances, zero = [], []
            for obj in image["objects"]:
                entity = obj["entity_id"]
                row, paired = native_index[(index, entity)], paired_index[(index, entity)]
                require(row["sequence_source_frame"] == paired["source_frame"] == source_frame
                        and [point["world_m"] for point in paired["keypoints"]] == row["world_m"]
                        and {key: paired["camera"][key] for key in ("intrinsics", "extrinsics")} == view,
                        "配对相机/world/源帧与native或view不同")
                target = image_object_target(obj, world_objects[entity], row, view, seed, camera, source_frame)
                if target is None:
                    zero.append(entity)
                else:
                    instances.append(target)
            require(image["rgb_ref"] == f"rgb/{index:06d}.png" and image["mask_ref"] == f"mask/{index:06d}.png" and image["rgb_source_ref"] is None,
                    "只接收已有拷贝RGB和canonical mask引用")
            rgb = saved(camera_dir / image["rgb_ref"])
            mask = saved(camera_dir / image["mask_ref"])
            require(rgb.is_file() and mask.is_file(), "导出媒体引用不存在，不补图")
            frame = {"schema": "separated_pose_location_frame_v1", "frame_key": f"seed_{seed:08d}:{camera}:source{source_frame:06d}",
                     "root_seed": seed, "camera_id": camera, "frame_index": index, "source_frame_index": source_frame,
                     "image_file": str(rgb), "image_sha256": None, "width": 3840, "height": 2160,
                     "bbox_coverage": "all_noncrowd_person_instances", "crowd_present": False,
                     "expected_person_annotation_ids": [row["annotation_id"] for row in instances], "instances": instances,
                     "nonvisible_entities_retained_in_world": zero, "source": "synthetic_location_pilot_candidate",
                     "diagnostic_only": True, "not_training": True, "MRQ_sync_verified": False,
                     "source_dataset_ref": str(release / "dataset.json"), "source_group_ref": str(group_dir / "group.json"),
                     "source_evidence_ref": str(output / "candidate_receipt.json") + "#seed=" + str(seed),
                     "source_world_ref": str(camera_dir / "pose3d_world.jsonl") + "#frame_index=" + str(index)}
            collate_location_contract([frame])
            frames.append(frame)
    stats = summarize_frames(frames)
    require(stats["images"] == 480 and stats["world_entity_records"] == 4800, "本组总量不完整")
    evidence = {"seed": seed, "split": split, "attempt": attempt_name, "source_export": str(release),
                "source_dataset_ref": str(release / "dataset.json"), "source_group_ref": str(group_dir / "group.json"),
                "source_scope": group["source_scope"], "location_pilot": group["location_pilot"],
                "source_flags": {"diagnostic_only": True, "not_training": True, "production_accepted": False, "MRQ_sync_verified": False},
                "evidence_files": list(reader.files.values()), "counts": stats,
                "text_checks": {"passed": True, "native_bundle": True, "world_and_projection_exact": True,
                                "all_10_entities_retained_per_source_frame": True, "media_hash_or_pixels_read": False}}
    return frames, evidence


def write_jsonl(path, rows):
    path = common.plain(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")


def prepare(output):
    output = common.root_path(output)
    require(not output.exists(), "候选输出目录已存在；不覆盖、续写或读val结果重抽")
    require(common.sha(CANDIDATE_PROTOCOL) == PROTOCOL_SHA, "候选COCO协议SHA变化")
    source_reader = TextEvidence()
    driver_path = PROJECT / "Saved/Pose17Development/location_pilot_driver_a01.json"
    driver = source_reader.json(driver_path, "driver")
    require(driver.get("schema") == "futsalmot_location_pilot_driver_v1" and driver.get("status") == "complete_capture_and_diagnostic_export_not_training"
            and driver.get("gpu_training_started") is False and driver.get("error") is None and driver.get("current") is None,
            "四组捕获/导出尚未完成")
    entries = driver["completed"]
    require([row["seed"] for row in entries] == list(PILOT_WINDOWS), "只接收原定四seed，不能读旧40图或扩容")
    source_candidate_path = PROJECT / "Saved/Pose17Development/location_pilot_4seed_candidate_a01/candidate_plan.json"
    source_candidate = source_reader.json(source_candidate_path, "capture_candidate")
    require(source_reader.files[str(source_candidate_path)]["sha256"] == SOURCE_CANDIDATE_SHA, "源候选清单变化")
    expected = {row["root_seed"]: row for row in source_candidate["groups"]}
    real_path = common.plain(REAL_ROOT / "subset/train2017.location.jsonl")
    real_bytes = real_path.read_bytes()
    real_sha = hashlib.sha256(real_bytes).hexdigest()
    require(real_sha == "81a54940011f8b10dbf9907b79ac74c0130002a203e14d8d9cf0f4f2cc11e97c", "真实1000图JSONL SHA变化")
    real = [json.loads(line) for line in real_bytes.splitlines() if line.strip()]
    require(len(real) == 1000 and len({row["image_id"] for row in real}) == 1000, "真实1000ID不唯一")
    output.mkdir(parents=True)
    common.dump_new(output / "started.json", {"status": "text_evidence_adaptation_started_not_training", "training_authorized": False})
    train, dev, groups = [], [], []
    for entry in entries:
        frames, evidence = adapt_group(entry, output, expected[entry["seed"]])
        (train if entry["seed"] in TRAIN_SEEDS else dev).extend(frames)
        groups.append(evidence)
        print(json.dumps({"seed": entry["seed"], "text_evidence_adapted_frames": len(frames), "media_reread": False}), flush=True)
    require(len(train) == 1440 and len(dev) == 480, "训练/dev组总数不符")
    train_path, dev_path = output / "train_all.location.jsonl", output / "synthetic_dev.location.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(dev_path, dev)
    selected, quotas = balanced_selection(train)
    selected_keys = [train[index]["frame_key"] for index in selected]
    selection = {"schema": "futsalmot_pose_location_synthetic_selection_v1", "status": "candidate_selected_before_training",
                 "training_authorized": False, "train_seeds": TRAIN_SEEDS, "synthetic_dev_seeds": [1102],
                 "selection_salt": SALT, "selection_rule": "seed_then_camera_balanced_independent_SHA_with_hashed_remainder_ties",
                 "seed_camera_quotas": quotas, "selected_synthetic_indices": selected, "selected_synthetic_frame_keys": selected_keys,
                 "source_jsonl": {"path": str(train_path), "sha256": common.sha(train_path), "images": 1440}}
    selection_path = output / "selection_manifest.json"
    common.dump_new(selection_path, selection)
    schedules = []
    for seed in TRAINING_SEEDS:
        rows = paired_schedule([frame["image_id"] for frame in real], selected, selected_keys, seed)
        shared = Counter(index for row in rows for index in row["shared_real_indices"])
        extra = Counter(row["extra_real_index"] for row in rows)
        require(shared == Counter({index: 3 for index in range(1000)}) and extra == Counter({index: 1 for index in range(1000)})
                and Counter(row["synthetic_index"] for row in rows) == Counter(selected), "采样表真实曝光或合成覆盖计数不符")
        path = output / "schedules" / f"seed_{seed}.jsonl"
        write_jsonl(path, rows)
        for arm in ("R", "RS25"):
            schedules.append({"seed": seed, "arm": arm, "path": str(path), "sha256": common.sha(path), "updates": 1000, "batch": 4})
    real_ref = {"path": str(real_path), "sha256": real_sha, "images": 1000}
    train_ref = {"path": str(train_path), "sha256": common.sha(train_path), "images": 1440}
    dev_ref = {"path": str(dev_path), "sha256": common.sha(dev_path), "images": 480}
    manifest = {"schema": "futsalmot_pose_location_paired_schedule_manifest_v1", "protocol_sha256": None,
                "candidate_protocol_sha256": PROTOCOL_SHA, "training_authorized": False, "batch_id": "location_pilot_4seed_a01",
                "real_jsonl": real_ref, "synthetic_jsonl": train_ref, "synthetic_dev_jsonl": dev_ref,
                "selection_manifest": {"path": str(selection_path), "sha256": common.sha(selection_path)},
                "selected_synthetic_indices": selected, "selected_synthetic_frame_keys": selected_keys, "schedules": schedules,
                "schedule_index_semantics": "real_indices_address_1000_R_JSONL_synthetic_indices_address_full1440_S_JSONL",
                "real_exposures_R": 4000, "real_exposures_RS25": 3000, "synthetic_exposures_RS25": 1000,
                "media_policy": MEDIA_POLICY, "source_diagnostic_flags_preserved": True}
    manifest_path = output / "schedule_manifest.json"
    common.dump_new(manifest_path, manifest)
    candidate = {"schema": "futsalmot_pose17_location_training_candidate_v1", "status": "candidate_text_evidence_adapted_requires_root_location_only_admission",
                 "training_authorized": False, "batch_id": "location_pilot_4seed_a01", "driver_evidence": list(source_reader.files.values()),
                 "candidate_protocol_sha256": PROTOCOL_SHA, "real_jsonl": real_ref, "synthetic_jsonl": train_ref, "synthetic_dev_jsonl": dev_ref,
                 "schedule_manifest": {"path": str(manifest_path), "sha256": common.sha(manifest_path)},
                 "selection_manifest": {"path": str(selection_path), "sha256": common.sha(selection_path)}, "groups": groups,
                 "statistics": {"real_train": summarize_frames(real), "synthetic_train_all": summarize_frames(train),
                                "synthetic_train_selected": summarize_frames([train[index] for index in selected]), "synthetic_dev": summarize_frames(dev)},
                 "media_policy": MEDIA_POLICY, "image_sha256_available_for_synthetic": False, "media_pixels_or_hashes_read": False,
                 "source_diagnostic_flags_preserved": True, "MRQ_sync_verified": False, "formal_pose_release": False,
                 "location_only_root_admission_required": str(PROJECT / "Saved/Pose17Development/location_training_admission_a01.json"),
                 "cannot_authorize_visibility_supervision": True, "cannot_authorize_formal_GT_release": True,
                 "GPU_or_UE_called": False, "test_SN116_SN187_read": False}
    common.dump_new(output / "candidate_receipt.json", candidate)
    print(json.dumps({"output": str(output), "status": candidate["status"], "schedule_manifest_sha256": common.sha(manifest_path)}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    prepare(args.output)


if __name__ == "__main__":
    main()
