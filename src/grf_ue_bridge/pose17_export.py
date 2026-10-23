"""v2 COCO17 导出准备：严格帧契约、可见实例 mask 与不确定可见性。

不读取或修改 v1 标注；不把历史文件名 coco17_3d 当成真实 17 点。
该模块不证明 UE 采样、关键点语义或渲染对齐，发布前仍需真实 UE 验收。
"""

import argparse
import copy
import json
import math
from pathlib import Path, PurePosixPath


COCO17_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
FLIP_INDEX = (0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15)
SKELETON = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
            [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11],
            [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
STATES = {"visible", "occluded", "outside", "behind_camera", "missing", "unknown"}
# 这些名称只允许提供证据的调用方使用；本模块不能验证证据本身。
KNOWN_METHODS = {"surface_depth_test", "manual_verified"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _vector(value, length):
    return (isinstance(value, (list, tuple)) and len(value) == length
            and all(type(v) in (int, float) and math.isfinite(v) for v in value))


def _relative_path(value):
    _require(isinstance(value, str) and value and "\\" not in value,
             "媒体路径必须是非空 POSIX 相对路径")
    path = PurePosixPath(value)
    _require(not path.is_absolute() and ".." not in path.parts and ":" not in value,
             "媒体路径不得越界")


def project_landmarks(world_points, camera):
    """将已采样世界米坐标投影为像素，未知遮挡始终保持 unknown。

    camera 与现有 camera.json 的 intrinsics/extrinsics 字段一致。拒绝非正交
    坐标基，避免将 OpenCV 右手外参静默当作本项目 UE 外参使用。
    """
    intr, extr = camera["intrinsics"], camera["extrinsics"]
    width, height = intr["width"], intr["height"]
    _require(_integer(width, 1) and _integer(height, 1), "相机尺寸无效")
    _require(_vector([intr[k] for k in ("fx", "fy", "cx", "cy")], 4)
             and intr["fx"] > 0 and intr["fy"] > 0, "相机内参无效")
    location = extr["world_location_m"]
    basis = [extr[k] for k in ("forward", "right", "up")]
    _require(_vector(location, 3) and all(_vector(v, 3) for v in basis), "相机外参无效")
    for i in range(3):
        for j in range(3):
            dot = sum(a * b for a, b in zip(basis[i], basis[j]))
            _require(abs(dot - int(i == j)) < 1e-5, "相机基必须正交归一")
    output = []
    for point in world_points:
        row = copy.deepcopy(point)
        world = row.get("world_m")
        row.update(uv_px=None, visibility_state="missing", visibility_method="not_measured")
        if world is not None:
            _require(_vector(world, 3), "世界点必须为有限米坐标")
            delta = [a - b for a, b in zip(world, location)]
            depth, right, up = [sum(a * b for a, b in zip(axis, delta)) for axis in basis]
            row["camera_forward_depth_m"] = depth
            if depth <= 0:
                row["visibility_state"] = "behind_camera"
            else:
                u, v = intr["cx"] + intr["fx"] * right / depth, intr["cy"] - intr["fy"] * up / depth
                row.update(uv_px=[u, v], visibility_state=("unknown" if 0 <= u < width and 0 <= v < height else "outside"))
        output.append(row)
    return output


def validate_capture(frame):
    """校验单帧，拒绝缺少同步身份或将 mask 邻域当作确定可见性。"""
    _require(frame.get("schema") == "futsalmot_pose17_capture_v2", "需要 v2 capture schema")
    _require(frame.get("keypoint_names") == list(COCO17_NAMES), "必须显式使用 COCO17 固定顺序")
    for name in ("seed", "frame_index", "source_frame_index", "sequence_frame_index"):
        _require(_integer(frame.get(name)), name + " 必须为非负整数")
    for name in ("clip_id", "camera_id", "capture_id", "render_sample_id", "camera_calibration_ref"):
        _require(isinstance(frame.get(name), str) and frame[name], name + " 缺失")
    _require(frame["capture_id"] == frame["render_sample_id"], "Pose/RGB/mask 采样身份不一致")
    _require(frame.get("world_coordinates") == "ue_left_handed_x_forward_y_right_z_up_m",
             "必须声明 UE 左手系世界米坐标")
    _require(_vector([frame.get("time_seconds")], 1), "时间必须为有限秒数")
    for name in ("width", "height"):
        _require(_integer(frame.get(name), 1), name + " 必须为正整数")
    _relative_path(frame.get("file_name"))
    _relative_path(frame.get("mask_file_name"))
    _require(isinstance(frame.get("objects"), list), "objects 必须是列表")
    excluded = frame.get("excluded_mask_ids", [])
    _require(isinstance(excluded, list) and all(_integer(v, 1) for v in excluded), "排除 mask ID 必须显式列正整数")
    seen = set()
    for obj in frame["objects"]:
        entity = obj.get("entity_id")
        _require(entity in {s + str(i) for s in "LR" for i in range(5)}, "球员实体必须为 L0..R4")
        expected = int(entity[1]) + (1 if entity[0] == "L" else 6)
        _require(obj.get("track_id") == expected and type(obj.get("track_id")) is int,
                 "track_id 必须固定 L0..L4=1..5，R0..R4=6..10")
        _require(_integer(obj.get("mask_id"), 1), "mask_id 必须为正整数")
        _require(entity not in seen, "重复球员实体")
        seen.add(entity)
        points = obj.get("keypoints")
        _require(isinstance(points, list) and len(points) == 17, "必须有 17 点，不可填充头中心")
        for point in points:
            state = point.get("visibility_state")
            _require(state in STATES, "未知可见性枚举")
            _require(isinstance(point.get("source_method"), str) and point["source_method"], "缺少点来源")
            world, uv = point.get("world_m"), point.get("uv_px")
            _require(world is None or _vector(world, 3), "world_m 必须为有限三维米坐标或 null")
            _require(uv is None or _vector(uv, 2), "uv_px 必须为有限二维像素或 null")
            if state in {"visible", "occluded"}:
                _require(_vector(world, 3) and _vector(uv, 2), "已知可见性需要有效 3D/2D")
                _require(0 <= uv[0] < frame["width"] and 0 <= uv[1] < frame["height"], "已标注点必须在图像内")
                _require(point.get("visibility_method") in KNOWN_METHODS and point.get("visibility_evidence_ref"),
                         "已知可见性需要表面深度或人工验收证据；mask 邻域不能证明自遮挡")
            else:
                _require(isinstance(point.get("visibility_method"), str) and point["visibility_method"],
                         "unknown 也必须记录可见性方法")
    _require(len({o["mask_id"] for o in frame["objects"]}) == len(frame["objects"]), "mask_id 重复")
    _require(not set(excluded) & {o["mask_id"] for o in frame["objects"]}, "不能排除球员 mask ID")


def binary_mask_geometry(mask, mask_id):
    """从二维实例 ID 计算像素面积、半开 xywh bbox、COCO 列优先未压缩 RLE。"""
    height = len(mask)
    _require(height > 0 and len(mask[0]) > 0, "mask 不能为空")
    width = len(mask[0])
    _require(all(len(row) == width for row in mask), "mask 行宽不一致")
    counts, previous, run, area = [], 0, 0, 0
    left, top, right, bottom = width, height, -1, -1
    for x in range(width):
        for y in range(height):
            value = int(mask[y][x] == mask_id)
            if value:
                area += 1
                left, top, right, bottom = min(left, x), min(top, y), max(right, x), max(bottom, y)
            if value == previous:
                run += 1
            else:
                counts.append(run)
                run, previous = 1, value
    counts.append(run)
    bbox = [left, top, right - left + 1, bottom - top + 1] if area else [0, 0, 0, 0]
    return area, bbox, {"size": [height, width], "counts": counts}


def validate_capture_coverage(frames, expected_cameras, expected_frame_indices):
    """单 seed/clip 小批次完整性门禁：逐机位逐帧十实体，不替代真实采样验收。"""
    cameras, indices = set(expected_cameras), set(expected_frame_indices)
    _require(cameras and indices and len(cameras) == len(expected_cameras)
             and len(indices) == len(expected_frame_indices), "期望帧和机位不能为空或重复")
    _require(all(_integer(index) for index in indices), "期望帧号无效")
    found, group = set(), None
    for frame in frames:
        validate_capture(frame)
        identity = (frame["seed"], frame["clip_id"])
        if group is None:
            group = identity
        _require(group == identity, "coverage 单次只允许一个 seed/clip")
        key = (frame["camera_id"], frame["frame_index"])
        _require(key not in found, "重复机位帧")
        found.add(key)
        _require({obj["entity_id"] for obj in frame["objects"]}
                 == {s + str(i) for s in "LR" for i in range(5)}, "逐帧必须覆盖十名球员，缺失关节点保留 null")
    _require(found == {(camera, index) for camera in cameras for index in indices}, "帧集合与计划不一致")


def build_coco(frames, masks):
    """以 (seed, clip, camera, frame) 排序产生稳定 ID，返回 COCO 与无损 Pose sidecar。

    masks 与 frames 一一对应。输出不接触磁盘；空可见 mask 的球员只保留 sidecar。
    """
    _require(len(frames) == len(masks) and len(frames) > 0, "帧和 mask 数量必须一致且非空")
    paired = sorted(zip(frames, masks), key=lambda pair: (
        pair[0]["seed"], pair[0]["clip_id"], pair[0]["camera_id"], pair[0]["frame_index"]))
    images, annotations, sidecar, identities, worlds = [], [], [], set(), {}
    for image_id, (frame, mask) in enumerate(paired, 1):
        validate_capture(frame)
        identity = (frame["seed"], frame["clip_id"], frame["camera_id"], frame["frame_index"])
        _require(identity not in identities, "重复相机帧")
        identities.add(identity)
        _require(len(mask) == frame["height"] and all(len(row) == frame["width"] for row in mask), "mask 分辨率与 RGB 不一致")
        allowed = {0} | set(frame.get("excluded_mask_ids", [])) | {obj["mask_id"] for obj in frame["objects"]}
        _require(all(_integer(pixel) and pixel in allowed for row in mask for pixel in row), "mask 含未映射 ID 或非整数")
        image = {k: frame[k] for k in ("file_name", "width", "height", "seed", "clip_id", "camera_id", "frame_index",
                                      "source_frame_index", "sequence_frame_index", "time_seconds", "capture_id",
                                      "camera_calibration_ref", "mask_file_name")}
        image["id"] = image_id
        images.append(image)
        for obj in sorted(frame["objects"], key=lambda o: o["track_id"]):
            world_key = (frame["seed"], frame["clip_id"], frame["frame_index"], obj["entity_id"])
            shared = (frame["source_frame_index"], frame["sequence_frame_index"], frame["time_seconds"],
                      [p["world_m"] for p in obj["keypoints"]])
            _require(world_key not in worlds or worlds[world_key] == shared,
                     "同一世界帧跨机位 Pose 或时间不一致，不得复用首相机世界姿态")
            worlds[world_key] = shared
            area, bbox, rle = binary_mask_geometry(mask, obj["mask_id"])
            entry = copy.deepcopy(obj)
            entry.update(image_id=image_id, bbox_visible_xywh=bbox, visible_mask_area=area)
            sidecar.append(entry)
            if not area:
                continue
            triples = []
            for point in obj["keypoints"]:
                state = point["visibility_state"]
                triples.extend([*point["uv_px"], 2 if state == "visible" else 1]
                               if state in {"visible", "occluded"} else [0, 0, 0])
            annotations.append({"id": len(annotations) + 1, "image_id": image_id,
                                "category_id": 1, "track_id": obj["track_id"], "entity_id": obj["entity_id"],
                                "mask_id": obj["mask_id"], "bbox": bbox, "area": area,
                                "iscrowd": 0, "segmentation": rle, "keypoints": triples,
                                "num_keypoints": sum(v > 0 for v in triples[2::3])})
    coco = {"info": {"schema": "futsalmot_pose17_coco_v2", "status": "draft_requires_ue_acceptance",
                     "bbox_semantics": "visible_instance_mask_half_open_xywh",
                     "area_semantics": "visible_instance_pixel_count", "unknown_visibility": "coco_v0_sidecar_preserved"},
            "images": images, "annotations": annotations,
            "categories": [{"id": 1, "name": "person", "supercategory": "person",
                            "keypoints": list(COCO17_NAMES), "skeleton": SKELETON}]}
    validate_coco(coco)
    return coco, {"schema": "futsalmot_pose17_sidecar_v2", "objects": sidecar}


def validate_coco(coco):
    """严格检查生成 JSON 的 ID、帧关系、RLE 面积与 bbox、COCO17 数量。"""
    _require(coco["categories"][0]["keypoints"] == list(COCO17_NAMES), "COCO17 类别顺序不符")
    images = {im["id"]: im for im in coco["images"]}
    _require(len(images) == len(coco["images"]), "image_id 重复")
    ids = set()
    for ann in coco["annotations"]:
        _require(ann["id"] not in ids, "annotation_id 重复")
        ids.add(ann["id"])
        _require(ann["image_id"] in images and ann["category_id"] == 1 and ann["iscrowd"] == 0, "标注引用错误")
        image = images[ann["image_id"]]
        width, height = image["width"], image["height"]
        rle = ann["segmentation"]
        _require(rle["size"] == [height, width], "RLE 尺寸错误")
        counts = rle["counts"]
        _require(isinstance(counts, list) and counts and all(_integer(c) for c in counts), "RLE runs 错误")
        _require(sum(counts) == width * height and sum(counts[1::2]) == ann["area"] > 0, "RLE 面积错误")
        left, top, right, bottom, offset = width, height, -1, -1, 0
        for index, count in enumerate(counts):
            if index % 2 and count:
                first_x, first_y = divmod(offset, height)
                last_x, last_y = divmod(offset + count - 1, height)
                left, right = min(left, first_x), max(right, last_x)
                if first_x != last_x:
                    top, bottom = 0, height - 1
                else:
                    top, bottom = min(top, first_y), max(bottom, last_y)
            offset += count
        _require(ann["bbox"] == [left, top, right - left + 1, bottom - top + 1], "bbox 与 mask 不符")
        values = ann["keypoints"]
        _require(len(values) == 51 and all(type(v) in (int, float) and math.isfinite(v) for v in values), "关键点必须为 51 个有限数")
        for x, y, visibility in zip(values[::3], values[1::3], values[2::3]):
            _require(visibility in (0, 1, 2), "COCO visibility 错误")
            _require((visibility == 0 and x == y == 0) or (visibility > 0 and 0 <= x < width and 0 <= y < height), "关键点坐标越界")
        _require(ann["num_keypoints"] == sum(v > 0 for v in values[2::3]), "num_keypoints 错误")


def main():
    """从独立 v2 capture bundle 导出新目录；拒绝覆盖或写入正式 v1。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    from PIL import Image
    frames, masks = bundle["frames"], []
    for frame in frames:
        _relative_path(frame["mask_file_name"])
        with Image.open(args.bundle.parent / frame["mask_file_name"]) as image:
            _require(image.mode in ("I", "I;16", "L"), "需要单通道整数实例 mask")
            data = list(image.getdata())
            masks.append([data[y * image.width:(y + 1) * image.width] for y in range(image.height)])
    coco, sidecar = build_coco(frames, masks)
    _require("v1_no_numbers_4k" not in {p.lower() for p in args.output.resolve().parts}, "禁止写入 v1 正式数据")
    args.output.mkdir(parents=True, exist_ok=False)
    for name, data in (("person_keypoints.json", coco), ("pose17_sidecar.json", sidecar)):
        (args.output / name).write_text(json.dumps(data, ensure_ascii=False, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
