"""仅从已落盘native JSON读取Pose17；不导入unreal或访问销毁的PIE对象。"""

import math
import re


def normalize_keys(value):
    if isinstance(value, list):
        return [normalize_keys(item) for item in value]
    if not isinstance(value, dict):
        return value
    output = {}
    for key, item in value.items():
        if len(key) > 1 and key[0] == "b" and key[1].isupper():
            key = key[1:]
        key = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", key)
        key = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key).lower()
        output[key] = normalize_keys(item)
    return output


def vector(value, size=3):
    names = "xyz"[:size]
    if isinstance(value, dict):
        values = [value[name] for name in names]
    elif isinstance(value, str):
        parsed = {axis.lower(): float(number) for axis, number in re.findall(r"([XYZxyz])=([-+0-9.eE]+)", value)}
        if set(parsed) != set(names):
            raise ValueError("不支持的native向量编码")
        values = [parsed[name] for name in names]
    else:
        values = list(value)
    if len(values) != size or any(type(item) not in (int, float) or not math.isfinite(item) for item in values):
        raise ValueError("native向量非有限或维度错误")
    return values


def normalize_native_receipt(raw):
    receipt = normalize_keys(raw)
    if receipt.get("schema") != "futsalmot_pose17_native_receipt_v1":
        raise ValueError("不支持的native收据schema")
    for row in receipt["records"]:
        sample = row.pop("sample")
        row.update(sample)
        row["world_m"] = [vector(point) for point in row.pop("world_points_m")]
        row["visibility_state"] = ["unknown"] * len(row["world_m"])
        for key in ("camera_world_location_m", "camera_forward", "camera_right", "camera_up"):
            if key in row:
                row[key] = vector(row[key])
        for key in ("projection_jitter", "off_center_projection_offset", "image_size"):
            if key in row:
                row[key] = vector(row[key], 2)
        if row.get("sample_succeeded") and (len(row["world_m"]) != 17 or len(row.get("keypoint_names", [])) != 17):
            raise ValueError("成功记录不是17点")
        if row.get("mrq_synchronization_verified"):
            raise ValueError("诊断native收据不得自行声明MRQ配对已验收")
    return receipt
