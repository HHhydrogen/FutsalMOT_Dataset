"""官方COCO小规模位置监督准备：CPU下载/选择/JSONL，不启动训练或读取SNMOT。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE / "src"))
from grf_ue_bridge.pose_location_supervision_v2 import coco_person_location

VERSION = "official_coco2017_location_subset_candidate_v1"
DEFAULT_ROOT = Path("D:/projects/FutsalMOT_Research/pose/coco2017_location_a01")
OFFICIAL_DOWNLOAD_PAGE = "https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/download.htm"
ANNOTATION_OBJECT = "annotations/annotations_trainval2017.zip"
# 原站域名HTTPS证书不匹配；同一官方S3桶的path-style HTTPS已核实，保留官网原始HTTP链接回退。
# 不关闭TLS校验，不采用第三方镜像；每个成功对象记录实际传输URL和本地SHA。
HOSTS = ("https://s3.amazonaws.com/images.cocodataset.org/", "http://images.cocodataset.org/")
SALT = "futsalmot:coco17:location:subset:v1"
COCO17 = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow",
          "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
MEMBERS = tuple("annotations/" + kind + "_" + split + ".json"
                for split in ("train2017", "val2017") for kind in ("instances", "person_keypoints"))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def plain(path):
    path = Path(path).absolute()
    for item in (path, *path.parents):
        if item.exists():
            require(not item.is_symlink() and not (getattr(item.stat(), "st_file_attributes", 0) & 0x400), "拒绝联接/符号链接: " + str(item))
    return path


def root_path(value):
    root = plain(value)
    allowed = plain("D:/projects/FutsalMOT_Research/pose")
    require(root != allowed and allowed in root.parents, "只允许Research/pose下独立数据目录")
    require(not any(part.lower() in {"soccernet_tracking", "v1_no_numbers_4k"} for part in root.parts), "拒绝正式v1或足球封存集")
    return root


def dump_new(path, value):
    path = plain(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def fetch_object(object_name, destination, limit_bytes, opener=urllib.request.urlopen):
    """仅COCO官方桶对象，保留失败partial；成功后才发布目标文件。"""
    require(re.fullmatch(r"(?:annotations/annotations_trainval2017\.zip|(?:train2017|val2017)/\d{12}\.jpg)", object_name) is not None,
            "拒绝非官方固定对象或任意URL")
    destination = plain(destination)
    require(not destination.exists(), "下载目标已存在，不覆盖")
    partial = destination.with_suffix(destination.suffix + ".part")
    require(not partial.exists(), "先前partial保留，需显式新目录或人工核对，不隐式覆盖")
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = None
    errors = []
    for host in HOSTS:
        url = host + object_name
        try:
            response = opener(urllib.request.Request(url, headers={"User-Agent": "FutsalMOT-pose-data-preparation/1"}), timeout=45)
            require(response.status == 200, "只接受完整对象200响应")
            break
        except (urllib.error.URLError, TimeoutError) as error:
            errors.append(repr(error))
    if response is None:
        raise RuntimeError("官方COCO传输失败: " + "; ".join(errors))
    total = 0
    digest = hashlib.sha256()
    with response:
        expected = int(response.headers.get("Content-Length", "0"))
        require(0 <= expected <= limit_bytes, "对象超出下载预算")
        with partial.open("xb") as handle:
            while True:
                block = response.read(min(1024 * 1024, limit_bytes - total + 1))
                if not block:
                    break
                total += len(block)
                require(total <= limit_bytes, "下载超过固定对象预算，保留partial")
                handle.write(block)
                digest.update(block)
        require(total > 0 and (not expected or total == expected), "传输截断，保留partial")
        final_url = response.geturl()
        require(any(final_url == host + object_name for host in HOSTS), "重定向脱离固定官方对象")
    partial.rename(destination)
    return {"path": str(destination), "object": object_name, "transport_url": final_url, "bytes": total,
            "sha256": digest.hexdigest(), "publisher_sha256_available": False, "official_bucket": "images.cocodataset.org"}


def prepare_annotations(root, local_archive=None):
    receipt_path = root / "annotations_receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        for member in receipt["members"]:
            require(sha(plain(member["path"])) == member["sha256"], "已缓存注释SHA变化")
        return receipt
    archive = root / "cache/annotations_trainval2017.zip"
    require(local_archive is not None or not archive.exists(), "无成功收据的缓存ZIP保留；需显式local-archive核对")
    if local_archive:
        source = plain(local_archive)
        require(zipfile.is_zipfile(source), "本机缓存不是ZIP")
        archive_record = {"path": str(source), "sha256": sha(source), "reused_local_archive": True,
                          "official_origin_unverified_for_supplied_local_file": True}
        archive = source
    else:
        archive_record = fetch_object(ANNOTATION_OBJECT, archive, 300_000_000)
    members = []
    with zipfile.ZipFile(archive) as zipped:
        require(len(set(zipped.namelist())) == len(zipped.namelist()), "ZIP成员重复")
        require(set(MEMBERS) <= set(zipped.namelist()), "缺官方instances或keypoints成员")
        for name in MEMBERS:
            info = zipped.getinfo(name)
            require(info.file_size < 700_000_000, "注释成员大小异常")
            destination = plain(root / name)
            require(not destination.exists(), "注释文件已存在，不覆盖partial")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open(name) as src, destination.open("xb") as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
            members.append({"path": str(destination), "zip_member": name, "zip_crc32": info.CRC, "size_bytes": destination.stat().st_size,
                            "sha256": sha(destination), "zip_member_crc_checked_by_zipfile": True})
    receipt = {"schema": VERSION, "action": "official_annotations_prepared", "official_download_page": OFFICIAL_DOWNLOAD_PAGE,
               "archive": archive_record, "members": members, "images_downloaded": 0, "training_started": False}
    dump_new(receipt_path, receipt)
    return receipt


def unique_index(rows, key):
    index = {}
    for row in rows:
        value = row[key]
        require(type(value) is int and value not in index, "注释或图像ID重复/非法")
        index[value] = row
    return index


def choose_subset(instances, keypoints, split, count, seed):
    """先整图排crowd，再哈希排序；保留选图全部noncrowd人物，包括全缺点。"""
    require(split in {"train2017", "val2017"} and type(count) is int and count > 0 and type(seed) is int, "选择参数非法")
    person_categories = [category for category in keypoints["categories"] if category.get("id") == 1]
    require(len(person_categories) == 1 and person_categories[0].get("name") == "person"
            and person_categories[0].get("keypoints") == COCO17, "需要官方COCO17 person顺序")
    images = unique_index(instances["images"], "id")
    kp_images = unique_index(keypoints["images"], "id")
    require(set(images) == set(kp_images), "instances/keypoints图像集合不同")
    for image_id, image in images.items():
        other = kp_images[image_id]
        require(all(image[field] == other[field] for field in ("width", "height", "file_name")), "图像元数据不一致")
        require(image["file_name"] == f"{image_id:012d}.jpg", "官方图像文件名与ID不符")
    persons = [ann for ann in instances["annotations"] if ann["category_id"] == 1]
    person_index = unique_index(persons, "id")
    kp_index = unique_index(keypoints["annotations"], "id")
    require(set(kp_index) == set(person_index), "全部person ann_id严格联结失败，不假造缺失注释")
    groups = {image_id: [] for image_id in images}
    for annotation_id, ann in person_index.items():
        kp = kp_index[annotation_id]
        require(all(ann[field] == kp[field] for field in ("image_id", "category_id", "bbox", "iscrowd")), "person框/身份/crowd联结不一致")
        require(kp["image_id"] in groups, "注释引用不存在的图像")
        groups[kp["image_id"]].append(kp)
    excluded = {"no_person": 0, "person_crowd_image": 0, "invalid_person_box_image": 0, "no_in_frame_labeled_keypoint": 0}
    invalid_box_examples = []
    eligible = []
    for image_id, anns in groups.items():
        if not anns:
            excluded["no_person"] += 1
            continue
        if any(ann["iscrowd"] for ann in anns):
            excluded["person_crowd_image"] += 1
            continue
        image = images[image_id]
        invalid = []
        for ann in anns:
            box = ann["bbox"]
            valid = isinstance(box, list) and len(box) == 4 and all(type(v) in (int, float) and math.isfinite(v) for v in box)
            if valid:
                x, y, w, h = box
                valid = w > 0 and h > 0 and min(image["width"], x + w) > max(0, x) and min(image["height"], y + h) > max(0, y)
            if not valid:
                invalid.append({"annotation_id": ann["id"], "bbox": box, "num_keypoints": ann["num_keypoints"]})
        if invalid:
            # 整图在排序前排除；绝不只删这个person后把它当背景，也不修补原bbox。
            excluded["invalid_person_box_image"] += 1
            if len(invalid_box_examples) < 8:
                invalid_box_examples.append({"image_id": image_id, "annotations": invalid})
            continue
        converted = [coco_person_location(ann, image["width"], image["height"]) for ann in anns]
        if not any(any(ann["location_valid"]) for ann in converted):
            excluded["no_in_frame_labeled_keypoint"] += 1
            continue
        rank = canonical_hash({"salt": SALT, "selection_seed": seed, "split": split, "image_id": image_id})
        eligible.append((rank, image_id))
    require(len(eligible) >= count, "元数据资格样本不足，不静默缩减目标")
    selected = sorted(eligible)[:count]
    ids = [image_id for _, image_id in selected]
    subset = {"info": keypoints.get("info", {}), "licenses": keypoints.get("licenses", []),
              "categories": keypoints["categories"], "images": [images[i] for i in ids],
              "annotations": [ann for i in ids for ann in sorted(groups[i], key=lambda x: x["id"])],
              "source_instance_area_by_annotation_id": {str(ann["id"]): person_index[ann["id"]]["area"]
                                                        for i in ids for ann in groups[i]}}
    selected_stats = {"images": count, "person_boxes": len(subset["annotations"]),
                      "all_missing_pose_person_boxes_retained": sum(ann["num_keypoints"] == 0 for ann in subset["annotations"]),
                      "source_labeled_keypoints": sum(ann["num_keypoints"] for ann in subset["annotations"])}
    return subset, {"source_image_count": len(images), "eligible_image_count": len(eligible), "excluded_image_counts": excluded,
                    "selected": selected_stats, "image_ids_in_sha_rank_order": ids,
                    "selection_ranks_sha256": canonical_hash(selected), "selection_seed": seed, "selection_salt": SALT,
                    "invalid_box_examples": invalid_box_examples,
                    "join_evidence": {"person_id_bbox_category_image_crowd_exact": True,
                                      "area_exact_disagreement_count_all_source_people": sum(person_index[i]["area"] != kp_index[i]["area"] for i in person_index),
                                      "area_policy": "retain_keypoints_original_area_for_evaluation_preserve_instances_original_area_separately_no_epsilon"}}


def select(root, train_count=1000, val_count=200, seed=20261024):
    target = root / "subset"
    require(not target.exists(), "选择输出已存在，不覆盖或根据结果重抽")
    receipt = json.loads(plain(root / "annotations_receipt.json").read_text(encoding="utf-8"))
    provenance = {row["zip_member"]: row for row in receipt["members"]}
    result = {"schema": VERSION, "status": "candidate_ids_selected_not_training_frozen", "training_authorized": False,
              "official_download_page": OFFICIAL_DOWNLOAD_PAGE, "selection_seed": seed, "selection_salt": SALT,
              "crowd_policy": "exclude_whole_image_before_ranking_if_any_person_iscrowd",
              "bbox_policy": "keep_all_annotated_noncrowd_person_bboxes_including_num_keypoints_zero",
              "pretraining_disclosure": "initial_checkpoint_was_COCO_pretrained_not_total_annotation_budget_of_1000",
              "val_role": "public_pose_development_only_not_SoccerNet_final_test", "splits": {}}
    for split, count in (("train2017", train_count), ("val2017", val_count)):
        names = ["annotations/" + kind + "_" + split + ".json" for kind in ("instances", "person_keypoints")]
        for name in names:
            require(sha(plain(provenance[name]["path"])) == provenance[name]["sha256"], "源注释SHA变化")
        instances = json.loads(Path(provenance[names[0]]["path"]).read_text(encoding="utf-8"))
        keypoints = json.loads(Path(provenance[names[1]]["path"]).read_text(encoding="utf-8"))
        subset, stats = choose_subset(instances, keypoints, split, count, seed)
        del instances, keypoints
        subset_path = target / (split + "_person_annotations.json")
        dump_new(subset_path, subset)
        stats.update(annotation_subset_path=str(subset_path), annotation_subset_sha256=sha(subset_path),
                     source_annotations=[provenance[name] for name in names], role="train" if split == "train2017" else "pose_dev_only")
        result["splits"][split] = stats
    require(not set(result["splits"]["train2017"]["image_ids_in_sha_rank_order"]) & set(result["splits"]["val2017"]["image_ids_in_sha_rank_order"]), "train/val ID相交")
    dump_new(target / "selection_manifest.json", result)
    return result


def download_selected(root, workers=4, reuse_images=None):
    require(type(workers) is int and 1 <= workers <= 4, "下载并发只允许1..4")
    subset_root = root / "subset"
    selection_path = plain(subset_root / "selection_manifest.json")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    require(selection.get("schema") == VERSION and selection.get("training_authorized") is False, "只处理本候选选择清单")
    receipt_path = root / "images_receipt.json"
    require(not receipt_path.exists(), "图片完成收据已存在，拒绝覆盖")
    reuse_root = plain(reuse_images) if reuse_images else None
    records = []
    work = []
    for split, stats in selection["splits"].items():
        source = plain(stats["annotation_subset_path"])
        require(sha(source) == stats["annotation_subset_sha256"], "冻结子注释变化")
        subset = json.loads(source.read_text(encoding="utf-8"))
        require([image["id"] for image in subset["images"]] == stats["image_ids_in_sha_rank_order"], "选择顺序变化")
        by_image = {}
        for ann in subset["annotations"]:
            by_image.setdefault(ann["image_id"], []).append(ann)
        for image in subset["images"]:
            work.append((split, image, by_image[image["id"]]))

    def one(item):
        from PIL import Image
        split, image, annotations = item
        destination = plain(root / "images" / split / image["file_name"])
        sidecar = destination.with_suffix(".jpg.source.json")
        if destination.exists():
            require(sidecar.is_file(), "既有图片无来源收据，不隐式接受")
            record = json.loads(sidecar.read_text(encoding="utf-8"))
            require(record["image_id"] == image["id"] and record["split"] == split and sha(destination) == record["sha256"], "已完成图片SHA/身份变化")
        elif reuse_root and (reuse_root / split / image["file_name"]).is_file():
            original = plain(reuse_root / split / image["file_name"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            with original.open("rb") as src, destination.open("xb") as dst:
                shutil.copyfileobj(src, dst)
            record = {"path": str(destination), "sha256": sha(destination), "bytes": destination.stat().st_size,
                      "reused_existing_local_image": str(original), "publisher_sha256_available": False}
        else:
            record = fetch_object(split + "/" + image["file_name"], destination, 20_000_000)
        with Image.open(destination) as bitmap:
            require(bitmap.format == "JPEG" and bitmap.size == (image["width"], image["height"]), "实际JPEG格式/尺寸与COCO不符")
            bitmap.verify()
        record.update(image_id=image["id"], split=split, original_coco_url=image.get("coco_url"), license_id=image.get("license"),
                      width=image["width"], height=image["height"])
        if not sidecar.exists():
            dump_new(sidecar, record)
        frame = {"schema": "separated_pose_location_frame_v1", "image_id": image["id"], "source": "coco_real",
                 "image_file": "../images/" + split + "/" + image["file_name"], "image_sha256": record["sha256"],
                 "width": image["width"], "height": image["height"], "bbox_coverage": "all_noncrowd_person_instances", "crowd_present": False,
                 "expected_person_annotation_ids": [ann["id"] for ann in annotations],
                 "instances": [coco_person_location(ann, image["width"], image["height"]) for ann in annotations]}
        return record, frame

    frames = {"train2017": [], "val2017": []}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, (record, frame) in enumerate(pool.map(one, work), 1):
            records.append(record)
            frames[record["split"]].append(frame)
            if index % 100 == 0:
                print(json.dumps({"downloaded_and_verified": index, "selected": len(work)}), flush=True)
    for split, rows in frames.items():
        path = subset_root / (split + ".location.jsonl")
        require(not path.exists(), "位置JSONL已存在，不覆盖")
        with path.open("x", encoding="utf-8") as handle:
            for frame in rows:
                handle.write(json.dumps(frame, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    result = {"schema": VERSION, "status": "candidate_images_and_location_jsonl_prepared_not_training_frozen", "training_started": False,
              "selection_manifest_sha256": sha(selection_path), "workers": workers, "images": records,
              "location_jsonl": [{"path": str(subset_root / (split + ".location.jsonl")), "sha256": sha(subset_root / (split + ".location.jsonl")),
                                  "images": len(rows), "person_boxes": sum(len(row["instances"]) for row in rows),
                                  "all_missing_people_retained": sum(not any(person["location_valid"]) for row in rows for person in row["instances"])}
                                 for split, rows in frames.items()]}
    dump_new(receipt_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("annotations", "select", "images"))
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--local-archive")
    parser.add_argument("--reuse-images")
    parser.add_argument("--train-count", type=int, default=1000)
    parser.add_argument("--val-count", type=int, default=200)
    parser.add_argument("--selection-seed", type=int, default=20261024)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root = root_path(args.root)
    if args.action == "annotations":
        result = prepare_annotations(root, args.local_archive)
    elif args.action == "select":
        result = select(root, args.train_count, args.val_count, args.selection_seed)
    else:
        result = download_selected(root, args.workers, args.reuse_images)
    print(json.dumps({"action": args.action, "root": str(root), "status": result.get("status", result.get("action")), "training_started": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
