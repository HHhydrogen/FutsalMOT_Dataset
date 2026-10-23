"""官方COCO子集选择/下载的纯CPU测试；不下载网络数据或训练。"""
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import prepare_coco_pose_subset as module


def fixture():
    images = [{"id": i, "width": 100, "height": 200, "file_name": f"{i:012d}.jpg"} for i in range(1, 6)]
    def person(annotation_id, image_id, missing=False, crowd=False):
        return {"id": annotation_id, "image_id": image_id, "category_id": 1, "bbox": [5, 10, 70, 170], "area": 11900,
                "iscrowd": int(crowd), "keypoints": [0, 0, 0] * 17 if missing else [20, 30, 1] + [0, 0, 0] * 16,
                "num_keypoints": 0 if missing else 1}
    anns = [person(11, 1), person(12, 1, missing=True), person(21, 2), person(31, 3), person(32, 3, crowd=True), person(41, 4, missing=True)]
    kp = {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "person", "keypoints": module.COCO17}], "licenses": []}
    instances = copy.deepcopy(kp)
    for ann in instances["annotations"]:
        ann.pop("keypoints")
        ann.pop("num_keypoints")
    return instances, kp


class TestSelection(unittest.TestCase):
    def test_stable_id_rank_all_people_and_whole_crowd_image(self):
        instances, kp = fixture()
        original = copy.deepcopy(kp)
        subset, stats = module.choose_subset(instances, kp, "train2017", 2, 20261024)
        self.assertEqual({image["id"] for image in subset["images"]}, {1, 2})
        self.assertEqual({ann["id"] for ann in subset["annotations"]}, {11, 12, 21})
        self.assertEqual(stats["selected"]["all_missing_pose_person_boxes_retained"], 1)
        self.assertEqual(stats["excluded_image_counts"], {"no_person": 1, "person_crowd_image": 1, "invalid_person_box_image": 0, "no_in_frame_labeled_keypoint": 1})
        instances["images"].reverse()
        kp["images"].reverse()
        kp["annotations"].reverse()
        _, second = module.choose_subset(instances, kp, "train2017", 2, 20261024)
        self.assertEqual(stats, second)
        self.assertEqual(original["annotations"][0]["keypoints"], [20, 30, 1] + [0, 0, 0] * 16)

    def test_missing_person_join_is_not_silently_converted_to_background(self):
        instances, kp = fixture()
        kp["annotations"] = [ann for ann in kp["annotations"] if ann["id"] != 12]
        with self.assertRaises(ValueError):
            module.choose_subset(instances, kp, "train2017", 1, 1)

    def test_wrong_bbox_duplicate_id_or_joint_order_refused(self):
        for damage in ("bbox", "duplicate", "order"):
            instances, kp = fixture()
            if damage == "bbox":
                kp["annotations"][0]["bbox"][0] += 1
            elif damage == "duplicate":
                kp["annotations"].append(copy.deepcopy(kp["annotations"][0]))
            else:
                kp["categories"][0]["keypoints"] = list(reversed(module.COCO17))
            with self.assertRaises(ValueError):
                module.choose_subset(instances, kp, "train2017", 1, 1)

    def test_not_enough_eligible_images_refused(self):
        instances, kp = fixture()
        with self.assertRaises(ValueError):
            module.choose_subset(instances, kp, "train2017", 3, 1)

    def test_original_areas_preserved_without_epsilon_or_changing_keypoint_eval_area(self):
        instances, kp = fixture()
        instances["annotations"][0]["area"] = 11900.000000001
        subset, stats = module.choose_subset(instances, kp, "train2017", 2, 1)
        self.assertEqual(subset["source_instance_area_by_annotation_id"]["11"], 11900.000000001)
        self.assertEqual(next(row for row in subset["annotations"] if row["id"] == 11)["area"], 11900)
        self.assertEqual(stats["join_evidence"]["area_exact_disagreement_count_all_source_people"], 1)

    def test_degenerate_person_box_excludes_whole_image_before_ranking(self):
        instances, kp = fixture()
        for data in (instances, kp):
            next(row for row in data["annotations"] if row["id"] == 12)["bbox"][2] = 0
        subset, stats = module.choose_subset(instances, kp, "train2017", 1, 1)
        self.assertEqual([image["id"] for image in subset["images"]], [2])
        self.assertEqual(stats["excluded_image_counts"]["invalid_person_box_image"], 1)
        self.assertEqual(stats["invalid_box_examples"][0]["annotations"][0]["annotation_id"], 12)

    def test_visibility_one_is_valid_location_and_all_missing_box_remains(self):
        _, kp = fixture()
        converted = [module.coco_person_location(ann, 100, 200) for ann in kp["annotations"][:2]]
        self.assertTrue(converted[0]["location_valid"][0])
        self.assertEqual(converted[0]["visibility_state"][0], "occluded")
        self.assertFalse(any(converted[1]["location_valid"]))
        self.assertTrue(converted[1]["detection_target_retained"])


class FakeResponse(io.BytesIO):
    status = 200
    def __init__(self, data, url, expected=None):
        super().__init__(data)
        self.url = url
        self.headers = {"Content-Length": str(len(data) if expected is None else expected)}
    def geturl(self):
        return self.url


class TestDownload(unittest.TestCase):
    def test_exact_official_object_exclusive_download_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "image.jpg"
            def opener(request, timeout):
                self.assertEqual(request.full_url, module.HOSTS[0] + "train2017/000000000001.jpg")
                return FakeResponse(b"abc", request.full_url)
            result = module.fetch_object("train2017/000000000001.jpg", destination, 10, opener)
            self.assertEqual(destination.read_bytes(), b"abc")
            self.assertEqual(result["sha256"], module.sha(destination))
            with self.assertRaises(ValueError):
                module.fetch_object("train2017/000000000001.jpg", destination, 10, opener)

    def test_arbitrary_url_or_path_refused_without_request(self):
        for value in ("https://example.com/a", "../train2017/x.jpg", "test2017/000000000001.jpg", "train2017.zip"):
            with self.assertRaises(ValueError):
                module.fetch_object(value, Path("unused.jpg"), 10, lambda *args, **kwargs: self.fail("不能发请求"))

    def test_truncation_preserves_partial_and_does_not_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "image.jpg"
            with self.assertRaises(ValueError):
                module.fetch_object("train2017/000000000001.jpg", destination, 10,
                                    lambda request, timeout: FakeResponse(b"abc", request.full_url, expected=5))
            self.assertFalse(destination.exists())
            self.assertTrue(destination.with_suffix(".jpg.part").exists())

    def test_only_four_fixed_archive_members_extracted_with_crc_and_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "source.zip"
            with zipfile.ZipFile(archive, "w") as zipped:
                for member in module.MEMBERS:
                    zipped.writestr(member, "{}")
                zipped.writestr("../../not_extracted.txt", "no")
            receipt = module.prepare_annotations(root / "output", archive)
            self.assertEqual(len(receipt["members"]), 4)
            self.assertTrue(all(row["zip_member_crc_checked_by_zipfile"] for row in receipt["members"]))
            self.assertEqual(module.prepare_annotations(root / "output"), receipt)


if __name__ == "__main__":
    unittest.main()
