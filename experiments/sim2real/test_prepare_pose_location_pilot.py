"""位置试产适配/采样的CPU小核心测试，不读取媒体、不训练。"""
from collections import Counter
import random
import unittest

import prepare_pose_location_pilot as module


def frames():
    return [{"root_seed": seed, "camera_id": camera, "frame_key": f"seed_{seed:08d}:{camera}:source{100+index:06d}"}
            for seed in module.TRAIN_SEEDS for camera in module.CAMERAS for index in range(120)]


class TestSampling(unittest.TestCase):
    def test_balanced_deterministic_selection_independent_of_order_and_global_rng(self):
        rows = frames()
        random.seed(123)
        old_state = random.getstate()
        indices, quotas = module.balanced_selection(rows)
        keys = [rows[index]["frame_key"] for index in indices]
        self.assertEqual(len(indices), len(set(indices)))
        self.assertEqual(len(indices), 1000)
        self.assertEqual(sorted(Counter(rows[index]["root_seed"] for index in indices).values()), [333, 333, 334])
        self.assertTrue(all(quota in (83, 84) for value in quotas.values() for quota in value.values()))
        reversed_rows = list(reversed(rows))
        reversed_indices, second = module.balanced_selection(reversed_rows)
        self.assertEqual(keys, [reversed_rows[index]["frame_key"] for index in reversed_indices])
        self.assertEqual(quotas, second)
        self.assertEqual(old_state, random.getstate())

    def test_dev_seed_missing_frame_or_duplicate_key_refused(self):
        for damage in ("dev", "gap", "duplicate"):
            rows = frames()
            if damage == "dev":
                rows[0]["root_seed"] = 1102
            elif damage == "gap":
                rows.pop()
            else:
                rows[0]["frame_key"] = rows[1]["frame_key"]
            with self.assertRaises(ValueError):
                module.balanced_selection(rows)

    def test_three_seed_exact_paired_real_exposure_and_sparse_S_indices(self):
        rows = frames()
        selected, _ = module.balanced_selection(rows)
        keys = [rows[index]["frame_key"] for index in selected]
        real = list(range(10000, 11000))
        previous = None
        for seed in module.TRAINING_SEEDS:
            schedule = module.paired_schedule(real, selected, keys, seed)
            self.assertEqual([row["update_index"] for row in schedule], list(range(1000)))
            self.assertEqual(Counter(i for row in schedule for i in row["shared_real_indices"]), Counter({i: 3 for i in range(1000)}))
            self.assertEqual(Counter(row["extra_real_index"] for row in schedule), Counter({i: 1 for i in range(1000)}))
            self.assertEqual(Counter(row["synthetic_index"] for row in schedule), Counter(selected))
            self.assertTrue(any(row["synthetic_index"] >= 1000 for row in schedule))
            self.assertEqual(schedule, module.paired_schedule(real, selected, keys, seed))
            if previous is not None:
                self.assertNotEqual(previous, schedule)
            previous = schedule


class TestTargets(unittest.TestCase):
    def sample(self):
        points3d = [[0., 0., 0.]] * 17
        native = {"world_m": points3d, "sampling_method": "gpu_render_cpu_lbs", "mesh_identity": {"asset": "toy"}}
        world = {"track_id": 1, "world_m": points3d, "world_location_valid": [True] * 17, "mesh_identity": native["mesh_identity"], "native_record_ref": "toy_native#frame=0,L0"}
        view = {"intrinsics": {"width": 3840, "height": 2160, "fx": 1000., "fy": 1000., "cx": 1919.5, "cy": 1079.5},
                "extrinsics": {"world_location_m": [-1., 0., 0.], "forward": [1., 0., 0.], "right": [0., 1., 0.], "up": [0., 0., 1.]}}
        points = module.project_landmarks([{"world_m": point, "source_method": native["sampling_method"]} for point in points3d], view)
        obj = {"entity_id": "L0", "track_id": 1, "mask_id": 1, "location_valid": [True] * 17,
               "uv_px": [point["uv_px"] for point in points], "visibility_state": [point["visibility_state"] for point in points],
               "camera_forward_depth_m": [point["camera_forward_depth_m"] for point in points], "visibility_method": "not_measured",
               "bbox_visible_xywh": [1900, 1060, 40, 40], "visible_mask_area": 100}
        return obj, world, native, view

    def test_unknown_not_promoted_and_area0_kept_only_in_world(self):
        obj, world, native, view = self.sample()
        target = module.image_object_target(obj, world, native, view, 1042, "CineCam_01", 100)
        self.assertEqual(target["visibility_state"], ["unknown"] * 17)
        self.assertIsNone(target["source_coco_v"])
        self.assertEqual(sum(target["location_valid"]), 17)
        obj.update(visible_mask_area=0, bbox_visible_xywh=None)
        self.assertIsNone(module.image_object_target(obj, world, native, view, 1042, "CineCam_01", 100))
        self.assertEqual(len(world["world_m"]), 17)

    def test_bbox_person_with_all_missing_points_retained(self):
        obj, world, native, view = self.sample()
        view["extrinsics"]["world_location_m"] = [1., 0., 0.]
        points = module.project_landmarks([{"world_m": point, "source_method": native["sampling_method"]} for point in native["world_m"]], view)
        obj.update(location_valid=[False] * 17, uv_px=[point.get("uv_px") for point in points],
                   visibility_state=[point["visibility_state"] for point in points], camera_forward_depth_m=[point.get("camera_forward_depth_m") for point in points])
        target = module.image_object_target(obj, world, native, view, 1042, "CineCam_01", 100)
        self.assertFalse(any(target["location_valid"]))
        self.assertTrue(target["detection_target_retained"])

    def test_changed_world_visibility_or_empty_bbox_refused(self):
        for damage in ("world", "visibility", "bbox"):
            obj, world, native, view = self.sample()
            if damage == "world":
                world["world_m"] = [[1., 0., 0.]] * 17
            elif damage == "visibility":
                obj["visibility_state"][0] = "visible"
            else:
                obj["bbox_visible_xywh"] = [0, 0, 0, 3]
            with self.assertRaises(ValueError):
                module.image_object_target(obj, world, native, view, 1042, "CineCam_01", 100)

    def test_summary_uses_same_640_geometry_without_selection(self):
        obj, world, native, view = self.sample()
        target = module.image_object_target(obj, world, native, view, 1042, "CineCam_01", 100)
        frame = {"width": 3840, "height": 2160, "source": "synthetic_location_pilot_candidate", "instances": [target], "nonvisible_entities_retained_in_world": ["R1"]}
        stats = module.summarize_frames([frame])
        self.assertAlmostEqual(stats["bbox_height_after_640_letterbox_geometry"]["median"], 40/6)
        self.assertEqual(stats["world_entity_records"], 2)
        self.assertEqual(stats["area0_fraction_all_world_entities"], .5)
        self.assertIs(stats["distribution_used_for_selection"], False)


if __name__ == "__main__":
    unittest.main()
