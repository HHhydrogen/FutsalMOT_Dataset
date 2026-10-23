"""固定检测候选的纯CPU边界契约；不读取研究媒体/推理库。"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

MODULE = Path(__file__).with_name('prepare_pose_association_dev.py')
SPEC = importlib.util.spec_from_file_location('association_prepare', MODULE)
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


class FixedDetectionContract(unittest.TestCase):
    def setUp(self):
        self.source = {'sequence_id': 'SNMOT-151', 'camera_id': 'broadcast', 'width': 1920, 'height': 1080}
        self.frame = {'frame_number': 1, 'frame_index': 0, 'frame_key': 'SNMOT-151:000001', 'image_file': 'readonly.jpg'}

    def test_empty_frame_retained(self):
        row = tool.frame_record(self.source, self.frame, [])
        self.assertEqual(row['detections'], [])
        self.assertEqual(row['frame_index'], 0)

    def test_native_float_no_decimal_rounding(self):
        x = 1.123456789123
        row = tool.frame_record(self.source, self.frame, [([x, 2., 32., 42.], .123456789, 0)])
        det = json.loads(json.dumps(row))['detections'][0]
        self.assertEqual(det['bbox_xywh_px'][0], x)
        self.assertEqual(det['score'], .123456789)
        self.assertEqual(det['detection_id'], 'SNMOT-151:000001:000000')
        self.assertNotIn('track_id', det)

    def test_id_not_from_GT_and_order_retained(self):
        inputs = [([1, 2, 30, 40], .2, 0), ([5, 6, 10, 12], .9, 0)]
        rows = tool.frame_record(self.source, self.frame, inputs)['detections']
        self.assertEqual([row['source_index'] for row in rows], [0, 1])
        self.assertEqual([row['score'] for row in rows], [.2, .9])

    def test_wrong_scope_or_frame_rejected(self):
        for sequence in ('SNMOT-116', 'SNMOT-187', 'SNMOT-150'):
            with self.assertRaises(ValueError):
                tool.source_paths(sequence)
        for changed in ({'frame_index': 1}, {'frame_number': 151}, {'frame_key': 'other'}):
            with self.assertRaises(ValueError):
                tool.frame_record(self.source, {**self.frame, **changed}, [])

    def test_invalid_bbox_score_rejected(self):
        for box, score, cls in (([1, 2, 1, 5], .8, 0), ([-1, 2, 4, 5], .8, 0),
                                ([1, 2, 1921, 5], .8, 0), ([1, 2, 4, 5], .09, 0),
                                ([1, 2, 4, 5], float('nan'), 0), ([1, 2, 4, 5], .8, 1)):
            with self.assertRaises(ValueError):
                tool.frame_record(self.source, self.frame, [(box, score, cls)])

    def test_official_GT_copy_detected_not_eligible(self):
        gt = tool.mot_rows('1,10,1,2,3,4,1,-1,-1,-1\n2,10,2,2,3,4,1,-1,-1,-1\n')
        det = tool.mot_rows('2,-1,2,2,3,4,1,-1,-1,-1\n1,-1,1,2,3,4,1,-1,-1,-1\n')
        audit = tool.official_det_audit(det, gt)
        self.assertEqual(audit['status'], 'gt_bbox_oracle_do_not_use')
        self.assertFalse(audit['mainline_eligible'])
        self.assertTrue(audit['frame_bbox_multiset_exactly_matches_all_gt'])

    def test_unverified_det_also_not_auto_admitted(self):
        rows = tool.mot_rows('1,-1,1,2,3,4,0.9,-1,-1,-1')
        self.assertFalse(tool.official_det_audit(rows, [])['mainline_eligible'])

    def test_no_execution_flag_no_library_import_or_output(self):
        with self.assertRaisesRegex(ValueError, 'execute-inference'):
            tool.export_cache(None, None, None, None, None, execute=False)

    def test_exclusive_json_write(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'receipt.json'
            tool.write_new(path, {'candidate': True})
            with self.assertRaises(FileExistsError):
                tool.write_new(path, {'candidate': False})
            self.assertTrue(tool.read(path)['candidate'])

    def empty_cache(self, directory, mutate=None):
        path = Path(directory) / 'detections.jsonl'
        frames = []
        for sequence in tool.SEQUENCES:
            for number in range(1, 151):
                source = {**self.source, 'sequence_id': sequence}
                frame = {**self.frame, 'frame_number': number, 'frame_index': number - 1,
                         'frame_key': f'{sequence}:{number:06d}'}
                frames.append(tool.frame_record(source, frame, []))
        if mutate:
            mutate(frames)
        path.write_text(''.join(json.dumps(frame) + '\n' for frame in frames), encoding='utf-8')
        manifest = Path(directory) / 'cache_manifest.json'
        tool.write_new(manifest, {'schema': 'futsalmot_fixed_detection_cache_manifest_v1',
                                 'status': 'complete_fixed_raw_detections', 'tracking_applied': False,
                                 'pose_applied': False, 'GT_used_in_prediction': False, 'test_split_used': False,
                                 'detections_jsonl': {'path': str(path), 'sha256': tool.sha(path), 'frames': 450}})
        return manifest

    def test_consumer_preserves_all_450_empty_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.empty_cache(directory)
            _, frames = tool.load_fixed_cache(manifest, tool.sha(manifest))
            self.assertEqual(len(frames), 450)
            self.assertEqual(frames[-1]['frame_key'], 'SNMOT-153:000150')

    def test_consumer_rejects_changed_file_and_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.empty_cache(directory)
            identity = tool.sha(manifest)
            with self.assertRaises(ValueError):
                tool.load_fixed_cache(manifest, '0' * 64)
            path = Path(directory) / 'detections.jsonl'
            path.write_text(path.read_text(encoding='utf-8') + '\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'JSONL'):
                tool.load_fixed_cache(manifest, identity)

    def test_consumer_rejects_duplicate_missing_or_test_frame(self):
        for change in (lambda frames: frames.pop(), lambda frames: frames.append(frames[0]),
                       lambda frames: frames[0].update(sequence_id='SNMOT-116')):
            with tempfile.TemporaryDirectory() as directory:
                manifest = self.empty_cache(directory, change)
                with self.assertRaisesRegex(ValueError, '帧集合'):
                    tool.load_fixed_cache(manifest, tool.sha(manifest))

    def test_consumer_rejects_wrong_detection_id(self):
        def change(frames):
            frames[0]['detections'] = [{'detection_id': 'gt_12', 'source_index': 0,
                                        'bbox_xywh_px': [10, 20, 30, 40], 'score': .8, 'class_id': 0}]
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.empty_cache(directory, change)
            with self.assertRaisesRegex(ValueError, '检测ID'):
                tool.load_fixed_cache(manifest, tool.sha(manifest))


if __name__ == '__main__':
    unittest.main()
