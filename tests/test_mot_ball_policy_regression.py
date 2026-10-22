"""MOT 的球 track_id 只由第二列决定，坐标 100 不能开启球导出。"""
import json
from grf_ue_bridge.dataset_regression import collect_dataset_regression_errors


def test_player_coordinate_100_does_not_enable_ball_export(tmp_path):
    camera = tmp_path / 'Camera_01'
    (camera / 'gt').mkdir(parents=True)
    (camera / 'camera.json').write_text(json.dumps({'image_width': 1920, 'image_height': 1080, 'intrinsics': {'width': 1920, 'height': 1080}}), encoding='utf-8')
    objects = [
        {'entity_id': 'L0', 'track_id': 1, 'mask_id': 1, 'class': 'player',
         'in_frame': True, 'bbox_xyxy': [100, 10, 120, 40], 'bbox_xywh': [100, 10, 20, 30]},
        {'entity_id': 'BALL', 'track_id': 100, 'mask_id': 11, 'class': 'ball',
         'in_frame': True, 'bbox_xyxy': [200, 20, 205, 25], 'bbox_xywh': [200, 20, 5, 5]},
    ]
    (camera / 'annotations.jsonl').write_text(json.dumps({'frame_index': 1, 'objects': objects})+'\n', encoding='utf-8')
    (camera / 'gt/gt.txt').write_text('1,1,100,10,20,30,1,1,1.00\n', encoding='utf-8')
    errors = collect_dataset_regression_errors(tmp_path, require_mask=False, require_yolo_det=False, require_yolo_seg=False)
    assert not errors, errors
