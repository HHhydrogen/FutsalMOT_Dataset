"""固定足球开发检测的候选登记/缓存；默认只读文本，不启动推理。"""
import argparse
import configparser
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

RESEARCH = Path('D:/projects/FutsalMOT_Research')
PROJECT = Path('D:/projects/FutsalMOT_UEDataset')
CODE = Path(__file__).resolve().parents[2]
SEQUENCES = ('SNMOT-151', 'SNMOT-152', 'SNMOT-153')
CHECKPOINT = RESEARCH / 'weights/yolo11n.pt'
FRAME_SCHEMA = 'futsalmot_fixed_detection_frame_v1'
INFERENCE = {'ultralytics_version': '8.4.60', 'imgsz': 640, 'conf': .1, 'iou': .7,
             'classes': [0], 'max_det': 300, 'agnostic_nms': False, 'augment': False,
             'half': False, 'rect': True, 'batch': 1}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def plain(path):
    path = Path(path).absolute()
    for parent in (path, *path.parents):
        if parent.exists():
            require(not parent.is_symlink() and not (getattr(parent.stat(), 'st_file_attributes', 0) & 0x400),
                    '拒绝联接或符号链接: ' + str(parent))
    return path


def output_path(path):
    path = plain(path)
    roots = (PROJECT / 'Saved/Pose17Development', RESEARCH / 'pose')
    require(any(root in path.parents for root in roots), '只允许独立Saved/研究pose子目录')
    require(not path.exists(), '目标已存在，拒绝覆盖或隐式续写')
    return path


def write_new(path, value):
    with plain(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def source_paths(sequence):
    require(sequence in SEQUENCES, '只允许既定三个开发序列，封存test禁读')
    base = 'public/soccernet_tracking' if sequence == 'SNMOT-151' else 'public/development_extension_v1'
    return RESEARCH / base / 'train' / sequence


def mot_rows(text):
    """只解析MOT文本；保留原始坐标与行号，不读取图像。"""
    rows = []
    for line_index, line in enumerate(text.splitlines()):
        if not line.strip():
            continue
        values = [float(v) for v in line.split(',')]
        require(len(values) >= 7 and all(math.isfinite(v) for v in values), 'MOT行非法')
        require(values[0].is_integer() and values[1].is_integer(), '帧/ID须为整数')
        rows.append({'line_index': line_index, 'frame': int(values[0]), 'identity': int(values[1]),
                     'bbox': values[2:6], 'score': values[6], 'raw': line})
    return rows


def official_det_audit(det, gt):
    """只辨别现有det是否复制全部GT框；未知来源也不能自动准入。"""
    key = lambda row: (row['frame'], *row['bbox'])
    same = Counter(map(key, det)) == Counter(map(key, gt))
    return {'rows': len(det), 'all_scores_one': all(row['score'] == 1 for row in det),
            'frame_bbox_multiset_exactly_matches_all_gt': same,
            'status': 'gt_bbox_oracle_do_not_use' if same else 'unverified_official_detections_do_not_use',
            'mainline_eligible': False}


def prepare(output):
    output = output_path(output)
    rows = []
    for sequence in SEQUENCES:
        source = plain(source_paths(sequence))
        info = configparser.ConfigParser()
        info.read(source / 'seqinfo.ini', encoding='utf-8')
        game = configparser.ConfigParser()
        game.read(source / 'gameinfo.ini', encoding='utf-8')
        require(info['Sequence']['name'] == sequence and game['Sequence']['gameid'] == '9', '开发身份或game9不匹配')
        require(info.getint('Sequence', 'seqLength') >= 150, '不足150帧')
        width, height = info.getint('Sequence', 'imWidth'), info.getint('Sequence', 'imHeight')
        require((width, height, info.getint('Sequence', 'frameRate')) == (1920, 1080, 25), '源尺寸/帧率改变')
        players = sorted(int(k.split('_')[1]) for k, v in game['Sequence'].items()
                         if k.startswith('trackletid_') and v.lower().startswith(('player ', 'goalkeeper')))
        gt_path = source / 'gt/gt.txt'
        det_path = source / 'det/det.txt'
        gt = mot_rows(gt_path.read_text(encoding='utf-8'))
        official_det = mot_rows(det_path.read_text(encoding='utf-8'))
        selected = [row for row in gt if 1 <= row['frame'] <= 150 and row['identity'] in players]
        derived_path = RESEARCH / 'round4_combined_eval/datasets/real_player_pilot_v1/mot_gt' / sequence / 'gt/gt.txt'
        derived = mot_rows(derived_path.read_text(encoding='utf-8'))
        key = lambda row: (row['frame'], row['identity'], *row['bbox'])
        require(Counter(map(key, selected)) == Counter(map(key, derived)), '派生GT与既有player过滤不一致')
        images = []
        for frame in range(1, 151):
            image = plain(source / 'img1' / f'{frame:06d}.jpg')
            require(image.is_file(), '开发帧不存在: ' + str(image))
            images.append({'frame_number': frame, 'frame_index': frame - 1,
                           'frame_key': f'{sequence}:{frame:06d}', 'image_file': str(image)})
        metadata = [source / 'seqinfo.ini', source / 'gameinfo.ini', gt_path, det_path, derived_path]
        rows.append({'sequence_id': sequence, 'official_split': 'train', 'role': 'development_only',
                     'game_id': '9', 'camera_id': 'broadcast', 'width': width, 'height': height, 'fps': 25,
                     'selected_frame_numbers': [1, 150], 'range_end_inclusive': True, 'frames': images,
                     'player_goalkeeper_track_ids': players, 'selected_player_gt_rows': len(selected),
                     'derived_gt_file': str(derived_path),
                     'gt_filter': 'gameinfo tracklet value lower startswith player-space or goalkeeper; frame1..150; preserve original bbox; output mark/class/visibility=1,1,1; DO_PREPROC=False',
                     'text_sources': [{'path': str(path), 'sha256': sha(path)} for path in metadata],
                     'official_det_audit': official_det_audit(official_det, gt)})
    require(CHECKPOINT.is_file(), '官方预训练YOLO11n权重不存在')
    checkpoint = {'path': str(CHECKPOINT), 'sha256': sha(CHECKPOINT),
                  'identity': 'official_COCO_pretrained_yolo11n_person_class0', 'finetuned_on_this_project': False,
                  'checkpoint_choice_status': 'root_requested_candidate_not_yet_inference_authorized'}
    output.mkdir(parents=True)
    manifest = {'schema': 'futsalmot_pose_association_dev_source_candidate_v1',
                'status': 'candidate_not_frozen_not_inferred', 'sequences': rows, 'frames': 450,
                'detector': checkpoint, 'inference': INFERENCE, 'tracker_config': {
                    'path': str(CODE / 'experiments/sim2real/bytetrack_fixed.yaml'),
                    'sha256': sha(CODE / 'experiments/sim2real/bytetrack_fixed.yaml')},
                'images_read_or_hashed': 0, 'test_split_used': False,
                'limitation': '三个开发片段均game9；源seqLength750但仅前150；非独立比赛/最终测试/五人制收益证明',
                'baseline': '新raw检测缓存共享ByteTrack，不与旧R4/R5端到端检测/追踪数值直接比较',
                'raw_tracking_outputs_reusable_as_detections': False}
    write_new(output / 'source_manifest.json', manifest)
    protocol = {'schema': 'futsalmot_pose_association_protocol_candidate_v1', 'status': 'candidate_not_frozen',
                'source_manifest': {'path': str(output / 'source_manifest.json'), 'sha256': sha(output / 'source_manifest.json')},
                'fixed_detection_frame_schema': FRAME_SCHEMA, 'pose_attachment_schema_status': 'pending_independent_prediction_interface',
                'checkpoint_roles': 'detector固定官方COCO预训练；Pose初始化/R/RS三模型共享全部检测与关联参数；检测器不随Pose模型变化',
                'pose_training_seeds': [20261024, 20261025, 20261026],
                'arms': ['bbox_only_lambda0', 'initial_pose', 'real_pose', 'real_plus_synthetic_pose'],
                'tracking': {'reset_each_sequence': True, 'tracker_config': manifest['tracker_config'],
                             'cost_stage': 'pending_exact_matcher_interface_freeze', 'all_non_pose_settings_identical': True},
                'pose_cost_candidate': {'lambda_candidates': [0, .1, .25], 'joint_indices': list(range(5, 17)),
                                        'min_common_geometry_joints': 6, 'sigma': .1, 'max_frame_gap': 30,
                                        'missing_pose': 'bbox_only', 'validity_source': 'prediction_geometry_only_no_GT_no_visibility',
                                        'cost_reference': 'grf_ue_bridge.pose_association_contract_v2.pose_pair_cost_candidate'},
                'prediction_crop_strategy': 'pending_root_freeze; no_GT_crops; preserve fixed detection bbox and IDs',
                'freeze_prerequisites': ['root签固定source/checkpoint/cache身份和执行授权',
                                         'lambda0与相同raw输入的标准ByteTrack逐行等价，含空帧/丢失/低分/多阶段',
                                         'Pose附件17xy/几何mask/ID/模型SHA/缓存SHA精确对接并拒绝错配',
                                         'root冻结唯一lambda/crop/几何规则后方可按R/RS结果比较，不能挑有利lambda'],
                'metrics': {'primary': 'AssA', 'aggregation': 'TrackEval three-sequence COMBINED_SEQ; same source derived player GT; DO_PREPROC=False',
                            'report': ['AssA', 'HOTA', 'DetA', 'IDF1', 'MOTA', 'IDSW'],
                            'candidate_progress_rule': {'paired_RS_minus_R_AssA_mean_gain_pp': .5,
                                                        'AssA_positive_seeds_min': 2, 'HOTA_mean_gain_pp_min': 0.,
                                                        'IDF1_mean_drop_pp_max': .5},
                            'units': 'TrackEval百分数pp；同时报告原始值和各seed/片段负向结果',
                            'detection_AP': '共享检测缓存应保持不变，不能把Pose关联收益写成检测AP收益'},
                'limitations': ['151/152/153均game9且参与开发，三个训练seed不是三场比赛',
                                'COCO位置AP只为通用姿态保留代理；固定检测AssA才是本候选足球关联主线',
                                '不读取SNMOT116/187，不宣称最终测试、跨比赛显著或五人制泛化'],
                'training_authorized': False, 'inference_authorized': False, 'test_split_used': False}
    write_new(output / 'protocol_candidate.json', protocol)
    return {'output': str(output), 'source_manifest_sha256': sha(output / 'source_manifest.json'),
            'protocol_candidate_sha256': sha(output / 'protocol_candidate.json'),
            'official_det_status': {row['sequence_id']: row['official_det_audit'] for row in rows},
            'detector': checkpoint, 'inference_started': False}


def frame_record(source, frame, detections):
    """无GT参与的原始检测行；frame_index为0基，frame_number为MOT的1基。"""
    require(source['sequence_id'] in SEQUENCES and source['camera_id'] == 'broadcast', '非固定开发域')
    require(type(frame['frame_number']) is int and 1 <= frame['frame_number'] <= 150
            and type(frame['frame_index']) is int and frame['frame_index'] == frame['frame_number'] - 1, '帧身份非法')
    key = f"{source['sequence_id']}:{frame['frame_number']:06d}"
    require(frame['frame_key'] == key, '帧key错配')
    output = []
    for index, detection in enumerate(detections):
        box, score, cls = detection
        require(len(box) == 4 and all(type(v) in (int, float) and math.isfinite(v) for v in box), '框坐标非法')
        x0, y0, x1, y1 = box
        require(0 <= x0 < x1 <= source['width'] and 0 <= y0 < y1 <= source['height'], '检测框不在原图像素域')
        require(type(score) in (int, float) and math.isfinite(score) and INFERENCE['conf'] <= score <= 1
                and type(cls) in (int, float) and cls == 0,
                '检测分数/类别不符')
        output.append({'detection_id': f'{key}:{index:06d}', 'source_index': index, 'class_id': 0,
                       'bbox_xywh_px': [x0, y0, x1 - x0, y1 - y0], 'score': score})
    return {'schema': FRAME_SCHEMA, 'sequence_id': source['sequence_id'], 'camera_id': source['camera_id'],
            **frame, 'width': source['width'], 'height': source['height'], 'coordinate_space': 'image_pixels',
            'detections': output}


def export_cache(source_manifest, protocol, admission, output, device, execute=False):
    """显式授权后才导入推理库；本函数不启动追踪或训练，失败保留partial且无完成manifest。"""
    require(execute, '必须显式execute-inference；prepare绝不推理')
    source_manifest, protocol, admission = map(plain, (source_manifest, protocol, admission))
    source, candidate, approval = read(source_manifest), read(protocol), read(admission)
    require(source['schema'] == 'futsalmot_pose_association_dev_source_candidate_v1', '未知source')
    require(candidate['schema'] == 'futsalmot_pose_association_protocol_candidate_v1'
            and candidate['source_manifest']['sha256'] == sha(source_manifest), '候选source身份变化')
    require(approval.get('schema') == 'futsalmot_fixed_detection_cache_admission_v1'
            and approval.get('inference_authorized') is True
            and approval.get('source_manifest_sha256') == sha(source_manifest)
            and approval.get('protocol_candidate_sha256') == sha(protocol)
            and approval.get('detector_sha256') == source['detector']['sha256']
            and approval.get('test_split_used') is False, '缺root签署的缓存限定准入或身份错配')
    require(source['inference'] == INFERENCE and source['frames'] == 450
            and [row['sequence_id'] for row in source['sequences']] == list(SEQUENCES), '来源/推理范围改变')
    require(device in ('cpu', '0'), '设备必须显式cpu或0')
    require(plain(source['detector']['path']) == plain(CHECKPOINT)
            and sha(CHECKPOINT) == source['detector']['sha256'], '固定官方checkpoint变化')
    for row in source['sequences']:
        require(row['game_id'] == '9' and row['camera_id'] == 'broadcast' and len(row['frames']) == 150, '源域改变')
        for text in row['text_sources']:
            require(sha(plain(text['path'])) == text['sha256'], '源文本变化')
        require([frame['frame_number'] for frame in row['frames']] == list(range(1, 151)), '帧集合改变')
        for frame in row['frames']:
            require(plain(frame['image_file']) == plain(source_paths(row['sequence_id']) / 'img1' / f"{frame['frame_number']:06d}.jpg"),
                    '图像词法路径离开固定dev源')
    output = output_path(output)
    import cv2
    import torch
    import ultralytics
    from ultralytics import YOLO
    require(ultralytics.__version__ == INFERENCE['ultralytics_version'], 'Ultralytics版本未核验')
    output.mkdir(parents=True)
    write_new(output / 'started.json', {'status': 'running_not_complete', 'source_manifest_sha256': sha(source_manifest),
                                      'admission_sha256': sha(admission), 'device': device, 'test_split_used': False})
    model = YOLO(str(CHECKPOINT))
    require(model.names.get(0) == 'person' and len(model.names) == 80, '要求原始COCO80类官方检测头，非项目微调player头')
    partial = output / 'detections.partial.jsonl'
    counts = Counter()
    with partial.open('x', encoding='utf-8') as handle:
        for row in source['sequences']:
            for frame in row['frames']:
                image = cv2.imread(frame['image_file'], cv2.IMREAD_COLOR)
                require(image is not None and image.shape[:2] == (row['height'], row['width']), '源图像解码/尺寸不符')
                kwargs = {k: v for k, v in INFERENCE.items() if k not in ('ultralytics_version', 'batch')}
                kwargs.update(save=False, save_txt=False, save_crop=False, show=False)
                result = model.predict(image, device=device, verbose=False, **kwargs)[0]
                boxes = result.boxes
                detections = [] if boxes is None else list(zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist()))
                record = frame_record(row, frame, detections)
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')
                counts[row['sequence_id']] += len(detections)
    final = output / 'detections.jsonl'
    partial.rename(final)
    manifest = {'schema': 'futsalmot_fixed_detection_cache_manifest_v1', 'status': 'complete_fixed_raw_detections',
                'detections_jsonl': {'path': str(final), 'sha256': sha(final), 'frames': 450},
                'source_manifest': {'path': str(source_manifest), 'sha256': sha(source_manifest)},
                'protocol_candidate_sha256': sha(protocol), 'admission_sha256': sha(admission),
                'detector': source['detector'], 'inference': INFERENCE, 'device': device,
                'runtime': {'torch': torch.__version__, 'ultralytics': ultralytics.__version__, 'opencv': cv2.__version__},
                'prediction_output_options': {'save': False, 'save_txt': False, 'save_crop': False, 'show': False},
                'cache_builder': {'path': str(Path(__file__).absolute()), 'sha256': sha(__file__)},
                'detections_by_sequence': dict(counts), 'sort_order': 'sequence151,152,153; frame1..150; model_output_index',
                'scores_and_coordinates': 'python serialized native float values without decimal rounding',
                'tracking_applied': False, 'pose_applied': False, 'GT_used_in_prediction': False, 'test_split_used': False,
                'frame_number_base': 1, 'frame_index_base': 0,
                'consumer_identity': 'SHA256 of this complete cache_manifest.json is fixed_detection_set_sha256; JSONL rows carry no circular self hash'}
    write_new(output / 'cache_manifest.json', manifest)
    return {'cache_manifest': str(output / 'cache_manifest.json'), 'sha256': sha(output / 'cache_manifest.json'), 'counts': dict(counts)}


def load_fixed_cache(manifest_path, expected_manifest_sha256):
    """Matcher/Pose附件共用的纯CPU文本读取器；不打开图像，不接受partial或GT检测替代品。"""
    manifest_path = plain(manifest_path)
    require(manifest_path.name == 'cache_manifest.json' and sha(manifest_path) == expected_manifest_sha256,
            '固定检测cache manifest身份不符')
    manifest = read(manifest_path)
    require(manifest.get('schema') == 'futsalmot_fixed_detection_cache_manifest_v1'
            and manifest.get('status') == 'complete_fixed_raw_detections'
            and manifest.get('tracking_applied') is False and manifest.get('pose_applied') is False
            and manifest.get('GT_used_in_prediction') is False and manifest.get('test_split_used') is False,
            '只接收完整独立raw检测缓存')
    source = manifest['detections_jsonl']
    path = plain(source['path'])
    require(path.parent == manifest_path.parent and path.name == 'detections.jsonl'
            and source['frames'] == 450 and sha(path) == source['sha256'], '固定JSONL身份/边界改变')
    frames = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    expected = [(sequence, frame) for sequence in SEQUENCES for frame in range(1, 151)]
    require([(row['sequence_id'], row['frame_number']) for row in frames] == expected, '序列/帧集合不连续或重复')
    for row in frames:
        require(row['schema'] == FRAME_SCHEMA and row['coordinate_space'] == 'image_pixels'
                and (row['width'], row['height']) == (1920, 1080), '检测帧schema/像素域改变')
        raw = []
        for index, det in enumerate(row['detections']):
            require(det['source_index'] == index and det['detection_id'] == f"{row['frame_key']}:{index:06d}",
                    '检测ID/顺序改变')
            require(len(det['bbox_xywh_px']) == 4 and all(type(v) in (int, float) and math.isfinite(v)
                    for v in det['bbox_xywh_px']), '固定框必须有限数值')
            x, y, w, h = det['bbox_xywh_px']
            raw.append(([x, y, x + w, y + h], det['score'], det['class_id']))
        # 重新验证数值/ID但不重新浮点减加写入原框，返回原记录，避免舍入引入新值。
        frame_record(row, {key: row[key] for key in ('frame_number', 'frame_index', 'frame_key', 'image_file')}, raw)
    return manifest, frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'export'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-manifest', type=Path)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--admission', type=Path)
    parser.add_argument('--device', choices=['cpu', '0'])
    parser.add_argument('--execute-inference', action='store_true')
    args = parser.parse_args()
    result = prepare(args.output) if args.action == 'prepare' else export_cache(
        args.source_manifest, args.protocol, args.admission, args.output, args.device, args.execute_inference)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
