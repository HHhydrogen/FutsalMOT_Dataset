"""固定检测附件纯CPU夹具；不读真实图片/标签/封存test，不使用GPU。"""
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

from grf_ue_bridge import fixed_detection_pose_attach_v2 as a
from grf_ue_bridge.pose_association_contract_v2 import pose_pair_cost_candidate


def frame(boxes=([10.,10.,30.,70.],),number=1):
    key=f'SNMOT-151:{number:06d}'
    return dict(schema='futsalmot_fixed_detection_frame_v1',sequence_id='SNMOT-151',camera_id='broadcast',
        frame_index=number-1,frame_number=number,frame_key=key,coordinate_space='image_pixels',
        width=100,height=100,image_file='toy_only_not_read.jpg',detections=[dict(detection_id=f'{key}:{i:06d}',
        source_index=i,class_id=0,bbox_xywh_px=list(box),score=.7123456789) for i,box in enumerate(boxes)])


def prediction(box=(10.,10.,30.,70.)):
    return dict(bbox_xywh_px=list(box),score=.8,xy_px=[[20.+j%2*5,20.+j*3] for j in range(17)])


def attach(f,p=None,config=None):
    return a.attach_frame(f,[prediction()] if p is None else p,model_sha256='a'*64,
        fixed_detection_set_sha256='b'*64,config=config or a.candidate_configuration())


class FixedDetectionAttachmentTests(unittest.TestCase):
    def test_original_boxes_ids_scores_order_untouched(self):
        f=frame();original=deepcopy(f);p=prediction();original_prediction=deepcopy(p)
        result=attach(f,[p]);row=result['attachments'][0]
        self.assertEqual(f,original);self.assertEqual(p,original_prediction)
        self.assertEqual(row['bbox_xywh_px'],original['detections'][0]['bbox_xywh_px'])
        self.assertEqual(row['fixed_detection_score'],.7123456789)
        self.assertEqual(row['detection_id'],'SNMOT-151:000001:000000')
        self.assertTrue(row['prediction_valid']);self.assertEqual(row['visibility_state'],['unknown']*17)
        self.assertIsNone(row['joint_confidence'])
        a.validate_attachment_frame(f,result,model_sha256='a'*64,fixed_detection_set_sha256='b'*64,
                                    config=a.candidate_configuration())

    def test_iou_one_to_one_never_reuses_a_pose(self):
        f=frame(([10.,10.,30.,70.],[10.,10.,30.,70.]))
        rows=attach(f)['attachments']
        self.assertTrue(rows[0]['prediction_valid']);self.assertFalse(rows[1]['prediction_valid'])
        self.assertEqual(rows[0]['model_prediction_index'],0)
        self.assertEqual(rows[1]['prediction_rejection'],'no_model_bbox_match')

    def test_ties_are_fixed_key_then_model_order_not_score_or_gt(self):
        f=frame(([10.,10.,30.,70.],[10.,10.,30.,70.]))
        first=prediction();second=prediction();first['score']=.2;second['score']=.99
        rows=attach(f,[first,second])['attachments']
        self.assertEqual([r['model_prediction_index'] for r in rows],[0,1])

    def test_empty_fixed_frame_is_retained(self):
        result=attach(frame(()),[])
        self.assertEqual(result['attachments'],[]);self.assertEqual(result['fixed_detection_count'],0)
        self.assertEqual(result['frame_key'],'SNMOT-151:000001')

    def test_unmatched_pose_is_all_false_and_bbox_only_cost(self):
        first=attach(frame())['attachments'][0]
        second=attach(frame(number=2),[])['attachments'][0]
        self.assertFalse(second['prediction_valid']);self.assertEqual(second['xy_px'],[[0.,0.]]*17)
        self.assertEqual(second['geometry_valid'],[False]*17)
        cost=pose_pair_cost_candidate(first,second,[0.,0.])
        self.assertIsNone(cost['pose_cost']);self.assertEqual(cost['fallback'],'bbox_only')

    def test_finite_inside_margin_mask_is_not_visibility(self):
        p=prediction();p['xy_px'][0]=[float('nan'),10.];p['xy_px'][1]=[-1.,30.]
        p['xy_px'][2]=[90.,90.];p['xy_px'][3]=[45.,50.]
        row=attach(frame(),[p])['attachments'][0]
        self.assertEqual(row['geometry_valid'][:4],[False,False,False,True])
        self.assertEqual(row['point_geometry_rejection'][:4],
            ['nonfinite_prediction','outside_image','outside_fixed_bbox_margin',None])
        self.assertEqual(row['xy_px'][:3],[[0.,0.]]*3);self.assertIsNone(row['raw_prediction_xy_px'][0])
        self.assertEqual(row['physical_visibility'],'unknown')

    def test_too_few_limbs_invalidates_person_including_remaining_valid_points(self):
        p=prediction();p['xy_px'][5:]=[[-1.,10.]]*12
        row=attach(frame(),[p])['attachments'][0]
        self.assertFalse(row['prediction_valid']);self.assertFalse(any(row['geometry_valid']))
        self.assertEqual(sum(row['geometry_mask_before_global_gate']),5)

    def test_scale_and_degenerate_gates_are_explicit_not_hidden_defaults(self):
        p=prediction();p['xy_px']=[[20.,30.]]*17
        self.assertTrue(attach(frame(),[p])['attachments'][0]['prediction_valid'])
        config=a.candidate_configuration();config['minimum_limb_spread_bbox_diagonal_fraction']=.05
        row=attach(frame(),[p],config)['attachments'][0]
        self.assertFalse(row['prediction_valid']);self.assertEqual(row['prediction_rejection'],'degenerate_limb_spread')
        config=a.candidate_configuration();config['minimum_bbox_height_input_px']=640
        row=attach(frame(),[prediction()],config)['attachments'][0]
        self.assertEqual(row['prediction_rejection'],'bbox_too_small_at_model_input')

    def test_bad_frame_schema_and_sealed_sequence_rejected(self):
        for key,value in [('sequence_id','SNMOT-116'),('frame_index',1),('frame_key','SNMOT-151:000003')]:
            f=frame();f[key]=value
            with self.assertRaises(ValueError):attach(f)
        f=frame();f['detections'][0]['source_index']=True
        with self.assertRaises(ValueError):attach(f)

    def test_attachment_tamper_wrong_model_box_and_visibility_fail(self):
        f=frame();result=attach(f)
        for field,value in [('model_sha256','c'*64),('bbox_xywh_px',[11.,10.,30.,70.]),
                            ('visibility_state',['visible']*17),('joint_confidence',[.8]*17)]:
            bad=deepcopy(result);bad['attachments'][0][field]=value
            with self.assertRaises(ValueError):a.validate_attachment_frame(f,bad,model_sha256='a'*64,
                fixed_detection_set_sha256='b'*64,config=a.candidate_configuration())

    def test_false_geometry_requires_zero_placeholder(self):
        f=frame();result=attach(f);result['attachments'][0]['geometry_valid'][0]=False
        with self.assertRaises(ValueError):a.validate_attachment_frame(f,result,model_sha256='a'*64,
            fixed_detection_set_sha256='b'*64,config=a.candidate_configuration())

    def test_invalid_model_bbox_rejected_without_dropping_fixed_detection(self):
        p=prediction();p['bbox_xywh_px']=[0.,0.,-1.,3.]
        result=attach(frame(),[p])
        self.assertEqual(result['rejected_model_bbox_indices'],[0])
        self.assertFalse(result['attachments'][0]['prediction_valid'])

    def test_bad_joint_layout_is_error_not_silent_fallback(self):
        p=prediction();p['xy_px']=p['xy_px'][:12]
        with self.assertRaises(ValueError):attach(frame(),[p])

    def test_crop_strategy_or_fake_visibility_configuration_rejected(self):
        config=a.candidate_configuration();config['strategy']='GT_bbox_crop'
        with self.assertRaises(ValueError):attach(frame(),config=config)
        config=a.candidate_configuration();config['physical_visibility']='visible'
        with self.assertRaises(ValueError):attach(frame(),config=config)

    def test_inverse_odd_letterbox_and_outside_keypoints(self):
        ratio=640/321;raw=[0.,120.,640.,120.+ratio*200,.8,0]+[-ratio,120.+ratio*201]*17
        p=a.original_prediction(raw,(321,200))
        self.assertAlmostEqual(p['xy_px'][0][0],-1.)
        self.assertAlmostEqual(p['xy_px'][0][1],201.)
        self.assertEqual(len(p['xy_px']),17)

    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('ultralytics'),
                         '研究环境提供Torch/Ultralytics')
    def test_toy_full_frame_through_actual_cpu_nms_then_attachment(self):
        import numpy as np
        import torch
        class ToyModel(torch.nn.Module):
            def forward(self,image):
                output=torch.zeros((1,39,2),device=image.device)
                output[:,0:4,:]=torch.tensor([320.,320.,512.,512.],device=image.device).view(1,4,1)
                output[:,4,:]=torch.tensor([.9,.8],device=image.device);output[:,5:,:]=320.
                return output,[]
        image=np.full((100,100,3),114,np.uint8)
        predictions=a.infer_image(image,ToyModel().eval(),'cpu',a.candidate_configuration())
        self.assertEqual(len(predictions),1);self.assertEqual(predictions[0]['bbox_xywh_px'],[10.,10.,80.,80.])
        rows=attach(frame(([10.,10.,80.,80.],)),predictions)['attachments']
        self.assertTrue(rows[0]['prediction_valid']);self.assertEqual(rows[0]['xy_px'],[[50.,50.]]*17)

    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('ultralytics')
                         and importlib.util.find_spec('lap'),'实际ByteTrack需要现有lap运行时路径')
    def test_actual_attachment_all_false_to_pose_tracker_matches_bbox_baseline(self):
        import numpy as np
        from grf_ue_bridge.pose_bytetrack_v2 import run_sequence
        frames=[frame(([10.+i,10.,30.,70.],),number=i+1) for i in range(3)]
        by_frame={f['frame_index']:{r['detection_id']:r for r in attach(f,[])['attachments']} for f in frames}
        baseline,base_states,_=run_sequence(frames,fixed_sha='b'*64,reference=True)
        result,states,audit=run_sequence(frames,fixed_sha='b'*64,pose_lambda=.25,model_sha='a'*64,
                                       poses_by_frame=by_frame)
        self.assertTrue(all(np.array_equal(x,y) for x,y in zip(baseline,result)))
        self.assertEqual(states,base_states);self.assertEqual(audit['pose_pairs_used'],0)
        self.assertGreater(audit['bbox_fallback_pairs'],0)


class AttachmentEntryBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path(__file__).resolve().parents[1]/'experiments/sim2real'
        sys.path.insert(0,str(path))
        import attach_fixed_detection_pose
        cls.entry=attach_fixed_detection_pose

    def test_execute_optin_is_checked_before_any_path_or_model_read(self):
        with self.assertRaisesRegex(ValueError,'execute-inference'):
            self.entry.execute('never_read','bad','never_read','bad','never_read','bad',
                               'ema','never_created','cpu',explicit=False)

    def test_sealed_path_rejected_before_stat(self):
        from unittest.mock import patch
        with patch.object(self.entry.fixed_cache,'plain',side_effect=AssertionError('不应触碰路径')):
            with self.assertRaisesRegex(ValueError,'封存'):
                self.entry.safe('D:/sealed/SNMOT-116/img1/000001.jpg')

    def test_cache_source_frame_image_binding_is_metadata_only(self):
        entry=self.entry;frames=[];sequences=[]
        with tempfile.TemporaryDirectory() as directory:
            for sequence in a.SEQUENCES:
                selected=[]
                for number in range(1,151):
                    f=frame((),number);f.update(sequence_id=sequence,frame_key=f'{sequence}:{number:06d}',width=1920,height=1080)
                    f['image_file']=str(entry.fixed_cache.source_paths(sequence)/'img1'/f'{number:06d}.jpg')
                    selected.append({k:f[k] for k in ('frame_index','frame_number','frame_key','image_file')})
                    frames.append(f)
                sequences.append(dict(sequence_id=sequence,game_id='9',role='development_only',
                    camera_id='broadcast',width=1920,height=1080,frames=selected))
            source=dict(schema='futsalmot_pose_association_dev_source_candidate_v1',frames=450,
                        test_split_used=False,sequences=sequences)
            path=Path(directory)/'source.json';path.write_text(json.dumps(source),encoding='utf-8')
            manifest=dict(source_manifest=dict(path=str(path),sha256=entry.fixed_cache.sha(path)))
            entry.validate_cache_source(manifest,frames)
            # 仍是合法dev路径，但错到另一帧，不能按图片相似性/GT纠正。
            changed=deepcopy(frames);changed[0]['image_file']=changed[1]['image_file']
            with self.assertRaisesRegex(ValueError,'错配'):entry.validate_cache_source(manifest,changed)


if __name__=='__main__':unittest.main()
