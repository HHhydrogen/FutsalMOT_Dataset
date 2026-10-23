"""不使用GT、拒绝帧错配及未知值回退的纯CPU关联接口测试。"""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from grf_ue_bridge.pose_association_contract_v2 import pose_pair_cost_candidate


def record(frame,x):
    return dict(source='model_prediction_xy',coordinate_space='image_pixels',validity_method='finite_bbox_relative_geometry',
        model_sha256='toy_model',fixed_detection_set_sha256='toy_fixed_detection',sequence_id='toy',camera_id='cam',
        frame_index=frame,xy_px=[[x,20.]]*17,geometry_valid=[True]*17,bbox_xywh_px=[0.,0.,50.,100.])


class AssociationCandidateTests(unittest.TestCase):
    def test_motion_compensated_pair_and_geometric_fallback(self):
        a=record(1,10.);b=record(2,15.)
        self.assertEqual(pose_pair_cost_candidate(a,b,[5.,0.])['pose_cost'],0.)
        self.assertGreater(pose_pair_cost_candidate(a,b,[0.,0.])['pose_cost'],0.)
        b['geometry_valid']=[False]*17
        result=pose_pair_cost_candidate(a,b,[5.,0.])
        self.assertIsNone(result['pose_cost']);self.assertEqual(result['fallback'],'bbox_only')

    def test_rejects_gt_cross_frame_camera_model_and_detection_mismatch(self):
        a=record(1,10.);b=record(2,15.)
        for field,value in (('source','ground_truth'),('coordinate_space','crop_pixels'),('frame_index',1),
                            ('sequence_id','other'),('camera_id','other'),('model_sha256','other'),('fixed_detection_set_sha256','other')):
            changed=deepcopy(b);changed[field]=value
            with self.assertRaises(ValueError):pose_pair_cost_candidate(a,changed,[5.,0.])

    def test_rejects_nonfinite_and_semantically_fake_visibility_mask(self):
        a=record(1,10.);b=record(2,15.)
        b['validity_method']='ground_truth_visibility'
        with self.assertRaises(ValueError):pose_pair_cost_candidate(a,b,[5.,0.])
        b=record(2,15.);b['xy_px'][0][0]=float('nan')
        with self.assertRaises(ValueError):pose_pair_cost_candidate(a,b,[5.,0.])


if __name__=='__main__':unittest.main()
