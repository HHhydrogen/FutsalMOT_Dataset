"""位置有效性/可见性分离的CPU候选契约；不加载真实数据或优化器。"""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from grf_ue_bridge.pose_location_supervision_v2 import coco_person_location,synthetic_person_location,collate_location_contract,pack_location_batch,separated_location_loss_class


def coco(flags):
    return dict(id=7,category_id=1,iscrowd=0,bbox=[5,5,80,90],num_keypoints=sum(v>0 for v in flags),
                keypoints=[n for i,v in enumerate(flags) for n in ((20+i,30+i,v) if v else (0,0,0))])


class LocationContractTests(unittest.TestCase):
    def test_coco_missing_is_not_origin_target_and_occluded_is_valid(self):
        target=coco_person_location(coco([0,1,2]+[0]*14),100,100)
        self.assertEqual(target['location_valid'],[False,True,True]+[False]*14)
        self.assertEqual(target['visibility_state'][:3],['unlabeled','occluded','visible'])
        self.assertEqual(target['source_coco_v'][:3],[0,1,2])

    def test_zero_keypoints_person_remains_detection_target(self):
        target=coco_person_location(coco([0]*17),100,100)
        self.assertFalse(any(target['location_valid']))
        frame=dict(width=100,height=100,bbox_coverage='all_noncrowd_person_instances',crowd_present=False,
                   instances=[target],expected_person_annotation_ids=[7])
        batch=collate_location_contract([frame])
        self.assertEqual(len(batch['bboxes']),1)
        self.assertEqual(batch['cls'],[[0]])

    def test_synthetic_unknown_has_valid_position_without_coco_v(self):
        target=synthetic_person_location(dict(id=4,bbox=[2,3,90,90],xy_px=[[25,35]]*17,
                     location_valid=[True]*17,visibility_state=['unknown']*17),100,100)
        self.assertTrue(all(target['location_valid']))
        self.assertEqual(target['visibility_state'],['unknown']*17)
        self.assertIsNone(target['source_coco_v'])

    def test_fail_closed_crowd_coverage_flags_and_wrong_v(self):
        annotation=coco([0]*17);annotation['iscrowd']=1
        with self.assertRaises(ValueError):coco_person_location(annotation,100,100)
        annotation=coco([0]*17);annotation['keypoints'][0]=20
        with self.assertRaises(ValueError):coco_person_location(annotation,100,100)
        target=coco_person_location(coco([0]*17),100,100)
        frame=dict(width=100,height=100,bbox_coverage='all_noncrowd_person_instances',crowd_present=False,
                   instances=[target],expected_person_annotation_ids=[7,8])
        with self.assertRaises(ValueError):collate_location_contract([frame])

    def test_collate_keeps_real_missing_and_synthetic_unknown_in_one_batch(self):
        real=coco_person_location(coco([2]+[0]*16),100,100)
        syn=synthetic_person_location(dict(id=9,bbox=[2,3,90,90],xy_px=[[25,35]]*17,
                     location_valid=[True]*17,visibility_state=['unknown']*17),100,100)
        def frame(instance):return dict(width=100,height=100,bbox_coverage='all_noncrowd_person_instances',crowd_present=False,
                                       instances=[instance],expected_person_annotation_ids=[instance['annotation_id']])
        batch=collate_location_contract([frame(real),frame(syn)])
        self.assertEqual(batch['batch_idx'],[0,1]);self.assertEqual(batch['target_ids'],[7,9])
        self.assertEqual(sum(batch['location_valid'][0]),1);self.assertEqual(sum(batch['location_valid'][1]),17)


class LocationLossCPUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            import ultralytics
        except ImportError:raise unittest.SkipTest('仅研究环境提供Torch/Ultralytics，P1不升级依赖')
        if ultralytics.__version__!='8.4.60':raise unittest.SkipTest('本机契约仅核验8.4.60')
        cls.torch=torch;torch.set_num_threads(1)

    def test_pack_mask_is_not_visibility_and_source_is_unchanged(self):
        t=self.torch
        batch={'keypoints':t.full((2,17,2),.5),'location_valid':t.zeros((2,17),dtype=t.bool),
               'visibility_state':[['unknown']*17]*2}
        batch['location_valid'][1]=True
        packed=pack_location_batch(batch)
        self.assertEqual(tuple(batch['keypoints'].shape),(2,17,2))
        self.assertEqual(tuple(packed['keypoints'].shape),(2,17,3))
        self.assertTrue(t.equal(packed['keypoints'][0],t.zeros(17,3)))
        self.assertTrue(t.equal(packed['keypoints'][1,:,2],t.ones(17)))
        with self.assertRaises(ValueError):pack_location_batch({**batch,'location_valid':batch['location_valid'].float()})

    def test_partial_and_zero_keypoint_targets_have_no_invalid_gradient(self):
        from ultralytics.utils.loss import KeypointLoss
        t=self.torch
        criterion=object.__new__(separated_location_loss_class())
        criterion.keypoint_loss=KeypointLoss(t.full((17,),1/17))
        criterion.bce_pose=lambda *a,**k:(_ for _ in ()).throw(AssertionError('禁止visibility loss'))
        valid=t.zeros((2,17),dtype=t.bool);valid[0,0]=True
        batch=pack_location_batch({'keypoints':t.full((2,17,2),.4),'location_valid':valid})
        pred=t.full((1,2,17,2),.6,requires_grad=True)
        position,visibility=criterion.calculate_keypoints_loss(t.ones((1,2),dtype=t.bool),t.tensor([[0,1]]),
            batch['keypoints'],t.tensor([0,0]),t.ones((2,1)),t.tensor([[[0.,0.,1.,1.],[0.,0.,1.,1.]]]),pred)
        self.assertGreater(position.item(),0);self.assertEqual(visibility,0)
        position.backward()
        self.assertGreater(pred.grad[0,0,0].abs().sum().item(),0)
        self.assertEqual(pred.grad[0,0,1:].abs().sum().item(),0)
        self.assertEqual(pred.grad[0,1].abs().sum().item(),0)

    def test_actual_random_model_batch_bridge_preserves_missing_person_detection(self):
        from copy import copy
        import ultralytics
        from ultralytics.nn.tasks import PoseModel
        from ultralytics.utils import DEFAULT_CFG,YAML
        from grf_ue_bridge.pose_xy_head_transfer import map_pose_xy
        t=self.torch;t.manual_seed(733)
        config=YAML.load(Path(ultralytics.__file__).parent/'cfg/models/11/yolo11-pose.yaml')
        config['scale']='n'
        model=map_pose_xy(PoseModel(config,ch=3,nc=1,verbose=False)).cpu().train()
        model.args=copy(DEFAULT_CFG)
        criterion=separated_location_loss_class()(model)
        criterion.bce_pose.register_forward_pre_hook(lambda *a:(_ for _ in ()).throw(AssertionError('禁止visibility BCE')))
        predictions=model(t.full((2,3,64,64),.5))
        predictions['kpts'].retain_grad();predictions['boxes'].retain_grad()
        mask=t.zeros((2,17),dtype=t.bool);mask[0,0]=True
        batch={'keypoints':t.full((2,17,2),.5),'location_valid':mask,'cls':t.zeros((2,1)),
               'bboxes':t.tensor([[.5,.5,.8,.8],[.5,.5,.8,.8]]),'batch_idx':t.tensor([0,1])}
        loss,items=criterion(predictions,batch)
        self.assertEqual(tuple(batch['keypoints'].shape),(2,17,2))
        self.assertEqual(items[2].item(),0);self.assertTrue(t.isfinite(loss).all())
        loss.sum().backward()
        grad=predictions['kpts'].grad.reshape(2,17,2,-1)
        self.assertGreater(grad[0,0].abs().sum().item(),0)
        self.assertEqual(grad[0,1:].abs().sum().item(),0)
        self.assertEqual(grad[1].abs().sum().item(),0)
        self.assertGreater(predictions['boxes'].grad[1].abs().sum().item(),0)


if __name__=='__main__':unittest.main()
