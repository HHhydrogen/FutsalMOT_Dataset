"""自绘数值夹具验证dev边界和官方COCOeval，不读任何真实图片/封存测试。"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from grf_ue_bridge import coco_pose_dev_eval as e


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,allow_nan=False),encoding='utf-8')
    return e.sha256(path)


def toy_gt(ids=(1,2)):
    points=[v for i in range(17) for v in (20+i,30+i,2)]
    return dict(info={},licenses=[],categories=[dict(id=1,name='person',keypoints=list(e.COCO17),skeleton=[])],
        images=[dict(id=i,width=100,height=100,file_name=f'{i:012d}.jpg') for i in ids],
        annotations=[dict(id=1,image_id=ids[0],category_id=1,iscrowd=0,bbox=[10,10,80,80],area=6400,
                          keypoints=points,num_keypoints=17),
                     dict(id=2,image_id=ids[-1],category_id=1,iscrowd=0,bbox=[10,10,80,80],area=6400,
                          keypoints=[0]*51,num_keypoints=0)])


def prediction(image_id=1):
    return dict(image_id=image_id,category_id=1,bbox=[10,10,80,80],score=.9,
                keypoints=[v for i in range(17) for v in (20+i,30+i,.9)])


def context_fixture(root):
    """只创建200行元数据；没有真实或伪装成真实的JPEG。"""
    root=Path(root);protocol_path=root/'protocol.json';data=root/'prepared'
    protocol=dict(schema='futsalmot_pose_location_coco_control_candidate_v1',status='candidate',data_root=str(data),
        real_selection=dict(train_split='train2017',train_images=1000,pose_dev_split='val2017',pose_dev_images=200,
                            selection_manifest=str(data/'subset'/'selection_manifest.json')),
        paired_training_candidate=dict(imgsz=640,actual_optimizer_updates=1000,seeds=[123]),arms={'R':{},'RS25':{}})
    protocol_sha=write(protocol_path,protocol);ids=list(range(1,201));gt=toy_gt(ids)
    annpath=data/'subset'/'val2017_person_annotations.json';annsha=write(annpath,gt)
    selection=dict(schema='official_coco2017_location_subset_candidate_v1',
        val_role='public_pose_development_only_not_SoccerNet_final_test',splits=dict(
        train2017=dict(image_ids_in_sha_rank_order=list(range(1001,2001))),
        val2017=dict(role='pose_dev_only',image_ids_in_sha_rank_order=ids,
            annotation_subset_path=str(annpath),annotation_subset_sha256=annsha)))
    selpath=data/'subset'/'selection_manifest.json';selsha=write(selpath,selection)
    receipt=dict(selection_manifest_sha256=selsha,images=[dict(image_id=i,split='val2017',width=100,height=100,
        object=f'val2017/{i:012d}.jpg',path=str(data/'images'/'val2017'/f'{i:012d}.jpg'),sha256='a'*64) for i in ids])
    recpath=data/'images_receipt.json';recsha=write(recpath,receipt)
    binding=dict(schema='futsalmot_coco_pose_candidate_binding_v1',protocol_path=str(protocol_path),
        protocol_sha256=protocol_sha,test_SN116_SN187_read=False,selection_manifest_path=str(selpath),
        selection_manifest_sha256=selsha,images_receipt_path=str(recpath),images_receipt_sha256=recsha)
    write(data/'candidate_protocol_manifest.json',binding)
    return protocol_path,protocol_sha,selection,binding


class DevBoundaryTests(unittest.TestCase):
    def test_locked_200_metadata_without_reading_jpegs(self):
        with tempfile.TemporaryDirectory() as root:
            path,sha,_,_=context_fixture(root)
            context=e.load_dev_context(path,sha)
            self.assertEqual(len(context['image_ids']),200)
            self.assertEqual(context['gt']['annotations'][1]['num_keypoints'],0)

    def test_protocol_sha_changed_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path,_,_,_=context_fixture(root)
            with self.assertRaisesRegex(ValueError,'SHA'):e.load_dev_context(path,'0'*64)

    def test_frozen_to_original_candidate_preparation_chain(self):
        with tempfile.TemporaryDirectory() as root:
            candidate,sha,_,_=context_fixture(root)
            frozen=e.read_json(candidate);frozen.update(status='root_frozen',training_authorized=True,
                root_freeze=dict(approved_by='root',candidate_protocol=dict(path=str(candidate),sha256=sha)))
            path=Path(root)/'frozen_protocol.json';frozen_sha=write(path,frozen)
            context=e.load_dev_context(path,frozen_sha)
            self.assertTrue(context['frozen_protocol'])
            self.assertEqual(context['binding']['protocol_sha256'],frozen_sha)
            self.assertEqual(context['binding']['candidate_protocol_sha256'],sha)
            self.assertEqual(context['candidate_protocol_path'],candidate.resolve())
            # 准备binding仍然指向原候选，绝不重写成新frozen SHA。
            prepared=e.read_json(Path(root)/'prepared'/'candidate_protocol_manifest.json')
            self.assertEqual(prepared['protocol_sha256'],sha)
            frozen['real_selection']['pose_dev_images']=199
            changed=write(path,frozen)
            with self.assertRaisesRegex(ValueError,'冻结协议改变'):e.load_dev_context(path,changed)

    def test_frozen_chain_rejects_bad_candidate_sha_or_missing_root_approval(self):
        with tempfile.TemporaryDirectory() as root:
            candidate,sha,_,_=context_fixture(root)
            frozen=e.read_json(candidate);frozen.update(training_authorized=True,
                root_freeze=dict(approved_by='root',candidate_protocol=dict(path=str(candidate),sha256='a'*64)))
            path=Path(root)/'frozen.json';frozen_sha=write(path,frozen)
            with self.assertRaisesRegex(ValueError,'SHA'):e.load_dev_context(path,frozen_sha)
            frozen['root_freeze']['candidate_protocol']['sha256']=sha
            frozen['root_freeze']['approved_by']='other'
            frozen_sha=write(path,frozen)
            with self.assertRaisesRegex(ValueError,'root'):e.load_dev_context(path,frozen_sha)

    def test_train_dev_overlap_rejected_even_with_updated_file_sha(self):
        with tempfile.TemporaryDirectory() as root:
            path,sha,selection,binding=context_fixture(root)
            selection['splits']['train2017']['image_ids_in_sha_rank_order'][0]=1
            binding['selection_manifest_sha256']=write(binding['selection_manifest_path'],selection)
            write(Path(root)/'prepared'/'candidate_protocol_manifest.json',binding)
            with self.assertRaisesRegex(ValueError,'泄漏'):e.load_dev_context(path,sha)

    def test_annotation_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path,sha,selection,_=context_fixture(root)
            Path(selection['splits']['val2017']['annotation_subset_path']).write_text('{}',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'SHA'):e.load_dev_context(path,sha)

    def test_no_test_paths(self):
        for path in ('C:/data/SNMOT-116/img1/a.jpg','C:/data/SNMOT187/img1/a.jpg','C:/data/test2017/a.jpg'):
            with self.assertRaisesRegex(ValueError,'禁止'):e.safe_path(path)

    def test_zero_detections_still_needs_all_processed_ids(self):
        e.validate_predictions([], [1,2], [1,2])
        with self.assertRaises(ValueError):e.validate_predictions([], [1], [1,2])
        with self.assertRaises(ValueError):e.validate_predictions([], [1,1], [1,2])

    def test_foreign_predictions_and_visibility_scores_rejected(self):
        bad=prediction(3)
        with self.assertRaises(ValueError):e.validate_predictions([bad],[1,2],[1,2])
        bad=prediction();bad['keypoints'][2]=2
        with self.assertRaisesRegex(ValueError,'伪COCO'):e.validate_predictions([bad],[1,2],[1,2])
        bad=prediction();bad['keypoints'][0]=float('nan')
        with self.assertRaises(ValueError):e.validate_predictions([bad],[1,2],[1,2])

    def test_prediction_cap_fixed(self):
        with self.assertRaisesRegex(ValueError,'20'):e.validate_predictions([prediction()]*21,[1,2],[1,2])

    def test_odd_letterbox_inverse_retains_outside_amodal_point(self):
        # 321×200 -> 640×399，上120/下121。原(-1,201)点不能被裁成画内。
        ratio=640/321
        row=e.inverse_prediction(1,[0,120,640,120+200*ratio],
            [[-ratio,120+201*ratio]]*17,.4,(321,200))
        self.assertAlmostEqual(row['keypoints'][0],-1)
        self.assertAlmostEqual(row['keypoints'][1],201)
        self.assertEqual(row['keypoints'][2],.4)
        for actual,expected in zip(row['bbox'],[0.,0.,321.,200.]):self.assertAlmostEqual(actual,expected)

    def test_location_denominator_includes_missed_labeled_people(self):
        gt=toy_gt();extra=deepcopy(gt['annotations'][0]);extra.update(id=3,image_id=2)
        gt['annotations'].append(extra)
        metrics=e.location_metrics(gt,[prediction()])
        self.assertEqual(metrics['labeled_GT_points'],34)
        self.assertEqual(metrics['matched_labeled_coverage'],.5)
        self.assertEqual(metrics['person_boxes'],3)
        self.assertEqual(metrics['all_missing_pose_person_boxes'],1)
        self.assertEqual(metrics['conditional_mean_L2_px'],0)

    def test_complete_checkpoint_and_run_binding(self):
        context=dict(protocol_sha256='a'*64,protocol=dict(arms={'R':{}},
            paired_training_candidate=dict(actual_optimizer_updates=1000,seeds=[123])))
        checkpoint=dict(schema='futsalmot_pose_location_control_checkpoint_v1',completed_updates=1000,
            planned_updates=1000,ema_updates=1000,bindings=dict(protocol_sha256='a'*64,schedule_manifest_sha256='b'*64),
            arm='R',seed=123,exposure_chain_sha256='c'*64,initial_state_sha256='d'*64)
        for key in ('admission_sha256','schedule_sha256','trainer_source_sha256','real_jsonl_sha256','synthetic_jsonl_sha256'):
            checkpoint['bindings'][key]='e'*64
        run={k:deepcopy(v) for k,v in checkpoint.items() if k!='ema_updates'}
        run.update(schema='futsalmot_pose_location_control_run_v1',complete=True,
            checkpoint_choice='last_exact_budget_EMA_no_best',visibility_supervised=False,real_test_read=False)
        e.validate_checkpoint_metadata(checkpoint,run,context,'b'*64)
        for key,value in [('completed_updates',999),('ema_updates',999),('seed',999),('exposure_chain_sha256','x')]:
            bad=deepcopy(checkpoint);bad[key]=value
            with self.assertRaises(ValueError):e.validate_checkpoint_metadata(bad,run,context,'b'*64)
        with self.assertRaises(ValueError):e.validate_checkpoint_metadata(checkpoint,run,context,'d'*64)

    def test_cached_prediction_receipt_cannot_change_score_policy(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'predictions.json';write(path,[prediction()])
            context=dict(binding={'protocol_sha256':'a'*64})
            receipt=dict(schema='futsalmot_coco_pose_dev_predictions_v1',binding=context['binding'],
                prediction_sha256=e.sha256(path),processed_image_ids=[1,2],checkpoint=dict(sha256='b'*64),
                postprocess=e.POSTPROCESS,score_policy=e.SCORE_POLICY,
                physical_visibility='unknown_not_predicted',test_read=False)
            self.assertEqual(e.validate_prediction_receipt(receipt,context,path),[1,2])
            receipt['score_policy']='GT_v'
            with self.assertRaises(ValueError):e.validate_prediction_receipt(receipt,context,path)

    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('ultralytics'),
                         '研究环境提供Torch和Ultralytics')
    def test_toy_image_through_real_cpu_nms_and_inverse(self):
        import torch
        import cv2
        import numpy as np
        class ToyDecodedPose(torch.nn.Module):
            def forward(self,image):
                # 真正走本机普通Pose解码输出布局与官方NMS，未伪装成真实模型精度。
                result=torch.zeros((1,39,2),device=image.device)
                result[:,0:4,:]=torch.tensor([320.,320.,512.,512.],device=image.device).view(1,4,1)
                result[:,4,:]=torch.tensor([.9,.8],device=image.device)
                result[:,5:,:]=320.
                return result,[]
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'images'/'val2017'/'000000000001.jpg';path.parent.mkdir(parents=True)
            image=np.full((100,100,3),114,np.uint8)
            cv2.line(image,(20,20),(80,80),(0,255,255),3)
            ok,raw=cv2.imencode('.jpg',image);self.assertTrue(ok);path.write_bytes(raw.tobytes())
            context=dict(image_ids=[1],images={1:dict(path=str(path),sha256=e.sha256(path),width=100,height=100)})
            predictions,processed,runtime=e.inference(context,ToyDecodedPose(),'cpu')
            self.assertEqual(processed,[1]);self.assertEqual(runtime['device'],'cpu')
            self.assertEqual(len(predictions),1)
            self.assertEqual(predictions[0]['bbox'],[10.,10.,80.,80.])
            self.assertEqual(predictions[0]['keypoints'][:2],[50.,50.])
            e.validate_predictions(predictions,processed,[1])

    @unittest.skipUnless(importlib.util.find_spec('pycocotools'),'研究环境提供官方pycocotools')
    def test_official_perfect_and_empty_results(self):
        gt=toy_gt();metrics,log=e.official_metrics(gt,[prediction(1),prediction(2)],[1,2])
        self.assertAlmostEqual(metrics['keypoints']['AP'],1.)
        self.assertAlmostEqual(metrics['bbox']['AP'],1.)
        self.assertIn('Average Precision',log)
        empty,_=e.official_metrics(gt,[],[1,2])
        self.assertEqual(empty['keypoints']['AP'],0.)
        self.assertEqual(empty['bbox']['AP'],0.)
        self.assertEqual(metrics['keypoints']['AP_large'],-1.)

    @unittest.skipUnless(importlib.util.find_spec('pycocotools'),'研究环境提供官方pycocotools')
    def test_original_instances_area_only_changes_bbox_area_groups(self):
        gt=toy_gt();original=deepcopy(gt)
        gt['source_instance_area_by_annotation_id']={'1':10000.,'2':10000.}
        metrics,_=e.official_metrics(gt,[prediction(1),prediction(2)],[1,2])
        self.assertEqual(metrics['keypoints']['AP_large'],-1.)
        self.assertAlmostEqual(metrics['bbox']['AP_large'],1.)
        self.assertEqual(metrics['bbox']['area_source'],'original_joined_instances_area')
        self.assertEqual(gt['annotations'],original['annotations'])
        gt['source_instance_area_by_annotation_id'].pop('2')
        with self.assertRaisesRegex(ValueError,'area映射'):e.official_metrics(gt,[prediction()],[1,2])


if __name__=='__main__':unittest.main()
