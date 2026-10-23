"""自绘图像JSONL加载和几何同步的CPU契约测试，无优化器/真实数据。"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from grf_ue_bridge.pose_location_dataset_v2 import LocationPoseDataset,letterbox_geometry,transform_location_instance,location_pose_trainer_class
ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('smoke_pose_location_jsonl_cpu',ROOT/'tools/smoke_pose_location_jsonl_cpu.py')
TOOL=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(TOOL)


class LocationDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            import cv2
        except ImportError:raise unittest.SkipTest('仅研究环境提供图像/Torch依赖')
        cls.torch=torch;torch.set_num_threads(1)

    def test_nonsquare_letterbox_pixels_match_native_ultralytics(self):
        import cv2
        import numpy as np
        from ultralytics.data.augment import LetterBox
        for width,height,size in ((192,96,128),(96,192,128),(37,65,128),(65,37,128)):
            image=np.arange(height*width*3,dtype=np.uint8).reshape(height,width,3)
            g=letterbox_geometry(width,height,size)
            ours=cv2.resize(image,g['resize_wh'],interpolation=cv2.INTER_LINEAR)
            ours=cv2.copyMakeBorder(ours,g['top'],g['bottom'],g['left'],g['right'],cv2.BORDER_CONSTANT,value=(114,114,114))
            native=LetterBox(new_shape=(size,size),auto=False,scale_fill=False,scaleup=True,center=True)(image=image)
            self.assertTrue(np.array_equal(ours,native))

    def test_dataset_preserves_all_boxes_missing_and_unknown(self):
        t=self.torch
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'fixture';TOOL.make_toy_jsonl(root)
            dataset=LocationPoseDataset(root/'train.jsonl')
            a=dataset[0];b=dataset[1]
            self.assertEqual(tuple(a['img'].shape),(3,128,128))
            self.assertEqual(a['keypoints'].shape[0],2)
            self.assertEqual(a['location_valid'].sum(1).tolist(),[8,0])
            self.assertTrue((a['keypoints'][~a['location_valid']]==0).all())
            # 横图y以2/3缩放后加32像素padding，invalid占位仍为0。
            expected=(20*(128/192)+32)/128
            self.assertAlmostEqual(a['keypoints'][0,0,1].item(),expected,places=6)
            self.assertTrue(all(v=='unknown' for v in b['visibility_state'][0]))
            packed=dataset.collate_fn([a,b])
            self.assertEqual(packed['batch_idx'].tolist(),[0,0,1])
            self.assertEqual(packed['target_ids'],[11,12,21])
            self.assertEqual(packed['location_valid'].dtype,t.bool)
            self.assertEqual(packed['location_valid'].sum().item(),25)

    def test_affine_outside_and_false_never_reappears(self):
        item=dict(annotation_id=1,xy_normalized=[[.9,.5],[0.,0.]]+[[.2,.3]]*15,
                  location_valid=[True,False]+[True]*15,visibility_state=['unknown']*17,source_coco_v=None,
                  bbox_xywh_px=[10.,10.,80.,80.])
        result=transform_location_instance(item,(100,100),[[1.,0.,20.],[0.,1.,10.],[0.,0.,1.]],(100,100))
        self.assertFalse(result['location_valid'][0]);self.assertFalse(result['location_valid'][1])
        self.assertEqual(result['xy_normalized'][:2],[[0.,0.],[0.,0.]])
        self.assertEqual(result['geometry_rejection'][:2],['transformed_outside','source_invalid'])
        self.assertEqual(result['bbox_xywh_px'],[30.,20.,70.,80.])
        self.assertTrue(item['location_valid'][0])
        self.assertIsNone(transform_location_instance(item,(100,100),[[1.,0.,200.],[0.,1.,0.],[0.,0.,1.]],(100,100)))

    def test_flip_reindexes_mask_and_source_semantics_together(self):
        item=dict(xy_normalized=[[.2,.3]]*17,location_valid=[True]*17,
                  visibility_state=['unknown']*17,source_coco_v=None,bbox_xywh_px=[10.,10.,80.,80.])
        item['location_valid'][1]=False;item['xy_normalized'][1]=[0.,0.]
        item['visibility_state'][2]='occluded'
        flip=[0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15]
        result=transform_location_instance(item,(100,100),[[-1.,0.,100.],[0.,1.,0.],[0.,0.,1.]],(100,100),flip_idx=flip)
        self.assertFalse(result['location_valid'][2]);self.assertTrue(result['location_valid'][1])
        self.assertEqual(result['visibility_state'][1],'occluded')
        self.assertAlmostEqual(result['xy_normalized'][1][0],.8)

    def test_rejects_missing_person_bad_sha_and_coco_v0_promoted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'fixture';TOOL.make_toy_jsonl(root)
            original=[json.loads(s) for s in (root/'train.jsonl').read_text(encoding='utf-8').splitlines()]
            for change in ('coverage','sha','v0','synthetic_outside'):
                frames=deepcopy(original)
                if change=='coverage':frames[0]['instances'].pop()
                if change=='sha':frames[0]['image_sha256']='0'*64
                if change=='v0':frames[0]['instances'][1]['location_valid'][0]=True
                if change=='synthetic_outside':frames[1]['instances'][0]['visibility_state'][0]='outside'
                path=root/(change+'.jsonl');path.write_text(''.join(json.dumps(f)+'\n' for f in frames),encoding='utf-8')
                with self.assertRaises(ValueError):LocationPoseDataset(path)

    def test_trainer_rejects_gpu_and_auto_device_before_initializing(self):
        trainer=location_pose_trainer_class()
        for device in ('0','cuda','',None):
            with self.assertRaisesRegex(ValueError,'初始化前'):
                trainer(overrides={'device':device})


if __name__=='__main__':unittest.main()
