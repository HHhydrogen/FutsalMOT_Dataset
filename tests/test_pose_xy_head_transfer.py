"""真实Torch/Ultralytics随机Pose的CPU迁移测试，不下载权重或训练。"""
import io
import os
from pathlib import Path
import sys
import unittest
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ.setdefault('YOLO_CONFIG_DIR',str(Path(__file__).resolve().parents[5]/'Saved/ResearchDiagnostics/pose_xy_transfer_config'))
os.environ['YOLO_AUTOINSTALL']='false'
try:
    import torch
    import ultralytics
except ImportError:
    raise unittest.SkipTest('需要独立研究环境Torch/Ultralytics；P1不安装或升级依赖')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from grf_ue_bridge.pose_xy_head_transfer import map_pose_xy,pose_xy_trainer_class,XY_ROWS
from ultralytics.nn.modules.head import Pose
from ultralytics.nn.tasks import PoseModel
from ultralytics.utils import YAML
torch.set_num_threads(1)


def head():
    torch.manual_seed(42)
    source=Pose(nc=1,kpt_shape=(17,3),ch=(64,128,256)).cpu().eval()
    source.stride=torch.tensor([8.,16.,32.])
    return source


def full_model():
    cfg=YAML.load(Path(ultralytics.__file__).parent/'cfg/models/11/yolo11-pose.yaml')
    cfg['scale']='n'
    return PoseModel(cfg,ch=3,nc=1,verbose=False).cpu().eval()


class PoseXYHeadTests(unittest.TestCase):
    def test_final_rows_and_hidden_weights_bitwise_preserved(self):
        source=head();snapshot={k:v.clone() for k,v in source.state_dict().items()}
        target=map_pose_xy(source,mode='compact_xy_rows')
        self.assertEqual(target.nk,34)
        self.assertEqual([b[-1].in_channels for b in target.cv4],[51]*3)
        naive=Pose(nc=1,kpt_shape=(17,2),ch=(64,128,256))
        self.assertEqual([b[-1].in_channels for b in naive.cv4],[34]*3)
        for key,value in target.state_dict().items():
            expected=snapshot[key][list(XY_ROWS)] if key.startswith('cv4.') and key.split('.')[2]=='2' else snapshot[key]
            self.assertTrue(torch.equal(value,expected),key)
        for key,value in source.state_dict().items():self.assertTrue(torch.equal(value,snapshot[key]),key)

    def test_raw_and_decoded_xy_and_boxes_cpu(self):
        source=head();target=map_pose_xy(source)
        for sizes in ((16,8,4),(24,12,6)):
            features=[torch.randn(1,c,size,size) for c,size in zip((64,128,256),sizes)]
            with torch.no_grad():
                a,raw_a=source([x.clone() for x in features]);b,raw_b=target([x.clone() for x in features])
            self.assertTrue(torch.equal(raw_a['kpts'][:,list(XY_ROWS)],raw_b['kpts']))
            self.assertTrue(torch.equal(a[:,:5],b[:,:5]))
            self.assertTrue(torch.equal(a[:,5:][:,list(XY_ROWS)],b[:,5:]))

    def test_export_decode_xy_bitwise_same(self):
        source=head();target=map_pose_xy(source)
        source.export=True;target.export=True
        source.anchors=torch.randn(2,6);target.anchors=source.anchors.clone()
        source.strides=torch.tensor([[8.,8.,16.,16.,32.,32.]])
        target.strides=source.strides.clone()
        raw=torch.randn(2,51,6)
        a=source.kpts_decode(raw);b=target.kpts_decode(raw[:,list(XY_ROWS)])
        self.assertTrue(torch.equal(a[:,list(XY_ROWS)],b))

    def test_standard_xy_or_unsupported_shape_rejected(self):
        with self.assertRaises(ValueError):map_pose_xy(Pose(nc=1,kpt_shape=(17,2),ch=(64,128,256)))
        with self.assertRaises(ValueError):map_pose_xy(Pose(nc=1,kpt_shape=(12,3),ch=(64,128,256)))
        with self.assertRaises(TypeError):map_pose_xy('yolo11n-pose.pt')

    def test_full_random_model_and_in_memory_checkpoint(self):
        torch.manual_seed(17);source=full_model();target=map_pose_xy(source)
        image=torch.randn(1,3,128,160)
        with torch.no_grad():a=source(image)[0];b=target(image)[0]
        self.assertTrue(torch.equal(a[:,:5],b[:,:5]))
        self.assertTrue(torch.equal(a[:,5:][:,list(XY_ROWS)],b[:,5:]))
        self.assertEqual(target.yaml['kpt_shape'],[17,2])
        buf=io.BytesIO();torch.save({'model':target},buf);buf.seek(0)
        restored=torch.load(buf,map_location='cpu',weights_only=False)['model']
        again=map_pose_xy(restored)
        with torch.no_grad():c=again(image)[0]
        self.assertTrue(torch.equal(b,c))

    def test_trainer_get_model_preserves_actual_tower_and_rejects_none(self):
        trainer=object.__new__(pose_xy_trainer_class())
        trainer.data={'kpt_shape':[17,2],'nc':1,'names':{0:'player'}}
        source=full_model();mapped=trainer.get_model(weights=source)
        self.assertEqual([b[-1][0].in_channels for b in mapped.model[-1].cv4],[51]*3)
        with self.assertRaises(ValueError):trainer.get_model(weights=None)

    def test_exact_mode_preserves_all_final_weights_and_rng(self):
        source=head();rng=torch.get_rng_state().clone();target=map_pose_xy(source)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for old,new in zip(source.cv4,target.cv4):
            self.assertTrue(torch.equal(old[-1].weight,new[-1][0].weight))
            self.assertTrue(torch.equal(old[-1].bias,new[-1][0].bias))
        compact=map_pose_xy(source,mode='compact_xy_rows')
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        with self.assertRaises(ValueError):map_pose_xy(compact)

    def test_xy_only_gradient_leaves_visibility_rows_zero(self):
        target=map_pose_xy(head()).train()
        output=target([torch.randn(1,c,s,s) for c,s in zip((64,128,256),(16,8,4))])
        output['kpts'].square().mean().backward()
        for branch in target.cv4:
            grad=branch[-1][0].weight.grad
            self.assertGreater(grad[list(XY_ROWS)].abs().sum().item(),0)
            self.assertEqual(grad[2::3].abs().sum().item(),0)


if __name__=='__main__':unittest.main()
