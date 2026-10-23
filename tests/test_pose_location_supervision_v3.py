"""与上游17×3实算位置loss/梯度对比；纯CPU随机toy，不改活跃实验。"""
from copy import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))


@pytest.fixture
def models():
    torch=pytest.importorskip("torch");ultralytics=pytest.importorskip("ultralytics")
    from ultralytics.nn.tasks import PoseModel
    from ultralytics.utils import DEFAULT_CFG,YAML
    from grf_ue_bridge.pose_xy_head_transfer import map_pose_xy
    torch.set_num_threads(1);torch.manual_seed(193)
    config=YAML.load(Path(ultralytics.__file__).parent/"cfg/models/11/yolo11-pose.yaml");config["scale"]="n"
    source=PoseModel(config,ch=3,nc=1,verbose=False).cpu().train();source.args=copy(DEFAULT_CFG)
    xy=map_pose_xy(source).cpu().train();xy.args=copy(DEFAULT_CFG)
    return torch,source,xy


def criteria(models):
    from ultralytics.utils.loss import v8PoseLoss
    from grf_ue_bridge.pose_location_supervision_v3 import coco_sigma_location_loss_class
    _,source,xy=models
    return v8PoseLoss(source),coco_sigma_location_loss_class()(xy)


def test_explicit_coco_sigmas_equal_actual_upstream_and_v2_remains_unchanged(models):
    from grf_ue_bridge.pose_location_supervision_v2 import separated_location_loss_class
    torch,_,xy=models
    original,new=criteria(models)
    assert torch.equal(original.keypoint_loss.sigmas,new.keypoint_loss.sigmas)
    assert new.location_contract["visibility_supervised"] is False
    v2=separated_location_loss_class()(xy)
    assert torch.equal(v2.keypoint_loss.sigmas,torch.ones(17)/17)
    assert not torch.equal(v2.keypoint_loss.sigmas,new.keypoint_loss.sigmas)


def test_position_loss_and_xy_gradients_equal_upstream_without_visibility_bce(models):
    torch,_,_=models;original,new=criteria(models)
    from grf_ue_bridge.pose_location_supervision_v2 import pack_location_batch
    valid=torch.ones((3,17),dtype=torch.bool);valid[0,::3]=False;valid[1]=False;valid[2,0]=False
    xy=torch.rand((3,17,2))*.1+.4
    raw_v=torch.where(valid,torch.tensor(2.),torch.tensor(0.));raw_v[2,1::2]=1.
    # v1与v2均有位置，v0仍保留人物；新接口独立bool，不借用visible概念。
    upstream=torch.cat((xy*64,raw_v[...,None]),dim=-1)
    packed=pack_location_batch(dict(keypoints=xy,location_valid=valid))["keypoints"]
    packed[...,:2]*=64
    masks=torch.tensor([[True,True,False],[True,False,True]])
    indices=torch.tensor([[0,1,0],[0,0,0]])
    batch_idx=torch.tensor([0,0,1]);stride=torch.ones((3,1))
    boxes=torch.tensor([[[0.,0.,64.,64.],[0.,0.,64.,64.],[0.,0.,64.,64.]]]*2)
    pred3=(torch.rand((2,3,17,3))*.2+27.).requires_grad_(True)
    pred2=pred3.detach()[...,:2].clone().requires_grad_(True)
    old_position,_=original.calculate_keypoints_loss(masks,indices,upstream,batch_idx,stride,boxes,pred3)
    new.bce_pose.register_forward_pre_hook(lambda *a:(_ for _ in ()).throw(AssertionError("禁止visibility BCE")))
    new_position,new_visibility=new.calculate_keypoints_loss(masks,indices,packed,batch_idx,stride,boxes,pred2)
    old_position.backward();new_position.backward()
    assert torch.equal(old_position,new_position) and new_visibility==0
    assert torch.equal(pred3.grad[...,:2],pred2.grad)
    assert pred3.grad[...,2].abs().sum()==0
    assert pred2.grad[0,1].abs().sum()==0  # 全缺点person仍在分配中，但pose梯度为0。
    assert pred2.grad[0,0,::3].abs().sum()==0 and pred2.grad[1,:,0].abs().sum()==0
    assert pred2.grad.abs().sum()>0


def test_full_model_upstream_nonvisibility_loss_and_physical_row_gradient_equivalent(models):
    torch,source,xy=models;original,new=criteria(models)
    image=torch.rand((2,3,64,64))
    mask=torch.zeros((2,17),dtype=torch.bool);mask[0,[0,5,6,11,12]]=True
    locations=torch.full((2,17,2),.5)
    batch=dict(keypoints=locations,location_valid=mask,batch_idx=torch.tensor([0,1]),
        cls=torch.zeros((2,1)),bboxes=torch.tensor([[.5,.5,.8,.8],[.5,.5,.8,.8]]))
    from grf_ue_bridge.pose_location_supervision_v2 import pack_location_batch
    upstream_batch=pack_location_batch(batch)
    loss3,items3=original(source(image),upstream_batch)
    new.bce_pose.register_forward_pre_hook(lambda *a:(_ for _ in ()).throw(AssertionError("禁止visibility BCE")))
    loss2,items2=new(xy(image),batch)
    # 对照直接调用完整上游分配/检测/位置loss，剥离它独有的visibility项。
    old=loss3[[0,1,3,4]].sum();current=loss2.sum()
    torch.testing.assert_close(old,current,rtol=1e-6,atol=1e-7)
    torch.testing.assert_close(items3[[0,1,3,4]],items2[[0,1,3,4]],rtol=1e-6,atol=1e-7)
    assert items2[2]==0
    old.backward();current.backward()
    for source_branch,xy_branch in zip(source.model[-1].cv4,xy.model[-1].cv4):
        expected=source_branch[-1].weight.grad;actual=xy_branch[-1][0].weight.grad
        torch.testing.assert_close(expected,actual,rtol=1e-5,atol=1e-7)
        assert actual[2::3].abs().sum()==0
