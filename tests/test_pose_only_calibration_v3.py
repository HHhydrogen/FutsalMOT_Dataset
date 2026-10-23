"""单个CPU随机toy步证明pose-only更新不改变检测输出，不运行真实实验。"""
from copy import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))


def test_pose_step_preserves_detector_values_bn_and_unused_physical_rows():
    torch=pytest.importorskip("torch");ultralytics=pytest.importorskip("ultralytics")
    from ultralytics.nn.tasks import PoseModel
    from ultralytics.utils import DEFAULT_CFG,YAML
    from grf_ue_bridge.pose_xy_head_transfer import map_pose_xy
    from grf_ue_bridge.pose_only_calibration_v3 import configure_pose_only
    from grf_ue_bridge.pose_location_supervision_v3 import coco_sigma_location_loss_class
    torch.set_num_threads(1);torch.manual_seed(23)
    config=YAML.load(Path(ultralytics.__file__).parent/"cfg/models/11/yolo11-pose.yaml");config["scale"]="n"
    model=map_pose_xy(PoseModel(config,ch=3,nc=1,verbose=False)).cpu();model.args=copy(DEFAULT_CFG)
    guard=configure_pose_only(model)
    assert guard.report["trainable_parameters"] and all(".cv4." in n for n in guard.report["trainable_parameters"])
    criterion=coco_sigma_location_loss_class()(model)
    criterion.bce_pose.register_forward_pre_hook(lambda *a:(_ for _ in ()).throw(AssertionError("禁止visibility BCE")))
    optimizer=torch.optim.SGD((p for p in model.parameters() if p.requires_grad),lr=.0001,momentum=.9,weight_decay=0.)
    image=torch.rand((2,3,64,64));before=model(image)
    frozen_detection=before[0][:,:5].detach().clone()
    old_boxes=before[1]["boxes"].detach().clone();old_scores=before[1]["scores"].detach().clone()
    old_pose=before[1]["kpts"].detach().clone()
    mask=torch.zeros((2,17),dtype=torch.bool);mask[0,[0,5,6,11,12]]=True
    batch=dict(keypoints=torch.full((2,17,2),.5),location_valid=mask,batch_idx=torch.tensor([0,1]),
        bboxes=torch.tensor([[.5,.5,.8,.8],[.5,.5,.8,.8]]),cls=torch.zeros((2,1)))
    loss,items=criterion(before,batch);assert items[1]>0 and items[2]==0
    loss.sum().backward()
    assert sum(p.grad.abs().sum().item() for p in model.model[-1].cv4.parameters() if p.grad is not None)>0
    assert all(p.grad is None for name,p in model.named_parameters() if ".cv4." not in name)
    optimizer.step();guard.assert_unchanged()
    after=model(image)
    assert torch.equal(frozen_detection,after[0][:,:5])
    assert torch.equal(old_boxes,after[1]["boxes"]) and torch.equal(old_scores,after[1]["scores"])
    assert not torch.equal(old_pose,after[1]["kpts"])
    model.train()
    with pytest.raises(AssertionError,match="BN进入训练态"):guard.assert_unchanged()
    guard.enforce_eval_mode();guard.assert_unchanged();guard.close()
