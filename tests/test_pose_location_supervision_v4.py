"""实际随机YOLO CPU数学夹具；不读真实图像、不启动训练或下载权重。"""
from copy import copy, deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def fixture():
    torch = pytest.importorskip("torch")
    ultralytics = pytest.importorskip("ultralytics")
    from ultralytics.nn.tasks import PoseModel
    from ultralytics.utils import DEFAULT_CFG, YAML
    from grf_ue_bridge.pose_xy_head_transfer import map_pose_xy
    from grf_ue_bridge.pose_location_supervision_v4 import SeparateDomainLocationObjective
    torch.set_num_threads(1); torch.manual_seed(20261029)
    config = YAML.load(Path(ultralytics.__file__).parent / "cfg/models/11/yolo11-pose.yaml")
    config["scale"] = "n"
    model = map_pose_xy(PoseModel(config, ch=3, nc=1, verbose=False)).cpu()
    model.args = copy(DEFAULT_CFG)
    objective = SeparateDomainLocationObjective(model)
    yield torch, model, objective
    objective.close()


def batch(torch, domain, persons=(2, 1, 2), *, all_missing=False):
    image_count = len(persons); indices = []; boxes = []; xy = []; masks = []
    for image, count in enumerate(persons):
        for person in range(count):
            # 网格位置由标签定义，不依赖模型分数；S可比R更稠密。
            center = torch.tensor([.2 + .2 * (person % 4), .3 + .35 * ((person // 4) % 2)])
            points = center + (torch.rand((17, 2)) - .5) * .12
            valid = torch.rand(17) > .2; valid[0] = False
            if all_missing or (image == 0 and person == 0): valid[:] = False
            points[~valid] = 0.
            indices.append(image); boxes.append([center[0], center[1], .34, .4])
            xy.append(points); masks.append(valid)
    valid = torch.stack(masks)
    states = [["unknown"] * 17 if domain == "synthetic" else
              ["unlabeled" if not bool(ok) else "visible" if i % 2 else "occluded"
               for i, ok in enumerate(row)] for row in valid]
    return dict(img=torch.rand((image_count, 3, 64, 64)), keypoints=torch.stack(xy),
        location_valid=valid, batch_idx=torch.tensor(indices), cls=torch.zeros((len(indices), 1)),
        bboxes=torch.tensor(boxes), sources=[domain] * len(indices), visibility_state=states)


def gradient_copy(torch, model):
    return {name: p.grad.detach().clone() if p.grad is not None else None for name, p in model.named_parameters()}


def assert_equal_gradients(torch, left, right):
    assert set(left) == set(right)
    for name in left:
        if left[name] is None:
            assert right[name] is None, name
        else:
            assert torch.equal(left[name], right[name]), name


def test_alpha_zero_real_loss_gradient_exact_and_dense_S_cannot_change_real_denominator(fixture):
    torch, model, objective = fixture
    real = batch(torch, "coco_real"); original = deepcopy(real)
    sparse = batch(torch, "synthetic", (1,)); dense = batch(torch, "synthetic", (10,))
    baseline = objective(real)
    baseline.loss.backward(); expected = gradient_copy(torch, model)
    assert sum(g.abs().sum().item() for g in expected.values() if g is not None) > 0
    for synthetic in (sparse, dense):
        model.zero_grad(set_to_none=True)
        zero = objective.alpha_zero_cpu_probe(real, synthetic)
        assert torch.equal(zero.real_mean, baseline.real_mean)
        assert torch.equal(zero.loss, baseline.loss)
        assert zero.report["real"] == baseline.report["real"]
        assert zero.report["alpha"] == 0 and zero.report["probe_only"]
        zero.loss.backward()
        assert_equal_gradients(torch, expected, gradient_copy(torch, model))
    assert objective.real_criterion is not objective.synthetic_criterion
    assert torch.equal(real["location_valid"], original["location_valid"])
    assert real["visibility_state"] == original["visibility_state"]
    objective.guard.assert_unchanged()


def test_real_term_equals_direct_v3_location_component_divided_by_three(fixture):
    torch, model, objective = fixture
    from grf_ue_bridge.pose_location_supervision_v3 import coco_sigma_location_loss_class
    from ultralytics.utils.metrics import OKS_SIGMA
    real = batch(torch, "coco_real")
    native = coco_sigma_location_loss_class()(model)
    loss, _ = native(model(real["img"]), real)
    direct = loss[1] / 3
    direct.backward(); gradients = gradient_copy(torch, model)
    model.zero_grad(set_to_none=True)
    result = objective(real)
    assert torch.equal(direct, result.loss)
    result.loss.backward()
    assert_equal_gradients(torch, gradients, gradient_copy(torch, model))
    assert torch.equal(objective.real_criterion.keypoint_loss.sigmas, torch.from_numpy(OKS_SIGMA))
    assert result.report["real"]["upstream_batch_size_divisor"] == 3


def test_fixed_alpha_unknown_S_has_location_gradient_and_one_step_preserves_detector_BN_vis(fixture):
    torch, model, objective = fixture
    from grf_ue_bridge.pose_location_supervision_v4 import SYNTHETIC_ALPHA
    real = batch(torch, "coco_real"); synthetic = batch(torch, "synthetic", (10,))
    before = model(real["img"])
    detection = before[0][:, :5].detach().clone()
    raw_boxes = before[1]["boxes"].detach().clone(); raw_scores = before[1]["scores"].detach().clone()
    pose = before[1]["kpts"].detach().clone()
    result = objective(real, synthetic)
    assert result.report["alpha"] == SYNTHETIC_ALPHA == 1 / 3
    # 固定登记算子为乘α；浮点除3虽数学等价，却不承诺逐bit等价。
    assert torch.equal(result.loss, result.real_mean + SYNTHETIC_ALPHA * result.synthetic_mean)
    assert result.report["synthetic"]["raw_valid_points"] > result.report["real"]["raw_valid_points"]
    assert result.report["real"]["visibility_loss"] == result.report["synthetic"]["visibility_loss"] == 0
    parameters = [p for p in model.parameters() if p.requires_grad]
    synthetic_gradient = torch.autograd.grad(result.synthetic_mean, parameters, retain_graph=True, allow_unused=True)
    assert sum(g.abs().sum().item() for g in synthetic_gradient if g is not None) > 0
    optimizer = torch.optim.SGD(parameters, lr=.0001, momentum=0, weight_decay=0)
    result.loss.backward()
    assert all(p.grad is None for name, p in model.named_parameters() if ".cv4." not in name)
    for branch in model.model[-1].cv4:
        assert branch[-1][0].out_channels == 51
        assert branch[-1][0].weight.grad[2::3].abs().sum() == 0
        assert branch[-1][0].bias.grad[2::3].abs().sum() == 0
    optimizer.step(); objective.guard.assert_unchanged()
    after = model(real["img"])
    assert torch.equal(detection, after[0][:, :5])
    assert torch.equal(raw_boxes, after[1]["boxes"]) and torch.equal(raw_scores, after[1]["scores"])
    assert not torch.equal(pose, after[1]["kpts"])
    assert not any(module.training for module in model.modules() if isinstance(module, torch.nn.BatchNorm2d))


def test_missing_assigned_points_have_zero_gradient_and_unknown_is_not_masked(fixture):
    torch, model, objective = fixture
    captured = []
    def capture(_module, args):
        prediction, _gt, mask, _area = args
        prediction.retain_grad(); captured.append((prediction, mask.detach().bool()))
    handles = [c.keypoint_loss.register_forward_pre_hook(capture) for c in
               (objective.real_criterion, objective.synthetic_criterion)]
    real = batch(torch, "coco_real"); synthetic = batch(torch, "synthetic", (4,))
    result = objective(real, synthetic); result.loss.backward()
    assert len(captured) == 2
    for prediction, valid in captured:
        assert valid.any() and (~valid).any()
        assert prediction.grad[~valid].abs().sum() == 0
        assert prediction.grad[valid].abs().sum() > 0
    for handle in handles: handle.remove()
    model.zero_grad(set_to_none=True)
    missing = batch(torch, "synthetic", (3,), all_missing=True)
    zero = objective(real, missing)
    assert zero.report["synthetic"]["person_boxes"] == 3
    assert zero.report["synthetic"]["assigned_valid_points"] == 0
    assert zero.synthetic_mean.item() == 0
    gradients = torch.autograd.grad(zero.synthetic_mean, [p for p in model.parameters() if p.requires_grad], allow_unused=True)
    assert all(g is None or g.abs().sum() == 0 for g in gradients)


def test_reject_bad_domain_size_visibility_promotion_and_mixed_index(fixture):
    torch, _model, objective = fixture
    real = batch(torch, "coco_real"); synthetic = batch(torch, "synthetic", (3,))
    wrong = deepcopy(real); wrong["img"] = torch.rand((4, 3, 64, 64))
    with pytest.raises(ValueError, match="R3或S1"): objective(wrong)
    wrong = deepcopy(synthetic); wrong["visibility_state"][0][0] = "visible"
    with pytest.raises(ValueError, match="不提升可见性"): objective(real, wrong)
    wrong = deepcopy(synthetic); wrong["batch_idx"] += 3
    with pytest.raises(ValueError, match="本域0"): objective(real, wrong)
    wrong = deepcopy(real); wrong["location_valid"] = wrong["location_valid"].float()
    with pytest.raises(ValueError, match="17bool"): objective(wrong)
    wrong = deepcopy(real); wrong["location_valid"][0, 0] = True
    with pytest.raises(ValueError, match="v0"): objective(wrong)
