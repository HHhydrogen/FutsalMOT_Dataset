"""独立v4数学候选：相同R3曝光，R/S分源计算位置项；不管理训练或协议。"""
from dataclasses import dataclass
from typing import Any, Optional

from .pose_location_supervision_v3 import coco_sigma_location_loss_class
from .pose_only_calibration_v3 import configure_pose_only

VERSION = "pose_location_separate_R3_S1_fixed_alpha_v4_candidate"
SYNTHETIC_ALPHA = 1. / 3.


@dataclass
class SeparatedLocationResult:
    """保留可求导分量；report仅为本次调用计数，不是训练准入。"""
    loss: Any
    real_mean: Any
    synthetic_mean: Optional[Any]
    report: dict


class SeparateDomainLocationObjective:
    """固定R3/S1，复用COCO sigma及pose-only冻结；绝不拼域共享FG分母。"""

    def __init__(self, model, freeze_guard=None):
        self.model = model
        if freeze_guard is not None and freeze_guard.model is not model:
            raise ValueError("冻结guard必须绑定同一模型")
        self.guard = freeze_guard if freeze_guard is not None else configure_pose_only(model)
        self.owns_guard = freeze_guard is None
        self.guard.assert_unchanged()
        self.handles = []
        self.observed = {"real": {}, "synthetic": {}}
        loss_type = coco_sigma_location_loss_class()
        self.real_criterion = loss_type(model)
        self.synthetic_criterion = loss_type(model)
        for domain, criterion in (("real", self.real_criterion), ("synthetic", self.synthetic_criterion)):
            def record(_module, args, domain=domain):
                mask = args[2].detach().bool()
                count = mask.shape[0]
                self.observed[domain] = dict(foreground_anchors=count,
                    foreground_with_valid_points=int(mask.any(1).sum().item()),
                    assigned_valid_points=int(mask.sum().item()),
                    keypoint_mean_elements=count * 17)
            self.handles.append(criterion.keypoint_loss.register_forward_pre_hook(record))
            self.handles.append(criterion.bce_pose.register_forward_pre_hook(self._reject_visibility))
        self.contract = dict(version=VERSION, candidate_only=True, training_authorized=False,
            real_batch_size=3, synthetic_batch_size=1, synthetic_alpha=SYNTHETIC_ALPHA,
            real_reduction="batch_scaled_location_component_divided_by_3",
            synthetic_reduction="batch_scaled_location_component_divided_by_1",
            foreground_denominator_shared_between_domains=False,
            location_valid="independent_bool_not_visibility", visibility_supervised=False,
            extra_synthetic_compute_is_equal_FLOPs=False,
            sigma_source="ultralytics.utils.metrics.OKS_SIGMA",
            optimizer_gradient_clip_EMA_schedule_and_admission_managed_by_caller=True)

    @staticmethod
    def _reject_visibility(*_args):
        raise AssertionError("v4禁止physical visibility BCE")

    def _validate_batch(self, batch, domain, count):
        import torch
        image = batch.get("img")
        parameter = next(self.model.parameters())
        if (not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[0] != count
                or image.shape[1] != 3 or image.dtype != parameter.dtype or image.device != parameter.device
                or not image.is_floating_point() or not torch.isfinite(image).all().item()
                or image.min().item() < 0 or image.max().item() > 1):
            raise ValueError("v4需要同模型设备/类型且已归一[0,1]的R3或S1图像")
        points = batch.get("keypoints"); valid = batch.get("location_valid")
        if (not isinstance(points, torch.Tensor) or points.ndim != 3 or points.shape[1:] != (17, 2)
                or not isinstance(valid, torch.Tensor) or valid.shape != points.shape[:2]
                or valid.dtype != torch.bool):
            raise ValueError("v4需要独立17×2及17bool有效性")
        sources = batch.get("sources"); visibility = batch.get("visibility_state")
        expected = "coco_real" if domain == "real" else "synthetic"
        if (not isinstance(sources, list) or len(sources) != len(points) or any(s != expected for s in sources)
                or not isinstance(visibility, list) or len(visibility) != len(points)
                or any(not isinstance(row, list) or len(row) != 17 for row in visibility)):
            raise ValueError("位置batch必须显式保留分域来源和原visibility")
        if domain == "synthetic" and any(state != "unknown" for row in visibility for state in row):
            raise ValueError("本v4候选仅接当前unknown合成位置记录，不提升可见性")
        if domain == "real":
            for row, mask in zip(visibility, valid):
                if any(state not in ("visible", "occluded", "unlabeled") for state in row):
                    raise ValueError("真实COCO需保留原v状态")
                if any(state == "unlabeled" and bool(flag.item()) for state, flag in zip(row, mask)):
                    raise ValueError("COCO v0不能提供位置监督")
        indices = batch.get("batch_idx")
        if (not isinstance(indices, torch.Tensor) or indices.numel() != len(points)
                or indices.ndim != 1 or not torch.isfinite(indices).all().item()
                or (len(indices) and (indices.min().item() < 0 or indices.max().item() >= count
                                     or not torch.equal(indices, indices.floor())))):
            raise ValueError("分域batch_idx必须从本域0起，不沿用混合batch偏移")
        if (points.device != parameter.device or valid.device != parameter.device
                or not torch.isfinite(points).all().item()
                or not ((points[valid] >= 0) & (points[valid] < 1)).all().item()
                or not (points[~valid] == 0).all().item()):
            raise ValueError("有效位置必须画内有限，原false为零占位")

    def _domain(self, batch, domain, count):
        import torch
        self._validate_batch(batch, domain, count)
        self.observed[domain] = dict(foreground_anchors=0, foreground_with_valid_points=0,
                                     assigned_valid_points=0, keypoint_mean_elements=0)
        criterion = self.real_criterion if domain == "real" else self.synthetic_criterion
        loss, items = criterion(self.model(batch["img"]), batch)
        if (tuple(loss.shape) != (5,) or tuple(items.shape) != (5,)
                or not torch.isfinite(loss).all().item() or not torch.isfinite(items).all().item()
                or items[2].item() != 0 or loss[2].item() != 0):
            raise AssertionError("v4上游返回契约改变、非有限或出现visibility loss")
        mean = loss[1] / count  # 上游8.4.60返回位置项已乘本域batch_size。
        observed = dict(self.observed[domain], images=count, person_boxes=len(batch["location_valid"]),
            raw_valid_points=int(batch["location_valid"].sum().item()),
            upstream_batch_size_divisor=count, upstream_batch_scaled_location=float(loss[1].detach().item()),
            mean_location=float(mean.detach().item()), visibility_loss=0.,
            source_visibility_preserved=True)
        return mean, observed

    def _compute(self, real_batch, synthetic_batch, *, zero_probe=False):
        self.guard.enforce_eval_mode()
        self.guard.assert_unchanged()
        real, real_report = self._domain(real_batch, "real", 3)
        synthetic = None; synthetic_report = None
        if synthetic_batch is not None:
            synthetic, synthetic_report = self._domain(synthetic_batch, "synthetic", 1)
        alpha = 0. if synthetic is None or zero_probe else SYNTHETIC_ALPHA
        total = real if synthetic is None else real + alpha * synthetic
        return SeparatedLocationResult(total, real, synthetic,
            dict(self.contract, mode="alpha_zero_cpu_probe" if zero_probe else "R3S1" if synthetic is not None else "R3",
                 alpha=alpha, real=real_report, synthetic=synthetic_report, probe_only=zero_probe,
                 returned_loss_is_location_only=True, gradients_are_before_caller_clip=True))

    def __call__(self, real_batch, synthetic_batch=None):
        """实际候选只允许R3基线或固定α=1/3的R3+S1；不接任意调参α。"""
        return self._compute(real_batch, synthetic_batch)

    def alpha_zero_cpu_probe(self, real_batch, synthetic_batch):
        """仅CPU数学验收：实际计算S但系数0，不能把此模式当正式采样臂。"""
        if next(self.model.parameters()).device.type != "cpu":
            raise ValueError("alpha0探针只允许CPU")
        if synthetic_batch is None:
            raise ValueError("探针必须提供S1以检验分母隔离")
        return self._compute(real_batch, synthetic_batch, zero_probe=True)

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        if self.owns_guard:
            self.guard.close()
