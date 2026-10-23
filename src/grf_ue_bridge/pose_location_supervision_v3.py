"""独立候选：17×2位置监督显式保留COCO17 sigma，不修改活跃v2。"""

from .pose_location_supervision_v2 import pack_location_batch

VERSION="coco17_separated_location_explicit_oks_sigmas_v3_candidate"
_LOSS_CLASS=None


def _loss_init(self,model,*args,**kwargs):
    import torch
    import ultralytics
    from ultralytics.utils.loss import v8PoseLoss
    from ultralytics.utils.metrics import OKS_SIGMA
    if ultralytics.__version__!="8.4.60" or tuple(model.model[-1].kpt_shape)!=(17,2):
        raise ValueError("v3候选仅核验8.4.60的COCO17×2位置模型")
    v8PoseLoss.__init__(self,model,*args,**kwargs)
    # 上游根据17×3选择COCO sigma；位置头没有第三输出，必须显式保留语义。
    sigma=torch.from_numpy(OKS_SIGMA.copy()).to(self.device)
    if tuple(sigma.shape)!=(17,) or not torch.isfinite(sigma).all() or not (sigma>0).all():
        raise ValueError("本机COCO17 sigma无效")
    self.keypoint_loss.sigmas=sigma
    self.location_contract=dict(version=VERSION,keypoint_shape=[17,2],
        location_valid="independent_bool_not_visibility",sigma_source="ultralytics.utils.metrics.OKS_SIGMA",
        sigmas=sigma.detach().cpu().tolist(),visibility_supervised=False,candidate_only=True)


def _loss_forward(self,preds,batch):
    from ultralytics.utils.loss import v8PoseLoss
    return v8PoseLoss.loss(self,preds,pack_location_batch(batch))


def coco_sigma_location_loss_class():
    """复用上游分配/归一化/位置loss；只修sigma身份，不训练可见性。"""
    global _LOSS_CLASS
    if _LOSS_CLASS is None:
        from ultralytics.utils.loss import v8PoseLoss
        _LOSS_CLASS=type("CocoSigmaLocationPoseLoss",(v8PoseLoss,),
            {"__module__":__name__,"__init__":_loss_init,"loss":_loss_forward})
        globals()["CocoSigmaLocationPoseLoss"]=_LOSS_CLASS
    return _LOSS_CLASS


def __getattr__(name):
    if name=="CocoSigmaLocationPoseLoss":return coco_sigma_location_loss_class()
    raise AttributeError(name)
