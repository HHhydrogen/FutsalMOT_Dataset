"""独立下一轮候选：只训练Pose cv4，固定共享网络、检测分支及BN统计。"""
from .pose_xy_head_transfer import _validate

VERSION="pose_cv4_only_all_bn_statistics_frozen_v3_candidate"


class PoseOnlyFreezeGuard:
    """保存冻结断言；不调优化器、不选协议、不自动启动训练。"""
    def __init__(self,model):
        import torch
        head=_validate(model)
        if tuple(head.kpt_shape)!=(17,2) or head.pose_xy_transfer.get("mode")!="exact_full_conv_then_select":
            raise ValueError("只接受已验exact_xy 17×2模型，不重建检测器")
        self.model=model;self.handles=[]
        model.eval()
        for parameter in model.parameters(): parameter.requires_grad_(False)
        for parameter in head.cv4.parameters(): parameter.requires_grad_(True)
        self.frozen_parameters={name:parameter.detach().clone() for name,parameter in model.named_parameters()
                                if not parameter.requires_grad}
        self.bn_buffers={name:buffer.detach().clone() for name,buffer in model.named_buffers()
                         if name.endswith(("running_mean","running_var","num_batches_tracked"))}
        self.visibility=[]
        for branch in head.cv4:
            conv=branch[-1][0]
            for parameter in (conv.weight,conv.bias):
                mask=torch.ones_like(parameter);mask[2::3]=0
                self.handles.append(parameter.register_hook(lambda grad,mask=mask:grad*mask))
                self.visibility.append((parameter,parameter[2::3].detach().clone()))
        self.report=dict(version=VERSION,candidate_only=True,all_bn_running_statistics_frozen=True,
            model_forward_mode="eval_with_autograd_not_no_grad",visibility_supervised=False,
            trainable_parameters=[name for name,p in model.named_parameters() if p.requires_grad],
            trainable_parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad),
            frozen_parameter_count=sum(p.numel() for p in model.parameters() if not p.requires_grad))

    def enforce_eval_mode(self):
        """若外层曾调用train()，须显式恢复eval再前向，保持所有BN统计固定。"""
        self.model.eval()

    def assert_unchanged(self):
        import torch
        require=lambda ok,message:None if ok else (_ for _ in ()).throw(AssertionError(message))
        parameters=dict(self.model.named_parameters());buffers=dict(self.model.named_buffers())
        require(not any(module.training for module in self.model.modules() if isinstance(module,torch.nn.BatchNorm2d)),
                "BN进入训练态，不能宣称固定共享网络")
        require(all(not parameters[name].requires_grad and torch.equal(parameters[name],value)
                    for name,value in self.frozen_parameters.items()),"冻结网络参数改变")
        require(all(torch.equal(buffers[name],value) for name,value in self.bn_buffers.items()),"BN统计改变")
        require(all(torch.equal(parameter[2::3],value) for parameter,value in self.visibility),"物理visibility输出行改变")
        return dict(frozen_parameters_unchanged=True,bn_buffers_unchanged=True,visibility_rows_unchanged=True)

    def close(self):
        for handle in self.handles: handle.remove()
        self.handles=[]


def configure_pose_only(model):
    """原地配置明确下一轮候选，调用方管理guard、冻结协议与新run目录。"""
    return PoseOnlyFreezeGuard(model)
