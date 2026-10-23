"""保留Pose特征塔容量的17×3→17×2输出适配；依赖延迟加载。

仅验证Ultralytics8.4.60普通Pose，拒绝Pose26/end2end/fused头或YAML重建。
工具保留实际模型，调用方仍须核验预训练来源确为COCO17以及固定checkpoint SHA。
"""
from copy import deepcopy

VERSION='coco17_xy_output_rows_preserve_pose_tower_v2'
ULTRALYTICS_VERSION='8.4.60'
XY_ROWS=tuple(3*joint+axis for joint in range(17) for axis in (0,1))
_TRAINER=None
_SELECT_CLASS=None


def _xy_select_init(self):
    import torch
    torch.nn.Module.__init__(self)
    self.register_buffer('rows',torch.tensor(XY_ROWS,dtype=torch.long))


def _xy_select_forward(self,value):
    return value.index_select(1,self.rows)


def _selector_class():
    """延迟建立可按模块名反序列化的无参数索引层，不要求P1安装Torch。"""
    global _SELECT_CLASS
    if _SELECT_CLASS is None:
        import torch
        _SELECT_CLASS=type('SelectPoseXYRows',(torch.nn.Module,),
            {'__module__':__name__,'__init__':_xy_select_init,'forward':_xy_select_forward})
        globals()['SelectPoseXYRows']=_SELECT_CLASS
    return _SELECT_CLASS


def __getattr__(name):
    if name=='SelectPoseXYRows':return _selector_class()
    raise AttributeError(name)


def _imports():
    import torch
    import ultralytics
    from ultralytics.nn.modules.head import Pose
    from ultralytics.nn.tasks import PoseModel
    if ultralytics.__version__!=ULTRALYTICS_VERSION:raise RuntimeError('仅验证Ultralytics'+ULTRALYTICS_VERSION)
    return torch,Pose,PoseModel


def _head(model):
    _,Pose,PoseModel=_imports()
    if type(model) is Pose:return model
    if type(model) is PoseModel and type(model.model[-1]) is Pose:return model.model[-1]
    raise TypeError('只接受实际普通Pose头或PoseModel，不接受路径/YAML/Pose26')


def _validate(model):
    torch,_,_=_imports();head=_head(model)
    if head.end2end or head.cv4 is None or head.nl!=3:raise ValueError('仅验证三尺度非end2end未融合Pose')
    if tuple(head.kpt_shape) not in ((17,3),(17,2)):raise ValueError('只允许17点三维或已迁移二维头')
    mapped=tuple(head.kpt_shape)==(17,2)
    if mapped and getattr(head,'pose_xy_transfer',{}).get('version')!=VERSION:
        raise ValueError('拒绝无标记的标准二维重建头')
    for branch in head.cv4:
        final=branch[-1]
        exact=mapped and head.pose_xy_transfer.get('mode')=='exact_full_conv_then_select'
        if exact:
            if type(final) is not torch.nn.Sequential or len(final)!=2 or type(final[1]) is not _selector_class():raise ValueError('缺少严格输出索引层')
            final=final[0]
        if len(branch)!=3 or type(final) is not torch.nn.Conv2d or final.kernel_size!=(1,1) or final.groups!=1 or final.bias is None:
            raise ValueError('末端Pose结构不支持')
        if final.out_channels!=(51 if exact else head.nk) or final.in_channels<51:raise ValueError('Pose隐藏容量或输出不符合原51维契约')
    if head.nk!=17*head.kpt_shape[1]:raise ValueError('nk与kpt_shape不一致')
    widths=[(branch[-1][0] if type(branch[-1]) is torch.nn.Sequential else branch[-1]).in_channels for branch in head.cv4]
    if mapped and getattr(head,'pose_xy_transfer',{}).get('hidden_channels')!=widths:
        raise ValueError('已迁移隐藏容量标记不一致')
    return head


def map_pose_xy(weights,mode='exact_full_conv_then_select'):
    """复制真实模型，默认保留原51通道末端卷积再无算术索引xy34。

    不归一化dtype，不静默随机重建；允许同助手迁移后的阶段权重再次复制。
    compact_xy_rows可直接裁末端conv，但改变kernel形状，未保证bitwise一致。
    """
    torch,_,PoseModel=_imports()
    if mode not in ('exact_full_conv_then_select','compact_xy_rows'):raise ValueError('不支持的迁移模式')
    if isinstance(weights,dict):weights=weights.get('ema') if isinstance(weights.get('ema'),torch.nn.Module) else weights.get('model')
    original=_validate(weights)
    if tuple(original.kpt_shape)==(17,2) and original.pose_xy_transfer.get('mode')!=mode:
        raise ValueError('已迁移模型模式不同，不能恢复被丢弃的visibility行或静默切换')
    model=deepcopy(weights)
    head=_head(model)
    if tuple(original.kpt_shape)==(17,3):
        for branch in head.cv4:
            old=branch[-1]
            if mode=='exact_full_conv_then_select':
                # 卷积计算形状保持51，随后无算术index_select，严格保留xy浮点路径。
                branch[-1]=torch.nn.Sequential(old,_selector_class()().to(old.weight.device))
                branch[-1].train(old.training)
            else:
                # meta构造不消耗随机状态；后续直接填入源行，没有随机重初始化。
                final=torch.nn.Conv2d(old.in_channels,34,1,bias=True,device='meta',dtype=old.weight.dtype).to_empty(device=old.weight.device)
                with torch.no_grad():
                    final.weight.copy_(old.weight[list(XY_ROWS)])
                    final.bias.copy_(old.bias[list(XY_ROWS)])
                final.train(old.training)
                final.weight.requires_grad_(old.weight.requires_grad)
                final.bias.requires_grad_(old.bias.requires_grad)
                branch[-1]=final
        head.kpt_shape=(17,2);head.nk=34
        head.pose_xy_transfer={'version':VERSION,'source_kpt_shape':[17,3],'target_kpt_shape':[17,2],
              'xy_rows':list(XY_ROWS),'hidden_channels':[(branch[-1][0] if type(branch[-1]) is torch.nn.Sequential else branch[-1]).in_channels for branch in head.cv4],
              'mode':mode,'method':'deepcopy_actual_model_preserve_pose_tower'}
    if type(model) is PoseModel:
        model.yaml=deepcopy(model.yaml);model.yaml['kpt_shape']=[17,2]
        model.yaml['pose_xy_transfer']=deepcopy(head.pose_xy_transfer)
        model.pose_xy_transfer=deepcopy(head.pose_xy_transfer)
        if hasattr(model,'criterion'):delattr(model,'criterion')
    _validate(model)
    return model


def pose_xy_trainer_class():
    """防止标准PoseTrainer依二维YAML把隐藏51重建为34；DDP未验证。"""
    global _TRAINER
    _imports()
    if _TRAINER is None:
        from ultralytics.models.yolo.pose import PoseTrainer
        class PreservedPoseXYTrainer(PoseTrainer):
            def get_model(self,cfg=None,weights=None,verbose=True):
                if self.data.get('kpt_shape')!=[17,2] or self.data.get('nc')!=1:
                    raise ValueError('仅验证单类17×2数据')
                if weights is None:raise ValueError('需要实际PoseModel权重，拒绝YAML随机重建')
                model=map_pose_xy(weights)
                _,_,PoseModel=_imports()
                if type(model) is not PoseModel:raise TypeError('Trainer需要完整PoseModel')
                if model.model[-1].nc!=1:raise ValueError('COCO Pose单类检测头不匹配')
                model.names=dict(self.data['names'])
                return model
        _TRAINER=PreservedPoseXYTrainer
    return _TRAINER
