"""独立v3候选：COCO sigma、cv4-only、BN固定、零重放预算与间隔断点。"""
import argparse
from copy import copy, deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import signal
import sys
import time

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE / 'src'))
_SPEC = importlib.util.spec_from_file_location('pose_v3_readonly_v2_source_helpers', Path(__file__).with_name('train_pose_location_control.py'))
BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(BASE)
_RETRY_SPEC = importlib.util.spec_from_file_location('pose_v3_owned_replace_helper', Path(__file__).with_name('run_pose_location_control_windows.py'))
RETRY = importlib.util.module_from_spec(_RETRY_SPEC)
_RETRY_SPEC.loader.exec_module(RETRY)

SCHEMA = 'futsalmot_pose_location_poseonly_checkpoint_v3'
PROTOCOL_SCHEMA = 'futsalmot_pose_location_poseonly_protocol_v3'
ADMISSION_SCHEMA = 'futsalmot_pose_location_poseonly_admission_v3'
RUN_SCHEMA = 'futsalmot_pose_location_poseonly_run_v3'
SETTINGS = dict(actual_optimizer_updates=1000, batch=4, nbs=4, gradient_accumulation=1, imgsz=640,
                optimizer='SGD', lr0=.0001, lrf=1., momentum=.9, weight_decay=0., warmup_epochs=0.,
                amp=False, multi_scale=False, early_stopping=False, augmentation=False, checkpoint_interval=50)
POLICY = dict(trainable='pose_cv4_only', model_mode='eval_with_autograd', all_bn_running_stats_frozen=True,
              backbone_neck_detector_frozen=True, visibility_supervision=False, physical_channels=51,
              output_keypoint_shape=[17, 2], sigma='explicit_COCO17_OKS_SIGMA', ema_dtype='float32',
              checkpoint_choice='last1000_actual_FP32_EMA_no_best', hard_crash_budget_replay=False)
require, sha, read, rows, canonical, bound_file = BASE.require, BASE.sha, BASE.read, BASE.rows, BASE.canonical, BASE.bound_file


def source_bindings():
    """冻结实际训练/损失/隔离/兼容依赖；不修改它们。"""
    names = [Path(__file__), Path(BASE.__file__), Path(RETRY.__file__),
             CODE / 'src/grf_ue_bridge/pose_location_supervision_v3.py',
             CODE / 'src/grf_ue_bridge/pose_only_calibration_v3.py',
             CODE / 'src/grf_ue_bridge/pose_location_dataset_v2.py',
             CODE / 'src/grf_ue_bridge/pose_location_supervision_v2.py',
             CODE / 'src/grf_ue_bridge/pose_xy_head_transfer.py']
    return [{'path': str(path.absolute()), 'sha256': sha(path)} for path in names]


def template(output, base_protocol, base_admission, schedule_manifest):
    """仅生成全新不授权模板；不签root、不启动训练。"""
    output = Path(output).absolute()
    require(not output.exists(), '模板目录已存在，拒绝覆盖')
    base_protocol, base_admission, schedule_manifest = map(Path, (base_protocol, base_admission, schedule_manifest))
    prior = read(base_protocol)
    bind = lambda path: dict(path=str(path.absolute()), sha256=sha(path))
    protocol = dict(schema=PROTOCOL_SCHEMA, status='candidate_not_frozen', training_authorized=False,
                    source_v2_protocol=bind(base_protocol), source_v2_admission=bind(base_admission),
                    schedule_manifest=bind(schedule_manifest), source_files=source_bindings(),
                    paired_training=dict(SETTINGS, seeds=[20261024, 20261025, 20261026]), policy=POLICY,
                    initialization=prior['initialization'], arms=prior['arms'],
                    comparison='same frozen R/S/schedule; R4000 vs RS3000R+1000S; explicit lower lr and sigma+freeze jointly change; no single-factor causal claim',
                    evaluation='only completed last1000 actual FP32 EMA; no best; COCO dev proxy not football AssA/final test',
                    test_split_used=False)
    output.mkdir(parents=True)
    BASE.atomic_json(output / 'protocol_template.json', protocol)
    admission = dict(schema=ADMISSION_SCHEMA, status='candidate_not_approved', approved_by=None,
                     training_authorized=False, location_only=True, visibility_supervision=False,
                     protocol=bind(output / 'protocol_template.json'), schedule_manifest=bind(schedule_manifest),
                     source_v2_protocol=bind(base_protocol), source_v2_admission=bind(base_admission),
                     source_files=source_bindings(), policy=POLICY,
                     UE_serial_gate=dict(all_rendering_finished=False, editor_safely_exited=False), test_split_used=False)
    BASE.atomic_json(output / 'admission_template.json', admission)
    return dict(output=str(output), training_authorized=False)


def validate_inputs(protocol_path, manifest_path, admission_path, seed, arm):
    """必须先通过独立v3签署，再只读复用旧输入验证；旧授权不能直接启动。"""
    protocol_path, manifest_path, admission_path = map(lambda p: Path(p).absolute(), (protocol_path, manifest_path, admission_path))
    protocol, admission = read(protocol_path), read(admission_path)
    require(protocol.get('schema') == PROTOCOL_SCHEMA and protocol.get('status') == 'frozen'
            and protocol.get('training_authorized') is True and protocol.get('test_split_used') is False, '需要独立冻结v3协议')
    require(admission.get('schema') == ADMISSION_SCHEMA and admission.get('status') == 'approved_poseonly_location_v3'
            and admission.get('approved_by') == 'root' and admission.get('training_authorized') is True
            and admission.get('location_only') is True and admission.get('visibility_supervision') is False
            and admission.get('test_split_used') is False, '旧v2或候选准入不能授权v3')
    require(bound_file(admission['protocol']) == protocol_path.resolve()
            and bound_file(admission['schedule_manifest']) == manifest_path.resolve()
            and bound_file(protocol['schedule_manifest']) == manifest_path.resolve(), 'v3协议/采样表不匹配')
    require(admission.get('policy') == protocol.get('policy') == POLICY, 'Pose-only/BN/sigma/EMA/预算策略变化')
    settings = protocol.get('paired_training', {})
    require(all(type(settings.get(k)) is type(v) and settings[k] == v for k, v in SETTINGS.items()), 'v3配置必须显式固定候选设置')
    require(settings.get('seeds') == [20261024, 20261025, 20261026], 'v3种子范围变化')
    require(protocol.get('source_files') == admission.get('source_files') == source_bindings(), 'v3代码或兼容依赖SHA改变')
    require(admission.get('UE_serial_gate') == dict(all_rendering_finished=True, editor_safely_exited=True), '需v3新签UE串行门禁')
    v2_protocol = bound_file(protocol['source_v2_protocol'])
    v2_admission = bound_file(protocol['source_v2_admission'])
    require(bound_file(admission['source_v2_protocol']) == v2_protocol
            and bound_file(admission['source_v2_admission']) == v2_admission, '旧来源只读绑定不一致')
    context = BASE.validate_inputs(v2_protocol, manifest_path, v2_admission, seed, arm)
    require(protocol.get('initialization') == context['protocol']['initialization'], '不能更换已冻结初始化')
    require(protocol.get('arms') == context['protocol']['arms'], '真实/合成曝光预算不能改变')
    source = context['bindings']
    context.update(protocol=protocol, admission=admission)
    context['bindings'] = dict(v3_protocol_sha256=sha(protocol_path), v3_admission_sha256=sha(admission_path),
        v3_trainer_source_sha256=sha(__file__), source_files=source_bindings(), source_v2_bindings=source,
        protocol_sha256=sha(protocol_path), admission_sha256=sha(admission_path), trainer_source_sha256=sha(__file__),
        schedule_manifest_sha256=source['schedule_manifest_sha256'], schedule_sha256=source['schedule_sha256'],
        real_jsonl_sha256=source['real_jsonl_sha256'], synthetic_jsonl_sha256=source['synthetic_jsonl_sha256'])
    return context


def replace_owned(source, target):
    """调用有界重试帮助函数，永不monkeypatch活跃v2或全局os.replace。"""
    return RETRY.retry_replace(os.replace, source, target, Path(target).absolute().parent,
                               report=lambda value: print(canonical(value), flush=True))


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    require(not temporary.exists(), '已有事务partial，拒绝覆盖: ' + str(temporary))
    with temporary.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(canonical(value) + '\n'); handle.flush(); os.fsync(handle.fileno())
    replace_owned(temporary, path)


def append_exposure(path, value):
    with Path(path).open('a', encoding='utf-8', newline='\n') as handle:
        handle.write(canonical(value) + '\n'); handle.flush(); os.fsync(handle.fileno())


def model_copy(model):
    result = deepcopy(model).cpu().float()
    if hasattr(result, 'criterion'):
        delattr(result, 'criterion')
    return result


def save_last(path, model, ema, optimizer, completed, context, initial_hash, chain, last_exposure):
    import numpy as np
    import torch
    path = Path(path); temporary = path.with_name(path.name + '.tmp')
    require(not temporary.exists(), '断点partial保留，禁止覆盖/猜测恢复')
    start = time.perf_counter()
    checkpoint = dict(schema=SCHEMA, model=model_copy(model), ema=model_copy(ema.ema),
        optimizer=deepcopy(optimizer.state_dict()), ema_updates=ema.updates, completed_updates=completed,
        arm=context['arm'], seed=context['seed'], bindings=context['bindings'], initial_state_sha256=initial_hash,
        exposure_chain_sha256=chain, last_exposure=last_exposure, planned_updates=1000, policy=POLICY,
        checkpoint_interval=50, checkpoint_choice='last_actual_FP32_EMA_no_best',
        actual_ema_fp32=True, keypoint_shape=[17, 2], physical_output_channels=51, visibility_supervised=False,
        rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                 cuda=torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else None))
    with temporary.open('xb') as handle:
        torch.save(checkpoint, handle); handle.flush(); os.fsync(handle.fileno())
    replace_owned(temporary, path)
    return time.perf_counter() - start


def observed_criterion(model):
    from grf_ue_bridge.pose_location_supervision_v3 import coco_sigma_location_loss_class
    criterion = coco_sigma_location_loss_class()(model)
    criterion.observed = {}
    def record(module, args):
        valid = args[2].bool(); count = valid.shape[0]; nonempty = int(valid.any(1).sum())
        criterion.observed = dict(foreground_anchors=count, foreground_with_valid_points=nonempty,
            assigned_valid_points=int(valid.sum()), effective_pose_weight=float(model.args.pose) * nonempty / max(1, count))
    criterion.keypoint_loss.register_forward_pre_hook(record)
    criterion.bce_pose.register_forward_pre_hook(lambda *args: (_ for _ in ()).throw(AssertionError('禁止visibility BCE')))
    return criterion


def update_ema_preserve_frozen(ema, model, guard):
    """EMA真实更新cv4；冻结项直接复制，避免EMA插值舍入令固定检测器漂移。"""
    import torch
    ema.update(model)
    source = model.state_dict(); target = ema.ema.state_dict()
    with torch.no_grad():
        for name in (*guard.frozen_parameters, *guard.bn_buffers):
            target[name].copy_(source[name])
        for branch, ema_branch in zip(model.model[-1].cv4, ema.ema.model[-1].cv4):
            for parameter, destination in zip((branch[-1][0].weight, branch[-1][0].bias),
                                               (ema_branch[-1][0].weight, ema_branch[-1][0].bias)):
                destination[2::3].copy_(parameter[2::3])
    require(all(torch.equal(target[name], source[name]) for name in (*guard.frozen_parameters, *guard.bn_buffers)), 'EMA冻结网络/BN改变')
    require(all(not value.is_floating_point() or value.dtype == torch.float32 for value in target.values()), 'EMA不是实际FP32')


def reconcile_resume(output, checkpoint, context):
    """仅严格同计数恢复；绝不截断/补写日志或从旧last重放已发生预算。"""
    require(checkpoint.get('schema') == SCHEMA and checkpoint.get('bindings') == context['bindings']
            and checkpoint.get('arm') == context['arm'] and checkpoint.get('seed') == context['seed']
            and checkpoint.get('planned_updates') == 1000 and checkpoint.get('policy') == POLICY, 'v3断点身份不符')
    completed = checkpoint['completed_updates']
    require(type(completed) is int and 0 <= completed <= 1000 and checkpoint['ema_updates'] == completed, '实际/EMA更新计数错误')
    output = Path(output)
    require(not list(output.glob('*.tmp')), '存在未完成写盘partial，拒绝自动回放/覆盖')
    exposures = rows(output / 'exposures.jsonl') if (output / 'exposures.jsonl').exists() else []
    require(len(exposures) == completed, '曝光预算超过或不同于last，硬崩溃尝试拒绝重放')
    transaction = read(output / 'update_transaction.json') if (output / 'update_transaction.json').exists() else None
    require((transaction is None and completed == 0) or (transaction and transaction.get('status') == 'committed'
            and transaction.get('completed_updates') == completed and transaction.get('update_index') == completed - 1
            and transaction.get('arm') == context['arm'] and transaction.get('seed') == context['seed']
            and transaction.get('exposure_chain_sha256') == checkpoint['exposure_chain_sha256']),
            '未提交/超出last的更新事务，拒绝预算重放')
    chain = '0' * 64
    for index, exposure in enumerate(exposures):
        value = dict(exposure); recorded = value.pop('chain_sha256')
        require(value['actual_optimizer_update'] == index + 1 and value['update_index'] == index, '曝光次序改变')
        require(value['arm'] == context['arm'] and value['seed'] == context['seed'], '曝光身份改变')
        chain = hashlib.sha256((chain + canonical(value)).encode()).hexdigest()
        require(chain == recorded, '曝光链改变')
    require(chain == checkpoint['exposure_chain_sha256'] and
            checkpoint.get('last_exposure') == (exposures[-1] if exposures else None), 'last/曝光链不一致')
    return completed


def train_loop(model, criterion, optimizer, ema, guard, real, synthetic, schedule, context, output,
               completed=0, chain='0' * 64, initial_hash=None, last_exposure=None, stop_after=None,
               stop_requested=None, checkpoint_interval=50, planned_updates=1000):
    """小型更新引擎；CPU toy可传2步，公开train始终1000步/50间隔。"""
    import torch
    from grf_ue_bridge.pose_location_dataset_v2 import LocationPoseDataset
    require(type(checkpoint_interval) is int and checkpoint_interval > 0, '断点间隔非法')
    output = Path(output); device = next(model.parameters()).device
    guard.enforce_eval_mode(); guard.assert_unchanged()
    initial_hash = initial_hash or BASE.state_hash(model)
    seconds = 0.; saves = []; start = time.perf_counter()
    def checkpoint():
        nonlocal seconds
        seconds += save_last(output / 'last.pt', model, ema, optimizer, completed, context, initial_hash, chain, last_exposure)
        saves.append(completed)
    if completed == 0:
        checkpoint()
    paused = False
    for row in schedule[completed:]:
        if (stop_requested and stop_requested()) or (stop_after is not None and completed >= stop_after):
            paused = True; break
        index = row['update_index']; require(index == completed, '更新不连续')
        start_update = time.perf_counter(); mixed = context['arm'] == 'RS25'
        indices = row['shared_real_indices']
        samples = [real[i] for i in indices]
        samples.append(synthetic[row['synthetic_index']] if mixed else real[row['extra_real_index']])
        batch = LocationPoseDataset.collate_fn(samples)
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)
        batch['img'] = batch['img'].float() / 255.
        atomic_json(output / 'update_transaction.json', dict(schema='poseonly_update_transaction_v3',
            status='started_not_committed', update_index=index, completed_updates_before=completed,
            arm=context['arm'], seed=context['seed']))
        guard.enforce_eval_mode(); optimizer.zero_grad(set_to_none=True); criterion.observed = {}
        loss, items = criterion(model(batch['img']), batch)
        require(torch.isfinite(loss).all().item() and torch.isfinite(items).all().item() and items[2].item() == 0., '非有限loss或visibility监督')
        if loss.requires_grad:
            loss.sum().backward()
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        require(all(parameter.grad is None or torch.isfinite(parameter.grad).all().item() for parameter in trainable), '非有限Pose梯度，停止且不重放预算')
        nonzero = sum(int(parameter.grad is not None and bool(parameter.grad.abs().sum())) for parameter in trainable)
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=10.)
        optimizer.step(); guard.assert_unchanged()
        update_ema_preserve_frozen(ema, model, guard)
        completed += 1; valid = batch['location_valid']
        exposure = dict(schema='poseonly_step_exposure_v3', actual_optimizer_update=completed, update_index=index,
            arm=context['arm'], seed=context['seed'], real_indices=indices + ([] if mixed else [row['extra_real_index']]),
            synthetic_indices=[row['synthetic_index']] if mixed else [], image_files=batch['im_file'],
            source_image_keys=[BASE.source_key(real, i) for i in indices] +
                [BASE.source_key(synthetic, row['synthetic_index']) if mixed else BASE.source_key(real, row['extra_real_index'])],
            real_images=3 if mixed else 4, synthetic_images=1 if mixed else 0, person_boxes=len(valid),
            people_with_valid_points=int(valid.any(1).sum()), all_missing_people=int((~valid.any(1)).sum()),
            valid_points=int(valid.sum()), foreground_anchors=0, effective_pose_weight=0.,
            learning_rate=optimizer.param_groups[0]['lr'], loss_items=items.tolist(), visibility_loss=0.,
            visibility_rows_unchanged=True, bn_buffers_unchanged=True, frozen_parameters_unchanged=True,
            trainable_parameters_with_nonzero_gradient=nonzero, zero_gradient_update=nonzero == 0,
            elapsed_seconds=time.perf_counter() - start_update)
        exposure.update(criterion.observed)
        chain = hashlib.sha256((chain + canonical(exposure)).encode()).hexdigest()
        last_exposure = dict(exposure, chain_sha256=chain)
        append_exposure(output / 'exposures.jsonl', last_exposure)
        atomic_json(output / 'update_transaction.json', dict(schema='poseonly_update_transaction_v3', status='committed',
            update_index=index, completed_updates=completed, exposure_chain_sha256=chain,
            arm=context['arm'], seed=context['seed']))
        if completed % checkpoint_interval == 0:
            checkpoint()
        if stop_after is not None and completed >= stop_after:
            paused = completed < planned_updates; break
    if not saves or saves[-1] != completed:
        checkpoint()
    return dict(completed_updates=completed, planned_updates=1000, complete=completed == 1000,
                status='complete' if completed == 1000 else 'safely_paused', paused=paused,
                checkpoint_seconds=seconds, checkpoints_saved_at_updates=saves,
                elapsed_seconds=time.perf_counter() - start, initial_state_sha256=initial_hash,
                exposure_chain_sha256=chain, guard=guard.assert_unchanged())


def train(context, output, device='cuda:0', resume=False, cache_mib=3072, stop_after=None):
    import numpy as np
    import torch
    import ultralytics
    from ultralytics.utils import DEFAULT_CFG
    from ultralytics.utils.torch_utils import ModelEMA
    from grf_ue_bridge.pose_location_dataset_v2 import LocationPoseDataset
    from grf_ue_bridge.pose_xy_head_transfer import map_pose_xy
    from grf_ue_bridge.pose_only_calibration_v3 import configure_pose_only
    require(ultralytics.__version__ == '8.4.60' and torch.__version__ == '2.6.0+cu124', '仅当前研究运行时')
    require(device in ('cpu', 'cuda:0') and not BASE.editor_running(), 'UE未安全退出或设备不支持')
    require(type(cache_mib) is int and 0 <= cache_mib <= 4096, '缓存须0..4096MiB')
    require(stop_after is None or type(stop_after) is int and 1 <= stop_after <= 1000, '暂停点须1..1000累计步')
    output = Path(output).absolute()
    for path in (output, *output.parents):
        require(not path.exists() or (not path.is_symlink() and not getattr(path.stat(), 'st_file_attributes', 0) & 0x400), '输出禁止联接')
    allowed = Path('D:/projects/FutsalMOT_Research/pose/poseonly_v3')
    require(allowed in output.parents, 'v3训练仅允许全新Research/pose/poseonly_v3子目录，禁止写v2')
    require((output / 'last.pt').is_file() if resume else not output.exists(), '必须新目录或明确last续跑')
    output.mkdir(parents=True, exist_ok=resume)
    seed = context['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.startswith('cuda'):
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    initialization = context['protocol']['initialization']; initial_path = Path(initialization['derived_checkpoint'])
    require(sha(initial_path) == initialization['derived_sha256'], '初始化SHA改变')
    loaded = torch.load(output / 'last.pt' if resume else initial_path, map_location='cpu', weights_only=False)
    completed = reconcile_resume(output, loaded, context) if resume else 0
    require(completed < 1000, 'last1000已完成，只能独立评价，禁止重复训练/覆盖最终断点')
    model = map_pose_xy(loaded['model'].float()).to(device); model.args = copy(DEFAULT_CFG)
    guard = configure_pose_only(model)
    optimizer = torch.optim.SGD((parameter for parameter in model.parameters() if parameter.requires_grad),
                                lr=.0001, momentum=.9, weight_decay=0.)
    ema = ModelEMA(model)
    if resume:
        optimizer.load_state_dict(loaded['optimizer']); ema.ema.load_state_dict(loaded['ema'].state_dict()); ema.updates = loaded['ema_updates']
        require(all(group['lr'] == .0001 and group['momentum'] == .9 and group['weight_decay'] == 0 for group in optimizer.param_groups), '恢复优化器配置改变')
        random.setstate(loaded['rng']['python']); np.random.set_state(loaded['rng']['numpy']); torch.set_rng_state(loaded['rng']['torch'])
        if loaded['rng']['cuda'] is not None:
            torch.cuda.set_rng_state_all(loaded['rng']['cuda'])
    real_base = LocationPoseDataset(context['real_jsonl'], 640, verify_image_sha256=True)
    synthetic_base = LocationPoseDataset(context['synthetic_jsonl'], 640, verify_image_sha256=False)
    limit = cache_mib * 1024 * 1024; mixed = context['arm'] == 'RS25'
    real = BASE.CachedDataset(real_base, limit // 2 if mixed else limit)
    synthetic = BASE.CachedDataset(synthetic_base, limit - limit // 2 if mixed else 0)
    cache = dict(limit_mib=cache_mib, real=real.prewarm(range(1000)),
                 synthetic=synthetic.prewarm(context['selected_synthetic_indices'] if mixed else []))
    criterion = observed_criterion(model)
    stopped = [False]; previous = signal.signal(signal.SIGINT, lambda *args: stopped.__setitem__(0, True))
    try:
        result = train_loop(model, criterion, optimizer, ema, guard, real, synthetic, context['schedule'], context, output,
            completed=completed, chain=loaded['exposure_chain_sha256'] if resume else '0' * 64,
            initial_hash=loaded['initial_state_sha256'] if resume else None,
            last_exposure=loaded.get('last_exposure') if resume else None, stop_after=stop_after,
            stop_requested=lambda: stopped[0])
    finally:
        signal.signal(signal.SIGINT, previous); guard.close()
    result.update(schema=RUN_SCHEMA, arm=context['arm'], seed=seed, bindings=context['bindings'], policy=POLICY,
        cache=cache, device=device, source_not_training_flags_preserved=True, real_test_read=False,
        AP_evaluated=False, visibility_supervised=False, actual_ema_fp32=True, keypoint_shape=[17, 2],
        physical_output_channels=51, checkpoint_choice='last1000_actual_FP32_EMA_no_best',
        actual_exposures=dict(real=result['completed_updates'] * (3 if mixed else 4), synthetic=result['completed_updates'] * int(mixed)),
        runtime=dict(torch=torch.__version__, ultralytics=ultralytics.__version__, cuda=torch.version.cuda))
    atomic_json(output / 'run_receipt.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['template', 'train'])
    for key in ('protocol', 'schedule-manifest', 'admission', 'output', 'base-protocol', 'base-admission'):
        parser.add_argument('--' + key, type=Path)
    parser.add_argument('--arm', choices=['R', 'RS25']); parser.add_argument('--seed', type=int)
    parser.add_argument('--device', choices=['cpu', 'cuda:0'], default='cuda:0')
    parser.add_argument('--resume', action='store_true'); parser.add_argument('--cache-mib', type=int, default=3072)
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    os.environ.update(YOLO_AUTOINSTALL='false', YOLO_OFFLINE='true', WANDB_MODE='disabled', COMET_MODE='DISABLED', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    if args.action == 'template':
        result = template(args.output, args.base_protocol, args.base_admission, args.schedule_manifest)
    else:
        context = validate_inputs(args.protocol, args.schedule_manifest, args.admission, args.seed, args.arm)
        result = train(context, args.output, args.device, args.resume, args.cache_mib, args.stop_after)
    print(canonical(result))


if __name__ == '__main__':
    main()
