"""v3独立入口CPU toy：新准入、2更新、无重放恢复与检测/EMA冻结。"""
from copy import copy, deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

CODE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('poseonly_v3_cpu_test', CODE / 'experiments/sim2real/train_pose_location_poseonly_v3.py')
TOOL = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(TOOL)


def dump(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')
    return dict(path=str(path), sha256=TOOL.sha(path))


def binding_fixture(root):
    spec = importlib.util.spec_from_file_location('pose_v3_old_source_fixture', CODE / 'tests/test_pose_location_control_trainer.py')
    fixture = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture)
    v2_protocol, manifest, v2_admission = fixture.bound_fixture(root)
    output = root / 'v3_templates'
    TOOL.template(output, v2_protocol, v2_admission, manifest)
    protocol_path, admission_path = output / 'protocol_template.json', output / 'admission_template.json'
    protocol, admission = TOOL.read(protocol_path), TOOL.read(admission_path)
    protocol.update(status='frozen', training_authorized=True)
    admission.update(status='approved_poseonly_location_v3', training_authorized=True, approved_by='root',
                     UE_serial_gate=dict(all_rendering_finished=True, editor_safely_exited=True))
    admission['protocol'] = dump(protocol_path, protocol)
    dump(admission_path, admission)
    return protocol_path, manifest, admission_path


def test_new_protocol_and_admission_exactly_bind_same_frozen_source(tmp_path):
    paths = binding_fixture(tmp_path)
    result = TOOL.validate_inputs(*paths, 20261024, 'RS25')
    assert result['protocol']['paired_training']['lr0'] == .0001
    assert result['bindings']['protocol_sha256'] == TOOL.sha(paths[0])
    assert result['bindings']['source_v2_bindings']['protocol_sha256'] != TOOL.sha(paths[0])
    assert len(result['schedule']) == 1000


@pytest.mark.parametrize('damage', ['v2_admission', 'candidate', 'policy', 'lr', 'source_code', 'source_binding'])
def test_old_or_changed_authorization_cannot_start_v3(tmp_path, damage):
    paths = binding_fixture(tmp_path)
    protocol, admission = TOOL.read(paths[0]), TOOL.read(paths[2])
    if damage == 'v2_admission':
        admission['schema'] = 'futsalmot_pose_location_training_admission_v1'
    elif damage == 'candidate':
        protocol['status'] = 'candidate_not_frozen'
    elif damage == 'policy':
        protocol['policy']['all_bn_running_stats_frozen'] = False
    elif damage == 'lr':
        protocol['paired_training']['lr0'] = .001
    elif damage == 'source_code':
        protocol['source_files'][0]['sha256'] = '0' * 64
    else:
        protocol['source_v2_admission']['sha256'] = '0' * 64
    admission['protocol'] = dump(paths[0], protocol); dump(paths[2], admission)
    with pytest.raises(ValueError):
        TOOL.validate_inputs(*paths, 20261024, 'RS25')


def minimal_checkpoint(context, completed=0):
    return dict(schema=TOOL.SCHEMA, bindings=context['bindings'], arm=context['arm'], seed=context['seed'],
                planned_updates=1000, policy=TOOL.POLICY, completed_updates=completed, ema_updates=completed,
                exposure_chain_sha256='0' * 64, last_exposure=None)


def test_hard_crash_beyond_interval_checkpoint_rejected_without_log_mutation(tmp_path):
    context = dict(arm='R', seed=23, bindings={'toy_only': True})
    checkpoint = minimal_checkpoint(context)
    TOOL.append_exposure(tmp_path / 'exposures.jsonl', {'actual_optimizer_update': 1})
    before = (tmp_path / 'exposures.jsonl').read_bytes()
    with pytest.raises(ValueError, match='预算超过'):
        TOOL.reconcile_resume(tmp_path, checkpoint, context)
    assert (tmp_path / 'exposures.jsonl').read_bytes() == before


def test_pending_update_or_partial_rejected_even_without_exposure(tmp_path):
    context = dict(arm='R', seed=23, bindings={'toy_only': True})
    checkpoint = minimal_checkpoint(context)
    TOOL.atomic_json(tmp_path / 'update_transaction.json', dict(status='started_not_committed', update_index=0))
    with pytest.raises(ValueError, match='事务'):
        TOOL.reconcile_resume(tmp_path, checkpoint, context)
    (tmp_path / 'last.pt.tmp').write_bytes(b'partial')
    with pytest.raises(ValueError, match='partial'):
        TOOL.reconcile_resume(tmp_path, checkpoint, context)


def test_owned_atomic_json_refuses_leftover_partial(tmp_path):
    path = tmp_path / 'run_receipt.json'
    TOOL.atomic_json(path, {'a': 1})
    (tmp_path / 'run_receipt.json.tmp').write_text('partial', encoding='utf-8')
    with pytest.raises(ValueError, match='partial'):
        TOOL.atomic_json(path, {'a': 2})
    assert TOOL.read(path) == {'a': 1}


def test_two_cpu_steps_safe_signal_pause_resume_ema_detector_and_interval(tmp_path):
    torch = pytest.importorskip('torch'); ultralytics = pytest.importorskip('ultralytics')
    from ultralytics.nn.tasks import PoseModel
    from ultralytics.utils import DEFAULT_CFG, YAML
    from ultralytics.utils.torch_utils import ModelEMA
    from grf_ue_bridge.pose_xy_head_transfer import map_pose_xy
    from grf_ue_bridge.pose_only_calibration_v3 import configure_pose_only
    from grf_ue_bridge.pose_location_dataset_v2 import LocationPoseDataset
    spec = importlib.util.spec_from_file_location('pose_v3_toy_images', CODE / 'tools/smoke_pose_location_jsonl_cpu.py')
    fixture = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture)
    fixture.make_toy_jsonl(tmp_path / 'toy')
    dataset = LocationPoseDataset(tmp_path / 'toy/train.jsonl', 64)
    torch.set_num_threads(1); torch.manual_seed(309)
    cfg = YAML.load(Path(ultralytics.__file__).parent / 'cfg/models/11/yolo11-pose.yaml'); cfg['scale'] = 'n'
    original = map_pose_xy(PoseModel(cfg, ch=3, nc=1, verbose=False)).cpu(); original.args = copy(DEFAULT_CFG)
    probe = torch.rand((2, 3, 64, 64)); original.eval()
    with torch.no_grad():
        initial_detection = original(probe)[0][:, :5].clone()
    context = dict(arm='RS25', seed=309, bindings=dict(toy_only=True, training_authorized=False))
    schedule = [dict(update_index=i, shared_real_indices=[0, 0, 0], extra_real_index=0, synthetic_index=1) for i in range(2)]
    def components(model):
        guard = configure_pose_only(model)
        optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=.0001, momentum=.9, weight_decay=0.)
        return guard, TOOL.observed_criterion(model), optimizer, ModelEMA(model)
    model = deepcopy(original); guard, criterion, optimizer, ema = components(model)
    full = tmp_path / 'full'; full.mkdir()
    result = TOOL.train_loop(model, criterion, optimizer, ema, guard, dataset, dataset, schedule, context,
                             full, checkpoint_interval=2, planned_updates=2)
    assert result['checkpoints_saved_at_updates'] == [0, 2]
    assert torch.equal(initial_detection, model(probe)[0][:, :5])
    assert torch.equal(initial_detection, ema.ema(probe)[0][:, :5])
    guard.close()
    model = deepcopy(original); guard, criterion, optimizer, ema = components(model)
    paused = tmp_path / 'paused'; paused.mkdir(); signal_checks = iter((False, True))
    result = TOOL.train_loop(model, criterion, optimizer, ema, guard, dataset, dataset, schedule, context,
                             paused, stop_requested=lambda: next(signal_checks), checkpoint_interval=50)
    assert result['completed_updates'] == 1 and result['paused'] and result['checkpoints_saved_at_updates'] == [0, 1]
    guard.close()
    saved = torch.load(paused / 'last.pt', map_location='cpu', weights_only=False)
    assert TOOL.reconcile_resume(paused, saved, context) == 1
    model = saved['model'].float(); guard, criterion, optimizer, ema = components(model)
    optimizer.load_state_dict(saved['optimizer']); ema.ema.load_state_dict(saved['ema'].state_dict()); ema.updates = saved['ema_updates']
    TOOL.train_loop(model, criterion, optimizer, ema, guard, dataset, dataset, schedule, context, paused,
        completed=1, chain=saved['exposure_chain_sha256'], initial_hash=saved['initial_state_sha256'],
        last_exposure=saved['last_exposure'], checkpoint_interval=50)
    a = torch.load(full / 'last.pt', map_location='cpu', weights_only=False)
    b = torch.load(paused / 'last.pt', map_location='cpu', weights_only=False)
    assert a['completed_updates'] == b['completed_updates'] == a['ema_updates'] == b['ema_updates'] == 2
    for field in ('model', 'ema'):
        for key, value in a[field].state_dict().items():
            assert torch.equal(value, b[field].state_dict()[key])
        assert torch.equal(initial_detection, b[field](probe)[0][:, :5])
    for key, value in a['optimizer']['state'].items():
        assert torch.equal(value['momentum_buffer'], b['optimizer']['state'][key]['momentum_buffer'])
    assert all(not value.is_floating_point() or value.dtype == torch.float32 for value in b['ema'].state_dict().values())
    exposures = TOOL.rows(paused / 'exposures.jsonl')
    assert len(exposures) == 2 and all(row['person_boxes'] == 7 and row['all_missing_people'] == 3 for row in exposures)
    assert all(row['valid_points'] == 41 and row['visibility_loss'] == 0 and row['bn_buffers_unchanged'] for row in exposures)
    assert TOOL.reconcile_resume(paused, b, context) == 2
    guard.close()
