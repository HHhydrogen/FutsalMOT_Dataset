"""正式无号码批次的UE侧入口；仅由本机MCP调用，P1不得导入。"""
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import unreal

project = Path(unreal.Paths.project_dir()).resolve()
private = project / 'Saved/Production/v1_no_numbers_4k'
runtime = private / 'runtime'
command = json.loads((private / 'control.json').read_text(encoding='utf-8'))
entry = command['entry']
settings = json.loads((private / 'production_plan.json').read_text(encoding='utf-8'))['settings']
assert project == Path(settings['project']).resolve()
assert Path(__file__).resolve() == (runtime / 'ue/production_control.py').resolve()
sys.path.insert(0, str(runtime / 'ue'))
if 'run_task' not in sys.modules:
    import run_task
else:
    run_task = sys.modules['run_task']
import render_episode
import episode_appearance as appearance
import episode_environment as environment
import diagnose_episode_appearance as appearance_diag
import diagnose_episode_environment as environment_diag
import import_grf_episode

episode = Path(entry['episode'])
assert episode.parent.resolve() == Path(settings['work_root']).resolve()
mapping = json.loads((runtime / 'ue/actor_mapping.example.json').read_text(encoding='utf-8'))
manifest = json.loads((episode / 'provenance/environment.json').read_text(encoding='utf-8'))
evidence_dir = private / 'evidence' / entry['task_id']
evidence_dir.mkdir(parents=True, exist_ok=True)


def dump(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(str(temporary), str(path))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fresh_gate():
    app = appearance_diag.collect_diagnostic(unreal, mapping)
    weather = environment_diag.collect_environment_diagnostic(unreal, manifest['parameters']['venue_fill_lighting'])
    assert app['current_level'] == settings['map'] and app['task7_preflight_can_start'], app
    assert not app['dirty_package_paths'] and weather['preflight_can_start'], (app, weather)
    assert not unreal.EditorLevelLibrary.get_pie_worlds(False)
    assert not unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem).is_rendering()
    assert getattr(render_episode, '_ACTIVE_RENDER', None) is None
    return {'appearance':app, 'environment':weather}


def baseline_check(fresh):
    baseline = json.loads((runtime / 'baselines/weather.json').read_text(encoding='utf-8'))
    for label, record in baseline['actors'].items():
        assert environment._values_match(record['baseline_readback'], fresh['environment']['actors'][label]['baseline_readback']), label
    manual = json.loads((runtime / 'baselines/manual_lights.json').read_text(encoding='utf-8'))
    current = {}
    for actor in unreal.EditorLevelLibrary.get_all_level_actors():
        if isinstance(actor, unreal.SpotLight):
            component = actor.get_component_by_class(unreal.SpotLightComponent)
            record = {'location':str(actor.get_actor_location()), 'rotation':str(actor.get_actor_rotation()), 'scale':str(actor.get_actor_scale3d())}
            for prop in ('intensity','intensity_units','light_color','attenuation_radius','inner_cone_angle','outer_cone_angle','cast_shadows'):
                record[prop] = str(component.get_editor_property(prop))
            current[actor.get_actor_label()] = record
    def stable(records):
        return {key:{name:re.sub(r'\(0x[0-9a-fA-F]+\)', '', value) for name,value in row.items()} for key,row in records.items()}
    assert stable(current) == stable(manual), '手动场地灯与原始基线不同'
    return stable(current)


action = command['action']
if action == 'render':
    assert not (episode / 'render_summary.json').exists()
    unreal.LevelSequenceEditorBlueprintLibrary.close_level_sequence()
    gate = fresh_gate()
    gate['manual_lights'] = baseline_check(gate)
    assert hasattr(unreal, 'FutsalMoviePipelinePNGOutput'), '当前FutsalMOT module缺少已验收的PNG writer'
    unreal.log('PRODUCTION_MODULE_VERIFIED FutsalMOT FutsalMoviePipelinePNGOutput')
    local = runtime / '.futsalmot/runtime' / entry['task_id']
    assert not list(local.glob('*rollback*'))
    queue = {name:unreal.SystemLibrary.get_console_variable_int_value(name) for name in ('ImageWriteQueue.MaxConcurrency','ImageWriteQueue.MaxQueueSize')}
    assert queue == {'ImageWriteQueue.MaxConcurrency':settings['image_write_concurrency'], 'ImageWriteQueue.MaxQueueSize':settings['image_write_queue_size']}, queue
    gate['image_write_queue'] = queue
    dump(evidence_dir / 'preflight.json', gate)
    create_with_preview = import_grf_episode.create_sequence
    def create_without_preview(*args, **kwargs):
        result = create_with_preview(*args, **kwargs)
        unreal.LevelSequenceEditorBlueprintLibrary.close_level_sequence()
        return result
    import_grf_episode.create_sequence = create_without_preview
    execution = {'schema':'futsalmot_render_execution','version':1, 'task_id':entry['task_id'],
        'runtime_source_sha256':{p.relative_to(runtime).as_posix():digest(p) for folder in ('src','ue') for p in (runtime / folder).rglob('*.py')},
        'image_write_queue':queue, 'module_dll_sha256':digest(project / 'Binaries/Win64/UnrealEditor-FutsalMOT.dll'),
        'output_policy':'FinalImage PNG only; native multilayer EXR; numbers/pose disabled',
        'orchestration_sha256':digest(Path(__file__)), 'jobs':[]}
    dump(episode / 'provenance/render_execution.json', execution)
    original_builder = render_episode._build_combined_mrq_job
    def filtered_builder(*args, **kwargs):
        job = original_builder(*args, **kwargs)
        config = render_episode._get_job_config(job)
        original = config.find_setting_by_class(unreal.MoviePipelineImageSequenceOutput_PNG)
        assert original is not None
        alpha = original.get_editor_property('write_alpha')
        config.remove_setting(original)
        filtered = config.find_or_add_setting_by_class(unreal.FutsalMoviePipelinePNGOutput)
        filtered.set_editor_property('write_alpha', alpha)
        assert config.find_setting_by_class(unreal.MoviePipelineImageSequenceOutput_EXR) is not None
        execution['jobs'].append({'settings':[s.get_class().get_name() for s in config.get_all_settings()]})
        dump(episode / 'provenance/render_execution.json', execution)
        assert render_episode._set_job_config(job, config)
        return job
    render_episode._build_combined_mrq_job = filtered_builder
    old_argv = sys.argv
    try:
        os.environ.pop('C5_CAMERA_IDS', None)
        sys.argv = ['run_task.py','--resolved-task',entry['resolved'],'--mode','full']
        result = run_task.main()
        assert result == 0, result
    finally:
        sys.argv = old_argv
        os.environ.pop('C5_RESOLVED_TASK', None)
        os.environ.pop('C5_RUN_MODE', None)
    dump(evidence_dir / 'submitted.json', {'submitted':True, 'task_id':entry['task_id']})
elif action == 'cancel':
    assert getattr(render_episode, '_ACTIVE_RENDER', None) is not None
    result = render_episode._ACTIVE_RENDER.request_cancel()
    dump(evidence_dir / 'cancel_requested.json', {'result':str(result)})
elif action == 'finalize':
    summary = json.loads((episode / 'render_summary.json').read_text(encoding='utf-8'))
    assert summary['status'] in ('success','failed','cancelled'), summary
    for kind in ('appearance','environment'):
        report = json.loads((episode / ('provenance/' + kind + '_application.json')).read_text(encoding='utf-8'))
        assert report['state'] == 'restored' and not report.get('errors') and not report.get('mrq_pending'), report
    gate = fresh_gate()
    gate['manual_lights'] = baseline_check(gate)
    local = runtime / '.futsalmot/runtime' / entry['task_id']
    rollback, report_path = local / 'appearance-rollback.json', local / 'appearance-application.json'
    if rollback.exists():
        snapshot = json.loads(rollback.read_text(encoding='utf-8'))
        report = json.loads(report_path.read_text(encoding='utf-8'))
        gate['snapshot_readback'] = appearance.preflight_episode_appearance_recovery(snapshot, mapping, True)
        assert gate['snapshot_readback']['ok'], gate['snapshot_readback']
        for path in local.glob('*.json'):
            (evidence_dir / ('restored_' + path.name)).write_bytes(path.read_bytes())
        app_manifest = json.loads((episode / 'provenance/appearance.json').read_text(encoding='utf-8'))
        resolved = json.loads(Path(entry['resolved']).read_text(encoding='utf-8'))
        run_task._recover_local_appearance_evidence(appearance, rollback, report_path,
            episode / 'provenance/appearance_application.json', snapshot, report, app_manifest,
            resolved['ue_profile']['appearance'], mapping)
    assert not list(local.glob('*rollback*')), '存在尚未退休的恢复证据'
    dump(evidence_dir / 'finalized.json', {'passed':True, 'render_status':summary['status'], **gate})
elif action == 'quit':
    assert json.loads((evidence_dir / 'finalized.json').read_text(encoding='utf-8'))['passed']
    gate = fresh_gate()
    assert not list((runtime / '.futsalmot/runtime').glob('*/*rollback*'))
    dump(evidence_dir / 'shutdown.json', {'normal_quit':True, **gate})
    unreal.SystemLibrary.quit_editor()
else:
    raise ValueError('未知正式调度动作：{}'.format(action))
print('PRODUCTION_UE_ACTION ' + action + ' ' + entry['task_id'])
