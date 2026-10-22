"""实验会话的MCP入口；正常完成时保留当前场景和UE进程。"""
import json
from pathlib import Path
import sys
import unreal

private = Path(unreal.Paths.project_dir()).resolve() / 'Saved/Production/v1_no_numbers_4k'
runtime = Path(json.loads((private / 'runtime_freeze.json').read_text(encoding='utf-8')).get('runtime_root', str(private / 'runtime')))
sys.path.insert(0, str(runtime / 'ue'))
import production_session
import import_grf_episode
import render_episode
from scene_retention import file_sha
command = json.loads((private / 'session_control.json').read_text(encoding='utf-8'))
entry = command['entry']
episode = Path(entry.get('episode', str(private)))
mapping = json.loads((runtime / 'ue/actor_mapping.example.json').read_text(encoding='utf-8'))
if not hasattr(production_session, '_SESSION'):
    production_session._SESSION = production_session.ProductionSession(private / 'sessions', mapping)
session = production_session._SESSION
if command['action'] == 'cancel':
    assert render_episode._ACTIVE_RENDER is not None
    assert render_episode._ACTIVE_RENDER.request_cancel()
    print('RETAINED_SESSION_CANCEL_REQUESTED')
    raise SystemExit(0)
assert not session.blocked and not session.active
assert not unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem).is_rendering()
if command['action'] in ('render', 'continue'):
    policy = json.loads((private / 'execution_policy.json').read_text(encoding='utf-8'))
    override = policy.get('task_write_queue_overrides', {}).get(entry['task_id'])
    if override:
        policy = dict(policy, **override)
    import production_write_queue
    write_queue = production_write_queue.configure(entry, policy)
    execution = {'schema': 'futsalmot_render_execution', 'version': 1, 'task_id': entry['task_id'],
        'orchestration_sha256': file_sha(Path(__file__)), 'scene_lifecycle': 'retain', 'session_id': session.session_id, 'raw_disk': entry['raw_disk'],
        'image_write_queue': write_queue,
        'module_dll_sha256': file_sha(Path(unreal.Paths.project_dir()) / 'Binaries/Win64/UnrealEditor-FutsalMOT.dll'),
        'runtime_source_sha256': {p.relative_to(runtime).as_posix(): file_sha(p) for folder in ('ue', 'src') for p in (runtime / folder).rglob('*.py')},
        'output_policy': 'FinalImage PNG only; native multilayer EXR; numbers/pose disabled', 'jobs': []}
    if command['action'] == 'continue':
        execution = json.loads((episode / 'provenance/render_execution.json').read_text(encoding='utf-8'))
        execution.setdefault('queue_tuning_events', []).append({'before_camera': len(execution['jobs']) + 1,
            'image_write_queue': write_queue, 'policy_revision': policy['revision'],
            'orchestration_sha256': file_sha(Path(__file__))})
    original = render_episode._build_combined_mrq_job
    def filtered(*args, **kwargs):
        job = original(*args, **kwargs)
        config = render_episode._get_job_config(job)
        png = config.find_setting_by_class(unreal.MoviePipelineImageSequenceOutput_PNG)
        assert png is not None
        alpha = png.get_editor_property('write_alpha')
        config.remove_setting(png)
        replacement = config.find_or_add_setting_by_class(unreal.FutsalMoviePipelinePNGOutput)
        replacement.set_editor_property('write_alpha', alpha)
        assert config.find_setting_by_class(unreal.MoviePipelineImageSequenceOutput_EXR) is not None
        execution['jobs'].append({'settings': [s.get_class().get_name() for s in config.get_all_settings()],
            'image_write_queue': write_queue})
        production_session.dump(episode / 'provenance/render_execution.json', execution)
        assert render_episode._set_job_config(job, config)
        return job
    render_episode._build_combined_mrq_job = filtered
    try:
        if command['action'] == 'render':
            session.render(entry['resolved'], raw_output_root=entry.get('raw_root'))
        else:
            session.continue_render()
        production_write_queue.watch()
    finally:
        render_episode._build_combined_mrq_job = original
elif command['action'] == 'collect':
    retired = production_session.release_finished_renderers()
    print('RETIRED_EXECUTORS ' + json.dumps(retired))
else:
    raise ValueError(command['action'])
print('RETAINED_SESSION_ACTION ' + command['action'] + ' ' + entry['task_id'])
