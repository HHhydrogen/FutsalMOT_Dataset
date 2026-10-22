"""UE 内持久生产会话：绝对参数覆盖、逐组 checkpoint、结束后保留场景。"""
import copy
import hashlib
import json
import os
from pathlib import Path
import uuid


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(str(temporary), str(path))


def release_finished_renderers():
    """拆除Python回调与UE executor之间的跨语言引用环；仅在PIE/MRQ结束后执行。"""
    import gc
    import unreal
    import render_episode
    assert not unreal.EditorLevelLibrary.get_pie_worlds(False)
    subsystem = unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem)
    assert not subsystem.is_rendering() and render_episode._ACTIVE_RENDER is None
    retired = []
    candidates = [obj for obj in gc.get_objects() if type(obj).__module__ == 'render_episode'
        and type(obj).__name__ == '_AsyncRenderPipeline' and obj.finished]
    for pipeline in candidates:
        executor = pipeline.executor
        if executor is not None:
            for name, callback in [('on_executor_finished_delegate', pipeline._on_finished),
                ('on_executor_errored_delegate', pipeline._on_errored)]:
                delegate = getattr(executor, name)
                delegate.remove_callable(callback)
            retired.append(executor.get_path_name())
        pipeline.executor = None
        pipeline.on_complete = None
        for info in pipeline.jobs + pipeline.requested_jobs:
            info.pop('job', None)
        pipeline.jobs = []
    executor = None
    delegate = None
    render_episode._clear_queue(render_episode._mrq_get_queue(subsystem))
    gc.collect()
    unreal.SystemLibrary.collect_garbage()
    return retired


class ProductionSession:
    """仅接管全新、已验证 clean 的会话；后续只接受本会话的原样 checkpoint。"""

    def __init__(self, private, mapping):
        self.private = Path(private)
        self.mapping = mapping
        self.session_id = hashlib.sha256(uuid.uuid4().bytes).hexdigest()
        self.checkpoint = None
        self.active = False
        self.blocked = False
        self.episode = None
        self.sequence = None
        self.owned_packages = set()
        self.pending_sequences = []
        self.camera_results = []

    def inspect(self, fill):
        import unreal
        import episode_appearance as appearance
        import episode_environment as environment
        import diagnose_episode_appearance as app_diag
        import diagnose_episode_environment as env_diag
        import render_episode
        assert not unreal.EditorLevelLibrary.get_pie_worlds(False)
        assert not unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem).is_rendering()
        assert getattr(render_episode, '_ACTIVE_RENDER', None) is None
        app = app_diag.collect_diagnostic(unreal, self.mapping)
        weather = env_diag.collect_environment_diagnostic(unreal, fill)
        assert app['current_level'] == '/Game/FutsalMOT/Maps/L_FutsalCourt'
        assert app['roster']['ok'] and app['key_asset_checks_ok'] and not app['dirty_key_asset_package_paths'], app
        expected_dirty_messages = {label + ' package dirty' for label in environment.ACTOR_CLASSES}
        assert not [error for error in weather.get('errors', []) if not error.startswith('存在 dirty package') and error not in expected_dirty_messages], weather
        actors, errors = appearance._actor_matches(unreal, self.mapping)
        assert not errors
        dirty = sorted(appearance._dirty_package_paths(unreal))
        player_state = {key: appearance._component_state(actor, appearance._component_for_actor(unreal, actor), appearance._actor_package_path(actor) in dirty)
            for key, actor in actors.items()}
        for key, actor in actors.items():
            component = appearance._component_for_actor(unreal, actor)
            slot = appearance._semantic_slot_data(component.get_skeletal_mesh_asset())['Body']
            material = component.get_material(slot)
            if isinstance(material, unreal.MaterialInstanceDynamic):
                player_state[key]['skin_texture_readback'] = appearance._package_path(material.get_texture_parameter_value('SkinTexture'))
        scene_actors = {}
        tracked_labels = set(self.mapping.values()) | set(environment.MANUAL_SPOTLIGHT_ACTOR_LABELS) | {'CineCam_%02d' % index for index in range(1,5)}
        for actor in unreal.EditorLevelLibrary.get_all_level_actors():
            label = actor.get_actor_label()
            if label in tracked_labels:
                record = {'location': [actor.get_actor_location().x, actor.get_actor_location().y, actor.get_actor_location().z],
                    'rotation': [actor.get_actor_rotation().pitch, actor.get_actor_rotation().yaw, actor.get_actor_rotation().roll]}
                if isinstance(actor, unreal.CineCameraActor):
                    camera = actor.get_cine_camera_component()
                    filmback = camera.get_editor_property('filmback')
                    record.update(focal_length=camera.get_editor_property('current_focal_length'), sensor=[filmback.sensor_width, filmback.sensor_height])
                scene_actors[label] = record
        if self.owned_packages:
            assert not set(dirty) - self.owned_packages, '发现本会话范围外的未保存改动'
        state = {'players': player_state, 'environment': weather['actors'],
            'manual_lights': weather['manual_spot_lighting'], 'scene_actors': scene_actors, 'dirty_packages': dirty, 'level': app['current_level']}
        return state, app, weather, actors

    def render(self, resolved_path, *, raw_output_root=None):
        import unreal
        import episode_appearance as appearance
        import episode_environment as environment
        import import_grf_episode
        import annotation_exporter
        import render_episode
        from scene_retention import file_sha
        assert not self.active and not self.blocked
        resolved = json.loads(Path(resolved_path).read_text(encoding='utf-8'))
        episode = Path(resolved['dataset_episode_dir'])
        profile = resolved['ue_profile']
        manifest = json.loads((episode / 'provenance/appearance.json').read_text(encoding='utf-8'))
        weather_manifest = json.loads((episode / 'provenance/environment.json').read_text(encoding='utf-8'))
        meta, frames = import_grf_episode.load_episode(episode)
        assert not (episode / 'render_summary.json').exists()
        assert not weather_manifest['parameters']['venue_fill_lighting']['enabled'], '本入口仅接受正式v1无临时RectLight配置'
        assert not appearance.validate_appearance_inputs(manifest, profile['appearance'], meta, episode / 'frames.jsonl')
        assert not environment.validate_environment_inputs(weather_manifest, profile['environment_profile'], meta, episode / 'frames.jsonl')
        unreal.LevelSequenceEditorBlueprintLibrary.close_level_sequence()
        state, app, weather, actors = self.inspect(weather_manifest['parameters']['venue_fill_lighting'])
        if self.checkpoint is None:
            assert app['task7_preflight_can_start'] and not state['dirty_packages'] and weather['preflight_can_start']
            snapshots = appearance._snapshot_roster(unreal, actors, set())
            dump(self.private / self.session_id / 'initial_player_baseline.json', snapshots)
            dump(self.private / self.session_id / 'initial_environment_baseline.json', weather)
        else:
            assert environment._values_match(state, self.checkpoint), '当前场景与本会话上次checkpoint不同，拒绝覆盖'
        if not self.owned_packages:
            labels = set(self.mapping.values()) | set(environment.ACTOR_CLASSES) | set(environment.MANUAL_SPOTLIGHT_ACTOR_LABELS)
            labels.update(row['camera_actor'] for row in profile['sequences'])
            self.owned_packages = {appearance._actor_package_path(actor)
                for actor in unreal.EditorLevelLibrary.get_all_level_actors() if actor.get_actor_label() in labels}
        dump(self.private / self.session_id / ('before_' + resolved['task_id'] + '.json'), state)
        loaded = appearance._preload_assignments(unreal, manifest, profile['appearance'])
        plan = environment.build_environment_property_write_plan(weather_manifest, weather)
        assert plan['ok'], plan
        app_report = appearance.create_application_report(manifest, self.mapping)
        weather_report = environment.build_environment_application_report(weather_manifest, self.session_id,
            'not_started', environment._application_actor_records(environment._diagnostic_actors(weather)), [], [])
        transport = environment.UEEnvironmentTransport()
        self.active = True
        self.episode = episode
        try:
            transport.track_environment_actors()
            for label, fields in plan['plan'].items():
                for field, entry in fields.items():
                    transport.write(label, entry, entry['value'])
                    assert environment._values_match(transport.read(label, entry), entry['value'], tolerance=environment._field_tolerance(field)), (label, field)
            transport.apply_manual_spot_lighting(weather_manifest['parameters']['venue_spot_lighting'])
            transport.commit()
            for entity in manifest['entities']:
                result = appearance._apply_entity(unreal, actors[entity['entity_id']], entity, loaded)
                record = next(row for row in app_report['actors'] if row['entity_id'] == entity['entity_id'])
                record.update(result)
            app_report.update(state='applied', transaction_id=self.session_id)
            weather_report.update(state='applied')
            for row in weather_report['actors']:
                row['apply_status'] = 'applied'
            for report in (app_report, weather_report):
                report.update(scene_lifecycle='retain', session_id=self.session_id, mrq_pending=True)
            self.reports = {'appearance': app_report, 'environment': weather_report}
            for kind, report in self.reports.items():
                dump(episode / ('provenance/' + kind + '_application.json'), report)
            self.sequence = profile['sequence_package_path']
            import_grf_episode.create_sequence(meta, frames, self.mapping, profile.get('replace_existing', True),
                self.sequence, profile['sequences'], profile.get('ball_rolling'))
            unreal.LevelSequenceEditorBlueprintLibrary.close_level_sequence()
            annotations = copy.deepcopy(profile['annotation_export'])
            annotations['resolved_appearance_policy'] = profile['appearance']['policy']
            annotations['export_mot'] = False
            annotation_exporter.export_annotations(episode, Path(resolved['actor_mapping']), episode.parent, annotations)
            if raw_output_root is not None:
                # MRQ原始输出直接写SSD；canonical RGB的移交仍由原有完成逻辑负责。
                # 输出目录同时用于MRQ与完成回调，使用junction保证两者读取同一组文件。
                for row in profile['sequences']:
                    camera = row['camera_actor']
                    link = episode / camera / 'render'
                    target = Path(raw_output_root) / resolved['task_id'] / camera
                    assert link.resolve() == target.resolve(), 'SSD实验必须由P1显式创建并核验render junction'
            self.pending_sequences = list(profile['sequences'])
            self.camera_results = []
            self.inter_camera_checkpoint = None
            self.render_options = dict(annotations=annotations, mapping_path=resolved['actor_mapping'],
                environment_policy=weather_manifest['policy'], fingerprint=weather_manifest['fingerprint'], preset=weather_manifest['preset'])
            self.continue_render()
        except BaseException:
            self.active = False
            self.blocked = True
            raise

    def continue_render(self):
        """相机之间回到独立MCP调用，释放上一次PIE/输出任务后再提交下一机位。"""
        import render_episode
        import unreal
        assert not self.blocked and self.pending_sequences
        assert not unreal.EditorLevelLibrary.get_pie_worlds(False)
        assert not unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem).is_rendering()
        assert render_episode._ACTIVE_RENDER is None
        if self.inter_camera_checkpoint is not None:
            import episode_environment as environment
            manifest = json.loads((self.episode / 'provenance/environment.json').read_text(encoding='utf-8'))
            current, _app, _weather, _actors = self.inspect(manifest['parameters']['venue_fill_lighting'])
            assert environment._values_match(current, self.inter_camera_checkpoint), '机位之间场景被修改，拒绝继续'
        release_finished_renderers()
        row = self.pending_sequences.pop(0)
        self.active = True
        options = self.render_options
        try:
            render_episode.render_sequences([row], options['annotations'], self.sequence, self.episode,
                self.episode.parent, mapping_path=Path(options['mapping_path']), on_complete=self.camera_complete,
                appearance_active=True, environment_active=True, environment_policy=options['environment_policy'],
                environment_manifest_fingerprint=options['fingerprint'], environment_preset=options['preset'],
                environment_preflight_verified=True, environment_transaction_state='applied')
        except BaseException:
            self.active = False
            self.blocked = True
            raise

    def camera_complete(self, result):
        """逐机位保存原始成功终态；最终报告明确说明四次串行MRQ组成一组。"""
        try:
            assert result['status'] == 'success', result
            self.camera_results.append(copy.deepcopy(result))
            task_dir = self.private / self.session_id / result['episode_id']
            dump(task_dir / ('camera_%02d.json' % len(self.camera_results)), result)
            self.active = False
            if self.pending_sequences:
                manifest = json.loads((self.episode / 'provenance/environment.json').read_text(encoding='utf-8'))
                self.inter_camera_checkpoint, _app, _weather, _actors = self.inspect(manifest['parameters']['venue_fill_lighting'])
                dump(self.episode / 'provenance/camera_progress.json', {'completed': len(self.camera_results),
                    'next_camera': self.pending_sequences[0]['camera_actor'], 'session_id': self.session_id})
                return
            summary = json.loads((self.episode / 'render_summary.json').read_text(encoding='utf-8'))
            combined = copy.deepcopy(result)
            combined['cameras'] = {key: value for item in self.camera_results for key, value in item['cameras'].items()}
            combined['jobs'] = [job for item in self.camera_results for job in item['jobs']]
            assert len(combined['cameras']) == len(combined['jobs']) == 4
            summary.update(combined)
            summary.update(total_img1_frames=sum(row['img1_frames'] for row in combined['cameras'].values()),
                total_mask_frames=sum(row['mask_frames'] for row in combined['cameras'].values()),
                finalization_mode='serial_camera_session', scene_lifecycle='retain')
            dump(self.episode / 'render_summary.json', summary)
            self.complete(combined)
        except BaseException:
            self.active = False
            self.blocked = True
            import traceback
            dump(self.episode / 'provenance/session_error.json', {'error': traceback.format_exc()})
            raise

    def complete(self, result):
        from scene_retention import file_sha
        import episode_environment as environment
        try:
            assert result['status'] == 'success', result
            for kind, report in self.reports.items():
                report.update(render_result=result, mrq_pending=False)
                if kind == 'appearance':
                    report['mrq_jobs'] = result['jobs']
                dump(self.episode / ('provenance/' + kind + '_application.json'), report)
            weather_manifest = json.loads((self.episode / 'provenance/environment.json').read_text(encoding='utf-8'))
            state, app, weather, actors = self.inspect(weather_manifest['parameters']['venue_fill_lighting'])
            self.checkpoint = state
            checkpoint_sha = canonical_sha(state)
            dump(self.private / self.session_id / 'checkpoint.json', state)
            files = ['render_summary.json', 'provenance/appearance_application.json', 'provenance/environment_application.json',
                'frames.jsonl', 'provenance/appearance.json', 'provenance/environment.json']
            receipt = {'schema': 'futsalmot_scene_completion', 'version': 1, 'lifecycle': 'retain', 'state': 'retained',
                'session_id': self.session_id, 'episode_id': result['episode_id'], 'checkpoint_sha256': checkpoint_sha,
                'live_readback_verified': True, 'files_sha256': {relative: file_sha(self.episode / relative) for relative in files}}
            dump(self.episode / 'provenance/scene_completion.json', receipt)
            self.active = False
        except BaseException:
            self.active = False
            self.blocked = True
            raise
