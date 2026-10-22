"""按 seed 去重发布的准入、原子性和独立读取回归。"""
import json
import errno
import os
from pathlib import Path

import pytest
from PIL import Image

from grf_ue_bridge import seed_dataset as module


def make_episode(tmp_path, seed=1001, task_id="clip_a", split="train"):
    source = tmp_path / task_id
    source.mkdir()
    entry = {"episode": str(source), "root_seed": seed, "task_id": task_id, "split": split,
             "frames_per_camera": 2, "camera_count": 1, "source_frame_range": [10, 12]}
    trajectory = source / "frames.jsonl"
    trajectory.write_text('{"step":0,"players":[]}\n{"step":1,"players":[]}\n', encoding="utf-8")
    catalog = {"policy": "appearance_v1", "models": [{"player_id": 3}], "skins": [], "kits": []}
    fingerprint = module._canonical_hash(catalog)
    appearance = {"policy": "appearance_v1", "catalog_fingerprint": fingerprint, "fingerprint": "a" * 64,
                  "trajectory_sha256": module._hash(trajectory), "entities": [{"entity_id": "L0", "player_id": 3,
                    "skin_id": "skin-01", "kit_set_id": "01", "kit_class": "left_goalkeeper"}]}
    environment = {"policy": "weather_v1", "fingerprint": "b" * 64, "trajectory_sha256": module._hash(trajectory),
                   "catalog_id": "weather", "catalog_fingerprint": "c" * 64, "environment_seed": seed,
                   "namespace_seed": 1, "preset": "clear", "parameters": {"sun": 1}}
    meta = {"randomness": {"root_seed": seed, "ue_visual_seed": 5}, "source": {"seed": seed, "scenario": "5_vs_5"},
            "timing": {"num_steps": 2, "playback_fps": 10}, "field": {"length_m": 40},
            "entities": [{"id": "L0", "team": "left", "role": "goalkeeper"}]}
    payloads = {"meta.json": meta, "pilot_ready.json": dict(entry, status="audit_passed", appearance_restored=True,
                environment_restored=True, rgb_frames=2, mask_frames=2), "render_summary.json": {"status": "success"},
                "audit/soak_audit_report.json": {"passed": True, "exit_code": 0, "errors": [],
                "validation": {"level": "full", "passed": True, "exit_code": 0}},
                "provenance/appearance.json": appearance, "provenance/environment.json": environment,
                "provenance/ue-profile.json": {"appearance": catalog},
                "provenance/appearance_application.json": {"state": "restored", "errors": [], "manifest_fingerprint": appearance["fingerprint"]},
                "provenance/environment_application.json": {"state": "restored", "errors": [], "environment_manifest_fingerprint": environment["fingerprint"]},
                "CineCam_01/camera.json": {"camera_id": "CineCam_01", "image_width": 2, "image_height": 2}}
    for relative, value in payloads.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
    camera = source / "CineCam_01"
    rows = []
    for frame in (1, 2):
        for folder in ("img1", "mask"):
            (camera / folder).mkdir(exist_ok=True)
            Image.new("RGB" if folder == "img1" else "L", (2, 2)).save(camera / folder / f"{frame:06d}.png")
        rows.append({"frame_index": frame, "source_step": frame - 1, "objects": [{"entity_id": "L0", "class": "player",
                     "team": "left", "role": "goalkeeper", "world_position": [1, 2, 3], "bbox_xyxy": [0, 0, 1, 1],
                     "bbox_xywh": [0, 0, 1, 1], "raw_bbox_xyxy": [0, 0, 2, 2], "in_frame": True,
                     "visible_pixel_count": 1, "bbox_source": "instance_mask"}]})
    (camera / "annotations.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")
    return entry


def test_shared_catalog_and_calibration_are_written_once_across_seeds(tmp_path):
    root = tmp_path / "published"
    first = module.promote_episode(make_episode(tmp_path, task_id="a"), root)
    second = module.promote_episode(make_episode(tmp_path, seed=1002, task_id="b"), root)
    assert len(list((root / "shared/catalogs").glob("*.json"))) == 1
    assert len(list((root / "shared/calibrations").glob("*.json"))) == 1
    index = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    assert index["counts"] == {"seed_groups": 2, "clips": 2, "rgb_frames": 4, "mask_frames": 4}
    assert first["group"].endswith("seed_00001001") and second["group"].endswith("seed_00001002")


def test_environment_restore_does_not_fall_back_to_appearance_field(tmp_path):
    entry = make_episode(tmp_path)
    path = Path(entry['episode']) / 'provenance/environment_application.json'
    value = module._read(path)
    value['manifest_fingerprint'] = value.pop('environment_manifest_fingerprint')
    path.write_text(json.dumps(value), encoding='utf-8')
    with pytest.raises(ValueError, match='指纹'):
        module.promote_episode(entry, tmp_path / 'published')


def test_environment_mismatch_rejected_even_if_generic_field_matches(tmp_path):
    entry = make_episode(tmp_path)
    path = Path(entry['episode']) / 'provenance/environment_application.json'
    value = module._read(path)
    value['manifest_fingerprint'] = value['environment_manifest_fingerprint']
    value['environment_manifest_fingerprint'] = 'd' * 64
    path.write_text(json.dumps(value), encoding='utf-8')
    with pytest.raises(ValueError, match='指纹'):
        module.promote_episode(entry, tmp_path / 'published')


def quick_policy_fixture(tmp_path):
    root = tmp_path / 'published'
    baseline = module.promote_episode(make_episode(tmp_path, task_id='baseline'), root)
    reference = (Path(baseline['clip']) / 'clip.json').relative_to(root).as_posix()
    entry = make_episode(tmp_path, seed=1002, task_id='quick')
    report_path = Path(entry['episode']) / 'audit/soak_audit_report.json'
    report = module._read(report_path)
    report['validation']['level'] = 'quick'
    report_path.write_text(json.dumps(report), encoding='utf-8')
    return root, entry, {'mode':'quick_after_full', 'full_audit_anchor_ref':reference}


def test_explicit_quick_policy_preserves_honest_admission(tmp_path):
    root, entry, policy = quick_policy_fixture(tmp_path)
    result = module.promote_episode(entry, root, audit_policy=policy)
    clip = module.verify_clip(Path(result['clip']))
    assert clip['admission']['full_audit_passed'] is False
    assert clip['admission']['validation_level'] == 'quick'
    assert clip['admission']['basic_checks_passed'] is True
    assert clip['admission']['full_audit_anchor']['clip_ref'] == policy['full_audit_anchor_ref']
    assert module.verify_clip(Path(result['clip']), verification_level='metadata') == clip


def test_explicit_basic_policy_publishes_without_claiming_annotation_audit(tmp_path):
    """用户可跳过逐组审计，但必须如实记录且仍要求完整帧集合。"""
    root, entry, policy = quick_policy_fixture(tmp_path)
    policy['mode'] = 'basic_after_full'
    episode = Path(entry['episode'])
    (episode / 'audit/soak_audit_report.json').unlink()
    ready = module._read(episode / 'pilot_ready.json')
    ready['status'] = 'postprocess_complete'
    (episode / 'pilot_ready.json').write_text(json.dumps(ready), encoding='utf-8')
    result = module.promote_episode(entry, root, audit_policy=policy)
    admission = module.verify_clip(Path(result['clip']))['admission']
    assert admission['validation_level'] == 'basic'
    assert admission['full_audit_passed'] is False
    assert admission['annotation_audit_performed'] is False
    assert 'audit' not in admission['source_sha256']


def test_retained_scene_publication_preserves_lifecycle(tmp_path):
    """保留场景可发布，公共准入必须如实记录未恢复及收据SHA。"""
    entry = make_episode(tmp_path)
    episode = Path(entry['episode'])
    summary_path = episode / 'render_summary.json'
    summary = module._read(summary_path)
    summary['episode_id'] = episode.name
    summary_path.write_text(json.dumps(summary), encoding='utf-8')
    files = ['render_summary.json', 'frames.jsonl', 'provenance/appearance.json', 'provenance/environment.json']
    for kind in ('appearance', 'environment'):
        path = episode / ('provenance/' + kind + '_application.json')
        report = module._read(path)
        report.update(state='applied', scene_lifecycle='retain', session_id='a' * 64,
            mrq_pending=False, render_result={'status': 'success'})
        path.write_text(json.dumps(report), encoding='utf-8')
        files.append(path.relative_to(episode).as_posix())
    ready_path = episode / 'pilot_ready.json'
    ready = module._read(ready_path)
    ready.update(appearance_restored=False, environment_restored=False, scene_retained=True)
    ready_path.write_text(json.dumps(ready), encoding='utf-8')
    receipt = {'schema': 'futsalmot_scene_completion', 'version': 1, 'state': 'retained', 'lifecycle': 'retain',
        'live_readback_verified': True, 'session_id': 'a' * 64, 'episode_id': episode.name,
        'checkpoint_sha256': 'b' * 64, 'files_sha256': {relative: module._hash(episode / relative) for relative in files}}
    (episode / 'provenance/scene_completion.json').write_text(json.dumps(receipt), encoding='utf-8')
    result = module.promote_episode(entry, tmp_path / 'published')
    clip = module.verify_clip(Path(result['clip']))
    assert clip['admission']['appearance_restored'] is False
    assert clip['admission']['environment_restored'] is False
    assert clip['admission']['scene_lifecycle'] == 'retain'
    assert clip['admission']['scene_retained'] is True
    assert len(clip['admission']['scene_completion_sha256']) == 64


def test_quick_without_explicit_policy_rejected(tmp_path):
    root, entry, _ = quick_policy_fixture(tmp_path)
    with pytest.raises(ValueError, match='full annotation'):
        module.promote_episode(entry, root)


def test_quick_rejects_missing_or_modified_full_baseline(tmp_path):
    root, entry, policy = quick_policy_fixture(tmp_path)
    anchor = root / policy['full_audit_anchor_ref']
    value = module._read(anchor)
    value['task_id'] = 'changed'
    anchor.write_text(json.dumps(value), encoding='utf-8')
    with pytest.raises(ValueError, match='基准描述已漂移'):
        module.promote_episode(entry, root, audit_policy=policy)


def test_quick_baseline_cannot_be_another_quick_clip(tmp_path):
    root, entry, policy = quick_policy_fixture(tmp_path)
    first = module.promote_episode(entry, root, audit_policy=policy)
    policy['full_audit_anchor_ref'] = (Path(first['clip'])/'clip.json').relative_to(root).as_posix()
    with pytest.raises(ValueError, match='基准组未通过完整'):
        module.promote_episode(entry, root, audit_policy=policy)


def test_metadata_verify_avoids_rehashing_png_but_checks_dimensions(tmp_path, monkeypatch):
    root, entry, policy = quick_policy_fixture(tmp_path)
    clip = Path(module.promote_episode(entry, root, audit_policy=policy)['clip'])
    original = module._hash
    def text_only(path):
        assert path.suffix != '.png', '轻量验证不应再次读取整张PNG'
        return original(path)
    monkeypatch.setattr(module, '_hash', text_only)
    module.verify_clip(clip, verification_level='metadata')
    Image.new('RGB',(3,2)).save(clip/'cameras/CineCam_01/rgb/000001.png')
    with pytest.raises(ValueError, match='分辨率'):
        module.verify_clip(clip, verification_level='metadata')


def test_quick_rejects_missing_frame_even_with_policy(tmp_path):
    root, entry, policy = quick_policy_fixture(tmp_path)
    (Path(entry['episode'])/'CineCam_01/mask/000002.png').unlink()
    with pytest.raises(ValueError, match='不完整'):
        module.promote_episode(entry, root, audit_policy=policy)


def test_research_player_roster_excludes_actual_uppercase_ball_id(tmp_path):
    """真实 meta 的球为 BALL 且 role=null，不能误当球员训练或追踪。"""
    import importlib.util
    script = Path(__file__).resolve().parents[1] / "experiments/sim2real/pipeline.py"
    spec = importlib.util.spec_from_file_location("ball_filter_pipeline", script)
    pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline)
    entry = make_episode(tmp_path)
    meta_path = Path(entry['episode']) / 'meta.json'
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    meta['entities'].append({'id': 'BALL', 'team': None, 'role': None, 'is_goalkeeper': False})
    meta_path.write_text(json.dumps(meta), encoding='utf-8')
    root = tmp_path / 'published'
    module.promote_episode(entry, root)
    clips, _ = pipeline.canonical_clips(root)
    assert clips[0][3] == {'L0'}


def test_declared_4k_cannot_publish_lower_resolution_camera(tmp_path):
    entry=make_episode(tmp_path)
    entry['resolution']=[3840,2160]
    with pytest.raises(ValueError,match='分辨率'):
        module.promote_episode(entry,tmp_path/'published')


def test_view_annotations_only_store_2d_observations(tmp_path):
    result = module.promote_episode(make_episode(tmp_path), tmp_path / "published")
    clip = Path(result["clip"])
    row = json.loads((clip / "cameras/CineCam_01/annotations.jsonl").read_text().splitlines()[0])
    assert set(row) == {"frame_index", "objects"}
    assert not {"world_position", "team", "role", "class", "bbox_xywh", "raw_bbox_xyxy"} & set(row["objects"][0])
    assert module.verify_clip(clip)["admission"]["full_audit_passed"] is True
    assert not list(clip.rglob("gt.txt")) and not list(clip.rglob("dataset_manifest.json"))


@pytest.mark.parametrize("field", ["ready", "audit", "appearance"])
def test_failed_admission_never_publishes_group(tmp_path, field):
    entry = make_episode(tmp_path)
    source = Path(entry["episode"])
    if field == "ready":
        path = source / "pilot_ready.json";value = module._read(path);value["status"] = "failed"
    elif field == "audit":
        path = source / "audit/soak_audit_report.json";value = module._read(path);value["validation"]["level"] = "quick"
    else:
        path = source / "provenance/appearance_application.json";value = module._read(path);value["state"] = "restore_failed"
    path.write_text(json.dumps(value))
    root = tmp_path / "published"
    with pytest.raises(ValueError):module.promote_episode(entry, root)
    assert not (root / "groups").exists()


def test_same_seed_cannot_cross_split(tmp_path):
    root = tmp_path / "published"
    module.promote_episode(make_episode(tmp_path, task_id="a"), root)
    entry = make_episode(tmp_path, task_id="b", split="validation")
    with pytest.raises(ValueError, match="跨 split"):module.promote_episode(entry, root)


def test_idempotent_same_task_and_second_clip(tmp_path):
    root = tmp_path / "published"
    entry = make_episode(tmp_path, task_id="a")
    first = module.promote_episode(entry, root)
    again = module.promote_episode(entry, root)
    assert again["idempotent"] is True and first["clip"] == again["clip"]
    second = module.promote_episode(make_episode(tmp_path, task_id="b"), root)
    assert second["clip"].endswith("clip_001")
    assert module._read(root / "dataset.json")["counts"]["clips"] == 2


def test_second_clip_same_seed_rejects_different_weather(tmp_path):
    root = tmp_path / "published"
    module.promote_episode(make_episode(tmp_path, task_id="a"), root)
    entry = make_episode(tmp_path, task_id="b")
    path = Path(entry["episode"]) / "provenance/environment.json"
    value = module._read(path);value["parameters"]["sun"] = 2
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="静态信息、外观或天气不一致"):
        module.promote_episode(entry, root)
    assert module._read(root / "dataset.json")["counts"]["clips"] == 1


def test_same_task_different_source_annotations_rejected(tmp_path):
    root = tmp_path / "published"
    entry = make_episode(tmp_path)
    module.promote_episode(entry, root)
    path = Path(entry["episode"]) / "CineCam_01/annotations.jsonl"
    path.write_text(path.read_text(encoding="utf-8").replace('"visible_pixel_count": 1', '"visible_pixel_count": 2'), encoding="utf-8")
    with pytest.raises(ValueError, match="同 task 已发布不同内容"):
        module.promote_episode(entry, root)


def test_missing_frame_does_not_leave_visible_partial_group(tmp_path):
    entry = make_episode(tmp_path)
    (Path(entry["episode"]) / "CineCam_01/mask/000002.png").unlink()
    root = tmp_path / "published"
    with pytest.raises(ValueError, match="不完整"):module.promote_episode(entry, root)
    assert not list((root / "groups").glob("seed_*"))
    assert not (root / ".seed_dataset.lock").exists()


def test_canonical_survives_removed_legacy_and_detects_pixel_mutation(tmp_path):
    entry = make_episode(tmp_path)
    result = module.promote_episode(entry, tmp_path / "published")
    source = Path(entry["episode"])
    for file in source.rglob("*"):
        if file.is_file():file.unlink()
    clip = Path(result["clip"])
    module.verify_clip(clip)
    (clip / "cameras/CineCam_01/mask/000001.png").write_bytes(b"bad")
    with pytest.raises(ValueError, match="已变化"):module.verify_clip(clip)


def test_canonical_research_packer_works_after_legacy_removed(tmp_path, monkeypatch):
    import importlib.util
    from types import SimpleNamespace
    script = Path(__file__).resolve().parents[1] / "experiments/sim2real/pipeline.py"
    spec = importlib.util.spec_from_file_location("sim2real_pipeline", script)
    pipeline = importlib.util.module_from_spec(spec);spec.loader.exec_module(pipeline)
    entry = make_episode(tmp_path)
    root = tmp_path / "published"
    module.promote_episode(entry, root)
    for path in Path(entry["episode"]).rglob("*"):
        if path.is_file():path.unlink()
    research = tmp_path / "research"
    monkeypatch.setattr(pipeline, "ROOT", research)
    pipeline.prepare_synthetic(SimpleNamespace(canonical_root=root))
    output = research / "datasets/synthetic_pilot_v1"
    images = (output / "train.txt").read_text().splitlines()
    assert len(images) == 2 and all(Path(path).is_file() for path in images)
    label = output / "labels/train" / (Path(images[0]).stem + ".txt")
    assert label.read_text().strip() == "0 0.25000000 0.25000000 0.50000000 0.50000000"
    provenance = module._read(output / "provenance.json")
    assert provenance["root_seed_split"] == {"1001": "train"}
    assert provenance["clips"][0]["boxes"] == 2
    pipeline.prepare_synthetic_mot(SimpleNamespace(canonical_root=root))
    mot_output = research / "datasets/synthetic_mot_v1"
    mot_provenance = module._read(mot_output / "provenance.json")
    sequence = mot_output / "train" / mot_provenance["sequences"][0]["sequence"]
    mot_rows = (sequence / "gt/gt.txt").read_text(encoding="utf-8").splitlines()
    numeric_rows = [list(map(float, row.split(","))) for row in mot_rows]
    assert numeric_rows == [[1, 1, 0, 0, 1, 1, 1, 1, -1], [2, 1, 0, 0, 1, 1, 1, 1, -1]]
    assert mot_provenance["sequences"][0]["entity_id_to_track_id"] == {"L0": 1}
    assert len(list((sequence / "img1").glob("*.png"))) == 2


def test_canonical_missing_checksum_entry_rejected(tmp_path):
    result = module.promote_episode(make_episode(tmp_path), tmp_path / "published")
    clip = Path(result["clip"])
    checksums = module._read(clip / "checksums.json")
    del checksums["cameras/CineCam_01/mask/000001.png"]
    (clip / "checksums.json").write_text(json.dumps(checksums))
    with pytest.raises(ValueError, match="完整覆盖"):module.verify_clip(clip)


def add_active_selection(entry):
    episode = Path(entry["episode"])
    value = {
        "schema": "futsalmot_active_episode", "version": 1, "root_seed": entry["root_seed"],
        "frame_count": 2, "frames_sha256": module._hash(episode / "frames.jsonl"),
        "source_episode_id": "full_source", "source_capture_steps": 300,
        "source_frames_sha256": "d" * 64, "source_meta_sha256": "e" * 64,
        "source_step_range": [10, 12], "source_time_range_s": [1., 1.2],
        "dt_s": .1, "policy_sha256": "f" * 64,
        "source_export_profile": {"num_steps": 300, "scenario": "5_vs_5"},
        "policy": {"max_duration_s": 30},
        "source_window": {"start_index": 10, "end_index_exclusive": 12, "frame_count": 2,
            "sample_duration_s": .2, "trajectory_span_s": .1, "team_active_coverage": 1.,
            "moving_outfield_ids": ["L1", "L2", "R1", "R2"],
            "local_motion": {"coverage": .9, "max_low_motion_streak_s": .1},
            "players": {"L1": {"path_length_m": 1., "net_displacement_m": .9,
                                  "spatial_extent_m": .9, "active_ratio": 1.}}},
    }
    (episode / "provenance/active_window.json").write_text(json.dumps(value), encoding="utf-8")
    return value


def test_active_selection_preserved_without_repeating_common_recipe(tmp_path):
    entry = make_episode(tmp_path)
    active = add_active_selection(entry)
    root = tmp_path / "published"
    result = module.promote_episode(entry, root)
    clip = Path(result["clip"])
    selection = module.verify_clip(clip)["source_selection"]
    for key in ("source_frames_sha256", "source_meta_sha256", "source_capture_steps",
                "source_step_range", "source_time_range_s", "dt_s", "policy_sha256", "source_window"):
        assert selection[key] == active[key]
    assert "source_export_profile" not in selection and "policy" not in selection
    assert "source_export_profile" not in (clip / "clip.json").read_text(encoding="utf-8")
    assert module.promote_episode(entry, root)["idempotent"] is True
    for path in Path(entry["episode"]).rglob("*"):
        if path.is_file():path.unlink()
    assert module.verify_clip(clip)["source_selection"] == selection


def test_legacy_without_active_window_retains_null_selection(tmp_path):
    entry = make_episode(tmp_path)
    path = Path(entry["episode"]) / "meta.json"
    meta = module._read(path)
    meta["source_selection"] = {"stale_legacy_field": True}
    path.write_text(json.dumps(meta), encoding="utf-8")
    result = module.promote_episode(entry, tmp_path / "published")
    assert module.verify_clip(Path(result["clip"]))["source_selection"] is None


@pytest.mark.parametrize("key,value", [
    ("frames_sha256", "0" * 64), ("source_frames_sha256", "invalid"),
    ("source_capture_steps", 11), ("source_step_range", [10, 13]),
    ("source_time_range_s", [1., 2.]), ("root_seed", 1002),
])
def test_invalid_active_evidence_fails_before_publication(tmp_path, key, value):
    entry = make_episode(tmp_path)
    active = add_active_selection(entry)
    active[key] = value
    (Path(entry["episode"]) / "provenance/active_window.json").write_text(json.dumps(active), encoding="utf-8")
    root = tmp_path / "published"
    with pytest.raises(ValueError, match="active_window"):
        module.promote_episode(entry, root)
    assert not (root / "groups").exists()


def test_changed_active_selection_cannot_reuse_old_task_admission(tmp_path):
    entry = make_episode(tmp_path)
    active = add_active_selection(entry)
    root = tmp_path / "published"
    module.promote_episode(entry, root)
    active["source_window"]["team_active_coverage"] = .8
    (Path(entry["episode"]) / "provenance/active_window.json").write_text(json.dumps(active), encoding="utf-8")
    with pytest.raises(ValueError, match="同 task 已发布不同内容"):
        module.promote_episode(entry, root)


def test_same_inode_publish_reuses_only_current_source_hash_and_verify_still_reads_all(tmp_path, monkeypatch):
    entry = make_episode(tmp_path)
    episode = Path(entry['episode'])
    expected_files = [episode/'meta.json', episode/'frames.jsonl',
                      episode/'CineCam_01/camera.json', episode/'CineCam_01/annotations.jsonl']
    expected_files += sorted((episode/'CineCam_01/img1').glob('*.png'))
    expected_files += sorted((episode/'CineCam_01/mask').glob('*.png'))
    expected_payload = module._canonical_hash({p.relative_to(episode).as_posix(): module._hash(p) for p in expected_files})
    original = module._hash
    hashed = []
    def counted(path):
        hashed.append(Path(path))
        return original(path)
    monkeypatch.setattr(module, '_hash', counted)
    result = module.promote_episode(entry, tmp_path/'published')
    clip = Path(result['clip'])
    assert module._read(clip/'clip.json')['admission']['source_payload_sha256'] == expected_payload
    assert '_source_hashes' not in (clip/'clip.json').read_text(encoding='utf-8')
    assert len([p for p in hashed if p.suffix == '.png']) == 4
    assert all(episode in p.parents for p in hashed if p.suffix == '.png')
    assert os.path.samefile(episode/'CineCam_01/img1/000001.png', clip/'cameras/CineCam_01/rgb/000001.png')
    before = len(hashed)
    module.verify_clip(clip)
    independently_hashed = hashed[before:]
    checksums = module._read(clip/'checksums.json')
    assert {p.relative_to(clip).as_posix() for p in independently_hashed} == set(checksums)
    assert len(independently_hashed) == len(checksums)


def test_cross_volume_fallback_independently_hashes_every_copied_frame(tmp_path, monkeypatch):
    entry = make_episode(tmp_path)
    def cross_volume(*args, **kwargs):
        raise OSError(errno.EXDEV, '跨卷链接')
    monkeypatch.setattr(module.os, 'link', cross_volume)
    original = module._hash
    hashed = []
    def counted(path):
        hashed.append(Path(path))
        return original(path)
    monkeypatch.setattr(module, '_hash', counted)
    result = module.promote_episode(entry, tmp_path/'published')
    clip = Path(result['clip'])
    assert len([p for p in hashed if p.suffix == '.png']) == 8
    source = Path(entry['episode'])/'CineCam_01/img1/000001.png'
    target = clip/'cameras/CineCam_01/rgb/000001.png'
    assert not os.path.samefile(source, target)
    assert source.read_bytes() == target.read_bytes()
    module.verify_clip(clip)


@pytest.mark.parametrize('moment', ('after_hash', 'later_hash', 'before_link', 'after_link', 'replace_source', 'later_frame'))
def test_source_identity_or_payload_drift_never_publishes_visible_group(tmp_path, monkeypatch, moment):
    entry = make_episode(tmp_path)
    episode = Path(entry['episode'])
    first = episode/'CineCam_01/img1/000001.png'
    original_hash, original_link = module._hash, module._link
    changed = []
    def mutate(path, replace=False):
        old = path.stat()
        if replace:
            replacement = path.with_suffix('.replacement')
            replacement.write_bytes(path.read_bytes())
            os.utime(replacement, ns=(old.st_atime_ns, old.st_mtime_ns))
            os.replace(replacement, path)
        else:
            path.write_bytes(path.read_bytes() + b'drift')
            os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns + 1000000))
        changed.append(True)
    def hashed(path):
        digest = original_hash(path)
        if moment == 'after_hash' and Path(path) == first and not changed:
            mutate(first)
        if moment == 'later_hash' and Path(path).name == '000002.png' and Path(path).parent.name == 'img1' and not changed:
            mutate(first)
        return digest
    def linked(source, target):
        source = Path(source)
        if source == first and moment == 'before_link':
            mutate(first)
        original_link(source, target)
        if source == first and moment in ('after_link', 'replace_source'):
            mutate(first, replace=moment == 'replace_source')
        if source.name == '000002.png' and source.parent.name == 'img1' and moment == 'later_frame':
            mutate(first)
    monkeypatch.setattr(module, '_hash', hashed)
    monkeypatch.setattr(module, '_link', linked)
    root = tmp_path/'published'
    with pytest.raises(ValueError, match='漂移|变化'):
        module.promote_episode(entry, root)
    assert changed
    assert not list((root/'groups').glob('seed_*'))
    assert not (root/'.seed_dataset.lock').exists()


def test_copied_target_corruption_is_rejected_before_publication(tmp_path, monkeypatch):
    entry = make_episode(tmp_path)
    def cross_volume(*args, **kwargs):
        raise OSError(errno.EXDEV, '跨卷链接')
    monkeypatch.setattr(module.os, 'link', cross_volume)
    original = module._link
    def corrupt_copy(source, target):
        original(source, target)
        if Path(source).suffix == '.png':
            data = bytearray(Path(target).read_bytes())
            data[-1] ^= 1
            Path(target).write_bytes(data)
    monkeypatch.setattr(module, '_link', corrupt_copy)
    root = tmp_path/'published'
    with pytest.raises(ValueError, match='不一致|变化'):
        module.promote_episode(entry, root)
    assert not list((root/'groups').glob('seed_*'))


def test_fresh_admission_does_not_reuse_previous_call_file_hashes(tmp_path):
    entry = make_episode(tmp_path)
    admitted = module._admit(entry)
    frame = Path(entry['episode'])/'CineCam_01/img1/000001.png'
    before = frame.stat()
    data = bytearray(frame.read_bytes())
    data[-1] ^= 1
    frame.write_bytes(data)
    os.utime(frame, ns=(before.st_atime_ns, before.st_mtime_ns))
    current = module._admit(entry)
    assert current['admission']['source_payload_sha256'] != admitted['admission']['source_payload_sha256']


def test_hash_rejects_source_modified_while_stream_is_open(tmp_path, monkeypatch):
    path = tmp_path/'frame.png'
    path.write_bytes(b'original bytes')
    original = module.hashlib.sha256
    class MutatingHash:
        def __init__(self):
            self.digest = original()
            self.changed = False
        def update(self, data):
            self.digest.update(data)
            if not self.changed:
                self.changed = True
                with path.open('ab') as stream:
                    stream.write(b'drift')
        def hexdigest(self):
            return self.digest.hexdigest()
    monkeypatch.setattr(module.hashlib, 'sha256', MutatingHash)
    with pytest.raises(ValueError, match='漂移'):
        module._hash(path)


def test_copy_target_changed_after_hash_cannot_publish_stale_digest(tmp_path, monkeypatch):
    entry = make_episode(tmp_path)
    episode = Path(entry['episode'])
    def cross_volume(*args, **kwargs):
        raise OSError(errno.EXDEV, '跨卷链接')
    monkeypatch.setattr(module.os, 'link', cross_volume)
    original = module._hash
    def changed_after_hash(path):
        digest = original(path)
        if path.suffix == '.png' and episode not in path.parents:
            path.write_bytes(path.read_bytes() + b'drift')
        return digest
    monkeypatch.setattr(module, '_hash', changed_after_hash)
    root = tmp_path/'published'
    with pytest.raises(ValueError, match='漂移'):
        module.promote_episode(entry, root)
    assert not list((root/'groups').glob('seed_*'))
