"""四seed源入口的CPU边界，不替代UE资产和120帧渲染验收。"""

import ast
import copy
from pathlib import Path
import runpy

import pytest

ROOT = Path(__file__).resolve().parents[1]
M = runpy.run_path(str(ROOT / "ue/pose17_location_pilot_sources.py"))
PROJECT = Path("D:/projects/FutsalMOT_UEDataset")


def source(seed):
    meta = {"schema": "grf_ue_episode", "source": {"seed": seed, "environment": "google_research_football", "scenario": "5_vs_5"},
            "timing": {"source_step_seconds": .1, "playback_fps": 10, "num_steps": 1800}}
    frames = [{"step": i, "time_seconds": i * .1, "players": [{"id": entity, "position_m": [0., 0., 0.]}
                                                                  for entity in M["joint"].JOINT_VISUAL_IDS],
               "ball": {"position_m": [0., 0., .11]}} for i in range(1800)]
    return meta, frames


def candidate():
    groups = []
    for seed, (start, end) in M["WINDOWS"].items():
        plan = M["joint"].make_joint_plan(PROJECT)
        package = M["PACKAGE"] + f"/seed_{seed:08d}"
        plan.update(attempt=f"location_pilot_s{seed:08d}_a01_gpu_lbs", start_frame=start, frame_count=120,
                    location_pilot_contract=True,
                    source_scope="real_GRF_motion_location_pilot_diagnostic_roster_override_current_scene_fixed_blue_location_only_not_original_weather_or_appearance",
                    review_frame_indices=[60])
        plan["location_pilot"] = {"schema": "futsalmot_pose17_location_pilot_v1", "batch_id": "location_pilot_4seed_a01", "seed": seed,
                                  "split": "train" if seed != 1102 else "synthetic_dev", "source_frame_range": [start, end],
                                  "location_supervision": True, "visibility_supervision": False, "formal_pose_release": False}
        plan["source_motion"] = {"root_seed": seed, "source_frames_sha256": "a" * 64, "source_meta_sha256": "b" * 64,
                                 "context_source_frame_range": [0, end], "selected_source_frame_range": [start, end],
                                 "sequence_frame_equals_source_step": True, "fps": 10,
                                 "source_build_receipt": (PROJECT / M["CANDIDATE_RELATIVE"]).with_name("source_prepare.json").as_posix()}
        plan["cameras"] = [{"camera_id": cam, "camera_actor_object_name": data[0], "source_sequence": package + "/LS_Cam_" + cam[-2:]}
                           for cam, data in M["joint"].CAMERAS.items()]
        source_dir = PROJECT / f"Saved/Production/v1_no_numbers_4k/sources/source_s{seed:08d}"
        groups.append({"root_seed": seed, "split": "train" if seed != 1102 else "synthetic-dev", "sequence_package": package,
                       "source_prefix_range": [0, end], "selected_source_frame_range": [start, end],
                       "source_frames_sha256": "a" * 64, "source_meta_sha256": "b" * 64,
                       "original_frozen_active_source_range": [0, 1800], "source_frames_path": (source_dir / "frames.jsonl").as_posix(),
                       "source_meta_path": (source_dir / "meta.json").as_posix(), "proposed_runtime_plan": plan,
                       "floor_manifest": {"application_status": "planned_not_applied", "selection_mode": "paired_override", "variant": {"id": "sport_blue_v1"}}})
    return {"schema": "futsalmot_pose17_location_pilot_candidate_v1", "status": "candidate_not_admitted_not_runnable",
            "production_admitted": False, "training_authorized_by_this_file": False, "groups": groups}


@pytest.mark.parametrize("seed", [1042, 1044, 1048, 1102])
def test_prefix_keeps_actual_source_zero_context_and_1044_307_start(seed):
    meta, frames = source(seed)
    old = copy.deepcopy(meta)
    out_meta, prefix = M["source_prefix"](meta, frames, seed)
    start, end = M["WINDOWS"][seed]
    assert len(prefix) == end and out_meta["timing"]["num_steps"] == end
    assert len(prefix[start:end]) == 120 and prefix[start]["step"] == start
    assert [f["step"] for f in prefix] == list(range(end))
    assert meta == old and len(frames) == 1800
    prefix[0]["players"][0]["position_m"][0] = 12
    assert frames[0]["players"][0]["position_m"][0] == 0


@pytest.mark.parametrize("damage", ["seed", "count", "gap", "time", "entity", "duplicate", "position", "ball"])
def test_refuses_changed_incomplete_or_noncontinuous_source(damage):
    meta, frames = source(1044)
    if damage == "seed":
        meta["source"]["seed"] = 1048
    elif damage == "count":
        frames.pop()
    elif damage == "gap":
        frames[306]["step"] = 307
    elif damage == "time":
        frames[306]["time_seconds"] = float("nan")
    elif damage == "entity":
        frames[306]["players"].pop()
    elif damage == "duplicate":
        frames[306]["players"][0]["id"] = "L1"
    elif damage == "position":
        frames[306]["players"][0]["position_m"][0] = float("inf")
    else:
        frames[306]["ball"]["position_m"] = []
    with pytest.raises(ValueError):
        M["source_prefix"](meta, frames, 1044)


def test_candidate_is_exact_four_seed_nonproduction_and_location_only():
    value = candidate()
    assert M["validate_candidate"](value, PROJECT) is value
    assert len(value["groups"]) == 4
    assert sum(len(row["proposed_runtime_plan"]["cameras"]) for row in value["groups"]) == 16
    assert [row["proposed_runtime_plan"]["location_pilot"]["split"] for row in value["groups"]] == ["train", "train", "train", "synthetic_dev"]


@pytest.mark.parametrize("damage", ["extra_seed", "count", "rebase1044", "camera", "roster", "source", "split", "visibility", "formal", "floor_applied", "active_window"])
def test_candidate_refuses_scope_expansion_and_fake_supervision(damage):
    value = candidate()
    row = value["groups"][1]
    plan = row["proposed_runtime_plan"]
    if damage == "extra_seed":
        value["groups"].append(copy.deepcopy(row))
    elif damage == "count":
        plan["frame_count"] = 121
    elif damage == "rebase1044":
        plan["start_frame"] = 100
    elif damage == "camera":
        plan["cameras"][0]["source_sequence"] = "/Game/FutsalMOT/Sequences/LS_Cam_01"
    elif damage == "roster":
        plan["targets"][0]["calibration_player_id"] = 9
    elif damage == "source":
        row["source_frames_path"] = "G:/FutsalMOT_Dataset/v1_no_numbers_4k/frames.jsonl"
    elif damage == "split":
        value["groups"][3]["proposed_runtime_plan"]["location_pilot"]["split"] = "train"
    elif damage == "visibility":
        plan["location_pilot"]["visibility_supervision"] = True
    elif damage == "formal":
        plan["location_pilot"]["formal_pose_release"] = True
    elif damage == "floor_applied":
        row["floor_manifest"]["application_status"] = "applied"
    else:
        row["original_frozen_active_source_range"] = [308, 1800]
    with pytest.raises(ValueError):
        M["validate_candidate"](value, PROJECT)


def test_source_entry_has_no_queue_render_or_actor_mutation_and_disables_importer_side_effects():
    tree = ast.parse((ROOT / "ue/pose17_location_pilot_sources.py").read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    names = {node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else "" for node in calls}
    assert not {"validate_plan", "prepare", "submit", "run_task", "render_sequences", "set_actor_location", "set_actor_transform", "save_map", "apply_preview"} & names
    create = [node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "create_sequence"]
    assert len(create) == 1
    values = {arg.arg: arg.value.value for arg in create[0].keywords if isinstance(arg.value, ast.Constant)}
    assert values["replace_existing"] is False and values["open_after_create"] is False and values["run_api_smoke"] is False
    assert "_dump_new" in names and "_scene" in names and "_dirty" in names


def test_plain_path_refuses_symlink(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    linked = tmp_path / "linked"
    try:
        linked.symlink_to(actual, target_is_directory=True)
    except OSError:
        pytest.skip("本机创建symlink权限不可用；入口仍显式检查reparse属性")
    with pytest.raises(ValueError):
        M["_plain_path"](linked / "input.json")
