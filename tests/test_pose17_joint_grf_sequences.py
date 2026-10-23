"""真实GRF联合源入口的纯CPU边界；玩具源不代表真实GRF或UE已验收。"""

import ast
import copy
import inspect
from pathlib import Path
import runpy

import pytest

ROOT = Path(__file__).resolve().parents[1]
M = runpy.run_path(str(ROOT / "ue/pose17_joint_grf_sequences.py"))


def fixture():
    meta = {"schema": "grf_ue_episode", "source": {"seed": 1048, "environment": "google_research_football", "scenario": "5_vs_5"},
            "timing": {"source_step_seconds": .1, "playback_fps": 10, "num_steps": 1800}}
    frames = [{"step": i, "time_seconds": i * .1, "players": [{"id": entity} for entity in M["JOINT_VISUAL_IDS"]]}
              for i in range(1800)]
    return meta, frames


def test_prefix_preserves_real_source_indices_and_motion_context_without_input_mutation():
    meta, frames = fixture()
    original = copy.deepcopy(meta)
    selected_meta, prefix = M["source_prefix"](meta, frames)
    assert meta == original and len(frames) == 1800
    assert selected_meta["timing"]["num_steps"] == 110
    assert [f["step"] for f in prefix] == list(range(110))
    assert [f["time_seconds"] for f in prefix[100:]] == [i * .1 for i in range(100, 110)]
    prefix[0]["step"] = 9
    assert frames[0]["step"] == 0


@pytest.mark.parametrize("damage", ["seed", "environment", "fps", "source_count", "gap", "time", "entity", "duplicate_entity"])
def test_refuses_wrong_source_or_noncontinuous_incomplete_prefix(damage):
    meta, frames = fixture()
    if damage == "seed":
        meta["source"]["seed"] = 42
    elif damage == "environment":
        meta["source"]["environment"] = "canonical_toy"
    elif damage == "fps":
        meta["timing"]["playback_fps"] = 30
    elif damage == "source_count":
        frames.pop()
    elif damage == "gap":
        frames[105]["step"] = 106
    elif damage == "time":
        frames[105]["time_seconds"] = float("nan")
    elif damage == "entity":
        frames[105]["players"].pop()
    else:
        frames[105]["players"][0]["id"] = "L1"
    with pytest.raises(ValueError):
        M["source_prefix"](meta, frames)


def test_joint_plan_is_exact_ten_people_four_original_cameras_with_separate_source_and_render_package():
    plan = M["make_joint_plan"](Path("D:/projects/FutsalMOT_UEDataset"))
    M["runner"].validate_plan(plan)
    assert plan["start_frame"] == 100 and plan["frame_count"] == 10
    assert plan["gpu_lbs_warmup_frames"] == plan["override_warmup_frames"] == 8
    assert plan["review_frame_indices"] == [5]
    assert "diagnostic_roster_override" in plan["source_scope"] and "not_original_weather_or_appearance" in plan["source_scope"]
    assert {t["entity_id"]: t["calibration_player_id"] for t in plan["targets"]} == M["JOINT_VISUAL_IDS"]
    assert len({t["actor_object_name"] for t in plan["targets"]}) == 10
    assert sum("diagnostic_visual_mesh_override" in t for t in plan["targets"]) == 8
    assert {c["camera_id"] for c in plan["cameras"]} == set(M["CAMERAS"])
    assert all(c["source_sequence"].startswith(M["PACKAGE"] + "/LS_Cam_") for c in plan["cameras"])
    assert M["ATTEMPT"] not in M["PACKAGE"]


def test_importer_new_options_preserve_defaults_and_guard_editor_open_and_temporary_smoke():
    import import_grf_episode as builder
    parameters = inspect.signature(builder.create_sequence).parameters
    for key in ("open_after_create", "run_api_smoke"):
        assert parameters[key].default is True and parameters[key].kind is inspect.Parameter.KEYWORD_ONLY
    tree = ast.parse(inspect.getsource(builder.create_sequence))
    function = tree.body[0]
    for flag, expected_call in (("open_after_create", "open_level_sequence"), ("run_api_smoke", "_smoke_test_sequencer_api")):
        guards = [node for node in ast.walk(function) if isinstance(node, ast.If)
                  and isinstance(node.test, ast.Name) and node.test.id == flag]
        assert len(guards) == 1
        names = [node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
                 for node in ast.walk(guards[0]) if isinstance(node, ast.Call)]
        assert expected_call in names


def test_joint_entry_does_not_call_scene_preview_or_mrq_and_uses_both_optouts():
    tree = ast.parse((ROOT / "ue/pose17_joint_grf_sequences.py").read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    names = [node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else "" for node in calls]
    assert not {"apply_preview", "open_level_sequence", "render_sequences", "submit", "set_actor_transform", "set_actor_location"} & set(names)
    create = [node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "create_sequence"]
    assert len(create) == 1
    flags = {keyword.arg: keyword.value.value for keyword in create[0].keywords if isinstance(keyword.value, ast.Constant)}
    assert flags["open_after_create"] is False and flags["run_api_smoke"] is False and flags["replace_existing"] is False
