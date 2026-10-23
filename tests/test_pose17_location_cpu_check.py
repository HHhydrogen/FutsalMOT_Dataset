"""真实诊断适配器的微型CPU工程反例；不读取实际40图或GPU。"""

import importlib.util
import json
from pathlib import Path

import pytest

from grf_ue_bridge.tools import export_pose17_diagnostic as exporter
from test_export_pose17_diagnostic import fixture

SPEC=importlib.util.spec_from_file_location("pose17_location_cpu_check",Path(__file__).resolve().parents[1]/"experiments/sim2real/check_pose17_location_cpu.py")
TOOL=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def prepared(tmp_path,monkeypatch):
    attempt,alignment,native,out,calls=fixture(tmp_path,monkeypatch)
    exporter.export_attempt(attempt,alignment,native,out,copy_rgb=True)
    acceptance=out.parent/"actual_cpu_acceptance.json"
    acceptance.write_text(json.dumps(dict(schema="pose17_real_diagnostic_export_cpu_acceptance_v1",passed=True,
        errors=[],dataset=str(out),source_attempt=str(attempt),world_objects_exact=400,independent_bool_location_masks=400)),encoding="utf-8")
    return out,acceptance


def rewrite_first(root,field,value):
    path=root/"groups/seed_00001048/cameras/CineCam_01/pose2d.jsonl"
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["objects"][0][field]=value
    path.write_text("".join(json.dumps(r)+"\n" for r in rows),encoding="utf-8")


def test_adapts_all_visible_people_unknown_and_evidence_contents(tmp_path,monkeypatch):
    root,acceptance=prepared(tmp_path,monkeypatch)
    frames,evidence=TOOL.adapt_records(root,acceptance)
    assert len(frames)==40 and sum(len(f["instances"]) for f in frames)==400
    assert all(f["not_training"] and f["MRQ_sync_verified"] is False for f in frames)
    assert all(i["source_coco_v"] is None and i["detection_target_retained"] for f in frames for i in f["instances"])
    assert all(i["visibility_state"]==["unknown"]*17 and i["location_valid"]==[True]*17 for f in frames for i in f["instances"])
    assert evidence["world_and_projection_rechecked"] and evidence["location_training_admission"] is False
    assert len(evidence["acceptance_sha256"])==64


@pytest.mark.parametrize("case",["nonempty_fake_ref","failed_receipt","omitted_person","false_visibility","nonbool_mask","changed_world","bad_bbox"])
def test_cannot_pass_with_refs_or_changed_native_semantics(tmp_path,monkeypatch,case):
    root,acceptance=prepared(tmp_path,monkeypatch)
    if case=="nonempty_fake_ref":
        value=json.loads(acceptance.read_text());value["source_attempt"]=str(root.parent/"not_present");acceptance.write_text(json.dumps(value))
    if case=="failed_receipt":
        value=json.loads(acceptance.read_text());value["passed"]=False;acceptance.write_text(json.dumps(value))
    if case=="omitted_person":
        path=root/"groups/seed_00001048/cameras/CineCam_01/pose2d.jsonl"
        rows=[json.loads(line) for line in path.read_text().splitlines()];rows[0]["objects"].pop()
        path.write_text("".join(json.dumps(r)+"\n" for r in rows))
    if case=="false_visibility":rewrite_first(root,"visibility_state",["visible"]*17)
    if case=="nonbool_mask":rewrite_first(root,"location_valid",[1]*17)
    if case=="bad_bbox":rewrite_first(root,"bbox_visible_xywh",[-1,0,2,3])
    if case=="changed_world":
        path=root/"groups/seed_00001048/cameras/CineCam_01/pose3d_world.jsonl"
        rows=[json.loads(line) for line in path.read_text().splitlines()];rows[0]["objects"][0]["world_m"][0][0]+=1
        path.write_text("".join(json.dumps(r)+"\n" for r in rows))
    with pytest.raises((ValueError,FileNotFoundError)):TOOL.adapt_records(root,acceptance)
