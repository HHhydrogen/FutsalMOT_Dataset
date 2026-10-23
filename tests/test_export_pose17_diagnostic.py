"""联合adapter的微型CPU夹具，不读取真实UE媒体或生成正式训练集。"""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from grf_ue_bridge.tools import export_pose17_diagnostic as tool
from test_pose17_joint_receipt import joint_bundle


def dump(path,value):
    path.write_text(json.dumps(value),encoding="utf-8")


def fixture(tmp_path,monkeypatch):
    saved = tmp_path/"Saved"
    saved.mkdir()
    monkeypatch.setattr(tool,"WORKSPACE_SAVED",saved)
    attempt = saved/"joint_attempt"
    attempt.mkdir()
    plan,state,before,after,manifests,hashes,receipts = joint_bundle()
    plan.update(source_motion=dict(root_seed=1048,fps=10,sequence_frame_equals_source_step=True),source_scope="real_GRF_with_diagnostic_roster")
    for entity in hashes:
        raw = json.dumps(manifests[entity]).encode()
        (attempt/(entity+"_landmark_snapshot.json")).write_bytes(raw)
        hashes[entity] = hashlib.sha256(raw).hexdigest()
    dump(attempt/"plan.json",plan)
    state["plan_sha256"] = hashlib.sha256((attempt/"plan.json").read_bytes()).hexdigest()
    dump(attempt/"state.json",state)
    dump(attempt/"preflight.json",before)
    dump(attempt/"editor_after.json",after)
    camera = dict(intrinsics=dict(width=16,height=8,fx=8.,fy=8.,cx=7.5,cy=3.5),
        extrinsics=dict(world_location_m=[0.,0.,0.],forward=[1.,0.,0.],right=[0.,1.,0.],up=[0.,0.,1.]))
    aligned = dict(attempt=str(attempt),structural_pairing_passed=True,MRQ_sync_verified=False,errors=[],cameras={})
    for camera_index,(camera_id,receipt) in enumerate(receipts.items()):
        directory = attempt/camera_id
        directory.mkdir()
        frames = []
        for frame in range(10):
            Image.new("RGB",(16,8)).save(directory/("FinalImage%06d.png"%frame))
            (directory/("%06d.exr"%frame)).write_bytes(b"TOY_NOT_EXR")
        raw_rows = []
        for row in receipt["records"]:
            entity,frame = row["entity_id"],row["output_frame"]
            row["manifest_sha256"] = hashes[entity]
            row["world_m"] = [[10.+camera_index*.000001,.1,.1] for _ in range(17)]
            row["world_time_seconds"] = 10.+frame*.1+camera_index*.000001
            row["m_rq_world_seconds"] = 9.9+frame*.1
            frames.append(dict(entity_id=entity,output_frame=frame,RGB=str(directory/("FinalImage%06d.png"%frame)),
                EXR=str(directory/("%06d.exr"%frame)),camera=copy.deepcopy(camera),source_frame=100+frame,
                keypoints=[{"world_m":copy.deepcopy(p)} for p in row["world_m"]]))
            # 真实normalizer处理原生sample层；不是直接用已flatten的成功数据绕过。
            raw = dict(row)
            raw["sample"] = {key:raw.pop(key) for key in ("world_m","mesh_identity","manifest_sha256","keypoint_names")}
            raw["sample"]["world_points_m"] = raw["sample"].pop("world_m")
            raw_rows.append(raw)
        native = dict(receipt,records=raw_rows)
        dump(attempt/(camera_id+"_native_pose_restore_receipt.json"),native)
        aligned["cameras"][camera_id] = {"frames":frames}
    alignment = saved/"alignment.json"
    dump(alignment,aligned)
    native_report = saved/"native_passed.json"
    dump(native_report,dict(attempt=str(attempt),acceptance_passed=True,component_transaction_passed=True,errors=[],MRQ_sync_verified=False))
    calls = []
    def decode(path,mapping):
        calls.append(Path(path))
        mask = np.zeros((8,16),dtype=np.uint8)
        for i in range(10): mask[0:2,i] = i+1
        return mask,{entity:2 for entity in mapping}
    monkeypatch.setattr(tool,"decode_mask",decode)
    return attempt,alignment,native_report,saved/"diagnostic_export",calls


def test_export_once_per_exr_keeps_micrometre_world_differences(tmp_path,monkeypatch):
    attempt,alignment,native,out,calls = fixture(tmp_path,monkeypatch)
    report = tool.export_attempt(attempt,alignment,native,out,copy_rgb=True)
    assert report["not_training"] and not report["MRQ_sync_verified"]
    assert len(calls)==len(set(calls))==40 and report["rgb_files_copied"]==40
    group = out/"groups/seed_00001048"
    worlds = []
    for camera_id in ("CineCam_01","CineCam_02","CineCam_03","CineCam_04"):
        directory = group/"cameras"/camera_id
        rows = [json.loads(line) for line in (directory/"pose3d_world.jsonl").read_text().splitlines()]
        assert len(rows)==10 and rows[5]["source_frame_index"]==105
        worlds.append(rows[0]["objects"][0]["world_m"][0][0])
        image = json.loads((directory/"pose2d.jsonl").read_text().splitlines()[0])
        assert len(image["objects"])==10 and image["objects"][0]["bbox_visible_xywh"]==[0,0,1,2]
        assert image["objects"][0]["visibility_state"]==["unknown"]*17
        assert image["objects"][0]["location_valid"]==[True]*17
        assert (directory/"view.json").exists() and len(list((directory/"mask").glob("*.png")))==10
    assert len(set(worlds))==4
    assert worlds==[10.,10.000001,10.000002,10.000003]
    assert not (group/"world17.jsonl").exists()
    assert len(list((group/"provenance/landmarks").glob("*.json")))==7
    with pytest.raises(ValueError): tool.export_attempt(attempt,alignment,native,out)


@pytest.mark.parametrize("case",["alignment_failed","restore_failed","wrong_attempt","mask_not_generated_frame","native_changed","outside_saved","world_changed_after_acceptance"])
def test_reject_unverified_or_unsafe_input(tmp_path,monkeypatch,case):
    attempt,alignment,native,out,calls = fixture(tmp_path,monkeypatch)
    if case=="alignment_failed":
        value=json.loads(alignment.read_text());value["structural_pairing_passed"]=False;dump(alignment,value)
    if case=="restore_failed":
        value=json.loads(native.read_text());value["acceptance_passed"]=False;dump(native,value)
    if case=="wrong_attempt":
        value=json.loads(alignment.read_text());value["attempt"]=str(attempt.parent/"other");dump(alignment,value)
    if case=="mask_not_generated_frame":
        (attempt/"CineCam_04/000009.exr").unlink()
    if case=="native_changed":
        path=attempt/"CineCam_04_native_pose_restore_receipt.json"
        value=json.loads(path.read_text());value["visual_mesh_overrides"][-1]["restore_succeeded"]=False;dump(path,value)
    if case=="outside_saved":out=tmp_path/"diagnostic_outside"
    if case=="world_changed_after_acceptance":
        path=attempt/"CineCam_04_native_pose_restore_receipt.json"
        value=json.loads(path.read_text());value["records"][-1]["sample"]["world_points_m"][0][0]+=1.;dump(path,value)
    with pytest.raises(ValueError):tool.export_attempt(attempt,alignment,native,out)
    assert not calls and not out.exists()


def test_location_valid_does_not_upgrade_unknown_or_outside(tmp_path,monkeypatch):
    attempt,alignment,native,out,calls = fixture(tmp_path,monkeypatch)
    path=attempt/"CineCam_01_native_pose_restore_receipt.json"
    value=json.loads(path.read_text())
    value["records"][0]["sample"]["world_points_m"][0]=[10.,100.,0.]
    dump(path,value)
    aligned=json.loads(alignment.read_text())
    aligned["cameras"]["CineCam_01"]["frames"][0]["keypoints"][0]["world_m"]=[10.,100.,0.]
    dump(alignment,aligned)
    tool.export_attempt(attempt,alignment,native,out)
    image=json.loads((out/"groups/seed_00001048/cameras/CineCam_01/pose2d.jsonl").read_text().splitlines()[0])
    assert image["objects"][0]["visibility_state"][0]=="outside"
    assert image["objects"][0]["location_valid"][0] is False
    assert image["objects"][0]["location_valid"][1] is True
