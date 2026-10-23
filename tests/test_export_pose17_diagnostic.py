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
from test_pose17_location_pilot_contract import pilot_plan


def dump(path,value):
    path.write_text(json.dumps(value),encoding="utf-8")


def fixture(tmp_path,monkeypatch,frame_count=10,seed=1048):
    saved = tmp_path/"Saved"
    saved.mkdir()
    monkeypatch.setattr(tool,"WORKSPACE_SAVED",saved)
    attempt = saved/"joint_attempt"
    attempt.mkdir()
    plan,state,before,after,manifests,hashes,receipts = joint_bundle()
    if frame_count==120:
        plan = pilot_plan(seed)
        for receipt in receipts.values():
            templates = {row["entity_id"]:row for row in receipt["records"]}
            receipt["records"] = []
            for entity,template in templates.items():
                for frame in range(frame_count):
                    row = copy.deepcopy(template)
                    row.update(output_frame=frame,sequence_source_frame=plan["start_frame"]+frame,
                        hook_engine_frame=110+frame,engine_frame_before=110+frame,engine_frame_after=110+frame,
                        gpu_lbs_warmup_engine_frames_observed=9+frame,override_warmup_engine_frames_observed=10+frame)
                    receipt["records"].append(row)
    else:
        plan.update(source_motion=dict(root_seed=seed,fps=10,sequence_frame_equals_source_step=True),source_scope="real_GRF_with_diagnostic_roster")
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
        for frame in range(frame_count):
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
                EXR=str(directory/("%06d.exr"%frame)),camera=copy.deepcopy(camera),source_frame=plan["start_frame"]+frame,
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


def test_frozen_120_pilot_exports_complete_sets_and_dynamic_counts(tmp_path,monkeypatch):
    attempt,alignment,native,out,calls = fixture(tmp_path,monkeypatch,frame_count=120,seed=1044)
    report = tool.export_attempt(attempt,alignment,native,out,copy_rgb=True)
    assert report["rgb_frames"]==report["mask_frames"]==report["rgb_files_copied"]==480
    assert report["native_pose_records"]==4800 and report["exr_decode_count"]==len(calls)==len(set(calls))==480
    assert report["not_training"] and not report["MRQ_sync_verified"]
    group = out/"groups/seed_00001044"
    timeline = [json.loads(line) for line in (group/"timeline.jsonl").read_text().splitlines()]
    assert len(timeline)==120 and timeline[0]["source_frame_index"]==307 and timeline[-1]["source_frame_index"]==426
    assert json.loads((group/"group.json").read_text())["location_pilot"]["split"]=="train"
    for camera in (group/"cameras").iterdir():
        expected = {"%06d.png"%i for i in range(120)}
        assert {p.name for p in (camera/"rgb").iterdir()}==expected
        assert {p.name for p in (camera/"mask").iterdir()}==expected
        rows = [json.loads(line) for line in (camera/"pose2d.jsonl").read_text().splitlines()]
        assert len(rows)==120 and sum(len(row["objects"]) for row in rows)==1200
        assert rows[-1]["objects"][0]["visibility_state"]==["unknown"]*17


@pytest.mark.parametrize("case",["missing_flag","missing_all_pilot_fields","wrong_window_1042","wrong_window_1044",
    "wrong_window_1048","wrong_window_1102","missing_media120","extra_media120"])
def test_reject_implicit_120_wrong_frozen_window_and_incomplete_media(tmp_path,monkeypatch,case):
    seed = int(case.rsplit("_",1)[-1]) if case.startswith("wrong_window_") else 1042
    attempt,alignment,native,out,calls = fixture(tmp_path,monkeypatch,frame_count=120,seed=seed)
    path = attempt/"plan.json"
    plan = json.loads(path.read_text())
    if case=="missing_flag": plan.pop("location_pilot_contract")
    elif case=="missing_all_pilot_fields":
        plan.pop("location_pilot_contract");plan.pop("location_pilot")
    elif case.startswith("wrong_window_"):
        plan["source_motion"]["selected_source_frame_range"][0]+=1
    elif case=="missing_media120": (attempt/"CineCam_04/000119.exr").unlink()
    else: (attempt/"CineCam_04/000120.exr").write_bytes(b"TOY_NOT_EXR")
    dump(path,plan)
    state = json.loads((attempt/"state.json").read_text())
    state["plan_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    dump(attempt/"state.json",state)
    with pytest.raises(ValueError): tool.export_attempt(attempt,alignment,native,out)
    assert not calls and not out.exists()


def tiny_real_exr_jobs(tmp_path):
    """自绘16×8真EXR；用于实际spawn合同，不读取UE或正式媒体。"""
    import OpenEXR
    mapping = {side+str(i):"Player_"+side+str(i) for side in ("L","R") for i in range(5)}
    manifest = {label:"%08x"%(0x3f800000+i*65536) for i,label in enumerate(mapping.values())}
    sources = []
    for frame in range(4):
        pixels = np.zeros((8,16,4),dtype=np.float32)
        for i,label in enumerate(mapping.values()):
            pixels[frame:frame+2,i,0] = np.array([int(manifest[label],16)],dtype=np.uint32).view(np.float32)[0]
        exr,rgb = tmp_path/("%06d.exr"%frame),tmp_path/("%06d.png"%frame)
        with OpenEXR.File({"cryptomatte/toy/manifest":json.dumps(manifest)},{"RGBA":pixels}) as encoded:
            encoded.write(str(exr))
        Image.new("RGB",(16,8)).save(rgb)
        sources.append((frame,exr,rgb,mapping))
    return sources


def mask_jobs(sources,directory):
    directory.mkdir()
    return [(frame,exr,rgb,directory/("%06d.png"%frame),mapping) for frame,exr,rgb,mapping in sources]


def test_spawn_serial_parallel_real_tiny_exr_equal_and_ordered(tmp_path):
    sources = tiny_real_exr_jobs(tmp_path)
    serial = list(tool._mask_results(mask_jobs(sources,tmp_path/"serial"),1))
    parallel = list(tool._mask_results(mask_jobs(sources,tmp_path/"parallel"),2))
    assert serial==parallel and [row[0] for row in parallel]==[0,1,2,3]
    for frame in range(4):
        assert (tmp_path/"serial"/("%06d.png"%frame)).read_bytes()==(tmp_path/"parallel"/("%06d.png"%frame)).read_bytes()


def test_spawn_decode_failure_keeps_partial_without_retry(tmp_path):
    sources = tiny_real_exr_jobs(tmp_path)
    sources[1][1].write_bytes(b"TOY_CORRUPT_EXR")
    destination = tmp_path/"partial"
    results = tool._mask_results(mask_jobs(sources,destination),2)
    assert next(results)[0]==0
    with pytest.raises(Exception): list(results)
    assert (destination/"000000.png").exists() and not (destination/"000001.png").exists()
    assert not (destination/"000003.png").exists()  # 两个在途任务，失败后不继续发帧。


def test_export_decode_failure_keeps_partial_without_complete_receipt(tmp_path,monkeypatch):
    attempt,alignment,native,out,calls = fixture(tmp_path,monkeypatch)
    decoder = tool.decode_mask
    def failing_decoder(path,mapping):
        if Path(path).stem=="000003": raise ValueError("TOY解码失败")
        return decoder(path,mapping)
    monkeypatch.setattr(tool,"decode_mask",failing_decoder)
    with pytest.raises(ValueError,match="TOY解码失败"):
        tool.export_attempt(attempt,alignment,native,out)
    assert (out/"export_started.json").exists() and not (out/"dataset.json").exists()
    assert len(list(out.rglob("mask/*.png")))==3


@pytest.mark.parametrize("workers",[0,7,True,1.5])
def test_workers_capped_before_writing_output(tmp_path,monkeypatch,workers):
    monkeypatch.setattr(tool,"WORKSPACE_SAVED",tmp_path)
    out = tmp_path/"diagnostic_bad_workers"
    with pytest.raises(ValueError,match="workers"):
        tool.export_attempt(tmp_path/"unused",tmp_path/"unused",tmp_path/"unused",out,workers=workers)
    assert not out.exists()
