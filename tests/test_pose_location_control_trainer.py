"""固定更新研究入口的CPU合同；无真实训练、GPU或协议签署。"""
from copy import copy,deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("pose_location_control_test",ROOT/"experiments/sim2real/train_pose_location_control.py")
TOOL=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(TOOL)


def dump(path,value):
    path.write_text(json.dumps(value),encoding="utf-8");return {"path":str(path),"sha256":TOOL.sha(path)}


def bound_fixture(tmp_path):
    protocol=json.loads((ROOT/"experiments/sim2real/pose_location_coco_candidate_protocol.json").read_text(encoding="utf-8"))
    protocol_path=tmp_path/"protocol.json";protocol_binding=dump(protocol_path,protocol)
    real=[dict(image_id=i,source="coco_real",instances=[dict(source="coco_real")]) for i in range(1000)]
    synthetic=[];groups=[]
    flags=dict(diagnostic_only=True,not_training=True,production_accepted=False,MRQ_sync_verified=False)
    for seed in (1042,1044,1048,1102):
        group=dict(seed=seed,source_dataset_ref="dataset%d.json"%seed,source_group_ref="group%d.json"%seed,
            source_flags=flags,text_checks=dict(passed=True,native_bundle=True,world_and_projection_exact=True),
            counts=dict(images=480,world_entity_records=4800))
        groups.append(group)
        if seed==1102:continue
        for camera in range(1,5):
            for frame in range(120):
                synthetic.append(dict(frame_key="%d:%d:%d"%(seed,camera,frame),root_seed=seed,camera_id="CineCam_%02d"%camera,
                    source_frame_index=(307 if seed==1044 else 100)+frame,not_training=True,MRQ_sync_verified=False,
                    source_dataset_ref=group["source_dataset_ref"],source_group_ref=group["source_group_ref"],
                    instances=[dict(source="synthetic",source_coco_v=None,visibility_state=["unknown"]*17)]))
    def jsonl(path,values):
        path.write_text("".join(json.dumps(row)+"\n" for row in values),encoding="utf-8")
        return dict(path=str(path),sha256=TOOL.sha(path),images=len(values))
    real_binding=jsonl(tmp_path/"real.jsonl",real);synthetic_binding=jsonl(tmp_path/"synthetic.jsonl",synthetic)
    selected=list(range(1000));keys=[synthetic[i]["frame_key"] for i in selected]
    selection=dump(tmp_path/"selection.json",dict(selected_synthetic_indices=selected,selected_synthetic_frame_keys=keys))
    schedule=[dict(schema="futsalmot_pose_location_paired_schedule_row_v1",update_index=i,
        shared_real_indices=[i,(i+1)%1000,(i+2)%1000],extra_real_index=(i+3)%1000,synthetic_index=i) for i in range(1000)]
    schedule_binding=jsonl(tmp_path/"schedule.jsonl",schedule);schedule_binding.pop("images")
    manifest=dict(schema="futsalmot_pose_location_paired_schedule_manifest_v1",protocol_sha256=None,
        candidate_protocol_sha256=TOOL.CANDIDATE_SHA,real_jsonl=real_binding,synthetic_jsonl=synthetic_binding,
        selection_manifest=selection,selected_synthetic_indices=selected,selected_synthetic_frame_keys=keys,
        schedules=[dict(schedule_binding,seed=20261024,arm=arm,updates=1000,batch=4) for arm in ("R","RS25")])
    manifest_path=tmp_path/"manifest.json";manifest_binding=dump(manifest_path,manifest)
    shutdown=dump(tmp_path/"shutdown.json",dict(normal_exit_confirmed=True))
    source=dump(tmp_path/"source_evidence.json",dict(schema="futsalmot_pose17_location_training_candidate_v1",groups=groups))
    admission=dict(schema="futsalmot_pose_location_training_admission_v1",status="approved_location_only",approved_by="root",
        training_authorized=True,location_only=True,visibility_supervision=False,location_alignment_accepted=True,
        source_not_training_flags_preserved=True,protocol=protocol_binding,schedule_manifest=manifest_binding,
        bound_candidate_protocol_sha256=TOOL.CANDIDATE_SHA,real_jsonl=real_binding,synthetic_jsonl=synthetic_binding,
        media_policy=TOOL.MEDIA_POLICY,UE_serial_gate=dict(all_rendering_finished=True,editor_safely_exited=True),
        shutdown_receipt=shutdown,source_evidence=source)
    admission_path=tmp_path/"admission.json";dump(admission_path,admission)
    return protocol_path,manifest_path,admission_path


def test_consumes_bound_schedule_without_changing_candidate_or_source_flags(tmp_path):
    paths=bound_fixture(tmp_path)
    value=TOOL.validate_inputs(*paths,20261024,"RS25")
    assert len(value["schedule"])==1000 and value["admission"]["visibility_supervision"] is False
    assert sum(row["synthetic_index"]>=0 for row in value["schedule"])==1000
    assert TOOL.rows(value["synthetic_jsonl"])[0]["not_training"] is True


@pytest.mark.parametrize("damage",["candidate_only","visibility","protocol_sha","source_evidence","shutdown","media_policy"])
def test_rejects_missing_or_wrong_root_admission(tmp_path,damage):
    paths=bound_fixture(tmp_path);admission=TOOL.read(paths[2])
    if damage=="candidate_only": admission["training_authorized"]=False
    elif damage=="visibility": admission["visibility_supervision"]=True
    elif damage=="protocol_sha": admission["protocol"]["sha256"]="0"*64
    elif damage=="source_evidence": admission["source_evidence"]["sha256"]="0"*64
    elif damage=="shutdown":
        binding=admission["shutdown_receipt"];dump(Path(binding["path"]),dict(normal_exit_confirmed=False));binding["sha256"]=TOOL.sha(binding["path"])
    else: admission["media_policy"]="skip_everything"
    dump(paths[2],admission)
    with pytest.raises(ValueError):TOOL.validate_inputs(*paths,20261024,"RS25")


@pytest.mark.parametrize("damage",["missing_update","source_index","real_coverage","duplicate_S"])
def test_sampling_table_requires_exact_update_and_exposure_budget(damage):
    schedule=[dict(schema="futsalmot_pose_location_paired_schedule_row_v1",update_index=i,
        shared_real_indices=[i,i,i],extra_real_index=i,synthetic_index=i) for i in range(1000)]
    if damage=="missing_update":schedule.pop()
    elif damage=="source_index":schedule[5]["extra_real_index"]=1000
    elif damage=="real_coverage":schedule[5]["shared_real_indices"][1]=0
    else:schedule[5]["synthetic_index"]=0
    with pytest.raises(ValueError):TOOL.validate_schedule(schedule,1000,1440,list(range(1000)))


def test_actual_cpu_two_update_resume_equal_including_ema_momentum_and_visibility(tmp_path):
    torch=pytest.importorskip("torch");pytest.importorskip("ultralytics")
    from ultralytics.nn.tasks import PoseModel
    from ultralytics.utils import DEFAULT_CFG,YAML
    from ultralytics.utils.torch_utils import ModelEMA
    import ultralytics
    from grf_ue_bridge.pose_xy_head_transfer import map_pose_xy
    from grf_ue_bridge.pose_location_dataset_v2 import LocationPoseDataset
    spec=importlib.util.spec_from_file_location("pose_toy_fixture",ROOT/"tools/smoke_pose_location_jsonl_cpu.py")
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    fixture.make_toy_jsonl(tmp_path/"images")
    dataset=LocationPoseDataset(tmp_path/"images/train.jsonl",64)
    torch.set_num_threads(1);torch.manual_seed(89)
    cfg=YAML.load(Path(ultralytics.__file__).parent/"cfg/models/11/yolo11-pose.yaml");cfg["scale"]="n"
    original=map_pose_xy(PoseModel(cfg,ch=3,nc=1,verbose=False)).cpu().train()
    original.requires_grad_(True);original.args=copy(DEFAULT_CFG)
    context=dict(seed=89,arm="RS25",bindings={"toy_only":True,"training_authorized":False})
    schedule=[dict(update_index=i,shared_real_indices=[0,0,0],extra_real_index=0,synthetic_index=1) for i in range(2)]
    cached=TOOL.CachedDataset(dataset,1024*1024)
    warmup=cached.prewarm([0,1]);assert warmup["images"]==2 and cached.bytes<=1024*1024
    sample=cached[0];sample["img"].zero_();assert cached[0]["img"].sum()>0
    def components(model):
        return TOOL.observed_criterion(model),torch.optim.SGD(model.parameters(),lr=.001,momentum=.9),ModelEMA(model)
    uninterrupted=deepcopy(original);criterion,optimizer,ema=components(uninterrupted)
    full=tmp_path/"full";full.mkdir()
    TOOL._train_loop(uninterrupted,criterion,optimizer,ema,cached,cached,schedule,context,full)
    interrupted=deepcopy(original);criterion,optimizer,ema=components(interrupted)
    resume=tmp_path/"resume";resume.mkdir()
    TOOL._train_loop(interrupted,criterion,optimizer,ema,cached,cached,schedule,context,resume,stop_after=1)
    saved=torch.load(resume/"last.pt",map_location="cpu",weights_only=False)
    assert TOOL.reconcile_resume(resume,saved,context)==1
    model=saved["model"].float().train();model.requires_grad_(True)
    criterion,optimizer,ema=components(model);optimizer.load_state_dict(saved["optimizer"])
    ema.ema.load_state_dict(saved["ema"].state_dict());ema.updates=saved["ema_updates"]
    TOOL._train_loop(model,criterion,optimizer,ema,cached,cached,schedule,context,resume,completed=1,
        chain=saved["exposure_chain_sha256"],initial_hash=saved["initial_state_sha256"],last_exposure=saved["last_exposure"])
    a=torch.load(full/"last.pt",map_location="cpu",weights_only=False)
    b=torch.load(resume/"last.pt",map_location="cpu",weights_only=False)
    assert a["completed_updates"]==b["completed_updates"]==a["ema_updates"]==b["ema_updates"]==2
    for key,value in a["model"].state_dict().items():assert torch.equal(value,b["model"].state_dict()[key])
    for key,value in a["ema"].state_dict().items():assert torch.equal(value,b["ema"].state_dict()[key])
    for key,state in a["optimizer"]["state"].items():
        assert torch.equal(state["momentum_buffer"],b["optimizer"]["state"][key]["momentum_buffer"])
    exposures=TOOL.rows(resume/"exposures.jsonl")
    assert len(exposures)==2 and all(e["person_boxes"]==7 and e["all_missing_people"]==3 for e in exposures)
    assert all(e["real_images"]==3 and e["synthetic_images"]==1 and e["visibility_loss"]==0 for e in exposures)
    assert all(e["visibility_rows_unchanged"] and e["foreground_anchors"]>0 for e in exposures)
    assert a["planned_updates"]==1000 and not TOOL.read(resume/"update_transaction.json")["status"]=="started_not_committed"
    TOOL.atomic_json(resume/"update_transaction.json",dict(update_index=2,status="started_not_committed"))
    with pytest.raises(ValueError,match="未提交优化更新"):TOOL.reconcile_resume(resume,b,context)
