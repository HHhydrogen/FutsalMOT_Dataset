"""有限位置监督对照：消费冻结采样表，固定更新预算，支持完整状态续跑。"""
import argparse
from collections import Counter, OrderedDict
from copy import copy, deepcopy
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
CANDIDATE_SHA = "b2946b58b8e2b10510f185815018e698cee25b705433590494ae3913f7d15c33"
CHECKPOINT_SCHEMA = "futsalmot_pose_location_control_checkpoint_v1"
MEDIA_POLICY = "export_mask_decode_and_text_evidence_no_media_rehash"


def require(ok,message):
    if not ok: raise ValueError(message)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def rows(path): return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
def canonical(value): return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False)


def bound_file(binding):
    require(isinstance(binding,dict) and set(binding)>={"path","sha256"},"缺少实际文件SHA绑定")
    path=Path(binding["path"]).resolve()
    require(path.is_file() and sha(path)==binding["sha256"],"绑定文件缺失或SHA变化: "+str(path))
    return path


def atomic_json(path,value):
    temporary=Path(str(path)+".tmp")
    with temporary.open("w",encoding="utf-8",newline="\n") as handle:
        handle.write(canonical(value)+"\n");handle.flush();os.fsync(handle.fileno())
    os.replace(temporary,path)


def editor_running():
    """启动GPU/CPU研究前的新鲜只读进程门禁，不关闭任何进程。"""
    result=subprocess.run(["tasklist","/FI","IMAGENAME eq UnrealEditor.exe","/FO","CSV","/NH"],
                          check=True,capture_output=True,text=True)
    return any(row and row[0].lower()=="unrealeditor.exe" for row in csv.reader(result.stdout.splitlines()))


def validate_schedule(schedule,real_count,synthetic_count,selected):
    require(len(schedule)==1000 and len(selected)==len(set(selected))==1000,"需要冻结1000个更新及1000个不同S")
    shared=[[],[],[]];extra=[];synthetic=[]
    for index,row in enumerate(schedule):
        require(row.get("schema")=="futsalmot_pose_location_paired_schedule_row_v1"
                and type(row.get("update_index")) is int and row["update_index"]==index,"采样表行身份/次序错误")
        indices=row.get("shared_real_indices")
        require(isinstance(indices,list) and len(indices)==3,"每更新需要三个共享真实槽")
        require(all(type(i) is int and 0<=i<real_count for i in indices+[row.get("extra_real_index")]),"真实索引越界/非整数")
        require(type(row.get("synthetic_index")) is int and 0<=row["synthetic_index"]<synthetic_count,"合成索引越界")
        for slot,value in enumerate(indices): shared[slot].append(value)
        extra.append(row["extra_real_index"]);synthetic.append(row["synthetic_index"])
    require(real_count==1000 and all(sorted(column)==list(range(1000)) for column in shared+[extra]),"真实四槽各须覆盖相同1000 IDs一次")
    require(len(set(synthetic))==1000 and set(synthetic)==set(selected),"S采样不是冻结1000选择的完整一次曝光")


def validate_inputs(protocol_path,manifest_path,admission_path,seed,arm):
    """读取收据和绑定文件内容；candidate、仅路径或改源标志均不能准入。"""
    protocol_path,manifest_path,admission_path=map(lambda p:Path(p).resolve(),(protocol_path,manifest_path,admission_path))
    protocol,manifest,admission=map(read,(protocol_path,manifest_path,admission_path))
    require(admission.get("schema")=="futsalmot_pose_location_training_admission_v1"
        and admission.get("status")=="approved_location_only" and admission.get("approved_by")=="root"
        and admission.get("training_authorized") is True and admission.get("location_only") is True
        and admission.get("visibility_supervision") is False and admission.get("location_alignment_accepted") is True
        and admission.get("source_not_training_flags_preserved") is True,"缺root签署的有限位置监督准入，禁止训练")
    require(bound_file(admission["protocol"])==protocol_path and bound_file(admission["schedule_manifest"])==manifest_path,"准入不属于当前冻结协议/采样表")
    require(admission.get("media_policy")==MEDIA_POLICY,"S无媒体SHA策略必须由root明确限定准入")
    require(admission.get("bound_candidate_protocol_sha256")==CANDIDATE_SHA,"准入未绑定当前候选来源")
    gate=admission.get("UE_serial_gate",{})
    require(gate.get("all_rendering_finished") is True and gate.get("editor_safely_exited") is True,"UE渲染/安全退出门禁未签署")
    shutdown=read(bound_file(admission["shutdown_receipt"]))
    require(shutdown.get("normal_exit_confirmed") is True,"实际正常退出收据未通过")
    source_evidence=read(bound_file(admission["source_evidence"]))
    require(source_evidence.get("schema")=="futsalmot_pose17_location_training_candidate_v1","合成源证据不是本轮位置候选")
    groups=source_evidence.get("groups",[])
    require(len(groups)==4 and {g.get("seed") for g in groups}=={1042,1044,1048,1102},"实际位置源证据必须包含本轮四seed")
    for group in groups:
        checks=group.get("text_checks",{});flags=group.get("source_flags",{})
        require(all(checks.get(k) is True for k in ("passed","native_bundle","world_and_projection_exact"))
            and flags==dict(diagnostic_only=True,not_training=True,production_accepted=False,MRQ_sync_verified=False)
            and group.get("counts",{}).get("images")==480 and group["counts"].get("world_entity_records")==4800,
            "源文本/native/投影证据未通过或源状态被提升")
    require(manifest.get("schema")=="futsalmot_pose_location_paired_schedule_manifest_v1"
        and manifest.get("candidate_protocol_sha256")==CANDIDATE_SHA
        and manifest.get("protocol_sha256") in (None,sha(protocol_path)),"采样表协议身份不同")
    settings=protocol["paired_training_candidate"]
    expected=dict(actual_optimizer_updates=1000,batch=4,nbs=4,gradient_accumulation=1,imgsz=640,
        optimizer="SGD",lr0=.001,lrf=1.,momentum=.9,weight_decay=0.,warmup_epochs=0.,amp=False,
        multi_scale=False,early_stopping=False)
    require(all(settings.get(k)==v and (type(settings[k]) is bool if type(v) is bool else True) for k,v in expected.items()),"冻结协议超出已实现固定训练配置")
    require(seed in settings["seeds"] and type(seed) is int and arm in ("R","RS25"),"训练seed/arm不在冻结范围")
    require(protocol["arms"]["R"]["real_exposures"]==4000 and protocol["arms"]["RS25"]["real_exposures"]==3000
        and protocol["arms"]["RS25"]["synthetic_exposures"]==1000,"对照曝光预算发生变化")
    real_path,syn_path=bound_file(manifest["real_jsonl"]),bound_file(manifest["synthetic_jsonl"])
    require(bound_file(admission["real_jsonl"])==real_path and bound_file(admission["synthetic_jsonl"])==syn_path,"准入数据集不同")
    real,synthetic=rows(real_path),rows(syn_path)
    require(len(real)==manifest["real_jsonl"]["images"]==1000 and len(synthetic)==manifest["synthetic_jsonl"]["images"]==1440,"R/S完整图数不同")
    require(all(frame.get("source")=="coco_real" and all(person.get("source")=="coco_real" for person in frame["instances"]) for frame in real),"R不是完整COCO来源")
    require(all(type(frame.get("image_id")) is int for frame in real) and len({frame["image_id"] for frame in real})==1000,"真实COCO图像ID不完整或重复")
    require(all(frame.get("not_training") is True and frame.get("MRQ_sync_verified") is False
        and all(person.get("source")=="synthetic" and person.get("source_coco_v") is None
                and all(v in ("unknown","outside","behind_camera","missing") for v in person["visibility_state"])
                for person in frame["instances"]) for frame in synthetic),"不得翻转诊断状态或给S伪造COCO可见性")
    group_refs={g["seed"]:g for g in groups}
    require(Counter((frame.get("root_seed"),frame.get("camera_id")) for frame in synthetic)==
        Counter({(s,"CineCam_%02d"%c):120 for s in (1042,1044,1048) for c in range(1,5)}),"S训练集合不是三train seed完整四相机，禁止混入dev")
    for frame in synthetic:
        source=group_refs[frame["root_seed"]];start=307 if frame["root_seed"]==1044 else 100
        require(frame["source_dataset_ref"]==source["source_dataset_ref"] and frame["source_group_ref"]==source["source_group_ref"]
            and type(frame["source_frame_index"]) is int and start<=frame["source_frame_index"]<start+120,"S来源引用/连续窗口不同")
    selection=read(bound_file(manifest["selection_manifest"]))
    selected=manifest["selected_synthetic_indices"]
    require(isinstance(selected,list) and len(selected)==len(set(selected))==1000
        and all(type(i) is int and 0<=i<len(synthetic) for i in selected),"选中S索引不完整/重复/越界")
    require(selection.get("selected_synthetic_indices")==selected and selection.get("selected_synthetic_frame_keys")==manifest["selected_synthetic_frame_keys"],"选择清单与采样manifest不同")
    require([synthetic[i]["frame_key"] for i in selected]==manifest["selected_synthetic_frame_keys"],"选中索引与真实帧key不同")
    entries=[entry for entry in manifest["schedules"] if entry["seed"]==seed]
    require(len(entries)==2 and {entry["arm"] for entry in entries}=={"R","RS25"}
        and len({(entry["path"],entry["sha256"]) for entry in entries})==1,"配对两arm必须读取同一采样表")
    entry=next(e for e in entries if e["arm"]==arm)
    require(entry.get("updates")==1000 and entry.get("batch")==4,"采样表更新/batch不符")
    schedule_path=bound_file(entry);schedule=rows(schedule_path)
    validate_schedule(schedule,len(real),len(synthetic),selected)
    return dict(protocol=protocol,manifest=manifest,admission=admission,schedule=schedule,arm=arm,seed=seed,
        real_jsonl=real_path,synthetic_jsonl=syn_path,selected_synthetic_indices=selected,
        bindings=dict(protocol_sha256=sha(protocol_path),schedule_manifest_sha256=sha(manifest_path),
            admission_sha256=sha(admission_path),schedule_sha256=sha(schedule_path),
            trainer_source_sha256=sha(__file__),real_jsonl_sha256=sha(real_path),synthetic_jsonl_sha256=sha(syn_path)))


class CachedDataset:
    """仅缓存确定性LetterBox结果；按tensor字节上限LRU，不修改原图。"""
    def __init__(self,dataset,max_bytes):
        self.dataset=dataset;self.cache=OrderedDict();self.max_bytes=max_bytes;self.bytes=0
        self.hits=self.misses=self.evictions=0
    def __getitem__(self,index):
        if index in self.cache:
            self.hits+=1;value,size=self.cache.pop(index);self.cache[index]=(value,size)
        else:
            self.misses+=1;value=self.dataset[index]
            size=sum(v.numel()*v.element_size() for v in value.values() if hasattr(v,"numel"))
            if size<=self.max_bytes:
                while self.cache and self.bytes+size>self.max_bytes:
                    _,(_,previous)=self.cache.popitem(last=False);self.bytes-=previous;self.evictions+=1
                self.cache[index]=(value,size);self.bytes+=size
        return {k:v.clone() if hasattr(v,"clone") else deepcopy(v) for k,v in value.items()}
    def prewarm(self,indices):
        started=time.perf_counter()
        for index in sorted(set(indices)): self[index]
        return dict(seconds=time.perf_counter()-started,images=len(set(indices)),bytes=self.bytes,evictions=self.evictions)


def freeze_visibility_rows(model):
    """物理51通道保留，未输出的visibility行梯度为零并逐步核对不变。"""
    import torch
    references=[];handles=[]
    for branch in model.model[-1].cv4:
        conv=branch[-1][0]
        for parameter in (conv.weight,conv.bias):
            mask=torch.ones_like(parameter);mask[2::3]=0
            handles.append(parameter.register_hook(lambda grad,mask=mask:grad*mask))
            references.append((parameter,parameter[2::3].detach().clone()))
    return references,handles


def observed_criterion(model):
    from grf_ue_bridge.pose_location_supervision_v2 import separated_location_loss_class
    criterion=separated_location_loss_class()(model)
    criterion.observed={}
    def record(module,args):
        valid=args[2].bool();foreground=valid.shape[0];nonempty=int(valid.any(1).sum())
        criterion.observed=dict(foreground_anchors=foreground,foreground_with_valid_points=nonempty,
            assigned_valid_points=int(valid.sum()),effective_pose_weight=float(model.args.pose)*nonempty/max(1,foreground))
    criterion.keypoint_loss.register_forward_pre_hook(record)
    criterion.bce_pose.register_forward_pre_hook(lambda *a:(_ for _ in ()).throw(AssertionError("禁止visibility BCE")))
    return criterion


def model_copy(model):
    result=deepcopy(model).cpu().float()
    if hasattr(result,"criterion"): delattr(result,"criterion")
    return result


def save_last(path,model,ema,optimizer,completed,context,initial_hash,chain,last_exposure):
    import numpy as np
    import torch
    started=time.perf_counter()
    checkpoint=dict(schema=CHECKPOINT_SCHEMA,model=model_copy(model),ema=model_copy(ema.ema),
        optimizer=deepcopy(optimizer.state_dict()),ema_updates=ema.updates,completed_updates=completed,
        arm=context["arm"],seed=context["seed"],bindings=context["bindings"],initial_state_sha256=initial_hash,
        exposure_chain_sha256=chain,last_exposure=last_exposure,planned_updates=1000,
        rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                 cuda=torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else None))
    temporary=path.with_suffix(".pt.tmp");torch.save(checkpoint,temporary);os.replace(temporary,path)
    return time.perf_counter()-started


def state_hash(model):
    digest=hashlib.sha256()
    for name,value in model.state_dict().items():
        array=value.detach().cpu().contiguous().numpy()
        digest.update(canonical([name,str(array.dtype),list(array.shape)]).encode());digest.update(array.tobytes())
    return digest.hexdigest()


def source_key(dataset,index):
    base=dataset.dataset if isinstance(dataset,CachedDataset) else dataset
    record=base.records[index]
    return record.get("frame_key", "coco_train2017:%s"%record["image_id"] if "image_id" in record else record["image_file"])


def _train_loop(model,criterion,optimizer,ema,real,synthetic,schedule,context,output,completed=0,chain="0"*64,
                initial_hash=None,last_exposure=None,stop_after=None,stop_requested=None):
    """最小更新引擎；公开CLI只能经validate_inputs进入，CPU toy测试不签正式准入。"""
    import torch
    from grf_ue_bridge.pose_location_dataset_v2 import LocationPoseDataset
    output=Path(output);last=output/"last.pt";exposure_path=output/"exposures.jsonl"
    device=next(model.parameters()).device;model.train()
    frozen,handles=freeze_visibility_rows(model)
    initial_hash=initial_hash or state_hash(model)
    checkpoint_seconds=0.;start=time.perf_counter()
    if completed==0: checkpoint_seconds+=save_last(last,model,ema,optimizer,0,context,initial_hash,chain,None)
    try:
        for row in schedule[completed:]:
            if (stop_requested and stop_requested()) or (stop_after is not None and completed>=stop_after): break
            started=time.perf_counter();index=row["update_index"]
            require(index==completed,"恢复更新索引不连续")
            samples=[real[i] for i in row["shared_real_indices"]]
            is_mixed=context["arm"]=="RS25"
            samples.append(synthetic[row["synthetic_index"]] if is_mixed else real[row["extra_real_index"]])
            batch=LocationPoseDataset.collate_fn(samples)
            for key,value in batch.items():
                if isinstance(value,torch.Tensor): batch[key]=value.to(device)
            batch["img"]=batch["img"].float()/255.
            atomic_json(output/"update_transaction.json",dict(update_index=index,status="started_not_committed"))
            optimizer.zero_grad(set_to_none=True)
            criterion.observed={}
            loss,items=criterion(model(batch["img"]),batch)
            require(torch.isfinite(loss).all().item() and items[2].item()==0.,"非有限loss或意外visibility监督")
            loss.sum().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=10.)
            optimizer.step()
            require(all(torch.equal(parameter[2::3],reference) for parameter,reference in frozen),"物理visibility行意外改变")
            ema.update(model)
            # 未输出行连EMA也保持原值，避免插值舍入造成物理行漂移。
            with torch.no_grad():
                for branch,ema_branch in zip(model.model[-1].cv4,ema.ema.model[-1].cv4):
                    for parameter,ema_parameter in zip((branch[-1][0].weight,branch[-1][0].bias),
                                                      (ema_branch[-1][0].weight,ema_branch[-1][0].bias)):
                        ema_parameter[2::3].copy_(parameter[2::3])
            completed+=1
            valid=batch["location_valid"]
            exposure=dict(actual_optimizer_update=completed,update_index=index,arm=context["arm"],seed=context["seed"],
                real_indices=row["shared_real_indices"]+([] if is_mixed else [row["extra_real_index"]]),
                synthetic_indices=[row["synthetic_index"]] if is_mixed else [],image_files=batch["im_file"],
                source_image_keys=[source_key(real,i) for i in row["shared_real_indices"]]
                    +[source_key(synthetic,row["synthetic_index"]) if is_mixed else source_key(real,row["extra_real_index"])],
                real_images=3 if is_mixed else 4,synthetic_images=1 if is_mixed else 0,person_boxes=len(valid),
                people_with_valid_points=int(valid.any(1).sum()),all_missing_people=int((~valid.any(1)).sum()),valid_points=int(valid.sum()),
                foreground_anchors=0,effective_pose_weight=0.,learning_rate=optimizer.param_groups[0]["lr"],
                loss_items=items.tolist(),visibility_loss=0.,visibility_rows_unchanged=True,elapsed_seconds=time.perf_counter()-started)
            exposure.update(criterion.observed)
            chain=hashlib.sha256((chain+canonical(exposure)).encode()).hexdigest()
            last_exposure=dict(exposure,chain_sha256=chain)
            checkpoint_seconds+=save_last(last,model,ema,optimizer,completed,context,initial_hash,chain,last_exposure)
            with exposure_path.open("a",encoding="utf-8") as handle:
                handle.write(canonical(last_exposure)+"\n");handle.flush()
            atomic_json(output/"update_transaction.json",dict(update_index=index,status="committed",completed_updates=completed))
            if stop_after is not None and completed>=stop_after: break
    finally:
        for handle in handles: handle.remove()
    return dict(completed_updates=completed,planned_updates=1000,complete=completed==1000,checkpoint_seconds=checkpoint_seconds,
                elapsed_seconds=time.perf_counter()-start,initial_state_sha256=initial_hash,exposure_chain_sha256=chain)


def reconcile_resume(output,checkpoint,context):
    require(checkpoint.get("schema")==CHECKPOINT_SCHEMA and checkpoint.get("bindings")==context["bindings"]
        and checkpoint.get("arm")==context["arm"] and checkpoint.get("seed")==context["seed"]
        and checkpoint.get("planned_updates")==1000,"断点不属于同一代码/协议/数据/arm/seed")
    completed=checkpoint["completed_updates"]
    require(type(completed) is int and 0<=completed<=1000 and checkpoint["ema_updates"]==completed,"断点实际更新/EMA计数错误")
    transaction=Path(output)/"update_transaction.json"
    if transaction.exists():
        pending=read(transaction)
        require(pending.get("status")=="committed" or pending.get("update_index",1000)<completed,
                "发现未提交优化更新，不能自动重放并伪称固定实际预算；保留现场交root决定")
    path=Path(output)/"exposures.jsonl";exposures=rows(path) if path.exists() else []
    if len(exposures)==completed-1 and checkpoint.get("last_exposure"):
        with path.open("a",encoding="utf-8") as handle: handle.write(canonical(checkpoint["last_exposure"])+"\n")
        exposures.append(checkpoint["last_exposure"])
    require(len(exposures)==completed,"原始曝光日志与实际优化断点不同")
    chain="0"*64
    for index,record in enumerate(exposures):
        value=dict(record);recorded=value.pop("chain_sha256")
        require(value["actual_optimizer_update"]==index+1 and value["update_index"]==index,"曝光日志次序变化")
        chain=hashlib.sha256((chain+canonical(value)).encode()).hexdigest()
        require(recorded==chain,"曝光日志链已改变")
    require(chain==checkpoint["exposure_chain_sha256"],"曝光链与last断点不一致")
    return completed


def train(context,output,device="cuda:0",resume=False,cache_mib=3072,stop_after=None):
    import numpy as np
    import torch
    import ultralytics
    from ultralytics.utils import DEFAULT_CFG
    from ultralytics.utils.torch_utils import ModelEMA
    from grf_ue_bridge.pose_location_dataset_v2 import LocationPoseDataset
    from grf_ue_bridge.pose_xy_head_transfer import map_pose_xy
    require(ultralytics.__version__=="8.4.60" and torch.__version__=="2.6.0+cu124","仅当前已冻结研究运行时")
    require(device in ("cpu","cuda:0") and not editor_running(),"设备不支持或UE仍存在，禁止研究训练")
    require(type(cache_mib) is int and 0<=cache_mib<=4096,"缓存只允许0..4096MiB")
    require(stop_after is None or type(stop_after) is int and 1<=stop_after<=1000,"暂停点必须在1..1000累计更新内")
    output=Path(output).resolve()
    require((output/"last.pt").is_file() if resume else not output.exists(),"需全新输出或显式有效last续跑")
    output.mkdir(parents=True,exist_ok=resume)
    settings=context["protocol"]["paired_training_candidate"]
    seed=context["seed"];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if device.startswith("cuda"): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    initialization=context["protocol"]["initialization"]
    checkpoint_path=Path(initialization["derived_checkpoint"])
    require(sha(checkpoint_path)==initialization["derived_sha256"],"初始化实际SHA变化")
    loaded=torch.load(output/"last.pt" if resume else checkpoint_path,map_location="cpu",weights_only=False)
    completed=reconcile_resume(output,loaded,context) if resume else 0
    model=map_pose_xy(loaded["model"].float()).to(device).train();model.requires_grad_(True);model.args=copy(DEFAULT_CFG)
    for name,parameter in model.named_parameters():
        if ".dfl." in name: parameter.requires_grad_(False)  # 保持上游固定DFL投影卷积。
    optimizer=torch.optim.SGD(model.parameters(),lr=settings["lr0"],momentum=settings["momentum"],weight_decay=0.)
    ema=ModelEMA(model)
    if resume:
        optimizer.load_state_dict(loaded["optimizer"]);ema.ema.load_state_dict(loaded["ema"].state_dict());ema.updates=loaded["ema_updates"]
        random.setstate(loaded["rng"]["python"]);np.random.set_state(loaded["rng"]["numpy"]);torch.set_rng_state(loaded["rng"]["torch"])
        if loaded["rng"]["cuda"] is not None: torch.cuda.set_rng_state_all(loaded["rng"]["cuda"])
    real_base=LocationPoseDataset(context["real_jsonl"],640,verify_image_sha256=True)
    synthetic_base=LocationPoseDataset(context["synthetic_jsonl"],640,verify_image_sha256=False)
    # 两域共享总缓存上限，R只需真实缓存；RS为1000R+1000S各一半。
    limit=cache_mib*1024*1024
    real=CachedDataset(real_base,limit if context["arm"]=="R" else limit//2)
    synthetic=CachedDataset(synthetic_base,0 if context["arm"]=="R" else limit-limit//2)
    cache={"strategy":"bounded_uint8_deterministic_letterbox_LRU","limit_mib":cache_mib,
           "real":real.prewarm(range(1000)),"synthetic":synthetic.prewarm(context["selected_synthetic_indices"] if context["arm"]=="RS25" else [])}
    criterion=observed_criterion(model)
    stopped=[False];previous=signal.signal(signal.SIGINT,lambda *a:stopped.__setitem__(0,True))
    try:
        result=_train_loop(model,criterion,optimizer,ema,real,synthetic,context["schedule"],context,output,completed,
            loaded["exposure_chain_sha256"] if resume else "0"*64,loaded["initial_state_sha256"] if resume else None,
            loaded.get("last_exposure") if resume else None,stop_after,lambda:stopped[0])
    finally: signal.signal(signal.SIGINT,previous)
    result.update(schema="futsalmot_pose_location_control_run_v1",arm=context["arm"],seed=seed,bindings=context["bindings"],
        cache=cache,checkpoint_choice="last_exact_budget_EMA_no_best",device=device,visibility_supervised=False,
        runtime=dict(torch=torch.__version__,ultralytics=ultralytics.__version__,cuda=torch.version.cuda,
            device_name=torch.cuda.get_device_name(0) if device.startswith("cuda") else "CPU"),
        source_not_training_flags_preserved=True,real_test_read=False,AP_evaluated=False,
        actual_exposures=dict(real=result["completed_updates"]*(4 if context["arm"]=="R" else 3),
            synthetic=result["completed_updates"]*(0 if context["arm"]=="R" else 1)),
        exposure_budget=dict(real=4000 if context["arm"]=="R" else 3000,synthetic=0 if context["arm"]=="R" else 1000))
    atomic_json(output/"run_receipt.json",result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("protocol","schedule-manifest","admission","output"): parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--arm",choices=("R","RS25"),required=True)
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--device",choices=("cpu","cuda:0"),default="cuda:0")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--cache-mib",type=int,default=3072)
    parser.add_argument("--stop-after",type=int,help="安全暂停于该累计更新；不改变1000步总预算或生成完成声明")
    args=parser.parse_args()
    os.environ.update(YOLO_AUTOINSTALL="false",YOLO_OFFLINE="true",WANDB_MODE="disabled",COMET_MODE="DISABLED",CUBLAS_WORKSPACE_CONFIG=":4096:8")
    context=validate_inputs(args.protocol,args.schedule_manifest,args.admission,args.seed,args.arm)
    print(canonical(train(context,args.output,args.device,args.resume,args.cache_mib,args.stop_after)))


if __name__=="__main__": main()
