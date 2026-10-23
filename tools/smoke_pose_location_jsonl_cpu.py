"""全新自绘JSONL夹具→LetterBox→位置loss→一更新和resume，严格CPU。"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import socket
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
SOURCE_SHA='869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_toy_jsonl(root):
    """自绘竖/横图和部分/全缺点真实源语义、unknown合成源语义夹具。"""
    from PIL import Image,ImageDraw
    from grf_ue_bridge.pose_location_supervision_v2 import coco_person_location,synthetic_person_location
    root=Path(root)
    if root.exists():raise FileExistsError(root)
    root.mkdir(parents=True)
    files=[];frames=[]
    for index,(width,height) in enumerate(((192,96),(96,192))):
        image=Image.new('RGB',(width,height),(25+index*30,60,80));draw=ImageDraw.Draw(image)
        draw.text((2,2),'TOY LOCATION CPU',fill='white')
        if index==0:
            v=[1,2]*4+[0]*9
            real=coco_person_location(dict(id=11,category_id=1,iscrowd=0,bbox=[6,6,70,78],num_keypoints=8,
                 keypoints=[n for j,flag in enumerate(v) for n in ((20+j*3,20+j*5,flag) if flag else (0,0,0))]),width,height)
            missing=coco_person_location(dict(id=12,category_id=1,iscrowd=0,bbox=[110,8,60,80],num_keypoints=0,keypoints=[0]*51),width,height)
            instances=[real,missing]
        else:
            instances=[synthetic_person_location(dict(id=21,bbox=[10,10,70,165],xy_px=[[30+j,30+j*7] for j in range(17)],
                        location_valid=[True]*17,visibility_state=['unknown']*17),width,height)]
        for item in instances:
            x,y,w,h=item['bbox_xywh_px'];draw.rectangle((x,y,x+w,y+h),outline='yellow',width=2)
            for (px,py),valid in zip(item['xy_normalized'],item['location_valid']):
                if valid:
                    x=px*width;y=py*height;draw.ellipse((x-2,y-2,x+2,y+2),fill='red')
        path=root/f'toy_{index}.png';image.save(path);files.append(path)
        frames.append(dict(schema='separated_pose_location_frame_v1',frame_key=f'toy_{index}',image_file=path.name,image_sha256=sha(path),
                           width=width,height=height,bbox_coverage='all_noncrowd_person_instances',crowd_present=False,
                           expected_person_annotation_ids=[i['annotation_id'] for i in instances],instances=instances))
    # 兼容夹具允许train/val同图；没有泛化评估含义。
    for split in ('train','val'):
        path=root/(split+'.jsonl');path.write_text(''.join(json.dumps(f,ensure_ascii=False)+'\n' for f in frames),encoding='utf-8');files.append(path)
    config=root/'toy_dataset.json';config.write_text(json.dumps(dict(schema='pose_location_jsonl_cpu_toy_v1',usage_scope='cpu_toy_compatibility_only',train='train.jsonl',val='val.jsonl'),indent=2),encoding='utf-8');files.append(config)
    return config,{str(p.relative_to(root)):sha(p) for p in files}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,default=Path('D:/projects/FutsalMOT_Research/weights/yolo11n-pose.pt'))
    args=parser.parse_args();out=args.output.resolve()
    if out.exists():raise FileExistsError(out)
    if sha(args.checkpoint)!=SOURCE_SHA:raise ValueError('源官方checkpoint SHA不符')
    out.mkdir(parents=True)
    os.environ.update(CUDA_VISIBLE_DEVICES='-1',YOLO_AUTOINSTALL='false',YOLO_OFFLINE='true',YOLO_CONFIG_DIR=str(out/'settings'),WANDB_MODE='disabled',COMET_MODE='DISABLED')
    attempts=[]
    def deny_network(sock,address):attempts.append(str(address));raise RuntimeError('禁止网络连接')
    socket.socket.connect=deny_network
    sys.path.insert(0,str(ROOT/'src'))
    import torch
    import ultralytics
    from ultralytics.utils import callbacks
    from grf_ue_bridge.pose_location_dataset_v2 import location_pose_trainer_class,VERSION
    from grf_ue_bridge.pose_xy_head_transfer import XY_ROWS
    callbacks.add_integration_callbacks=lambda instance:None
    torch.set_num_threads(1)
    config,fixture_sha=make_toy_jsonl(out/'fixture')
    state=dict(schema='pose_location_jsonl_trainer_cpu_compatibility',version=VERSION,python=sys.version,torch=torch.__version__,
               ultralytics=ultralytics.__version__,device='cpu',gpu_used=False,research_training=False,real_test_read=False,
               source_checkpoint=str(args.checkpoint.resolve()),source_sha256=SOURCE_SHA,fixture_sha256=fixture_sha,
               network_attempts=attempts,phases=[])
    Base=location_pose_trainer_class()

    class Instrumented(Base):
        def optimizer_step(self):
            weights=[b[-1][0].weight for b in self.model.model[-1].cv4]
            old=[p.detach().clone() for p in weights]
            xygrad=sum(p.grad[list(XY_ROWS)].abs().sum().item() for p in weights)
            visgrad=sum(p.grad[2::3].abs().sum().item() for p in weights)
            assert xygrad>0 and visgrad==0 and self.loss_items[2].item()==0
            super().optimizer_step()
            xychange=sum((p.detach()[list(XY_ROWS)]-q[list(XY_ROWS)]).abs().sum().item() for p,q in zip(weights,old))
            vischange=sum((p.detach()[2::3]-q[2::3]).abs().sum().item() for p,q in zip(weights,old))
            assert xychange>0 and vischange==0
            self.smoke['updates'].append(dict(xy_gradient_abs_sum=xygrad,visibility_gradient_abs_sum=visgrad,
                xy_parameter_change_abs_sum=xychange,visibility_parameter_change_abs_sum=vischange,loss_items=self.loss_items.tolist()))

    def before(trainer):
        head=trainer.model.model[-1];record=trainer.smoke
        assert trainer.device.type=='cpu' and len(trainer.train_loader)==1
        assert [b[-1][0].in_channels for b in head.cv4]==[51]*3
        assert [b[-1][0].out_channels for b in head.cv4]==[51]*3
        record.update(start_epoch=trainer.start_epoch,hidden_channels=[51]*3,physical_output_channels=[51]*3,
                      criterion=type(trainer.model.criterion).__name__,configuration=vars(trainer.args).copy())
        batch=next(iter(trainer.train_loader))
        assert tuple(batch['keypoints'].shape)==(3,17,2) and batch['location_valid'].dtype==torch.bool
        assert sorted(batch['location_valid'].sum(1).tolist())==[0,8,17]
        assert all(state=='unknown' for source,states in zip(batch['sources'],batch['visibility_state']) if source=='synthetic' for state in states)
        record['loader_targets']=[dict(annotation_id=i,source=s,valid_point_count=int(v.sum())) for i,s,v in zip(batch['target_ids'],batch['sources'],batch['location_valid'])]
        record['actual_batch_img_shape']=list(batch['img'].shape)
        record['invalid_xy_zero']=bool((batch['keypoints'][~batch['location_valid']]==0).all())
        def reject_visibility(*a,**k):raise AssertionError('禁止visibility BCE')
        trainer.model.criterion.bce_pose.register_forward_pre_hook(reject_visibility)
        if trainer.resume:
            ckpt=torch.load(trainer.args.resume,map_location='cpu',weights_only=False)
            actual=trainer.optimizer.state_dict();saved=ckpt['optimizer'];count=0
            assert actual['param_groups']==saved['param_groups'] and actual['state'].keys()==saved['state'].keys()
            for key,values in saved['state'].items():
                for name,value in values.items():
                    observed=actual['state'][key][name]
                    if isinstance(value,torch.Tensor):assert torch.equal(observed,value.to(observed.dtype));count+=1
                    else:assert observed==value
            assert count>0 and trainer.ema.updates==ckpt['updates']
            record.update(optimizer_restored_tensor_count=count,ema_updates_restored=trainer.ema.updates)

    def saved(trainer):
        import shutil
        path=out/(trainer.smoke['phase']+'_unstripped.pt')
        if path.exists():raise FileExistsError(path)
        shutil.copyfile(trainer.last,path)
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        assert checkpoint['optimizer']
        trainer.smoke.update(checkpoint=str(path),checkpoint_sha256=sha(path),saved_epoch=checkpoint['epoch'],saved_ema_updates=checkpoint['updates'])

    common=dict(model=str(args.checkpoint.resolve()),data=str(config),project=str(out),name='trainer_run',exist_ok=False,
        epochs=2,batch=2,nbs=2,imgsz=128,device='cpu',workers=0,amp=False,plots=False,optimizer='SGD',lr0=.001,lrf=1.,momentum=.9,
        weight_decay=0.,warmup_epochs=0.,seed=817,deterministic=True,save=True,save_period=-1,val=True,cache=False,close_mosaic=0,
        mosaic=0.,mixup=0.,cutmix=0.,copy_paste=0.,degrees=0.,translate=0.,scale=0.,shear=0.,perspective=0.,fliplr=0.,flipud=0.,
        hsv_h=0.,hsv_s=0.,hsv_v=0.,erasing=0.,multi_scale=0.,compile=False,verbose=False,rect=False,cls_pw=0.)
    try:
        for phase,resume in (('initial_one_update',None),('resume_one_update',out/'initial_one_update_unstripped.pt')):
            overrides=dict(common)
            if resume:overrides.update(model=str(resume),resume=str(resume),exist_ok=True)
            trainer=Instrumented(overrides=overrides)
            trainer.smoke={'phase':phase,'updates':[]};state['phases'].append(trainer.smoke)
            trainer.add_callback('on_pretrain_routine_end',before)
            trainer.add_callback('on_model_save',saved)
            trainer.add_callback('on_fit_epoch_end',lambda t:setattr(t,'stop',True))
            trainer.train()
            assert len(trainer.smoke['updates'])==1 and trainer.smoke['start_epoch']==(1 if resume else 0)
            trainer.smoke['final_eval_scope']=trainer.location_final_eval_scope
            trainer.smoke['returned_normally']=True
        assert not attempts and sha(args.checkpoint)==SOURCE_SHA
        state.update(passed=True,resume_scope='官方resume加载保存FP16优化器/EMA后完成下一更新；非不中断轨迹等价',
                     evaluation_scope='mask正确的loss_only；没有AP/跟踪/研究结论')
    except Exception:
        state.update(passed=False,exception=traceback.format_exc());raise
    finally:
        state['implementation_sha256']={str(p.relative_to(ROOT)):sha(p) for p in
            (Path(__file__),ROOT/'src/grf_ue_bridge/pose_location_dataset_v2.py',ROOT/'src/grf_ue_bridge/pose_location_supervision_v2.py',ROOT/'src/grf_ue_bridge/pose_xy_head_transfer.py')}
        (out/'receipt.json').write_text(json.dumps(state,ensure_ascii=False,indent=2,default=str),encoding='utf-8')


if __name__=='__main__':main()
