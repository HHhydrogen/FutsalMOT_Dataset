"""仅用自建toy夹具验证位置头Trainer/优化器/续载；不是数据集实验。"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import socket
import sys
import traceback

SOURCE_SHA='869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0'
TOY_POINTS=((.50,.22),(.46,.20),(.54,.20),(.42,.22),(.58,.22),(.37,.36),(.63,.36),(.30,.49),(.70,.49),(.25,.61),(.75,.61),(.43,.59),(.57,.59),(.40,.74),(.60,.74),(.37,.91),(.63,.91))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_toy_label(text):
    """强制纯xy39列和全部画内；此处没有COCO可见性值。"""
    row=[float(v) for v in text.split()]
    if len(row)!=39 or row[0]!=0 or any(not 0<v<1 for v in row[1:]):
        raise ValueError('toy标签必须为单类39列画内位置')
    return row


def make_fixture(root):
    """生成四张明确标记TOY的128像素图，绝不读取媒体目录。"""
    from PIL import Image,ImageDraw
    root=Path(root)
    if root.exists():raise FileExistsError(root)
    files=[]
    for split in ('train','val'):
        for index in range(2):
            image=Image.new('RGB',(128,128),(30+index*15,45,70))
            draw=ImageDraw.Draw(image)
            draw.text((3,3),'TOY CPU FIXTURE',fill='white')
            pts=[(int(x*128),int(y*128)) for x,y in TOY_POINTS]
            for a,b in ((5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)):
                draw.line((pts[a],pts[b]),fill=(240,180,40),width=5)
            for p in pts:draw.ellipse((p[0]-2,p[1]-2,p[0]+2,p[1]+2),fill=(255,80,60))
            im=root/'images'/split/f'toy_{index}.png'
            label=root/'labels'/split/f'toy_{index}.txt'
            im.parent.mkdir(parents=True,exist_ok=True);label.parent.mkdir(parents=True,exist_ok=True)
            image.save(im)
            text='0 0.5 0.56 0.6 0.8 '+' '.join(str(n) for p in TOY_POINTS for n in p)+'\n'
            validate_toy_label(text);label.write_text(text,encoding='utf-8')
            files.extend([im,label])
    config=root/'toy.yaml'
    config.write_text('path: '+root.as_posix()+'\ntrain: images/train\nval: images/val\nnames: {0: player}\nkpt_shape: [17, 2]\nflip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]\n',encoding='utf-8')
    files.append(config)
    return config,{str(p.relative_to(root)):sha(p) for p in files}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--checkpoint',type=Path,default=Path('D:/projects/FutsalMOT_Research/weights/yolo11n-pose.pt'))
    args=parser.parse_args()
    out=args.output.resolve()
    if out.exists():raise FileExistsError('要求全新输出目录：'+str(out))
    if sha(args.checkpoint)!=SOURCE_SHA:raise ValueError('非已验证官方checkpoint')
    out.mkdir(parents=True)
    os.environ.update(CUDA_VISIBLE_DEVICES='-1',YOLO_AUTOINSTALL='false',YOLO_OFFLINE='true',YOLO_CONFIG_DIR=str(out/'settings'),WANDB_MODE='disabled',COMET_MODE='DISABLED')
    network_attempts=[]
    def deny_connect(sock,address):
        network_attempts.append(str(address))
        raise RuntimeError('CPU smoke禁止网络连接：'+str(address))
    socket.socket.connect=deny_connect
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    import torch
    import ultralytics
    from ultralytics.utils import callbacks
    import ultralytics.data.utils as data_utils
    from grf_ue_bridge.pose_xy_head_transfer import pose_xy_trainer_class,XY_ROWS,VERSION
    torch.set_num_threads(1)
    # 禁用外围在线集成和无图场景字体下载；不改变Trainer/损失/优化器计算。
    callbacks.add_integration_callbacks=lambda instance:None
    data_utils.check_font=lambda *a,**k:None
    config,fixture_sha=make_fixture(out/'toy_fixture')
    state={'schema':'pose_xy_trainer_cpu_compatibility','adapter_version':VERSION,'torch':torch.__version__,
           'ultralytics':ultralytics.__version__,'device':'cpu','source_checkpoint':str(args.checkpoint.resolve()),
           'source_sha256':SOURCE_SHA,'fixture_sha256':fixture_sha,'toy_only':True,'research_training':False,
           'real_test_read':False,'gpu_used':False,'phases':[],'network_attempts':network_attempts}
    Trainer=pose_xy_trainer_class()

    class InstrumentedTrainer(Trainer):
        def optimizer_step(self):
            head=self.model.model[-1]
            params=[b[-1][0].weight for b in head.cv4]
            before=[p.detach().clone() for p in params]
            grad_xy=sum(p.grad[list(XY_ROWS)].abs().sum().item() for p in params)
            grad_vis=sum(p.grad[2::3].abs().sum().item() for p in params)
            assert grad_xy>0 and grad_vis==0
            assert float(self.loss_items[2])==0
            super().optimizer_step()
            delta_xy=sum((p.detach()[list(XY_ROWS)]-b[list(XY_ROWS)]).abs().sum().item() for p,b in zip(params,before))
            delta_vis=sum((p.detach()[2::3]-b[2::3]).abs().sum().item() for p,b in zip(params,before))
            assert delta_xy>0 and delta_vis==0
            assert all(torch.isfinite(p).all() for p in self.model.parameters())
            self.smoke['optimizer_updates'].append({'grad_xy_abs_sum':grad_xy,'grad_visibility_abs_sum':grad_vis,
                'xy_parameter_delta_abs_sum':delta_xy,'visibility_parameter_delta_abs_sum':delta_vis,
                'loss_items':self.loss_items.detach().tolist()})

    def setup(trainer):
        record=trainer.smoke
        assert trainer.device.type=='cpu' and not trainer.amp and trainer.args.workers==0 and not trainer.args.plots
        head=trainer.model.model[-1]
        assert tuple(head.kpt_shape)==(17,2)
        record['hidden_channels']=[b[-1][0].in_channels for b in head.cv4]
        record['physical_output_channels']=[b[-1][0].out_channels for b in head.cv4]
        assert record['hidden_channels']==[51]*3 and record['physical_output_channels']==[51]*3
        assert len(trainer.train_loader)==1
        record['start_epoch']=trainer.start_epoch
        record['optimizer_class']=type(trainer.optimizer).__name__
        record['optimizer_state_count_before']=len(trainer.optimizer.state)
        trainer.model.criterion=trainer.model.init_criterion()
        def reject_visibility(*a,**k):
            raise AssertionError('不允许调用visibility BCE')
        trainer.model.criterion.bce_pose.register_forward_pre_hook(reject_visibility)
        record['visibility_bce_fail_sentinel_installed']=True
        if trainer.resume:
            ckpt=torch.load(trainer.args.resume,map_location='cpu',weights_only=False)
            current=trainer.optimizer.state_dict()
            saved=ckpt['optimizer']
            assert current['param_groups']==saved['param_groups']
            assert current['state'].keys()==saved['state'].keys()
            compared=0
            for key,values in saved['state'].items():
                for name,value in values.items():
                    actual=current['state'][key][name]
                    if isinstance(value,torch.Tensor):
                        assert torch.equal(actual,value.to(actual.dtype));compared+=1
                    else:assert actual==value
            assert compared>0
            assert trainer.ema.updates==ckpt['updates']
            record['optimizer_restored_tensor_count']=compared
            record['optimizer_restored_equal_to_saved_state_after_dtype_cast']=True
            record['ema_updates_restored']=trainer.ema.updates

    def preserve_checkpoint(trainer):
        import shutil
        target=out/(trainer.smoke['phase']+'_unstripped.pt')
        if target.exists():raise FileExistsError(target)
        shutil.copyfile(trainer.last,target)
        ckpt=torch.load(target,map_location='cpu',weights_only=False)
        assert ckpt['optimizer'] and ckpt['epoch']==trainer.epoch
        trainer.smoke.update(checkpoint=str(target),checkpoint_sha256=sha(target),saved_epoch=ckpt['epoch'],
            saved_optimizer_state_count=len(ckpt['optimizer']['state']),saved_ema_updates=ckpt['updates'])

    common=dict(model=str(args.checkpoint.resolve()),data=str(config),project=str(out),name='trainer_run',
                exist_ok=False,epochs=2,batch=2,nbs=2,imgsz=128,device='cpu',workers=0,amp=False,plots=False,
                optimizer='SGD',lr0=.001,lrf=1.,momentum=.9,weight_decay=0.,warmup_epochs=0.,seed=714,
                deterministic=True,save=True,save_period=-1,val=True,cache=False,close_mosaic=0,
                mosaic=0.,mixup=0.,cutmix=0.,copy_paste=0.,degrees=0.,translate=0.,scale=0.,shear=0.,
                perspective=0.,fliplr=0.,flipud=0.,hsv_h=0.,hsv_s=0.,hsv_v=0.,erasing=0.,compile=False,verbose=False)
    try:
        for phase,resume in (('initial_one_update',None),('resume_one_update',out/'initial_one_update_unstripped.pt')):
            overrides=dict(common)
            if resume:overrides.update(model=str(resume),resume=str(resume),exist_ok=True)
            trainer=InstrumentedTrainer(overrides=overrides)
            trainer.smoke={'phase':phase,'optimizer_updates':[]}
            state['phases'].append(trainer.smoke)
            trainer.add_callback('on_pretrain_routine_end',setup)
            trainer.add_callback('on_model_save',preserve_checkpoint)
            # 声明2轮，仅完成当前一轮；保留epoch0未剥离checkpoint供真实resume API。
            trainer.add_callback('on_fit_epoch_end',lambda t:setattr(t,'stop',True))
            trainer.train()
            assert len(trainer.smoke['optimizer_updates'])==1
            assert trainer.smoke['start_epoch']==(1 if resume else 0)
            trainer.smoke['trainer_returned_normally']=True
        assert sha(args.checkpoint)==SOURCE_SHA
        state['passed']=True
        state['resume_scope']='官方resume API加载epoch0保存状态并完成epoch1一个更新；不是不中断训练数值等价证明'
    except Exception:
        state['passed']=False;state['exception']=traceback.format_exc()
        raise
    finally:
        (out/'receipt.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
