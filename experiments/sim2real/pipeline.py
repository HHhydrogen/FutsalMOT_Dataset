# -*- coding: utf-8 -*-
"""合成到真实足球的可复现 pilot：数据准备、等步数训练、检测和固定追踪器评估。"""
from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import math
import os
import random
import shutil
import sys
import uuid
from pathlib import Path

ROOT = Path('D:/projects/FutsalMOT_Research')
DEFAULT_INDEX = Path('D:/projects/FutsalMOT_UEDataset/Saved/PaperPilot/20261021_4k/index.json')
DEFAULT_PROTOCOL = Path(__file__).with_name('paper_pilot_protocol.json')
REAL_SEQUENCES = {'train': [('train', 'SNMOT-060'), ('train', 'SNMOT-097')],
                  'val': [('train', 'SNMOT-151')],
                  'test': [('test', 'SNMOT-116'), ('test', 'SNMOT-187')]}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, data):
    path=Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require_identical(path, expected):
    """存在的目标或缓存必须拥有预期完整 SHA256，绝不覆盖冲突内容。"""
    if digest(path)!=expected:raise ValueError(f'目标已有不同内容，拒绝覆盖：{path}')


def _atomic_copy_identical(source, target, expected):
    """先在目标目录完整复制并校验，再原子发布；临时文件不会作为图像暴露。"""
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        _require_identical(target,expected)
        return
    temporary=target.with_name('.'+target.name+'.tmp_'+uuid.uuid4().hex)
    try:
        with source.open('rb') as reader,temporary.open('xb') as writer:
            shutil.copyfileobj(reader,writer,length=1024*1024)
            writer.flush();os.fsync(writer.fileno())
        _require_identical(temporary,expected)
        try:
            # 同目录硬链接是原子的“不覆盖”发布；并发同内容写入也不会互相替换。
            os.link(temporary,target)
        except FileExistsError:
            _require_identical(target,expected)
        except OSError:
            # 文件系统完全不支持硬链接时，以独占锁串行执行同目录原子重命名。
            lock=target.with_name('.'+target.name+'.publish.lock')
            try:
                handle=os.open(str(lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY)
            except FileExistsError as exc:
                raise RuntimeError(f'图像发布锁已存在，需确认另一写入进程：{lock}') from exc
            os.close(handle)
            try:
                if target.exists():_require_identical(target,expected)
                else:temporary.rename(target)
            finally:lock.unlink()
    finally:
        if temporary.exists():temporary.unlink()


def link_or_copy(source, target):
    """同卷硬链接；跨卷经完整 SHA256 图像缓存只复制一次，再共享给派生格式。"""
    source, target = Path(source).resolve(), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if digest(source) != digest(target):
            raise ValueError(f'目标已有不同内容，拒绝覆盖：{target}')
        return 'existing'
    try:
        os.link(source, target)
        return 'hardlink'
    except FileExistsError:
        _require_identical(target,digest(source))
        return 'existing'
    except OSError:
        fingerprint=digest(source)
        cached=ROOT/'shared/image_cache'/fingerprint[:2]/fingerprint
        _atomic_copy_identical(source,cached,fingerprint)
        try:
            os.link(cached,target)
            return 'cache_hardlink'
        except FileExistsError:
            _require_identical(target,fingerprint)
            return 'existing'
        except OSError:
            # 目标另卷或其文件系统不支持链接时仍保留原始字节，并明确记录回退。
            _atomic_copy_identical(cached,target,fingerprint)
            print(f'IMAGE_COPY_FALLBACK {cached} -> {target}',flush=True)
            return 'copy_fallback'


def bbox_to_yolo(box, width, height):
    x, y, w, h = map(float, box)
    if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
        raise ValueError('bbox 非有限或面积无效')
    x0,y0,x1,y1=max(0,x),max(0,y),min(width,x+w),min(height,y+h)
    if x1<=x0 or y1<=y0:
        return None
    return f'0 {(x0+x1)/(2*width):.8f} {(y0+y1)/(2*height):.8f} {(x1-x0)/width:.8f} {(y1-y0)/height:.8f}'


def write_lists(output, groups):
    for name, paths in groups.items():
        # 保留images目录联接的词法路径，否则YOLO会去正式RGB目录寻找派生labels。
        (output/f'{name}.txt').write_text(''.join(Path(x).absolute().as_posix()+'\n' for x in paths), encoding='utf-8')
    # JSON 是合法 YAML 子集，无需在 P1 安装额外库。
    write_json(output/'dataset.yaml', {'path':output.resolve().as_posix(),'train':'train.txt','val':'val.txt','test':'test.txt','names':{0:'player'}})


def prepare_real(args):
    source=ROOT/'public/soccernet_tracking'
    output=ROOT/'datasets/real_player_pilot_v1'
    output.mkdir(parents=True,exist_ok=True)
    groups={k:[] for k in REAL_SEQUENCES}
    games={}; records=[]
    for split,sequences in REAL_SEQUENCES.items():
        for official_split,name in sequences:
            sequence=source/official_split/name
            game=configparser.ConfigParser();game.read(sequence/'gameinfo.ini',encoding='utf-8')
            info=configparser.ConfigParser();info.read(sequence/'seqinfo.ini',encoding='utf-8')
            width,height=info.getint('Sequence','imWidth'),info.getint('Sequence','imHeight')
            game_id=game.get('Sequence','gameID')
            if game_id in games and games[game_id]!=split:
                raise ValueError('同一真实比赛跨 split，禁止泄漏')
            games[game_id]=split
            players={int(k.split('_')[1]) for k,v in game['Sequence'].items() if k.startswith('trackletid_') and v.lower().startswith(('player ','goalkeeper'))}
            frame_labels={};mot_rows=[];ignored=0
            images=sorted((sequence/'img1').glob('*.jpg'))
            frame_ids={int(x.stem) for x in images}
            if frame_ids!=set(range(1,len(images)+1)):
                raise ValueError('真实子序列帧不连续')
            for line in (sequence/'gt/gt.txt').read_text().splitlines():
                row=line.split(',');frame,identity=int(float(row[0])),int(float(row[1]))
                if frame not in frame_ids:continue
                if identity not in players:
                    ignored+=1;continue
                label=bbox_to_yolo(row[2:6],width,height)
                if label is not None:frame_labels.setdefault(frame,[]).append(label)
                mot_rows.append(','.join(row[:6])+',1,1,1')
            for image in images:
                target=output/'images'/split/(name+'_'+image.name)
                link_or_copy(image,target)
                label=output/'labels'/split/(target.stem+'.txt');label.parent.mkdir(parents=True,exist_ok=True)
                label.write_text('\n'.join(frame_labels.get(int(image.stem),[]))+'\n',encoding='utf-8')
                groups[split].append(target)
            mot=output/'mot_gt'/name
            (mot/'gt').mkdir(parents=True,exist_ok=True)
            (mot/'gt/gt.txt').write_text('\n'.join(mot_rows)+'\n',encoding='utf-8')
            info['Sequence']['seqLength']=str(len(images))
            with (mot/'seqinfo.ini').open('w',encoding='utf-8') as stream:info.write(stream)
            records.append({'split':split,'official_split':official_split,'sequence':name,'game_id':game_id,'frames':len(images),'player_track_ids':sorted(players),'included_boxes':len(mot_rows),'excluded_referee_ball_other_boxes':ignored,'source_gt_sha256':digest(sequence/'gt/gt.txt'),'source_gameinfo_sha256':digest(sequence/'gameinfo.ini')})
    write_lists(output,groups)
    write_json(output/'provenance.json',{'protocol':'soccernet_player_only_first150_v1','warning':'派生 player/goalkeeper 子集，不是官方全类别 SoccerNet benchmark；前150帧结果仅为可行性pilot。','sequences':records,'game_split':games,'counts':{k:len(v) for k,v in groups.items()}})
    print('REAL_DATASET_READY',output,{k:len(v) for k,v in groups.items()})


def verify_episode(entry):
    episode=Path(entry['episode'])
    audit=read_json(episode/'audit/soak_audit_report.json')
    if audit.get('passed') is not True or audit.get('exit_code')!=0 or audit.get('errors'):
        raise ValueError(f'完整审计未通过：{episode}')
    validation=audit.get('validation') or {}
    if validation.get('level')!='full' or validation.get('passed') is not True or validation.get('exit_code')!=0:
        raise ValueError(f'没有 full annotation audit 证据：{episode}')
    render=read_json(episode/'render_summary.json')
    if render.get('status')!='success':raise ValueError('MRQ 未成功')
    for kind in ('appearance','environment'):
        report=read_json(episode/f'provenance/{kind}_application.json')
        if report.get('state')!='restored' or report.get('errors') or report.get('mrq_pending') is True:raise ValueError(f'{kind} 未恢复')
    ready=read_json(episode/'pilot_ready.json')
    if ready.get('status')!='audit_passed' or ready.get('appearance_restored') is not True or ready.get('environment_restored') is not True:
        raise ValueError('pilot_ready 未通过')
    if ready.get('root_seed')!=entry['root_seed'] or ready.get('task_id')!=entry.get('task_id'):
        raise ValueError('pilot_ready 与 entry 不一致')
    meta=read_json(episode/'meta.json')
    seed=meta.get('randomness',{}).get('root_seed',meta.get('seed'))
    if seed!=entry['root_seed']:raise ValueError('seed 缺失或与批次索引不一致')
    return episode


def prepare_synthetic(args):
    if getattr(args, 'canonical_root', None):
        return prepare_canonical_synthetic(args)
    entries=read_json(args.index)
    validate_entries(entries)
    output=ROOT/'datasets/synthetic_pilot_v1'
    output.mkdir(parents=True,exist_ok=True)
    groups={'train':[],'val':[],'test':[]};seeds={};records=[]
    for entry in entries:
        if entry['split']=='gate':continue
        split={'validation':'val','synthetic_test':'test'}.get(entry['split'],entry['split'])
        seed=entry['root_seed']
        if seed in seeds and seeds[seed]!=split:raise ValueError('同一 root_seed 跨 split')
        seeds[seed]=split
        episode=verify_episode(entry)
        cameras=sorted(episode.glob('CineCam_*'))
        if len(cameras)!=entry['camera_count']:raise ValueError('相机数量不匹配')
        frame_count=0;boxes=0
        for camera in cameras:
            calibration=read_json(camera/'camera.json')
            width=calibration.get('image_width') or calibration['intrinsics']['width']
            height=calibration.get('image_height') or calibration['intrinsics']['height']
            annotations=[json.loads(line) for line in (camera/'annotations.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
            if {x['frame_index'] for x in annotations}!=set(range(1,entry['frames_per_camera']+1)):raise ValueError('标注帧不完整')
            for frame in annotations:
                labels=[]
                for obj in frame['objects']:
                    if obj.get('class')!='player':continue
                    box=obj.get('bbox_xywh')
                    if box is None:continue
                    if obj.get('bbox_source')!='instance_mask':raise ValueError('训练必须使用已解码 Instance-ID 可见框')
                    label=bbox_to_yolo(box,width,height)
                    if label:labels.append(label)
                source=camera/'img1'/f"{frame['frame_index']:06d}.png"
                target=output/'images'/split/f"{episode.name}_{camera.name}_{source.name}"
                link_or_copy(source,target)
                labelpath=output/'labels'/split/(target.stem+'.txt');labelpath.parent.mkdir(parents=True,exist_ok=True)
                labelpath.write_text('\n'.join(labels)+'\n',encoding='utf-8')
                groups[split].append(target);frame_count+=1;boxes+=len(labels)
        records.append({'episode':str(episode),'root_seed':seed,'split':split,'preset':entry['preset'],'source_frame_range':entry.get('source_frame_range'),'images':frame_count,'boxes':boxes,'audit_sha256':digest(episode/'audit/soak_audit_report.json'),'appearance_sha256':digest(episode/'provenance/appearance.json'),'environment_sha256':digest(episode/'provenance/environment.json')})
    write_lists(output,groups)
    write_json(output/'provenance.json',{'protocol':'synthetic_episode_seed_grouped_v1','episodes':records,'root_seed_split':seeds,'counts':{k:len(v) for k,v in groups.items()}})
    print('SYNTHETIC_DATASET_READY',output,{k:len(v) for k,v in groups.items()})


def safe_relative(root, relative):
    """发布引用必须留在所属目录，避免外部路径混入研究数据。"""
    path=(Path(root)/relative).resolve()
    path.relative_to(Path(root).resolve())
    return path


def canonical_clips(root, verification_level='full'):
    """只读验证完整发布组，所有派生器共同使用同一 seed 准入。"""
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
    from grf_ue_bridge.seed_dataset import DATASET_SCHEMA, GROUP_SCHEMA, verify_clip
    root=Path(root).resolve()
    index=read_json(root/'dataset.json')
    if index.get('schema')!=DATASET_SCHEMA or index.get('version')!=1:
        raise ValueError('canonical dataset schema 不兼容')
    seeds={};seen_groups=set();clips=[]
    for reference in index['groups']:
        group_path=safe_relative(root,reference['path'])
        if group_path in seen_groups:raise ValueError('重复 canonical group')
        seen_groups.add(group_path)
        group=read_json(group_path)
        if group.get('schema')!=GROUP_SCHEMA or group.get('version')!=1:
            raise ValueError('canonical group schema 不兼容')
        seed,split=group['root_seed'],group['split']
        if type(seed) is not int or seed<0 or split not in {'train','val','test'}:
            raise ValueError('canonical seed/split 无效')
        if reference['root_seed']!=seed or reference['split']!=split:
            raise ValueError('canonical dataset/group seed 或 split 不一致')
        if seed in seeds:raise ValueError('同一 root_seed 出现多个 group，禁止泄漏')
        seeds[seed]=split
        # 源轨迹使用大写 BALL 且 role=null；发布格式不重复 class，须按实体ID排除球。
        players={entity['entity_id'] for entity in group['entities'] if entity['entity_id'].casefold()!='ball' and entity.get('role')!='ball'}
        if not players:raise ValueError('canonical group 没有球员实体')
        seen_clips=set()
        for clip_ref in group['clips']:
            clip_path=safe_relative(group_path.parent,clip_ref['path'])
            if clip_path.name!='clip.json' or clip_path in seen_clips:raise ValueError('重复或不合法 clip 引用')
            seen_clips.add(clip_path)
            clip=verify_clip(clip_path.parent,verification_level=verification_level)
            clips.append((group_path,seed,split,players,clip_path,clip))
    return clips,seeds


def canonical_view(root, camera, camera_id, clip):
    """解析共享相机与二维观测，不读取任何 legacy 路径。"""
    view=read_json(camera/'view.json')
    if view['camera_id']!=camera_id:raise ValueError('camera id 与 view 不一致')
    calibration_path=safe_relative(root,view['calibration_ref'])
    calibration=read_json(calibration_path)
    calibration_hash=hashlib.sha256(json.dumps(calibration,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()
    if calibration_path.name!=camera_id+'_'+calibration_hash+'.json':raise ValueError('共享 calibration 指纹不匹配')
    rows=[json.loads(line) for line in (camera/'annotations.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(rows)!=clip['frame_count'] or {row['frame_index'] for row in rows}!=set(range(1,clip['frame_count']+1)):
        raise ValueError('canonical 标注帧不完整或重复')
    return calibration,rows


def canonical_player_box(obj, players, width, height):
    """返回裁剪后的可见球员 xywh；不可见实体不生成伪造框。"""
    if obj['entity_id'] not in players or obj.get('in_frame') is False or obj.get('bbox_xyxy') is None:return None
    if obj.get('bbox_source')!='instance_mask':raise ValueError('训练必须使用 Instance-ID 可见框')
    x0,y0,x1,y1=map(float,obj['bbox_xyxy'])
    if not all(math.isfinite(x) for x in (x0,y0,x1,y1)) or x1<=x0 or y1<=y0:raise ValueError('无效 canonical bbox')
    x0,y0,x1,y1=max(0,x0),max(0,y0),min(width,x1),min(height,y1)
    if x1<=x0 or y1<=y0:return None
    return [x0,y0,x1-x0,y1-y0]


def verify_protocol_coverage(records, protocol, required_splits):
    """按事先选定的天气/连续片段核验覆盖；足够多邻帧不能代替缺失天气。"""
    if protocol.get('schema')!='sim2real_selection_v1' or not protocol.get('groups'):
        raise ValueError('实验选择协议 schema 无效')
    expected={}
    for entry in protocol['groups']:
        seed=entry.get('root_seed')
        if type(seed) is not int or seed<0 or seed in expected:raise ValueError('协议 seed 无效或重复')
        if entry.get('split') not in {'train','val','test'}:raise ValueError('协议 split 无效')
        count=entry.get('frame_count');window=entry.get('source_frame_range')
        if type(count) is not int or count<=0 or not isinstance(window,list) or len(window)!=2 or window[1]-window[0]!=count:
            raise ValueError('协议连续窗口无效')
        cameras=entry.get('cameras')
        if not isinstance(cameras,list) or not cameras or len(set(cameras))!=len(cameras):raise ValueError('协议相机无效')
        if not isinstance(entry.get('resolution'),list) or len(entry['resolution'])!=2:raise ValueError('协议分辨率无效')
        expected[seed]=entry
    seen={}
    for record in records:
        seed=record.get('root_seed')
        if seed in seen:raise ValueError(f'协议同 seed 出现重复 clip：{seed}')
        seen[seed]=record
        if seed not in expected:raise ValueError(f'canonical 来源含未声明 seed：{seed}')
        entry=expected[seed]
        for key in ('preset','split','frame_count','source_frame_range'):
            if record.get(key)!=entry[key]:raise ValueError(f'协议 {seed} 的 {key} 不一致')
        if sorted(record.get('cameras',[]))!=sorted(entry['cameras']):raise ValueError(f'协议 {seed} 相机不完整')
        if record.get('camera_resolutions')!={camera:entry['resolution'] for camera in entry['cameras']}:
            raise ValueError(f'协议 {seed} 各相机分辨率不一致')
        if record.get('images')!=entry['frame_count']*len(entry['cameras']):raise ValueError(f'协议 {seed} 图像数不完整')
    required={seed for seed,entry in expected.items() if entry['split'] in required_splits}
    missing=sorted(required-set(seen))
    if missing:raise ValueError(f'实验协议缺少已验收 seed：{missing}；不得用单天气图像数量提前冻结或启动 GPU')
    return {'groups':len(required),'root_seeds':sorted(required),
            'images':sum(seen[seed]['images'] for seed in required)}


def canonical_protocol_records(root, clips):
    """从已校验发布 clip 读取协议字段，同时核对每个相机的帧集合与标定。"""
    records=[]
    for group_path,seed,split,players,clip_path,clip in clips:
        resolutions={}
        for camera_id in clip['cameras']:
            calibration,_=canonical_view(root,clip_path.parent/'cameras'/camera_id,camera_id,clip)
            resolutions[camera_id]=[calibration['image_width'],calibration['image_height']]
        records.append({'root_seed':seed,'split':split,'preset':read_json(group_path).get('weather',{}).get('preset'),
                        'frame_count':clip['frame_count'],'source_frame_range':clip.get('source_frame_range'),
                        'cameras':clip['cameras'],'camera_resolutions':resolutions,
                        'images':clip['frame_count']*len(clip['cameras'])})
    return records


def verify_protocol(args):
    """只读完整准入核验；不创建派生文件、不加载训练框架、不访问 GPU。"""
    if not args.canonical_root:raise ValueError('协议核验需要 --canonical-root')
    root=Path(args.canonical_root).resolve()
    if not (root/'dataset.json').is_file():raise ValueError('canonical 尚未发布任何完整 seed 组')
    clips,_=canonical_clips(root,getattr(args,'verification_level','full'))
    result=verify_protocol_coverage(canonical_protocol_records(root,clips),read_json(args.protocol),{'train','val','test'})
    print('SERIAL_RENDER_DATA_READY',json.dumps(result),'protocol_sha256',digest(args.protocol))


def reference_image_directory(source, target):
    """Windows目录联接仅引用正式RGB，不跨卷复制4K数据；已有不同目标拒绝覆盖。"""
    import subprocess
    source,target=Path(source).resolve(),Path(target)
    if target.exists():
        if target.resolve()!=source:
            raise ValueError('图像引用目录指向不同来源')
        return
    if os.name!='nt':
        raise ValueError('--reference-images当前仅支持Windows目录联接')
    target.parent.mkdir(parents=True,exist_ok=True)
    quote=lambda value: "'"+str(value).replace("'","''")+"'"
    subprocess.run(['powershell','-NoProfile','-Command',
        "$ErrorActionPreference='Stop'; New-Item -ItemType Junction -Path "+quote(target)+' -Target '+quote(source)+' | Out-Null'],check=True)
    if target.resolve()!=source:
        raise ValueError('图像目录联接读回不一致')


def canonical_clip_id(group_path, clip_path):
    """逻辑片段ID来自manifest，不能把平铺seed目录名误作片段名。"""
    for reference in read_json(group_path)['clips']:
        if safe_relative(group_path.parent,reference['path'])==clip_path:
            return reference['clip_id']
    raise ValueError('clip未被所属group引用')


def prepare_canonical_synthetic(args):
    """只读 canonical 发布内容；删除 legacy staging 后仍可独立打包。"""
    root=Path(args.canonical_root).resolve()
    clips,seeds=canonical_clips(root,getattr(args,'verification_level','full'))
    # 所有发布组先验证通过，再创建研究派生文件。
    output=ROOT/'datasets/synthetic_pilot_v1';output.mkdir(parents=True,exist_ok=True)
    groups={'train':[],'val':[],'test':[]};records=[]
    for group_path,seed,split,players,clip_path,clip in clips:
        count=boxes=0;resolutions={}
        for camera_id in clip['cameras']:
            camera=clip_path.parent/'cameras'/camera_id
            clip_id=canonical_clip_id(group_path,clip_path)
            view_name=f'{group_path.parent.name}_{clip_id}_{camera_id}'
            use_reference=getattr(args,'reference_images',False)
            if use_reference:
                reference_image_directory(camera/'rgb',output/'images'/split/view_name)
            calibration,rows=canonical_view(root,camera,camera_id,clip)
            width,height=calibration['image_width'],calibration['image_height']
            resolutions[camera_id]=[width,height]
            for row in rows:
                labels=[]
                for obj in row['objects']:
                    box=canonical_player_box(obj,players,width,height)
                    if box is None:continue
                    label=bbox_to_yolo(box,width,height)
                    if label:labels.append(label)
                image=camera/'rgb'/f"{row['frame_index']:06d}.png"
                name=f'{view_name}_{image.name}'
                target=output/'images'/split/view_name/image.name if use_reference else output/'images'/split/name
                if not use_reference:
                    link_or_copy(image,target)
                label_path=(output/'labels'/split/view_name/(image.stem+'.txt')) if use_reference else output/'labels'/split/(target.stem+'.txt')
                label_path.parent.mkdir(parents=True,exist_ok=True)
                label_path.write_text('\n'.join(labels)+'\n',encoding='utf-8')
                groups[split].append(target);count+=1;boxes+=len(labels)
        records.append({'group':group_path.relative_to(root).as_posix(),'clip':clip_path.relative_to(root).as_posix(),'root_seed':seed,'split':split,'preset':read_json(group_path).get('weather',{}).get('preset'),'frame_count':clip['frame_count'],'cameras':clip['cameras'],'camera_resolutions':resolutions,'source_frame_range':clip.get('source_frame_range'),'images':count,'boxes':boxes,'checksums_sha256':digest(clip_path.with_name('checksums.json')),'admission':clip['admission']})
    write_lists(output,groups)
    write_json(output/'provenance.json',{'protocol':'canonical_seed_grouped_visible_player_v1','canonical_root':str(root),'dataset_index_sha256':digest(root/'dataset.json'),'clips':records,'root_seed_split':seeds,'counts':{k:len(v) for k,v in groups.items()}})
    print('SYNTHETIC_DATASET_READY',output,{k:len(v) for k,v in groups.items()})


def prepare_synthetic_mot(args):
    """逐 clip/相机生成 MOT 派生序列，保留 seed 分组和稳定实体 ID。"""
    if not args.canonical_root:raise ValueError('MOT 派生需要 --canonical-root')
    root=Path(args.canonical_root).resolve();clips,seeds=canonical_clips(root,getattr(args,'verification_level','full'))
    output=ROOT/'datasets/synthetic_mot_v1';output.mkdir(parents=True,exist_ok=True)
    seqmaps={'train':[],'val':[],'test':[]};records=[]
    for group_path,seed,split,players,clip_path,clip in clips:
        entity_map={name:index+1 for index,name in enumerate(sorted(players))}
        group=read_json(group_path)
        fps=group['timing']['playback_fps']
        if not isinstance(fps,(int,float)) or fps<=0:raise ValueError('MOT 需要有效 playback_fps')
        for camera_id in clip['cameras']:
            camera=clip_path.parent/'cameras'/camera_id
            calibration,rows=canonical_view(root,camera,camera_id,clip)
            width,height=calibration['image_width'],calibration['image_height']
            name=f'{group_path.parent.name}_{canonical_clip_id(group_path,clip_path)}_{camera_id}'
            sequence=output/split/name;(sequence/'gt').mkdir(parents=True,exist_ok=True)
            use_reference=getattr(args,'reference_images',False)
            if use_reference:
                reference_image_directory(camera/'rgb',sequence/'img1')
            mot=[]
            for row in sorted(rows,key=lambda x:x['frame_index']):
                frame=row['frame_index']
                if not use_reference:
                    link_or_copy(camera/'rgb'/f'{frame:06d}.png',sequence/'img1'/f'{frame:06d}.png')
                for obj in row['objects']:
                    box=canonical_player_box(obj,players,width,height)
                    if box is None:continue
                    visibility=obj.get('visibility')
                    if type(visibility) not in (float,int) or not math.isfinite(visibility) or not 0<=visibility<=1:visibility=-1
                    mot.append(','.join(map(str,[frame,entity_map[obj['entity_id']],*box,1,1,visibility])))
            (sequence/'gt/gt.txt').write_text('\n'.join(mot)+'\n',encoding='utf-8')
            info=configparser.ConfigParser()
            info['Sequence']={'name':name,'imDir':'img1','frameRate':str(fps),'seqLength':str(clip['frame_count']),'imWidth':str(width),'imHeight':str(height),'imExt':'.png'}
            with (sequence/'seqinfo.ini').open('w',encoding='utf-8') as stream:info.write(stream)
            seqmaps[split].append(name)
            records.append({'sequence':name,'split':split,'root_seed':seed,'entity_id_to_track_id':entity_map,'frames':clip['frame_count'],'boxes':len(mot),'clip':clip_path.relative_to(root).as_posix(),'checksums_sha256':digest(clip_path.with_name('checksums.json'))})
    for split,names in seqmaps.items():
        path=output/'seqmaps'/f'{split}.txt';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('name\n'+'\n'.join(names)+'\n',encoding='utf-8')
    write_json(output/'provenance.json',{'protocol':'canonical_player_visible_mot_v1','canonical_root':str(root),'dataset_index_sha256':digest(root/'dataset.json'),'root_seed_split':seeds,'sequences':records,'visibility_unknown_value':-1,'warning':'可见 Instance-ID 框；并非 amodal GT。MOT class=1 单类球员；未知 visibility=-1。'})
    print('SYNTHETIC_MOT_READY',output,{key:len(value) for key,value in seqmaps.items()})


def validate_entries(entries):
    """在创建研究输出前验证 seed 分组，禁止跨相机或相邻 clip 泄漏。"""
    seeds={};episodes=set()
    for entry in entries:
        if entry['split']=='gate':continue
        split={'validation':'val','synthetic_test':'test'}.get(entry['split'],entry['split'])
        if split not in {'train','val','test'}:raise ValueError('未知 synthetic split')
        seed=entry['root_seed']
        if type(seed) is not int or seed<0:raise ValueError('root_seed 必须为非负整数')
        if seed in seeds and seeds[seed]!=split:raise ValueError('同一 root_seed 跨 split')
        seeds[seed]=split
        episode=str(Path(entry['episode']).resolve())
        if episode in episodes:raise ValueError('索引含重复 episode')
        episodes.add(episode)
        if type(entry['frames_per_camera']) is not int or entry['frames_per_camera']<=0:raise ValueError('帧数必须为正整数')


def temporal_sample(images, count, seed):
    """按 clip/相机分配预算，在各连续序列的时间分层中点采样。"""
    if count<1 or len(set(images))!=len(images) or len(images)<count:raise ValueError('采样数量不足或存在重复图像')
    units={}
    for image in images:
        stem=Path(image).stem
        if stem.isdigit():
            unit,frame=Path(image).parent.name,stem
        else:
            unit,frame=stem.rsplit('_',1)
        if not frame.isdigit():raise ValueError('派生图像名缺少末尾帧编号')
        units.setdefault(unit,[]).append((int(frame),image))
    names=sorted(units);random.Random(seed).shuffle(names)
    quotas={name:0 for name in names}
    remaining=count
    while remaining:
        for name in names:
            if quotas[name]<len(units[name]):
                quotas[name]+=1;remaining-=1
                if not remaining:break
    selected=[];receipt={}
    for name in sorted(names):
        frames=sorted(units[name]);quota=quotas[name]
        indices=[int((index+0.5)*len(frames)/quota) for index in range(quota)]
        selected.extend(frames[index][1] for index in indices)
        receipt[name]={'available':len(frames),'selected':quota,'frame_indices':[frames[index][0] for index in indices]}
    return selected,receipt


def _real_inputs():
    """读取当前真实标注预算；返回原始字节供实验快照保存。"""
    real=ROOT/'datasets/real_player_pilot_v1'
    files={name:(real/name).read_bytes() for name in ('train.txt','val.txt','test.txt','provenance.json')}
    groups={split:files[split+'.txt'].decode('utf-8').splitlines() for split in ('train','val','test')}
    if not groups['train'] or len(set(groups['train']))!=len(groups['train']):raise ValueError('真实训练预算为空或重复')
    return real,files,groups


def verify_real_arm(require_completed=False):
    """只读验证 B 的原始四个输入及完成 receipt，不为匹配新代码改写旧 budget。"""
    real,_,groups=_real_inputs()
    directory=ROOT/'datasets/arm_real_only'
    config=read_json(directory/'dataset.yaml');budget=read_json(directory/'budget.json')
    expected={'path':directory.resolve().as_posix(),'train':'train.txt','val':'val.txt','test':'test.txt','names':{'0':'player'}}
    if config!=expected:raise ValueError('B dataset.yaml 与固定真实数据协议冲突')
    count=len(groups['train'])
    for split in ('train','val','test'):
        wanted=groups[split]*(2 if split=='train' else 1)
        if (directory/(split+'.txt')).read_text(encoding='utf-8').splitlines()!=wanted:
            raise ValueError('B 已冻结的 {} 清单与真实预算冲突'.format(split))
    expected_budget={'arm':'real_only','epoch_samples':2*count,'unique_real_annotation_budget':count,
                     'real_exposures_per_epoch':2*count,'synthetic_images_per_epoch':0,
                     'real_dataset_sha256':digest(real/'provenance.json'),'synthetic_dataset_sha256':None}
    if any(budget.get(key)!=value for key,value in expected_budget.items()):raise ValueError('B 已冻结 budget 与当前真实预算冲突')
    inputs={str((directory/name).resolve()):digest(directory/name) for name in ('dataset.yaml','budget.json','train.txt','val.txt')}
    receipts=[]
    for run in sorted((ROOT/'runs').glob('real_only_seed*_e*')):
        path=run/'training_receipt.json'
        if not path.is_file():raise ValueError('B 训练目录没有完成 receipt，拒绝修改或建立对照')
        receipt=read_json(path)
        recorded={str(Path(name).resolve()):value for name,value in receipt.get('input_sha256',{}).items()}
        if recorded!=inputs:raise ValueError('B 训练 receipt 的四个冻结输入 SHA256 冲突')
        if receipt.get('arm')!='real_only' or receipt.get('budget')!=budget or receipt.get('optimizer_steps',0)<=0 or receipt.get('optimizer_steps')!=receipt.get('expected_optimizer_steps'):
            raise ValueError('B 完成 receipt 或优化预算无效')
        receipts.append({'path':str(path.resolve()),'sha256':digest(path)})
    if require_completed and not receipts:raise ValueError('准备 C 前必须已有完成并核验的 B 训练 receipt')
    return receipts


def verify_frozen_arm(directory):
    """冻结文件仅校验并复用；损坏或旧版未冻结目录均拒绝自动修复覆盖。"""
    directory=Path(directory)
    manifest=read_json(directory/'freeze.json')
    if manifest.get('schema')!='sim2real_arm_freeze_v1' or not manifest.get('sha256'):
        raise ValueError('实验冻结清单 schema 无效')
    required={'dataset.yaml','budget.json','train.txt','val.txt','test.txt',
              'source_snapshot/synthetic_provenance.json','source_snapshot/synthetic_train.txt',
              'source_snapshot/selection_protocol.json'}
    if not required.issubset(manifest['sha256']):raise ValueError('实验冻结清单缺少输入或来源快照')
    for relative,expected in manifest['sha256'].items():
        path=safe_relative(directory,relative)
        if not path.is_file() or digest(path)!=expected:raise ValueError(f'冻结实验输入已改变：{relative}')
    budget=read_json(directory/'budget.json')
    if digest(directory/'source_snapshot/synthetic_provenance.json')!=budget.get('synthetic_dataset_sha256'):
        raise ValueError('冻结的 canonical provenance 与 budget 不一致')
    if digest(directory/'source_snapshot/selection_protocol.json')!=budget.get('selection_protocol_sha256'):
        raise ValueError('冻结的实验选择协议与 budget 不一致')
    coverage=verify_protocol_coverage(read_json(directory/'source_snapshot/synthetic_provenance.json').get('clips',[]),
                                      read_json(directory/'source_snapshot/selection_protocol.json'),{'train'})
    if budget.get('training_coverage')!=coverage:raise ValueError('冻结的天气覆盖记录与来源不一致')
    return manifest


def _publish_arm(directory,groups,budget,snapshots=None):
    """整目录构建后发布；已有目录绝不覆盖，避免中途失败破坏冻结输入。"""
    if directory.exists():raise ValueError(f'实验目录已存在，拒绝覆盖：{directory}')
    directory.parent.mkdir(parents=True,exist_ok=True)
    stage=directory.with_name('.staging_'+directory.name+'_'+uuid.uuid4().hex)
    stage.mkdir(exist_ok=False)
    write_lists(stage,groups)
    config=read_json(stage/'dataset.yaml');config['path']=directory.resolve().as_posix()
    write_json(stage/'dataset.yaml',config)
    write_json(stage/'budget.json',budget)
    if snapshots is not None:
        for name,data in snapshots.items():
            path=safe_relative(stage,'source_snapshot/'+name);path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(data)
        sha={path.relative_to(stage).as_posix():digest(path) for path in stage.rglob('*') if path.is_file()}
        write_json(stage/'freeze.json',{'schema':'sim2real_arm_freeze_v1','sha256':sha})
        verify_frozen_arm(stage)
    if directory.exists():raise ValueError(f'实验目录并发建立，拒绝覆盖：{directory}')
    stage.rename(directory)


def verify_arms(args):
    """GPU 窗口开始前只读核验完成 B、checkpoint 和已冻结 C；不补建任何文件。"""
    references=verify_real_arm(require_completed=True)
    for reference in references:
        path=Path(reference['path']);receipt=read_json(path)
        checkpoint=safe_relative(path.parent,receipt['evaluation_checkpoint'])
        if digest(checkpoint)!=receipt.get('last_checkpoint_sha256'):raise ValueError('B checkpoint 与完成 receipt 不一致')
    directory=ROOT/'datasets/arm_real_plus_synthetic'
    verify_frozen_arm(directory)
    if read_json(directory/'budget.json').get('selection_protocol_sha256')!=digest(args.protocol):
        raise ValueError('C 选择协议与当前计划不同')
    print('FROZEN_ARMS_VERIFIED','B checkpoints and C selection protocol')


def prepare_arms(args):
    # B 已经训练完毕；先校验四个原输入，整条路径没有任何 B 文件写操作。
    receipts=verify_real_arm(require_completed=True)
    real,real_files,real_groups=_real_inputs()
    synthetic=ROOT/'datasets/synthetic_pilot_v1'
    c_directory=ROOT/'datasets/arm_real_plus_synthetic'
    d_directory=ROOT/'datasets/arm_synthetic_only'
    if c_directory.exists():
        verify_frozen_arm(c_directory)
        frozen=read_json(c_directory/'budget.json')
        if frozen.get('sample_selection_seed')!=args.seed:raise ValueError('C 已冻结为不同采样 seed；请使用独立实验目录')
        if frozen.get('selection_protocol_sha256')!=digest(args.protocol):raise ValueError('C 已冻结为不同选择协议；请使用独立实验目录')
        arms=['real_only','real_plus_synthetic']
        if d_directory.exists():
            verify_frozen_arm(d_directory)
            if read_json(d_directory/'budget.json').get('synthetic_dataset_sha256')!=frozen['synthetic_dataset_sha256']:
                raise ValueError('C/D 冻结来源不一致')
            arms.append('synthetic_only')
        print('ARMS_FROZEN_REUSED',arms,'synthetic_snapshot_sha256',frozen['synthetic_dataset_sha256'],
              '后续发布组不会加入；如需新数据，请另建实验。')
        return
    provenance_bytes=(synthetic/'provenance.json').read_bytes()
    train_bytes=(synthetic/'train.txt').read_bytes()
    provenance=json.loads(provenance_bytes.decode('utf-8'))
    if provenance.get('protocol')!='canonical_seed_grouped_visible_player_v1':raise ValueError('C 必须来自已验收 canonical 派生 provenance')
    protocol_bytes=Path(args.protocol).read_bytes()
    coverage=verify_protocol_coverage(provenance.get('clips',[]),json.loads(protocol_bytes.decode('utf-8')),{'train'})
    synth_train=train_bytes.decode('utf-8').splitlines();real_train=real_groups['train'];count=len(real_train)
    if provenance.get('counts',{}).get('train')!=len(synth_train):raise ValueError('合成 provenance 与训练列表数量不一致')
    if len(synth_train)<count:raise ValueError('合成训练帧不足')
    selected,sampling=temporal_sample(synth_train,count,args.seed)
    arms={'real_plus_synthetic':real_train+selected};sampling_by_arm={'real_plus_synthetic':sampling}
    if len(synth_train)>=2*count:
        selected_d,sampling_d=temporal_sample(synth_train,2*count,args.seed)
        arms['synthetic_only']=selected_d;sampling_by_arm['synthetic_only']=sampling_d
    if d_directory.exists():raise ValueError('已有 D 目录而没有冻结 C，拒绝混合实验')
    snapshots={'synthetic_provenance.json':provenance_bytes,'synthetic_train.txt':train_bytes,'selection_protocol.json':protocol_bytes,
               **{'real_'+name:data for name,data in real_files.items()},
               'real_only_budget.json':(ROOT/'datasets/arm_real_only/budget.json').read_bytes()}
    for reference in receipts:
        path=Path(reference['path']);data=path.read_bytes()
        if hashlib.sha256(data).hexdigest()!=reference['sha256']:raise ValueError('B receipt 在取快照期间改变')
        snapshots['baseline_receipts/'+path.parent.name+'.json']=data
    # 来源在取快照期间若被另一打包进程更新，拒绝发布混合状态。
    if (synthetic/'provenance.json').read_bytes()!=provenance_bytes or (synthetic/'train.txt').read_bytes()!=train_bytes or Path(args.protocol).read_bytes()!=protocol_bytes:
        raise ValueError('合成来源在快照期间变化，拒绝准备实验')
    verify_real_arm(require_completed=True)
    for name,images in arms.items():
        budget={'arm':name,'sample_selection_seed':args.seed,'epoch_samples':len(images),
                'unique_real_annotation_budget':count if name!='synthetic_only' else 0,
                'real_exposures_per_epoch':count if name=='real_plus_synthetic' else 0,
                'synthetic_images_per_epoch':count if name=='real_plus_synthetic' else 2*count,
                'real_dataset_sha256':digest(real/'provenance.json'),
                'synthetic_dataset_sha256':hashlib.sha256(provenance_bytes).hexdigest(),
                'selection_protocol_sha256':hashlib.sha256(protocol_bytes).hexdigest(),'training_coverage':coverage,
                'synthetic_sampling_policy':'balanced_clip_camera_temporal_midpoints_v1',
                'synthetic_sampling':sampling_by_arm[name],'baseline_receipts':receipts,
                'source_snapshot':'source_snapshot/synthetic_provenance.json'}
        _publish_arm(ROOT/'datasets'/('arm_'+name),{'train':images,'val':real_groups['val'],'test':real_groups['test']},budget,snapshots)
    print('ARMS_READY',['real_only',*arms],'samples_per_epoch',2*count,'B原文件保持不变')


def prepare_real_arm(args):
    real,_,groups=_real_inputs();images=groups['train']
    output=ROOT/'datasets/arm_real_only'
    if output.exists():
        verify_real_arm()
        print('REAL_ARM_FROZEN_REUSED',output)
        return
    if list((ROOT/'runs').glob('real_only_seed*_e*')):raise ValueError('B 训练目录已存在但输入丢失，拒绝重建')
    _publish_arm(output,{'train':images*2,'val':groups['val'],'test':groups['test']},
                 {'arm':'real_only','sample_selection_seed':args.seed,'epoch_samples':len(images)*2,'unique_real_annotation_budget':len(images),'real_exposures_per_epoch':len(images)*2,'synthetic_images_per_epoch':0,'real_dataset_sha256':digest(real/'provenance.json'),'synthetic_dataset_sha256':None})
    print('REAL_ARM_READY',output,len(images)*2)


def setup_imports():
    sys.path.insert(0,str(ROOT/'runtime_dependencies'))
    os.environ['YOLO_AUTOINSTALL']='false'


def train(args):
    dataset=ROOT/'datasets'/('arm_'+args.arm)/'dataset.yaml'
    frozen=None
    if args.arm=='real_only':verify_real_arm()
    else:frozen=verify_frozen_arm(dataset.parent)
    setup_imports()
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    class BudgetTrainer(DetectionTrainer):
        def optimizer_step(self):
            super().optimizer_step()
            self.pilot_optimizer_steps=getattr(self,'pilot_optimizer_steps',0)+1
    budget=read_json(dataset.with_name('budget.json'))
    dataset_config=read_json(dataset)
    input_paths=[dataset,dataset.with_name('budget.json'),dataset.parent/dataset_config['train'],dataset.parent/dataset_config['val']]
    if frozen:
        input_paths=list(dict.fromkeys(input_paths+[dataset.parent/'freeze.json']+
                                      [safe_relative(dataset.parent,name) for name in frozen['sha256']]))
    before={str(path):digest(path) for path in input_paths}
    weight=ROOT/'weights/yolo11n.pt'
    model=YOLO(str(weight))
    runname=f'{args.arm}_seed{args.seed}_e{args.epochs}'
    directory=ROOT/'runs'/runname
    if directory.exists():raise ValueError('训练输出已存在，拒绝覆盖或静默续训')
    model.train(trainer=BudgetTrainer,data=str(dataset),epochs=args.epochs,batch=args.batch,nbs=args.batch,imgsz=args.imgsz,device=args.device,seed=args.seed,deterministic=True,workers=0,cache=False,amp=False,optimizer='SGD',lr0=0.001,lrf=0.1,momentum=0.937,weight_decay=0.0005,warmup_epochs=0,patience=0,close_mosaic=0,mosaic=0,mixup=0,copy_paste=0,rect=False,val=True,save=True,plots=False,project=str(ROOT/'runs'),name=runname,exist_ok=False,pretrained=True,verbose=False)
    steps=getattr(model.trainer,'pilot_optimizer_steps',0)
    expected=math.ceil(budget['epoch_samples']/args.batch)*args.epochs
    if steps!=expected:raise RuntimeError(f'optimizer 更新预算不一致：{steps}/{expected}')
    if {str(path):digest(path) for path in input_paths}!=before:raise RuntimeError('训练期间输入清单发生变化')
    write_json(directory/'training_receipt.json',{'arm':args.arm,'training_seed':args.seed,'epochs':args.epochs,'batch':args.batch,'imgsz':args.imgsz,'optimizer_steps':steps,'expected_optimizer_steps':expected,'initial_weights_sha256':digest(weight),'input_sha256':before,'dataset_yaml_sha256':digest(dataset),'budget':budget,'evaluation_checkpoint':'weights/last.pt','last_checkpoint_sha256':digest(directory/'weights/last.pt')})
    print('TRAIN_COMPLETE',directory,steps)


def evaluate(args):
    setup_imports()
    from ultralytics import YOLO
    if (ROOT/'evaluations'/args.name).exists():raise ValueError('评测输出已存在，拒绝覆盖或隐式改名')
    model=YOLO(str(args.weights))
    dataset=ROOT/'datasets/real_player_pilot_v1/dataset.yaml'
    metrics=model.val(data=str(dataset),split=args.split,imgsz=args.imgsz,batch=args.batch,device=args.device,classes=[0],single_cls=True,conf=0.001,iou=0.7,workers=0,plots=False,save_json=False,project=str(ROOT/'evaluations'),name=args.name,exist_ok=False,verbose=False)
    result={'name':args.name,'weights':str(args.weights),'weights_sha256':digest(args.weights),'split':args.split,'protocol':'soccernet_player_only_first150_v1','inference':{'device':str(args.device),'imgsz':args.imgsz,'batch':args.batch},'mAP50_95':float(metrics.box.map),'mAP50':float(metrics.box.map50),'precision':float(metrics.box.mp),'recall':float(metrics.box.mr),'metrics':metrics.results_dict,'dataset_provenance_sha256':digest(dataset.with_name('provenance.json'))}
    write_json(ROOT/'evaluations'/args.name/'detection_metrics.json',result)
    print('DETECTION_METRICS',json.dumps(result))


def track(args):
    setup_imports()
    import cv2
    from ultralytics import YOLO
    output=ROOT/'tracking'/args.name/'data';output.mkdir(parents=True,exist_ok=False)
    tracker=Path(__file__).with_name('bytetrack_fixed.yaml')
    for official_split,name in REAL_SEQUENCES[args.split]:
        model=YOLO(str(args.weights))
        source=ROOT/'public/soccernet_tracking'/official_split/name/'img1'
        rows=[]
        for image in sorted(source.glob('*.jpg')):
            frame=cv2.imread(str(image))
            if frame is None:raise ValueError(f'图片不可解码：{image}')
            results=model.track(frame,persist=True,tracker=str(tracker),conf=0.1,iou=0.7,imgsz=args.imgsz,device=args.device,classes=[0],verbose=False)
            boxes=results[0].boxes
            if boxes is None or boxes.id is None:continue
            for box,identity,confidence in zip(boxes.xyxy.cpu().tolist(),boxes.id.cpu().tolist(),boxes.conf.cpu().tolist()):
                x0,y0,x1,y1=box
                rows.append(f'{int(image.stem)},{int(identity)},{x0:.4f},{y0:.4f},{x1-x0:.4f},{y1-y0:.4f},{confidence:.8f},-1,-1,-1')
        (output/(name+'.txt')).write_text('\n'.join(rows)+'\n',encoding='utf-8')
    write_json(output.parent/'tracking_receipt.json',{'weights_sha256':digest(args.weights),'tracker_config_sha256':digest(tracker),'dataset_provenance_sha256':digest(ROOT/'datasets/real_player_pilot_v1/provenance.json'),'split':args.split,'device':str(args.device),'imgsz':args.imgsz,'confidence':0.1,'nms_iou':0.7,'protocol':'同一固定 ByteTrack；player-only 派生GT；每个序列重置追踪器'})
    print('TRACKING_READY',output)


def evaluate_tracks(args):
    setup_imports()
    import numpy as np
    import trackeval
    lengths={name:len(list((ROOT/'public/soccernet_tracking'/official/name/'img1').glob('*.jpg'))) for official,name in REAL_SEQUENCES[args.split]}
    config={'GT_FOLDER':str(ROOT/'datasets/real_player_pilot_v1/mot_gt'),'TRACKERS_FOLDER':str(ROOT/'tracking'),'OUTPUT_FOLDER':str(ROOT/'tracking_metrics'),'TRACKERS_TO_EVAL':[args.name],'SKIP_SPLIT_FOL':True,'SEQ_INFO':lengths,'DO_PREPROC':False,'BENCHMARK':'MOT17','SPLIT_TO_EVAL':args.split,'PRINT_CONFIG':False}
    evaluator=trackeval.Evaluator({'USE_PARALLEL':False,'PRINT_CONFIG':False,'PRINT_RESULTS':False,'PLOT_CURVES':False,'OUTPUT_SUMMARY':True,'OUTPUT_DETAILED':True})
    datasets=[trackeval.datasets.MotChallenge2DBox(config)]
    metrics=[trackeval.metrics.HOTA(),trackeval.metrics.CLEAR(),trackeval.metrics.Identity()]
    results,messages=evaluator.evaluate(datasets,metrics)
    combined=results['MotChallenge2DBox'][args.name]['COMBINED_SEQ']['pedestrian']
    summary={name:float(np.asarray(combined[group][name]).mean()) for group,names in [('HOTA',['HOTA','DetA','AssA']),('Identity',['IDF1']),('CLEAR',['MOTA','IDSW'])] for name in names}
    write_json(ROOT/'tracking_metrics'/args.name/'metrics.json',{'protocol':'soccernet_player_only_first150_v1','split':args.split,'metrics':summary,'messages':messages})
    print('TRACKING_METRICS',json.dumps(summary))


def compare(args):
    rows=[];reference=None
    for name in args.names:
        detection=read_json(ROOT/'evaluations'/name/'detection_metrics.json')
        tracking=read_json(ROOT/'tracking_metrics'/name/'metrics.json')
        receipt=read_json(ROOT/'tracking'/name/'tracking_receipt.json')
        if detection['split']!=args.split or tracking['split']!=args.split or receipt['split']!=args.split:
            raise ValueError('各组评测集合必须一致')
        if detection['weights_sha256']!=receipt['weights_sha256']:
            raise ValueError('检测与追踪使用的 checkpoint 不同')
        inference=detection.get('inference') or {}
        if not all(key in inference for key in ('device','imgsz')) or not all(key in receipt for key in ('device','imgsz','dataset_provenance_sha256','tracker_config_sha256')):
            raise ValueError('历史结果缺少设备或数据凭证；保留原结果，使用新名称重评')
        if any(inference[key]!=receipt[key] for key in ('device','imgsz')):
            raise ValueError('同组检测与追踪的推理设备或尺寸不同')
        if detection.get('dataset_provenance_sha256')!=receipt['dataset_provenance_sha256']:
            raise ValueError('同组检测与追踪的真实数据 provenance 不同')
        signature=(inference['device'],inference['imgsz'],receipt['dataset_provenance_sha256'],receipt['tracker_config_sha256'])
        if reference is None:reference=signature
        elif signature!=reference:raise ValueError('各组推理设备、尺寸、真实数据或追踪器不同')
        row={'name':name,'inference_device':inference['device'],'inference_imgsz':inference['imgsz'],**{key:detection[key] for key in ('mAP50_95','mAP50','precision','recall')},**tracking['metrics']}
        rows.append(row)
    baseline=rows[0]
    for row in rows:
        row['delta_vs_first']={key:row[key]-baseline[key] for key in ('mAP50_95','HOTA','IDF1','IDSW')}
    write_json(ROOT/'comparison'/('comparison_'+args.split+'.json'),{'protocol':'soccernet_player_only_first150_v1','warning':'小样本 pilot，不据此声称统计显著性；保留负向变化。','rows':rows})
    lines=['| 组 | AP50–95 | AP50 | HOTA | IDF1 | IDSW |','| --- | ---: | ---: | ---: | ---: | ---: |']
    lines.extend(f"| {row['name']} | {row['mAP50_95']:.4f} | {row['mAP50']:.4f} | {row['HOTA']:.4f} | {row['IDF1']:.4f} | {row['IDSW']:.0f} |" for row in rows)
    path=ROOT/'comparison'/('comparison_'+args.split+'.md')
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('COMPARISON_READY',path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['verify-protocol','verify-arms','prepare-real','prepare-synthetic','prepare-synthetic-mot','prepare-arms','prepare-real-arm','train','evaluate','track','evaluate-tracks','compare'])
    parser.add_argument('--index',type=Path,default=DEFAULT_INDEX)
    parser.add_argument('--canonical-root',type=Path,help='从规范化 seed 数据集读取，不依赖 legacy staging')
    parser.add_argument('--verification-level',choices=['full','metadata','structure'],default='full',help='structure复用既有发布证据，不重复读取媒体SHA或像素')
    parser.add_argument('--reference-images',action='store_true',help='通过Windows目录联接直接引用正式RGB，避免跨盘复制')
    parser.add_argument('--protocol',type=Path,default=DEFAULT_PROTOCOL,help='首次冻结 C 的完整天气选择协议')
    parser.add_argument('--arm',choices=['real_only','real_plus_synthetic','synthetic_only'],default='real_only')
    parser.add_argument('--seed',type=int,default=20261021)
    parser.add_argument('--epochs',type=int,default=5)
    parser.add_argument('--batch',type=int,default=4)
    parser.add_argument('--imgsz',type=int,default=640)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--weights',type=Path,default=ROOT/'weights/yolo11n.pt')
    parser.add_argument('--split',choices=['val','test'],default='test')
    parser.add_argument('--name',default='pretrained_baseline')
    parser.add_argument('--names',nargs='+',default=['pretrained_baseline','real_only','real_plus_synthetic'])
    args=parser.parse_args()
    actions={'verify-protocol':verify_protocol,'verify-arms':verify_arms,'prepare-real':prepare_real,'prepare-synthetic':prepare_synthetic,'prepare-synthetic-mot':prepare_synthetic_mot,'prepare-arms':prepare_arms,'prepare-real-arm':prepare_real_arm,'train':train,'evaluate':evaluate,'track':track,'evaluate-tracks':evaluate_tracks,'compare':compare}
    actions[args.action](args)


if __name__=='__main__':
    main()
