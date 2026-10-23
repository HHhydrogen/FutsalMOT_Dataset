"""固定检测关联结果的独立CPU TrackEval入口；不读媒体、不筛选预测。"""
import configparser
from contextlib import redirect_stdout
import hashlib
import inspect
import io
import json
import math
from pathlib import Path
import re

VERSION='futsalmot_fixed_pose_association_trackeval_v1'
SEQUENCES=('SNMOT-151','SNMOT-152','SNMOT-153')
GT_ROOT=Path('D:/projects/FutsalMOT_Research/round4_combined_eval/datasets/real_player_pilot_v1/mot_gt')


def require(ok,message):
    if not ok:raise ValueError(message)


def safe_path(path):
    """先词法拒绝封存域，再检查联接；不触碰被拒绝路径的stat/内容。"""
    path=Path(path).absolute()
    require('..' not in path.parts and not any(p.lower() in ('test','test2017','test2014')
        or re.search(r'snmot[-_]?0*(116|187)(?:\D|$)',p.lower()) for p in path.parts),'禁止封存test或父目录逃逸')
    for part in (path,*path.parents):
        if part.exists():require(not part.is_symlink() and not getattr(part.stat(),'st_file_attributes',0)&0x400,
                                 '禁止联接或符号链接: '+str(part))
    return path


def sha(path):return hashlib.sha256(safe_path(path).read_bytes()).hexdigest()


def read(path):
    def invalid(value):raise ValueError('JSON非有限常数: '+value)
    return json.loads(safe_path(path).read_text(encoding='utf-8-sig'),parse_constant=invalid)


def is_sha(value):return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None


def checked(path,expected):
    path=safe_path(path);require(is_sha(expected) and sha(path)==expected,'文件SHA变化: '+str(path))
    return path


def bound(value):return checked(value['path'],value['sha256'])


def validate_receipts(current,baseline,cache_sha,protocol_sha):
    """必须已在相同cache/实现/协议完成三序列lambda0逐值验证，才允许计算指标。"""
    for report in (current,baseline):
        require(report.get('schema')=='futsalmot_fixed_pose_association_candidate_v1'
            and report.get('status')=='candidate_cpu_tracking_complete_not_metric_evaluated'
            and report.get('protocol_sha256')==protocol_sha
            and report.get('identity',{}).get('fixed_detection_set_sha256')==cache_sha
            and report.get('GT_read') is False and report.get('test_read') is False
            and report.get('gpu_used') is False and report.get('physical_visibility_used') is False,
            '追踪收据身份/输入/无GT契约不符合')
        require(type(report.get('pose_lambda')) in (int,float) and report['pose_lambda'] in (0.,.1),
                '本轮只评价预定lambda0或唯一lambda0.1')
        require(set(report.get('sequences',{}))==set(SEQUENCES)
            and all(type(row.get('frames')) is int and row['frames']==150
                and type(row.get('tracked_rows')) is int and row['tracked_rows']>=0
                for row in report['sequences'].values()),'三序列必须完整150帧，不能按结果挑片段')
    require(baseline['pose_lambda']==0 and current['identity']==baseline['identity'],'lambda0不是同一固定检测/关联实现')
    for row in baseline['sequences'].values():
        evidence=row.get('lambda0_equivalence',{})
        require(isinstance(evidence,dict) and evidence.get('passed') is True and evidence.get('frames')==150
            and evidence.get('output_arrays_exact') is True and evidence.get('track_states_exact') is True,
            '缺实际lambda0输出与状态逐值等价证据')
    if current['pose_lambda']==0:
        for row in current['sequences'].values():
            evidence=row.get('lambda0_equivalence',{})
            require(isinstance(evidence,dict) and evidence.get('passed') is True
                and evidence.get('output_arrays_exact') is True and evidence.get('track_states_exact') is True,
                '当前lambda0没有通过等价门禁')
    else:
        require(is_sha(current.get('pose_model_sha256')) and is_sha(current.get('attachment_manifest_sha256')),
                'lambda0.1缺实际模型/附件SHA')


def parse_mot(text,frames,*,ground_truth=False):
    """只验全部原行，不删行、不改坐标/分数/ID；空预测允许。"""
    rows=[];keys=set()
    for line in text.splitlines():
        if not line.strip():continue
        values=[float(v) for v in line.split(',')]
        require(len(values)==(9 if ground_truth else 10) and all(math.isfinite(v) for v in values),'MOT列数或数值非法')
        require(values[0].is_integer() and 1<=values[0]<=frames and values[1].is_integer() and values[1]>0,
                'MOT帧/ID无效或超出已登记范围')
        key=(int(values[0]),int(values[1]));require(key not in keys,'重复的同帧track ID，禁止自动修复')
        keys.add(key);require(values[4]>0 and values[5]>0,'MOT框宽高无效')
        if ground_truth:require(values[6:9]==[1.,1.,1.],'派生player GT必须mark/class/visibility=1，避免隐式过滤')
        else:require(0<=values[6]<=1 and values[7:]==[-1.,-1.,-1.],'预测必须保留runner原MOT score/占位列')
        rows.append(values)
    return rows


def validate_inputs(tracking_dir,receipt_sha,cache_path,cache_sha,protocol_path,protocol_sha,baseline_path,baseline_sha):
    """生产门禁仅读固定开发文本、已登记player GT和MOT；不访问原图或其他GT。"""
    tracking_dir=safe_path(tracking_dir)
    current=read(checked(tracking_dir/'receipt.json',receipt_sha));baseline=read(checked(baseline_path,baseline_sha))
    cache=read(checked(cache_path,cache_sha));protocol=read(checked(protocol_path,protocol_sha))
    require(cache.get('schema')=='futsalmot_fixed_detection_cache_manifest_v1'
        and cache.get('status')=='complete_fixed_raw_detections'
        and cache.get('tracking_applied') is False and cache.get('pose_applied') is False
        and cache.get('GT_used_in_prediction') is False and cache.get('test_split_used') is False,
        '只能消费完整独立raw缓存')
    require(protocol.get('schema')=='futsalmot_pose_association_protocol_candidate_v1'
        and protocol.get('source_manifest')==cache.get('source_manifest')
        and protocol.get('test_split_used') is False
        and 0 in protocol['pose_cost_candidate']['lambda_candidates']
        and .1 in protocol['pose_cost_candidate']['lambda_candidates'],'协议/缓存来源或唯一lambda范围错误')
    validate_receipts(current,baseline,cache_sha,protocol_sha)
    source=read(bound(protocol['source_manifest']))
    require(source.get('schema')=='futsalmot_pose_association_dev_source_candidate_v1'
        and source.get('frames')==450 and source.get('test_split_used') is False
        and [row['sequence_id'] for row in source['sequences']]==list(SEQUENCES),'不是既定三序列源登记')
    require({p.name for p in (tracking_dir/'data').glob('*.txt')}=={s+'.txt' for s in SEQUENCES},
            '预测目录序列缺失或含未登记片段')
    gt_evidence={};prediction_evidence={}
    for entry in source['sequences']:
        sequence=entry['sequence_id']
        require(entry.get('game_id')=='9' and entry.get('role')=='development_only'
            and entry.get('frames') and len(entry['frames'])==150
            and [f['frame_number'] for f in entry['frames']]==list(range(1,151)),'源片段不是完整game9前150帧')
        expected=safe_path(GT_ROOT/sequence/'gt'/'gt.txt')
        require(safe_path(entry['derived_gt_file'])==expected,'GT不是原登记的round4combined派生player文件')
        references=[r for r in entry['text_sources'] if safe_path(r['path'])==expected]
        require(len(references)==1,'源登记缺唯一派生GT SHA')
        path=bound(references[0]);gt_rows=parse_mot(path.read_text(encoding='utf-8'),150,ground_truth=True)
        require(len(gt_rows)==entry['selected_player_gt_rows'] and {int(r[0]) for r in gt_rows}==set(range(1,151)),
                '派生GT数量/帧覆盖变化')
        info_path=safe_path(GT_ROOT/sequence/'seqinfo.ini');info=configparser.ConfigParser();info.read(info_path,encoding='utf-8')
        require(info['Sequence']['name']==sequence and info.getint('Sequence','seqLength')==150
            and info.getint('Sequence','imWidth')==1920 and info.getint('Sequence','imHeight')==1080
            and info.getint('Sequence','frameRate')==25,'派生seqinfo身份/长度变化')
        gt_evidence[sequence]=dict(path=str(path),sha256=references[0]['sha256'],rows=len(gt_rows),
                                   seqinfo=dict(path=str(info_path),sha256=sha(info_path)))
        pred=safe_path(tracking_dir/'data'/(sequence+'.txt'))
        pred_rows=parse_mot(pred.read_text(encoding='utf-8'),150)
        require(len(pred_rows)==current['sequences'][sequence]['tracked_rows'],'预测行数与原追踪收据不同')
        actual_sha=sha(pred);generation_sha=current['sequences'][sequence].get('mot_file_sha256')
        if generation_sha is not None:require(actual_sha==generation_sha,'预测内容与生成时SHA不同')
        prediction_evidence[sequence]=dict(path=str(pred),sha256=actual_sha,rows=len(pred_rows),
            original_generation_sha256=generation_sha)
    return dict(tracking_dir=tracking_dir,tracking=current,cache=cache,protocol=protocol,
                GT=gt_evidence,predictions=prediction_evidence,
                binding=dict(tracking_receipt_sha256=receipt_sha,cache_manifest_sha256=cache_sha,
                             protocol_sha256=protocol_sha,lambda0_receipt_sha256=baseline_sha))


def plain(value):
    """保留原指标数组与计数，转换NumPy类型，不压缩或均值替代原结果。"""
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    if hasattr(value,'tolist'):return plain(value.tolist())
    if hasattr(value,'item'):return value.item()
    return value


def summarize(raw):
    summaries={}
    for sequence,classes in raw.items():
        values=classes['pedestrian'];fraction={}
        for group,fields in (('HOTA',('HOTA','AssA','DetA')),('Identity',('IDF1',)),('CLEAR',('MOTA',))):
            for field in fields:
                value=plain(values[group][field])
                fraction[field]=sum(value)/len(value) if isinstance(value,list) else float(value)
        switches=float(values['CLEAR']['IDSW']);require(switches.is_integer(),'IDSW不是整数计数')
        summaries[sequence]=dict(metrics_fraction=fraction,metrics_percent={k:100*v for k,v in fraction.items()},
                                 IDSW_count=int(switches))
    return summaries


def run_trackeval(gt_folder,tracking_dir,seq_lengths,output):
    """薄数值入口；生产CLI另经validate_inputs门禁，CPU夹具可用更短序列。"""
    import trackeval
    require(trackeval.__version__=='1.3.0','仅验证本机TrackEval1.3.0')
    require(set(seq_lengths)==set(SEQUENCES) and all(type(n) is int and n>0 for n in seq_lengths.values()),
            '只允许登记的三序列，不接受test')
    gt_folder=safe_path(gt_folder);tracking_dir=safe_path(tracking_dir);output=safe_path(output)
    require(not output.exists(),'TrackEval结果必须新目录')
    output.mkdir(parents=True)
    dataset_config=dict(GT_FOLDER=str(gt_folder),TRACKERS_FOLDER=str(tracking_dir.parent),OUTPUT_FOLDER=str(output),
        TRACKERS_TO_EVAL=[tracking_dir.name],SKIP_SPLIT_FOL=True,SEQ_INFO=dict(seq_lengths),DO_PREPROC=False,
        BENCHMARK='MOT17',SPLIT_TO_EVAL='train',PRINT_CONFIG=False,CLASSES_TO_EVAL=['pedestrian'],
        TRACKER_SUB_FOLDER='data',INPUT_AS_ZIP=False)
    eval_config=dict(USE_PARALLEL=False,PRINT_CONFIG=False,PRINT_RESULTS=False,PLOT_CURVES=False,
        TIME_PROGRESS=False,OUTPUT_SUMMARY=True,OUTPUT_DETAILED=True,BREAK_ON_ERROR=True,LOG_ON_ERROR=str(output/'errors.txt'))
    capture=io.StringIO()
    with redirect_stdout(capture):
        evaluator=trackeval.Evaluator(eval_config);dataset=trackeval.datasets.MotChallenge2DBox(dataset_config)
        require(dataset.do_preproc is False,'实际TrackEval DO_PREPROC没有关闭')
        metrics=[trackeval.metrics.HOTA(),trackeval.metrics.CLEAR(dict(PRINT_CONFIG=False)),
                 trackeval.metrics.Identity(dict(PRINT_CONFIG=False))]
        results,messages=evaluator.evaluate([dataset],metrics)
    require(messages['MotChallenge2DBox'][tracking_dir.name]=='Success','TrackEval未成功')
    raw=plain(results['MotChallenge2DBox'][tracking_dir.name])
    require(set(raw)==set(SEQUENCES)|{'COMBINED_SEQ'},'TrackEval片段/合并结果不全')
    dependencies=[Path(inspect.getfile(type(evaluator))),Path(inspect.getfile(type(dataset)))]+[
        Path(inspect.getfile(type(metric))) for metric in metrics]
    return dict(raw_results=raw,summary=summarize(raw),messages=plain(messages),
        HOTA_alpha_thresholds=plain(metrics[0].array_labels),dataset_config=dataset.config,evaluator_config=evaluator.config,
        trackeval_version=trackeval.__version__,numerical_source_sha256={str(p):sha(p) for p in dependencies}),capture.getvalue()
