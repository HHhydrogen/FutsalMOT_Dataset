"""CPU TrackEval：固定开发检测关联的lambda0/.1及已登记player GT。"""
import argparse
import json
import os
from pathlib import Path
import sys

CODE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(CODE/'src'))
from grf_ue_bridge import pose_association_trackeval_v1 as metric


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False);handle.write('\n')


def evaluate(tracking_dir,cache_path,cache_sha,protocol_path,protocol_sha,lambda0_receipt,output,
             tracking_receipt_sha=None,lambda0_receipt_sha=None):
    tracking_dir=metric.safe_path(tracking_dir)
    tracking_receipt_sha=tracking_receipt_sha or metric.sha(tracking_dir/'receipt.json')
    lambda0_receipt_sha=lambda0_receipt_sha or metric.sha(lambda0_receipt)
    context=metric.validate_inputs(tracking_dir,tracking_receipt_sha,cache_path,cache_sha,protocol_path,
        protocol_sha,lambda0_receipt,lambda0_receipt_sha)
    output=metric.safe_path(output);metric.require(not output.exists(),'必须使用新评价目录，不覆盖原结果')
    metric.require(output!=tracking_dir and tracking_dir not in output.parents,'评价目录不能写入原追踪结果')
    output.mkdir(parents=True)
    try:
        result,log=metric.run_trackeval(metric.GT_ROOT,tracking_dir,{seq:150 for seq in metric.SEQUENCES},output/'trackeval')
        raw=result['raw_results']
        for sequence in metric.SEQUENCES:
            counts=raw[sequence]['pedestrian']['Count']
            metric.require(counts['Dets']==context['predictions'][sequence]['rows']
                and counts['GT_Dets']==context['GT'][sequence]['rows'] and counts['Frames']==150,
                '实际TrackEval丢失预测/GT或帧，不能宣称无过滤')
        complete=all(row['original_generation_sha256'] is not None for row in context['predictions'].values())
        result.update(schema=metric.VERSION,status='complete_fixed_development_TrackEval',
            primary='AssA',pose_lambda=context['tracking']['pose_lambda'],
            pose_model_sha256=context['tracking'].get('pose_model_sha256'),
            input_binding=context['binding'],GT=context['GT'],predictions=context['predictions'],
            original_tracking_receipt_bound_all_MOT_SHA=complete,DO_PREPROC=False,
            units=dict(raw_results='TrackEval原生比例/数组/计数；不缩放',
                       summary_metrics_fraction='HOTA/AssA/DetA跨alpha均值；IDF1/MOTA原生比例，MOTA可负',
                       summary_metrics_percent='仅比例乘100；跨arm差为百分点pp',IDSW_count='整数次数，绝不乘100'),
            combined='官方combine_sequences；不是各片段指标算术平均',frames=450,sequences=list(metric.SEQUENCES),
            predictions_filtered=False,images_read=False,GPU_used=False,test_read=False,
            source_sha256={str(Path(__file__)):metric.sha(__file__),str(Path(metric.__file__)):metric.sha(metric.__file__)},
            limitations=['three_game9_development_clips_not_independent_matches_or_final_test',
                         'no_futsal_generalization_claim_from_these_three_clips']+
                        ([] if complete else ['original_tracking_receipt_did_not_bind_MOT_content_SHA_first_recorded_at_evaluation']))
        write(output/'report.json',result);(output/'trackeval_stdout.txt').write_text(log,encoding='utf-8')
        return result
    except Exception as error:
        write(output/'failure.json',dict(status='failed_no_metric_claim',error_type=type(error).__name__,error=str(error)))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('tracking-dir','cache-manifest','protocol','lambda0-receipt','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--cache-sha256',required=True);parser.add_argument('--protocol-sha256',required=True)
    parser.add_argument('--tracking-receipt-sha256');parser.add_argument('--lambda0-receipt-sha256')
    args=parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='-1',YOLO_AUTOINSTALL='false')
    sys.path.append('D:/projects/FutsalMOT_Research/runtime_dependencies')
    report=evaluate(args.tracking_dir,args.cache_manifest,args.cache_sha256,args.protocol,args.protocol_sha256,
        args.lambda0_receipt,args.output,args.tracking_receipt_sha256,args.lambda0_receipt_sha256)
    print(json.dumps(dict(status=report['status'],pose_lambda=report['pose_lambda'],
                         COMBINED_SEQ=report['summary']['COMBINED_SEQ']),ensure_ascii=False))


if __name__=='__main__':main()
