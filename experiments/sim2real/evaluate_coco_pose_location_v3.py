"""独立v3开发评价：新协议/EMA身份，复用旧COCO评价数值。"""
import argparse
import json
import os
from pathlib import Path
import sys

CODE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(CODE/'src'))
from grf_ue_bridge import coco_pose_dev_eval_v3 as adapter
from grf_ue_bridge import coco_pose_dev_eval as numeric


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False);handle.write('\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','admission','output'):parser.add_argument('--'+name,type=Path,required=True)
    for name in ('protocol-sha256','admission-sha256'):parser.add_argument('--'+name,required=True)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint',type=Path);source.add_argument('--predictions',type=Path)
    parser.add_argument('--checkpoint-sha256')
    parser.add_argument('--checkpoint-role',choices=('last_ema','pretrained_reference'),default='last_ema')
    parser.add_argument('--run-receipt',type=Path);parser.add_argument('--run-receipt-sha256')
    parser.add_argument('--prediction-receipt',type=Path);parser.add_argument('--prediction-receipt-sha256')
    parser.add_argument('--device',choices=('cpu','cuda:0'),default='cpu')
    args=parser.parse_args()
    os.environ.update(YOLO_AUTOINSTALL='false',YOLO_OFFLINE='true',WANDB_MODE='disabled',COMET_MODE='DISABLED',
                      CUBLAS_WORKSPACE_CONFIG=':4096:8')
    context=adapter.load_context(args.protocol,args.protocol_sha256,args.admission,args.admission_sha256)
    output=numeric.safe_path(args.output);numeric.require(not output.exists(),'v3评价必须使用新目录')
    output.mkdir(parents=True)
    try:
        if args.checkpoint:
            model,metadata=adapter.load_model(args.checkpoint,args.checkpoint_sha256,args.checkpoint_role,context,
                args.run_receipt,args.run_receipt_sha256)
            predictions,processed,runtime=numeric.inference(context,model,args.device)
            numeric.validate_predictions(predictions,processed,context['image_ids'])
            write(output/'predictions.json',predictions)
            receipt=dict(schema='futsalmot_coco_pose_dev_predictions_v3',binding=context['binding'],
                prediction_sha256=numeric.sha256(output/'predictions.json'),processed_image_ids=processed,
                checkpoint=metadata,runtime=runtime,postprocess=numeric.POSTPROCESS,score_policy=numeric.SCORE_POLICY,
                physical_visibility='unknown_not_predicted',test_read=False,
                numerical_evaluator_source_sha256=adapter.NUMERICAL_SOURCE_SHA256)
            write(output/'predictions_receipt.json',receipt)
        else:
            numeric.require(args.prediction_receipt is not None,'v3缓存需独立预测收据')
            path=numeric.safe_path(args.predictions)
            receipt=numeric.checked_json(args.prediction_receipt,args.prediction_receipt_sha256)
            processed=adapter.validate_prediction_receipt(receipt,context,path)
            predictions=numeric.read_json(path)
        report,log=adapter.evaluate(context,predictions,processed)
        report.update(prediction_receipt=receipt,cli_source_sha256=numeric.sha256(__file__))
        write(output/'report.json',report);(output/'official_COCOeval.txt').write_text(log,encoding='utf-8')
        print(json.dumps(dict(status=report['status'],schema=report['schema'],
            output=str(output),keypoints=report['official_COCOeval']['keypoints']),ensure_ascii=False))
    except Exception as error:
        write(output/'failure.json',dict(status='failed_no_v3_metrics_claimed',error_type=type(error).__name__,error=str(error)))
        raise


if __name__=='__main__':main()
