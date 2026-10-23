"""固定COCO dev200的官方AP/AR评价；仅显式推理才读取该开发集图片。"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from grf_ue_bridge import coco_pose_dev_eval as evaluator


def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--protocol-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint',type=Path)
    source.add_argument('--predictions',type=Path)
    parser.add_argument('--checkpoint-sha256')
    parser.add_argument('--checkpoint-role',choices=('last_ema','pretrained_reference'),default='last_ema')
    parser.add_argument('--run-receipt',type=Path)
    parser.add_argument('--run-receipt-sha256')
    parser.add_argument('--schedule-manifest-sha256')
    parser.add_argument('--prediction-receipt',type=Path)
    parser.add_argument('--prediction-receipt-sha256')
    parser.add_argument('--device',choices=('cpu','cuda:0'),default='cpu')
    args=parser.parse_args()
    os.environ.update(YOLO_AUTOINSTALL='false',YOLO_OFFLINE='true',WANDB_MODE='disabled',COMET_MODE='DISABLED',
                      CUBLAS_WORKSPACE_CONFIG=':4096:8')
    context=evaluator.load_dev_context(args.protocol,args.protocol_sha256)
    output=evaluator.safe_path(args.output)
    evaluator.require(not output.exists(),'必须使用全新评价目录，保留已有收据')
    output.mkdir(parents=True)
    try:
        if args.checkpoint:
            model,checkpoint=evaluator.load_evaluation_model(args.checkpoint,args.checkpoint_sha256,
                args.checkpoint_role,context,args.run_receipt,args.run_receipt_sha256,args.schedule_manifest_sha256)
            predictions,processed,runtime=evaluator.inference(context,model,args.device)
            write_json(output/'predictions.json',predictions)
            receipt=dict(schema='futsalmot_coco_pose_dev_predictions_v1',binding=context['binding'],
                prediction_sha256=evaluator.sha256(output/'predictions.json'),processed_image_ids=processed,
                checkpoint=checkpoint,runtime=runtime,postprocess=evaluator.POSTPROCESS,score_policy=evaluator.SCORE_POLICY,
                physical_visibility='unknown_not_predicted',test_read=False)
            evaluator.validate_predictions(predictions,processed,context['image_ids'])
            write_json(output/'predictions_receipt.json',receipt)
        else:
            evaluator.require(args.prediction_receipt is not None,'缓存评价必须提供预测来源收据')
            path=evaluator.safe_path(args.predictions)
            receipt=evaluator.checked_json(args.prediction_receipt,args.prediction_receipt_sha256)
            processed=evaluator.validate_prediction_receipt(receipt,context,path)
            predictions=evaluator.read_json(path)
        report,log=evaluator.evaluate(context,predictions,processed)
        report.update(prediction_receipt=receipt,evaluator_source_sha256=evaluator.sha256(evaluator.__file__))
        write_json(output/'report.json',report)
        (output/'official_COCOeval.txt').write_text(log,encoding='utf-8')
        print(json.dumps(dict(status=report['status'],output=str(output),keypoints=report['official_COCOeval']['keypoints']),
                         ensure_ascii=False))
    except Exception as error:
        write_json(output/'failure.json',dict(status='failed_no_metrics_claimed',error_type=type(error).__name__,error=str(error)))
        raise


if __name__=='__main__':main()
