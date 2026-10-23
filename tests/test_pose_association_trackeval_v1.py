"""独立TrackEval CPU小夹具；不读取真实序列、图片或封存测试。"""
from copy import deepcopy
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from grf_ue_bridge import pose_association_trackeval_v1 as m


def receipt(lam=0):
    return dict(schema='futsalmot_fixed_pose_association_candidate_v1',
        status='candidate_cpu_tracking_complete_not_metric_evaluated',protocol_sha256='a'*64,
        identity=dict(fixed_detection_set_sha256='b'*64,implementation_sha256='c'*64),pose_lambda=lam,
        pose_model_sha256='d'*64 if lam else None,attachment_manifest_sha256='e'*64 if lam else None,
        GT_read=False,test_read=False,gpu_used=False,physical_visibility_used=False,sequences={s:dict(frames=150,
        tracked_rows=1,lambda0_equivalence=dict(passed=True,frames=150,output_arrays_exact=True,track_states_exact=True)
        if not lam else None) for s in m.SEQUENCES})


class AssociationTrackEvalTests(unittest.TestCase):
    def test_only_fixed_zero_or_point_one_after_actual_equivalence(self):
        baseline=receipt();m.validate_receipts(receipt(.1),baseline,'b'*64,'a'*64)
        for field,value in [('pose_lambda',.25),('GT_read',True),('test_read',True)]:
            invalid=receipt(.1);invalid[field]=value
            with self.assertRaises(ValueError):m.validate_receipts(invalid,baseline,'b'*64,'a'*64)
        baseline['sequences'][m.SEQUENCES[0]]['lambda0_equivalence']['track_states_exact']=False
        with self.assertRaisesRegex(ValueError,'逐值'):m.validate_receipts(receipt(.1),baseline,'b'*64,'a'*64)

    def test_cross_cache_protocol_or_missing_sequence_rejected(self):
        for cache,protocol in [('f'*64,'a'*64),('b'*64,'f'*64)]:
            with self.assertRaises(ValueError):m.validate_receipts(receipt(.1),receipt(),cache,protocol)
        invalid=receipt(.1);invalid['sequences'].pop(m.SEQUENCES[0])
        with self.assertRaises(ValueError):m.validate_receipts(invalid,receipt(),'b'*64,'a'*64)

    def test_mot_parser_keeps_low_score_and_rejects_duplicate_or_distractor_GT(self):
        line='1,1,10,20,30,40,0.001,-1,-1,-1'
        self.assertEqual(len(m.parse_mot(line,3)),1)
        self.assertEqual(m.parse_mot('',3),[])
        with self.assertRaisesRegex(ValueError,'重复'):m.parse_mot(line+'\n'+line,3)
        with self.assertRaises(ValueError):m.parse_mot('4,1,10,20,30,40,.9,-1,-1,-1',3)
        with self.assertRaisesRegex(ValueError,'player'):m.parse_mot('1,1,10,20,30,40,1,2,1',3,ground_truth=True)

    def test_sealed_paths_rejected_before_stat(self):
        with patch.object(Path,'exists',side_effect=AssertionError('不能访问封存路径')):
            with self.assertRaisesRegex(ValueError,'封存'):m.safe_path('D:/sealed/SNMOT-187/gt/gt.txt')

    def test_fraction_percent_and_negative_MOTA_do_not_scale_IDSW(self):
        raw=dict(COMBINED_SEQ=dict(pedestrian=dict(HOTA=dict(HOTA=[.2,.4],AssA=[.5,.7],DetA=[.1,.3]),
            Identity=dict(IDF1=.8),CLEAR=dict(MOTA=-.2,IDSW=7))))
        summary=m.summarize(raw)['COMBINED_SEQ']
        self.assertAlmostEqual(summary['metrics_fraction']['AssA'],.6)
        self.assertAlmostEqual(summary['metrics_percent']['AssA'],60.)
        self.assertEqual(summary['metrics_percent']['MOTA'],-20.)
        self.assertEqual(summary['IDSW_count'],7)
        self.assertNotIn('IDSW',summary['metrics_percent'])

    @unittest.skipUnless(importlib.util.find_spec('trackeval'),'本机既有TrackEval运行时')
    def test_actual_TrackEval_no_preproc_raw_all_sequences_and_weighted_combined(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);gt=root/'gt';tracking=root/'toy_tracker';(tracking/'data').mkdir(parents=True)
            for i,sequence in enumerate(m.SEQUENCES):
                people=2 if i==1 else 1;truth=[];pred=[]
                for f in range(1,4):
                    for identity in range(1,people+1):
                        x=identity*30
                        truth.append(f'{f},{identity},{x},10,10,20,1,1,1')
                        track_id=2 if i==2 and f>1 else identity
                        pred.append(f'{f},{track_id},{x},10,10,20,0.001,-1,-1,-1')
                if i==1:pred.append('2,9,200,200,10,10,0.001,-1,-1,-1')
                folder=gt/sequence/'gt';folder.mkdir(parents=True)
                (folder/'gt.txt').write_text('\n'.join(truth)+'\n',encoding='utf-8')
                (tracking/'data'/(sequence+'.txt')).write_text('\n'.join(pred)+'\n',encoding='utf-8')
            result,log=m.run_trackeval(gt,tracking,{s:3 for s in m.SEQUENCES},root/'metrics')
            self.assertFalse(result['dataset_config']['DO_PREPROC'])
            raw=result['raw_results'];summary=result['summary']
            self.assertEqual(set(raw),set(m.SEQUENCES)|{'COMBINED_SEQ'})
            self.assertEqual(len(raw[m.SEQUENCES[0]]['pedestrian']['HOTA']['HOTA']),19)
            self.assertEqual(raw['COMBINED_SEQ']['pedestrian']['Count']['GT_Dets'],12)
            self.assertEqual(raw['COMBINED_SEQ']['pedestrian']['Count']['Dets'],13)
            self.assertEqual(summary[m.SEQUENCES[0]]['metrics_fraction']['AssA'],1.)
            self.assertEqual(summary[m.SEQUENCES[2]]['IDSW_count'],1)
            self.assertEqual(summary['COMBINED_SEQ']['IDSW_count'],1)
            naive=sum(summary[s]['metrics_fraction']['AssA'] for s in m.SEQUENCES)/3
            self.assertNotAlmostEqual(summary['COMBINED_SEQ']['metrics_fraction']['AssA'],naive)
            self.assertTrue((root/'metrics/toy_tracker/pedestrian_detailed.csv').is_file())


if __name__=='__main__':unittest.main()
