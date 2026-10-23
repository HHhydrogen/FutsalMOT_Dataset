"""v3独立身份链及旧数值复用的CPU小夹具；不读真实图片/checkpoint/test。"""
from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import unittest

from grf_ue_bridge import coco_pose_dev_eval_v3 as v3
from grf_ue_bridge import coco_pose_dev_eval as numeric
from test_coco_pose_dev_eval import context_fixture,toy_gt,prediction,write


def ref(path):
    return dict(path=str(path),sha256=numeric.sha256(path))


def fixture(directory):
    root=Path(directory);candidate,_,_,binding=context_fixture(root)
    original=numeric.read_json(candidate)
    original['initialization']=dict(official_sha256='a'*64,derived_sha256='b'*64)
    original['paired_training_candidate']['seeds']=v3.SETTINGS['seeds']
    candidate_sha=write(candidate,original)
    binding['protocol_sha256']=candidate_sha;write(root/'prepared/candidate_protocol_manifest.json',binding)
    old=deepcopy(original);old.update(status='root_frozen',training_authorized=True,
        root_freeze=dict(approved_by='root',candidate_protocol=ref(candidate)))
    oldpath=root/'frozen_v2.json';write(oldpath,old)
    real=root/'R.jsonl';real.write_text('{}\n',encoding='utf-8')
    syn=root/'S.jsonl';syn.write_text('{}\n',encoding='utf-8')
    schedules=[]
    for seed in v3.SETTINGS['seeds']:
        path=root/f'schedule_{seed}.jsonl';path.write_text('{}\n',encoding='utf-8')
        for arm in ('R','RS25'):schedules.append(dict(ref(path),seed=seed,arm=arm,updates=1000,batch=4))
    manifest=dict(schema='futsalmot_pose_location_paired_schedule_manifest_v1',
        candidate_protocol_sha256=candidate_sha,real_jsonl=ref(real),synthetic_jsonl=ref(syn),schedules=schedules)
    manifest_path=root/'schedule_manifest.json';write(manifest_path,manifest)
    old_admission=dict(schema='futsalmot_pose_location_training_admission_v1',status='approved_location_only',
        approved_by='root',training_authorized=True,protocol=ref(oldpath),schedule_manifest=ref(manifest_path),
        real_jsonl=ref(real),synthetic_jsonl=ref(syn))
    old_admission_path=root/'admission_v2.json';write(old_admission_path,old_admission)
    files=[]
    for name in ('train_pose_location_control.py','train_pose_location_poseonly_v3.py'):
        path=root/name;path.write_text('# fixture metadata only\n',encoding='utf-8');files.append(ref(path))
    protocol=dict(schema=v3.PROTOCOL_SCHEMA,status='frozen',training_authorized=True,test_split_used=False,
        source_v2_protocol=ref(oldpath),source_v2_admission=ref(old_admission_path),schedule_manifest=ref(manifest_path),
        source_files=files,paired_training=deepcopy(v3.SETTINGS),policy=deepcopy(v3.POLICY),
        initialization=deepcopy(original['initialization']),arms=deepcopy(original['arms']))
    protocol_path=root/'protocol_v3.json';write(protocol_path,protocol)
    admission=dict(schema=v3.ADMISSION_SCHEMA,status='approved_poseonly_location_v3',approved_by='root',
        training_authorized=True,location_only=True,visibility_supervision=False,test_split_used=False,
        protocol=ref(protocol_path),schedule_manifest=ref(manifest_path),source_v2_protocol=ref(oldpath),
        source_v2_admission=ref(old_admission_path),source_files=files,policy=deepcopy(v3.POLICY),
        UE_serial_gate=dict(all_rendering_finished=True,editor_safely_exited=True))
    admission_path=root/'admission_v3.json';write(admission_path,admission)
    return protocol_path,admission_path


def checkpoint_fixture(context):
    seed=v3.SETTINGS['seeds'][0];old=context['source_v2_bindings_by_seed'][seed]
    bindings=dict(source_v2_bindings=deepcopy(old),source_files=deepcopy(context['v3_source_files']))
    for key in ('schedule_manifest_sha256','schedule_sha256','real_jsonl_sha256','synthetic_jsonl_sha256'):
        bindings[key]=old[key]
    for key,value in dict(protocol_sha256=context['protocol_sha256'],admission_sha256=context['v3_admission_sha256'],
            trainer_source_sha256=context['v3_trainer_source_sha256']).items():
        bindings[key]=value;bindings['v3_'+key]=value
    checkpoint=dict(schema=v3.CHECKPOINT_SCHEMA,completed_updates=1000,planned_updates=1000,ema_updates=1000,
        policy=deepcopy(v3.POLICY),checkpoint_choice='last_actual_FP32_EMA_no_best',actual_ema_fp32=True,
        keypoint_shape=[17,2],physical_output_channels=51,visibility_supervised=False,seed=seed,arm='R',bindings=bindings,
        initial_state_sha256='e'*64,exposure_chain_sha256='f'*64)
    run=deepcopy(checkpoint);run.update(schema=v3.RUN_SCHEMA,complete=True,real_test_read=False,
                                      actual_exposures=dict(real=4000,synthetic=0),
                                      checkpoint_choice='last1000_actual_FP32_EMA_no_best')
    return checkpoint,run


class V3EvaluationTests(unittest.TestCase):
    def context(self,directory):
        p,a=fixture(directory)
        return v3.load_context(p,numeric.sha256(p),a,numeric.sha256(a))

    def test_v3_to_v2_to_candidate_dev200_chain(self):
        with tempfile.TemporaryDirectory() as root:
            context=self.context(root)
            self.assertEqual(len(context['image_ids']),200)
            self.assertNotEqual(context['binding']['protocol_sha256'],context['binding']['source_v2_protocol_sha256'])
            self.assertNotEqual(context['binding']['source_v2_protocol_sha256'],context['binding']['candidate_protocol_sha256'])
            c,r=checkpoint_fixture(context);metadata=v3.validate_checkpoint(c,r,context)
            self.assertEqual(metadata['role'],'last_ema_v3')

    def test_old_protocol_or_approval_cannot_be_used_as_v3(self):
        with tempfile.TemporaryDirectory() as root:
            p,a=fixture(root);protocol=numeric.read_json(p)
            old=protocol['source_v2_protocol']
            with self.assertRaisesRegex(ValueError,'v3协议'):
                v3.load_context(old['path'],old['sha256'],a,numeric.sha256(a))
            admission=numeric.read_json(a);admission['approved_by']='other';write(a,admission)
            with self.assertRaisesRegex(ValueError,'root'):v3.load_context(p,numeric.sha256(p),a,numeric.sha256(a))

    def test_changed_source_schedule_and_sigma_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            p,a=fixture(root);protocol=numeric.read_json(p)
            target=Path(protocol['schedule_manifest']['path']);target.write_text('{}',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'SHA'):v3.load_context(p,numeric.sha256(p),a,numeric.sha256(a))
        with tempfile.TemporaryDirectory() as root:
            p,a=fixture(root);admission=numeric.read_json(a);admission['policy']['sigma']='uniform1/17';write(a,admission)
            with self.assertRaisesRegex(ValueError,'sigma'):v3.load_context(p,numeric.sha256(p),a,numeric.sha256(a))

    def test_checkpoint_alias_old_source_budget_and_dtype_flags_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            context=self.context(root);c,r=checkpoint_fixture(context)
            for key,value in [('schema','futsalmot_pose_location_control_checkpoint_v1'),('completed_updates',999),
                              ('actual_ema_fp32',False),('keypoint_shape',[17,3])]:
                bad=deepcopy(c);bad[key]=value
                with self.assertRaises(ValueError):v3.validate_checkpoint(bad,r,context)
            bad=deepcopy(c);wrong_run=deepcopy(r)
            for record in (bad,wrong_run):record['bindings']['v3_protocol_sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'别名'):v3.validate_checkpoint(bad,wrong_run,context)
            bad=deepcopy(c);wrong_run=deepcopy(r)
            for record in (bad,wrong_run):record['bindings']['source_v2_bindings']['real_jsonl_sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'原R/S'):v3.validate_checkpoint(bad,wrong_run,context)

    @unittest.skipUnless(importlib.util.find_spec('torch'),'研究环境提供Torch')
    def test_declared_fp32_cannot_hide_actual_fp16_model(self):
        import torch
        with self.assertRaisesRegex(ValueError,'实际EMA含非FP32'):
            v3.validate_model(torch.nn.Linear(2,2).half())

    @unittest.skipUnless(importlib.util.find_spec('pycocotools'),'官方COCOeval可用')
    def test_v3_reuses_identical_official_numbers_and_point_score_policy(self):
        context=dict(gt=toy_gt(),image_ids=[1,2],binding={'protocol_sha256':'a'*64},protocol=dict(status='frozen'))
        predictions=[prediction(1),prediction(2)]
        old,_=numeric.evaluate(context,predictions,[1,2]);new,_=v3.evaluate(context,predictions,[1,2])
        self.assertEqual(old['official_COCOeval'],new['official_COCOeval'])
        self.assertEqual(old['auxiliary_location'],new['auxiliary_location'])
        self.assertEqual(new['schema'],v3.VERSION)
        self.assertEqual(old['keypoint_score_policy'],new['keypoint_score_policy'])


if __name__=='__main__':unittest.main()
