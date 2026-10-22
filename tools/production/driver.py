"""一组一冷启动：UE渲染/恢复/退出→CPU审计→发布→清理；不启动GPU训练。"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import psutil
from common import canonical_sha, dump, load_settings, now, read, sha, require_child
from mcp_client import UnrealMCP


class PerformanceInfo(ctypes.Structure):
    _fields_ = [('cb',ctypes.c_ulong)] + [(key,ctypes.c_size_t) for key in (
        'CommitTotal','CommitLimit','CommitPeak','PhysicalTotal','PhysicalAvailable',
        'SystemCache','KernelTotal','KernelPaged','KernelNonpaged','PageSize')] + [
        ('HandleCount',ctypes.c_ulong),('ProcessCount',ctypes.c_ulong),('ThreadCount',ctypes.c_ulong)]


def memory():
    value = PerformanceInfo()
    value.cb = ctypes.sizeof(value)
    if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(value),value.cb):
        raise ctypes.WinError()
    return {'commit_headroom_bytes':(value.CommitLimit-value.CommitTotal)*value.PageSize,
        'physical_available_bytes':value.PhysicalAvailable*value.PageSize}


def editors():
    return [p for p in psutil.process_iter(['pid','name']) if (p.info['name'] or '').lower().startswith('unrealeditor')]


def assert_no_training():
    for process in psutil.process_iter(['pid','name','cmdline']):
        if 'python' not in (process.info['name'] or '').lower():
            continue
        command = ' '.join(process.info['cmdline'] or []).lower()
        if 'futsalmot_research' in command and any(word in command for word in ('train','yolo','trackeval')):
            raise RuntimeError('检测到研究GPU任务；正式渲染队列保持串行，停止启动')


def editor_command(settings, log):
    """UE的FParse要求等号后紧接引号；不能让list2cmdline把整个选项包住。"""
    base = subprocess.list2cmdline([settings['editor'],settings['uproject'],settings['map'],
        '-unattended','-nosplash','-NoSound','-NoLiveCoding','-NoVSync','-ModelContextProtocolStartServer','-abslog='+str(log)])
    return base + ' -LogCmds="LogModuleManager Verbose" -ExecCmds="ImageWriteQueue.MaxConcurrency {},ImageWriteQueue.MaxQueueSize {}"'.format(
        int(settings['image_write_concurrency']),int(settings['image_write_queue_size']))


class Driver:
    def __init__(self, config):
        self.config = Path(config).resolve()
        self.settings = load_settings(self.config)
        self.private = Path(self.settings['private_root'])
        receipt_path = self.private / 'runtime_freeze.json'
        receipt = read(receipt_path) if receipt_path.exists() else {}
        self.runtime = require_child(receipt.get('runtime_root', self.private/'runtime'), self.private)
        self.current = None
        self.lock = self.private/'driver.lock'

    def state(self, stage, **extra):
        record = {'stage':stage,'updated_utc':now(),'driver_pid':os.getpid(),
            'gpu_training':False,'current_entry':self.current,**extra}
        dump(self.private/'controller_state.json',record)
        print(now()+' '+stage+' '+str(extra),flush=True)

    def invoke(self, script, *args, tag):
        log = self.private/'logs'/tag
        log.parent.mkdir(parents=True,exist_ok=True)
        command = [self.settings['python'],'-X','utf8',str(self.runtime/'tools/production'/script),*map(str,args)]
        with log.with_suffix('.stdout.log').open('w',encoding='utf-8') as stdout, log.with_suffix('.stderr.log').open('w',encoding='utf-8') as stderr:
            process = subprocess.Popen(command,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
            self.state('p1_'+tag,child_pid=process.pid,stdout=str(log.with_suffix('.stdout.log')),stderr=str(log.with_suffix('.stderr.log')))
            while process.poll() is None:
                if editors():
                    raise RuntimeError('P1期间出现外部Editor，停止继续调度；当前P1保留完成证据')
                time.sleep(5)
            if process.returncode:
                raise RuntimeError('P1失败：{}；详见{}'.format(tag,log.with_suffix('.stderr.log')))

    def validate_freeze(self):
        receipt = read(self.private/'runtime_freeze.json')
        for relative, expected in receipt['files'].items():
            if sha(self.runtime/relative)!=expected:
                raise ValueError('冻结runtime发生变化：{}'.format(relative))
        if receipt.get('quality_policy_sha256') and sha(self.private/'quality_policy.json') != receipt['quality_policy_sha256']:
            raise ValueError('质量策略与用户授权的冻结记录不同')
        if sha(Path(self.settings['project'])/'Binaries/Win64/UnrealEditor-FutsalMOT.dll')!=receipt['module_dll_sha256']:
            raise ValueError('UE module DLL与验收版本不同')
        for relative, expected in receipt.get('asset_usage_sha256', {}).items():
            if sha(Path(self.settings['project']) / relative) != expected:
                raise ValueError('材质SkeletalMesh使用配置发生变化：{}'.format(relative))
        return receipt

    def control(self, client, action):
        dump(self.private/'control.json',{'action':action,'entry':self.current,'requested_utc':now()})
        result=client.run_file(self.runtime/'ue/production_control.py')
        dump(self.private/'evidence'/self.current['task_id']/(action+'_mcp.json'),result)
        return result

    def start_editor(self):
        if editors():
            raise RuntimeError('已有Editor，正式队列不抢占用户会话')
        assert_no_training()
        resources=memory()
        if resources['commit_headroom_bytes']<self.settings['minimum_commit_headroom_gib']*1024**3:
            raise RuntimeError('Editor冷启动前提交内存余量不足')
        if shutil.disk_usage('G:/').free<self.settings['g_free_reserve_gib']*1024**3:
            raise RuntimeError('G盘安全余量不足')
        receipt=self.validate_freeze()
        log=self.private/'evidence'/self.current['task_id']/'editor.log'
        log.parent.mkdir(parents=True,exist_ok=True)
        args=editor_command(self.settings,log)
        process=subprocess.Popen(args,creationflags=subprocess.CREATE_NO_WINDOW)
        self.state('editor_starting',editor_pid=process.pid,editor_log=str(log),memory=resources,
            frozen_dll=receipt['module_dll_sha256'])
        client=None
        # 仅启动探测可重试；任何写操作都不因超时重发。
        for _ in range(120):
            if process.poll() is not None:
                raise RuntimeError('Editor在MCP就绪前退出：{}'.format(process.returncode))
            try:
                client=UnrealMCP(timeout=20)
                break
            except (OSError,EOFError,http_error):
                time.sleep(5)
        if client is None:
            raise RuntimeError('MCP启动连接超时，保持Editor现场供诊断')
        client.timeout=600
        client.describe('editor_toolset.toolsets.scene.SceneTools')
        native_level=client.call('editor_toolset.toolsets.scene.SceneTools','get_current_level',{})
        client.describe('futsalmot_tools.FutsalMOTTools')
        client.describe('EditorToolset.EditorAppToolset')
        opened=client.call('EditorToolset.EditorAppToolset','GetOpenAssets',{})
        dump(log.parent/'startup_mcp.json',{'native_level':native_level,'open_assets':opened,
            'tool_descriptions':client.descriptions})
        if log.exists():
            text=log.read_text(encoding='utf-8',errors='replace')
            bad=('Still incompatible or missing module','Incompatible or missing module','different engine version')
            if any(token in text for token in bad):
                raise RuntimeError('本次启动日志存在module/engine不兼容')
        # P2脚本在新会话严格检查地图、十人、资产、dirty与完整环境基线。
        return process,client

    def render(self):
        process,client=self.start_editor()
        self.control(client,'render')
        self.state('rendering',editor_pid=process.pid,frames_per_camera=self.current['frames_per_camera'])
        episode=Path(self.current['episode'])
        cancelled=False
        start=time.monotonic()
        while True:
            if process.poll() is not None:
                raise RuntimeError('Editor渲染期间退出；不伪造成功/恢复证据，保留scratch')
            if time.monotonic()-start>12*3600:
                raise RuntimeError('单组渲染超过12小时，保留现场')
            summary_path=episode/'render_summary.json'
            if summary_path.exists():
                summary=read(summary_path)
                restored=True
                for kind in ('appearance','environment'):
                    path=episode/('provenance/'+kind+'_application.json')
                    report=read(path) if path.exists() else {}
                    if report.get('state')=='restore_failed' or report.get('errors'):
                        raise RuntimeError('{}恢复失败；保留Editor和全部证据'.format(kind))
                    restored=restored and report.get('state')=='restored' and not report.get('mrq_pending')
                if restored:
                    break
            resources=memory()
            if resources['commit_headroom_bytes']<self.settings['cancel_commit_headroom_gib']*1024**3 and not cancelled:
                self.control(client,'cancel')
                cancelled=True
            self.state('rendering',editor_pid=process.pid,memory=resources,cancellation_requested=cancelled,
                elapsed_seconds=round(time.monotonic()-start))
            time.sleep(30)
        self.control(client,'finalize')
        opened=client.call('EditorToolset.EditorAppToolset','GetOpenAssets',{})
        dump(self.private/'evidence'/self.current['task_id']/'open_assets_before_quit.json',opened)
        self.control(client,'quit')
        try:
            process.wait(180)
        except subprocess.TimeoutExpired:
            raise RuntimeError('正常退出未完成；不强制结束Editor')
        if editors():
            raise RuntimeError('仍有Editor，禁止P1')
        if summary['status']!='success':
            raise RuntimeError('本组渲染{}，已恢复并正常退出，未进入正式数据集'.format(summary['status']))

    def run(self):
        self.private.mkdir(parents=True,exist_ok=True)
        descriptor=os.open(str(self.lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        os.write(descriptor,json.dumps({'pid':os.getpid(),'created_utc':now()}).encode())
        os.close(descriptor)
        try:
            self.state('waiting_for_plan_and_legacy_cpu')
            while not (self.private/'production_plan.json').exists():
                if (self.private/'STOP_AFTER_CURRENT').exists():
                    self.state('stopped_by_user');return
                time.sleep(10)
            plan=read(self.private/'production_plan.json')
            if plan['settings_sha256']!=canonical_sha(self.settings):
                raise ValueError('正式清单与运行配置不同')
            while True:
                if (self.private/'STOP_AFTER_CURRENT').exists():
                    self.state('stopped_by_user');return
                statuses=[read(Path(self.settings['legacy_index']).parent/(task+'_continuation.json'))
                    for task in ('paper_4k_s1021_c000_q2','paper_4k_s1025_c000_ssd')]
                if any(record['status']=='failed_closed' for record in statuses):
                    raise RuntimeError('已渲染组的前置验收/发布失败；不能覆盖继续')
                if all(record['status']=='published_and_verified' for record in statuses):
                    break
                time.sleep(15)
            if editors():
                raise RuntimeError('前置CPU队列完成后出现外部Editor')
            self.validate_freeze()
            self.invoke('maintenance.py','clean-tests','--config',self.config,'--apply',tag='retired_test_cleanup')
            self.invoke('maintenance.py','adopt','--config',self.config,tag='adopt_verified_groups')
            self.invoke('maintenance.py','clean-adopted','--config',self.config,tag='adopted_scratch_cleanup')
            root=Path(self.settings['dataset_root'])
            recipe={'schema':'futsalmot_production_recipe','version':1,'release_id':self.settings['release_id'],
                'candidate_count':plan['candidate_count'],'selection':plan['selection'],
                'minimum_clip_frames':plan['minimum_clip_frames'],'fps':10,'resolution':[3840,2160],
                'planned_groups':len(plan['entries']),'plan_sha256':sha(self.private/'production_plan.json'),
                'runtime_freeze_sha256':sha(self.private/'runtime_freeze.json'),
                'runtime_source_sha256':read(self.private/'runtime_freeze.json')['files'],
                'motion_policy':plan['source_motion_policy'],'numbers_enabled':False,'pose_enabled':False,
                'legacy_generation_ref':'shared/legacy_generation_recipe.json',
                'reproducibility':'按root_seed采集完整1800帧源轨迹，核验源SHA，再按clip.source_selection重建连续窗及外观/天气；不同GPU/UE之间不保证PNG字节一致',
                'scheduling':'单组冷启动UE→恢复并退出→2-worker CPU后处理→按quality_policy检查/发布→清理；全部完成前不启动GPU训练'}
            policy_path=self.private/'quality_policy.json'
            if policy_path.exists():
                dump(root/'shared/quality_policy.json',read(policy_path))
                recipe['quality_policy_ref']='shared/quality_policy.json'
            dump(root/'shared/generation_recipe.json',recipe)
            dump(root/'splits.json',{split:[row['root_seed'] for row in plan['entries'] if row['split']==split] for split in ('train','val','test')})
            for row in plan['entries']:
                if (self.private/'STOP_AFTER_CURRENT').exists():
                    self.state('stopped_by_user');return
                canonical=root/'groups'/'seed_{:08d}'.format(row['root_seed'])/'clips/clip_000'
                if canonical.exists():
                    # 已有组必须对应本次合法采用证据，不根据目录存在跳过验证。
                    if row['root_seed'] not in self.settings['adopt_seeds'] and not (self.private/'published'/('fmot_v1_s{:08d}_c000_a01.json'.format(row['root_seed']))).exists():
                        raise RuntimeError('存在没有本次发布证据的同名组')
                    if row['root_seed'] not in self.settings['adopt_seeds']:
                        task_id='fmot_v1_s{:08d}_c000_a01'.format(row['root_seed'])
                        if not (self.private/'cleaned'/(task_id+'.json')).exists():
                            entry_path=self.private/'entries'/(task_id+'.json')
                            self.current=read(entry_path)
                            self.invoke('p1.py','clean','--config',self.config,'--entry',entry_path,tag='resume_clean_s{}'.format(row['root_seed']))
                    continue
                self.current=row
                row_path=self.private/'current_row.json';dump(row_path,row)
                self.invoke('p1.py','prepare','--config',self.config,'--row',row_path,tag='prepare_s{}'.format(row['root_seed']))
                entry_path=self.private/'entries'/('fmot_v1_s{:08d}_c000_a01.json'.format(row['root_seed']))
                self.current=read(entry_path)
                self.render()
                for action in ('postprocess','publish','clean'):
                    self.invoke('p1.py',action,'--config',self.config,'--entry',entry_path,tag=action+'_s{}'.format(row['root_seed']))
                self.state('group_complete',completed_counts=read(root/'dataset.json')['counts'])
            self.current=None
            index=read(root/'dataset.json')
            assert {group['root_seed'] for group in index['groups']} == {row['root_seed'] for row in plan['entries']}
            for row in plan['entries']:
                group_path=root/'groups'/'seed_{:08d}'.format(row['root_seed'])
                group=read(group_path/'group.json')
                assert group['split']==row['split'] and group['weather']['preset']==row['preset']
                clip_path=group_path/'clips/clip_000'
                clip=read(clip_path/'clip.json')
                assert clip['frame_count']==row['frames_per_camera'] and len(clip['cameras'])==4
                assert sha(clip_path/'clip.json')==read(clip_path/'checksums.json')['clip.json']
            expected_frames=sum(row['frames_per_camera']*4 for row in plan['entries'])
            assert index['counts']=={'seed_groups':len(plan['entries']),'clips':len(plan['entries']),
                'rgb_frames':expected_frames,'mask_frames':expected_frames}
            dump(self.private/'release_verified.json',{'verified_utc':now(),'counts':index['counts'],
                'plan_sha256':sha(self.private/'production_plan.json'),'all_groups_admitted':True,'gpu_training':False})
            self.state('completed',counts=index['counts'],planned_groups=len(plan['entries']))
        except Exception as error:
            self.state('failed_closed',error=str(error),traceback=traceback.format_exc())
            raise
        finally:
            self.lock.unlink()


# HTTP协议异常仅在初始化探测时可重试；写操作异常直接进入failed_closed。
from http.client import HTTPException as http_error

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    Driver(args.config).run()
