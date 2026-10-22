"""持久UE会话正式调度：逐机位MRQ→CPU后处理/发布→下一seed，不自动训练。"""
import argparse
import os
from pathlib import Path
import subprocess
import shutil
import time
import traceback
import psutil
from common import canonical_sha, dump, now, read, sha, task_id_for_row
from driver import Driver, assert_no_training, editors, memory
from mcp_client import UnrealMCP


class RetainedDriver(Driver):
    def cpu(self, action, *, entry=None, row=None):
        """仅允许本调度器已核验且空闲的Editor并存；P1不使用GPU。"""
        assert_no_training()
        log = self.private / 'logs' / (self.current['task_id'] + '_r5_' + action)
        log.parent.mkdir(parents=True, exist_ok=True)
        args = [self.settings['python'], '-X', 'utf8', str(self.runtime / 'tools/production/p1.py'), action,
            '--config', str(self.config)]
        if entry is not None:
            args += ['--entry', str(entry)]
        if row is not None:
            args += ['--row', str(row)]
        with log.with_suffix('.stdout.log').open('w', encoding='utf-8') as stdout, log.with_suffix('.stderr.log').open('w', encoding='utf-8') as stderr:
            process = subprocess.Popen(args, stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
            self.state('p1_' + action, child_pid=process.pid, scene_lifecycle='retain', editor_pid=self.editor_pid)
            while process.poll() is None:
                if any(p.pid != self.editor_pid for p in editors()):
                    raise RuntimeError('出现本会话之外的Editor，停止继续调度')
                time.sleep(3)
            if process.returncode:
                raise RuntimeError('P1 {} 失败，详见{}'.format(action, log.with_suffix('.stderr.log')))

    def session_control(self, action):
        dump(self.private / 'session_control.json', {'action': action, 'entry': self.current, 'requested_utc': now()})
        result = self.client.run_file(self.runtime / 'ue/production_session_control.py')
        dump(self.private / 'evidence' / self.current['task_id'] / (action + '_r5_mcp.json'), result)
        return result

    def connect_owned_editor(self):
        self.validate_freeze()
        existing = editors()
        ownership = self.private / 'session_editor.json'
        if existing:
            proof = read(ownership)
            assert [p.pid for p in existing] == [proof['editor_pid']], '不得接管其他Editor'
            process = psutil.Process(proof['editor_pid'])
            assert abs(process.create_time() - proof['process_created_unix']) < .001
            assert str(Path(process.exe()).resolve()).lower() == str(Path(self.settings['editor']).resolve()).lower()
            assert any(str(Path(token).resolve()).lower() == str(Path(self.settings['uproject']).resolve()).lower()
                for token in process.cmdline() if token.lower().endswith('.uproject'))
            self.editor_pid = process.pid
            self.client = UnrealMCP(timeout=600)
        else:
            process, self.client = self.start_editor()
            self.editor_pid = process.pid
            dump(ownership, {'editor_pid': process.pid, 'process_created_unix': psutil.Process(process.pid).create_time(),
                'recorded_utc': now(), 'runtime_freeze_sha256': sha(self.private / 'runtime_freeze.json')})
        self.client.describe('futsalmot_tools.FutsalMOTTools')
        self.client.describe('editor_toolset.toolsets.scene.SceneTools')
        level = self.client.call('editor_toolset.toolsets.scene.SceneTools', 'get_current_level', {})
        dump(self.private / 'evidence' / self.current['task_id'] / 'retained_editor_connection.json',
            {'native_level': level, 'editor_pid': self.editor_pid, 'verified_utc': now()})

    def render_retained(self):
        episode = Path(self.current['episode'])
        assert not (episode / 'render_summary.json').exists(), '未完成attempt须独立诊断，不得自动覆盖'
        self.session_control('render')
        progress_seen = 0
        start = time.monotonic()
        while not (episode / 'provenance/scene_completion.json').exists():
            if not psutil.pid_exists(self.editor_pid):
                raise RuntimeError('Editor退出，保留未完成数据')
            error_path = episode / 'provenance/session_error.json'
            if error_path.exists():
                raise RuntimeError('场景会话失败：' + error_path.read_text(encoding='utf-8'))
            resources = memory()
            if resources['commit_headroom_bytes'] < self.settings['cancel_commit_headroom_gib'] * 1024**3:
                self.session_control('cancel')
                raise RuntimeError('已因提交内存保护取消当前MRQ；保留场景和证据，不启动下一组')
            progress = episode / 'provenance/camera_progress.json'
            if progress.exists() and read(progress)['completed'] > progress_seen:
                progress_seen = read(progress)['completed']
                self.session_control('continue')
            self.state('rendering', editor_pid=self.editor_pid, completed_cameras=progress_seen,
                memory=resources, elapsed_seconds=round(time.monotonic()-start), scene_lifecycle='retain')
            assert time.monotonic() - start < 12 * 3600
            time.sleep(5)
        self.session_control('collect')

    def run_retained(self):
        descriptor = os.open(str(self.lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(descriptor, str(os.getpid()).encode()); os.close(descriptor)
        self.editor_pid = None
        try:
            receipt = self.validate_freeze()
            policy = read(self.private / 'execution_policy.json')
            assert sha(self.private / 'execution_policy.json') == receipt['execution_policy_sha256']
            assert policy['scene_lifecycle'] == 'retain' and policy['camera_batch_size'] == 1
            plan = read(self.private / 'production_plan.json')
            assert plan['settings_sha256'] == canonical_sha(self.settings)
            assert_no_training()
            for row in plan['entries']:
                self.validate_freeze()
                task_id = task_id_for_row(row)
                if any(p.exists() and read(p).get('verified') is True for p in [self.private / 'published' / (task_id + '.json')]):
                    continue
                # 兼容两组迁入的历史task；只认正式版本索引中实际存在的seed。
                index = read(Path(self.settings['dataset_root']) / 'dataset.json')
                if any(group.get('root_seed') == row['root_seed'] for group in index.get('groups', [])):
                    continue
                if (self.private / 'STOP_AFTER_CURRENT').exists():
                    self.state('stopped_by_user', scene_lifecycle='retain'); return
                self.current = dict(row, task_id=task_id)
                if self.editor_pid is None:
                    self.connect_owned_editor()
                self.session_control('collect')
                if memory()['commit_headroom_bytes'] < self.settings['minimum_commit_headroom_gib'] * 1024**3:
                    raise RuntimeError('空闲释放后提交内存余量仍不足；保留场景，不启动新seed')
                frames = row['frames_per_camera']
                ssd_needed = frames * 220000000 + self.settings['d_free_reserve_gib'] * 1024**3
                all_g = shutil.disk_usage('D:/').free < ssd_needed
                g_needed = (self.settings['g_free_reserve_gib'] * 1024**3
                    + frames * 4 * (self.settings['estimated_bytes_per_rgb_mask'] + (50000000 if all_g else 0)))
                if shutil.disk_usage('G:/').free < g_needed:
                    raise RuntimeError('G盘不足以保留安全余量并完成本组，停止启动新seed')
                row_path = self.private / 'rows' / (task_id + '.json')
                dump(row_path, row)
                self.cpu('prepare', row=row_path)
                entry_path = self.private / 'entries' / (task_id + '.json')
                self.current = read(entry_path)
                self.render_retained()
                for action in ('postprocess', 'publish', 'clean'):
                    self.cpu(action, entry=entry_path)
                self.state('published_and_cleaned', editor_pid=self.editor_pid, scene_lifecycle='retain')
            index = read(Path(self.settings['dataset_root']) / 'dataset.json')
            expected = sum(row['frames_per_camera'] * 4 for row in plan['entries'])
            assert index['counts']['rgb_frames'] == index['counts']['mask_frames'] == expected
            assert index['counts']['seed_groups'] == len(plan['entries'])
            dump(self.private / 'release_verified.json', {'finished_utc': now(), 'counts': index['counts'],
                'scene_lifecycle': 'retain', 'editor_pid': self.editor_pid, 'gpu_training': False})
            self.state('release_complete', editor_pid=self.editor_pid, scene_lifecycle='retain')
        except BaseException as exc:
            self.state('failed_closed', error=str(exc), traceback=traceback.format_exc(), scene_lifecycle='retain', editor_pid=self.editor_pid)
            raise
        finally:
            self.lock.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    RetainedDriver(parser.parse_args().config).run_retained()
