"""仅验证文件锁边界；不训练、不改真实检查点。"""
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('pose_windows_retry', Path(__file__).resolve().parents[1] /
    'experiments/sim2real/run_pose_location_control_windows.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class AtomicRetryTests(unittest.TestCase):
    def paths(self, root, name='last.pt'):
        return root / (name + '.tmp'), root / name

    def test_transient_lock_then_actual_atomic_replace(self):
        import os
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = self.paths(root)
            source.write_bytes(b'new-complete-checkpoint');target.write_bytes(b'old-checkpoint')
            calls = []
            def replace(a, b):
                calls.append(1)
                if len(calls) == 1:
                    error = PermissionError('locked');error.winerror = 5;raise error
                return os.replace(a, b)
            sleeps = []
            module.retry_replace(replace, source, target, root, sleep=sleeps.append)
            self.assertEqual(target.read_bytes(), b'new-complete-checkpoint')
            self.assertFalse(source.exists());self.assertEqual(sleeps, [.05])

    def test_exhausted_lock_keeps_source_and_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);source, target = self.paths(root)
            source.write_bytes(b'new');target.write_bytes(b'old')
            def locked(*args):
                error = PermissionError('sharing');error.winerror = 32;raise error
            sleeps = []
            with self.assertRaises(PermissionError):
                module.retry_replace(locked, source, target, root, sleep=sleeps.append)
            self.assertEqual(len(sleeps), 5)
            self.assertEqual(source.read_bytes(), b'new');self.assertEqual(target.read_bytes(), b'old')

    def test_other_permission_error_not_retried(self):
        calls = []
        def fail(*args):
            calls.append(1);raise PermissionError('real access policy')
        with self.assertRaises(PermissionError):
            module.retry_replace(fail, Path('/tmp/last.pt.tmp'), Path('/tmp/last.pt'), Path('/tmp'), sleep=lambda _:None)
        self.assertEqual(len(calls), 1)

    def test_outside_owned_output_not_retried(self):
        calls = []
        def fail(*args):
            calls.append(1);error = PermissionError('locked');error.winerror = 5;raise error
        with self.assertRaises(PermissionError):
            module.retry_replace(fail, Path('/tmp/elsewhere/last.pt.tmp'), Path('/tmp/elsewhere/last.pt'), Path('/tmp/owned'), sleep=lambda _:None)
        self.assertEqual(len(calls), 1)
