"""数值训练代码不变的Windows文件锁重试入口；只包装指定输出的原子替换。"""
import json
import os
from pathlib import Path
import runpy
import sys
import time

DELAYS = (0.05, 0.1, 0.2, 0.4, 0.8)


def retry_replace(original, source, target, output, *, sleep=time.sleep, report=None):
    """只重试本任务临时文件的WinError5/32；不删除目标或绕过权限。"""
    source, target, output = Path(source), Path(target), Path(output).resolve()
    owned = (target.parent.resolve() == output and source.parent.resolve() == output
             and target.name in ('last.pt', 'update_transaction.json', 'run_receipt.json')
             and source.name == target.name + '.tmp')
    for attempt in range(len(DELAYS) + 1):
        try:
            return original(source, target)
        except PermissionError as error:
            if not owned or getattr(error, 'winerror', None) not in (5, 32) or attempt == len(DELAYS):
                raise
            if report:
                report({'operation': 'atomic_replace_retry', 'target': target.name,
                        'winerror': error.winerror, 'retry': attempt + 1, 'delay_seconds': DELAYS[attempt]})
            sleep(DELAYS[attempt])


def main():
    arguments = sys.argv[1:]
    if arguments and arguments[0] == '--':
        arguments = arguments[1:]
    if arguments.count('--output') != 1:
        raise ValueError('必须明确唯一--output；其他参数完整传给冻结Trainer')
    output = Path(arguments[arguments.index('--output') + 1]).resolve()
    trainer = Path(__file__).with_name('train_pose_location_control.py')
    original = os.replace
    def report(value):
        print(json.dumps(value, ensure_ascii=False), flush=True)
    def wrapped(source, target, *args, **kwargs):
        if args or kwargs:
            return original(source, target, *args, **kwargs)
        return retry_replace(original, source, target, output, report=report)
    os.replace = wrapped
    previous = sys.argv
    try:
        sys.argv = [str(trainer), *arguments]
        runpy.run_path(str(trainer), run_name='__main__')
    finally:
        sys.argv = previous
        os.replace = original


if __name__ == '__main__':
    main()
