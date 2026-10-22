"""串行调度在出现外部Editor/GPU任务或冻结代码漂移时必须拒绝启动。"""
from pathlib import Path
from types import SimpleNamespace
import re
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/production'))
import driver
from common import dump, sha


def test_reject_research_training(monkeypatch):
    monkeypatch.setattr(driver.psutil,'process_iter',lambda fields:[SimpleNamespace(info={
        'name':'python.exe','cmdline':['python','D:/projects/FutsalMOT_Research/train_yolo.py']})])
    with pytest.raises(RuntimeError,match='GPU任务'):
        driver.assert_no_training()


def test_external_editor_blocks_cold_launch(monkeypatch):
    value=object.__new__(driver.Driver)
    monkeypatch.setattr(driver,'editors',lambda:[object()])
    with pytest.raises(RuntimeError,match='已有Editor'):
        value.start_editor()


def test_frozen_runtime_drift_rejected(tmp_path):
    value=object.__new__(driver.Driver)
    value.private=tmp_path
    value.runtime=tmp_path/'runtime';value.runtime.mkdir()
    code=value.runtime/'code.py';code.write_text('before')
    dump(tmp_path/'runtime_freeze.json',{'files':{'code.py':sha(code)}})
    code.write_text('after')
    with pytest.raises(ValueError,match='冻结runtime发生变化'):
        value.validate_freeze()


def test_unreal_exec_value_quotes_cover_both_commands():
    settings=dict(editor='E:/UE_5.8/UnrealEditor.exe',uproject='D:/Project Folder/Test.uproject',
        map='/Game/Maps/Court',image_write_concurrency=4,image_write_queue_size=2)
    command=driver.editor_command(settings,'D:/Project Folder/editor.log')
    assert re.search(r'-ExecCmds="([^"]+)"',command).group(1)=='ImageWriteQueue.MaxConcurrency 4,ImageWriteQueue.MaxQueueSize 2'
    assert '"D:/Project Folder/Test.uproject"' in command
    assert '-LogCmds="LogModuleManager Verbose"' in command
