"""PowerShell 计划和串行门禁只使用临时路径、假 UE 进程和无效 Python 标记。"""
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT=Path(__file__).with_name('run_serial_pilot.ps1').resolve()
PWSH=shutil.which('pwsh')
pytestmark=pytest.mark.skipif(not PWSH,reason='本机没有 PowerShell 7')


def test_default_plan_prints_commands_without_running_any_python():
    result=subprocess.run([PWSH,'-NoProfile','-File',str(SCRIPT),'-Python','PYTHON_MUST_NOT_RUN'],
                          capture_output=True,text=True,encoding='utf-8')
    assert result.returncode==0,result.stderr
    lines=[line for line in result.stdout.splitlines() if line.startswith('PYTHON_MUST_NOT_RUN ')]
    assert len(lines)==16
    assert 'verify-protocol' in lines[0]
    assert 'verify-arms' in lines[4]
    assert sum(' train ' in line for line in lines)==1
    assert all('--split test' not in line for line in lines)
    assert sum(' evaluate ' in line and '--device 0' in line for line in lines)==3


@pytest.mark.parametrize('phase', ['Gpu', 'All'])
def test_live_ue_blocks_gpu_phase_before_python_is_invoked(phase):
    escaped=str(SCRIPT).replace("'","''")
    command=("function global:Get-Process { [pscustomobject]@{ Id=123; ProcessName='UnrealEditor' } }; "
             "try { & '"+escaped+"' -Phase "+phase+" -Python PYTHON_MUST_NOT_RUN; exit 9 } "
             "catch { Write-Output $_.Exception.Message; exit 0 }")
    result=subprocess.run([PWSH,'-NoProfile','-Command',command],capture_output=True,text=True,encoding='utf-8')
    assert result.returncode==0,result.stderr
    assert 'UE' in result.stdout and 'PYTHON_MUST_NOT_RUN' not in result.stdout
