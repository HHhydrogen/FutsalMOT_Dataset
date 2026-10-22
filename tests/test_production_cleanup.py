"""删除只发生在指定工作树内，junction的外部目标必须保留。"""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import pytest

folder=Path(__file__).resolve().parents[1]/'tools/production'
sys.path.insert(0,str(folder))
from p1 import remove_tree


def test_reject_root_and_outside(tmp_path):
    owned=tmp_path/'owned';owned.mkdir()
    outside=tmp_path/'outside';outside.mkdir()
    for path in (owned,outside):
        with pytest.raises(ValueError,match='边界'):
            remove_tree(path,owned)
        assert path.exists()


@pytest.mark.skipif(os.name!='nt',reason='Windows junction边界')
def test_remove_junction_never_follows_external_target(tmp_path):
    owned=tmp_path/'owned';owned.mkdir()
    outside=tmp_path/'outside';outside.mkdir()
    sentinel=outside/'keep.txt';sentinel.write_text('keep')
    episode=owned/'episode';episode.mkdir()
    link=episode/'render_mask'
    def q(path):return "'"+str(path).replace("'","''")+"'"
    subprocess.run(['powershell','-NoProfile','-Command',
        "$ErrorActionPreference='Stop'; New-Item -ItemType Junction -Path "+q(link)+' -Target '+q(outside)+' | Out-Null'],check=True)
    remove_tree(episode,owned)
    assert sentinel.read_text()=='keep'
    assert not episode.exists()
