"""pytest 共享配置：把仓库根下的 ue/ 目录加入 sys.path。

ue/ 下的纯模块（camera_projection / annotation_utils / dataset_export）不依赖
unreal/numpy，可被普通 pytest 直接测试。
"""

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_UE_DIR = _REPO_ROOT / "ue"
if str(_UE_DIR) not in sys.path:
    sys.path.insert(0, str(_UE_DIR))


@pytest.fixture
def repo_root() -> Path:
    """仓库根目录。"""
    return _REPO_ROOT


@pytest.fixture
def pin_repo_root(monkeypatch, tmp_path):
    """把 default_repo_root 固定到 tmp_path/repo，隔离运行时路径（resolved task/active task）。"""
    from grf_ue_bridge.config import paths

    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    mapping = repo / "ue" / "actor_mapping.example.json"
    mapping.parent.mkdir(parents=True, exist_ok=True)
    mapping.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(paths, "default_repo_root", lambda: repo)
    return repo
