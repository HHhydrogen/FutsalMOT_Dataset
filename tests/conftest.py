"""pytest 共享配置：把仓库根下的 ue/ 目录加入 sys.path。

ue/ 下的纯模块（camera_projection / annotation_utils / dataset_export）不依赖
unreal/numpy，可被普通 pytest 直接测试。
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_UE_DIR = _REPO_ROOT / "ue"
if str(_UE_DIR) not in sys.path:
    sys.path.insert(0, str(_UE_DIR))
