"""UE/P1共用纯Python场景收据；此模块不加载NumPy或GRF。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from grf_ue_bridge.scene_lifecycle import retained_completion, admission_has_completed_scene, file_sha
