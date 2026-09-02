"""配置包：数据集 task（单 config）、resolved task 解析。

兼容导出：原 `grf_ue_bridge.config` 模块的 `ExportConfig` 移入本包，
`from grf_ue_bridge.config import ExportConfig` 仍可用。
"""

from grf_ue_bridge.config.models import (
    AuditTaskConfig,
    DatasetTaskConfig,
    ExportConfig,
    LocalMachineConfig,
    PostprocessTaskConfig,
    ResolvedTask,
    TaskConfigV3,
    UeProfile,
)
from grf_ue_bridge.config import loader, paths, resolver  # noqa: F401

__all__ = [
    "AuditTaskConfig",
    "DatasetTaskConfig",
    "ExportConfig",
    "LocalMachineConfig",
    "PostprocessTaskConfig",
    "ResolvedTask",
    "TaskConfigV3",
    "UeProfile",
    "loader",
    "paths",
    "resolver",
]
