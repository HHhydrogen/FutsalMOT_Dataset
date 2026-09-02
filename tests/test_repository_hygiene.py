"""仓库卫生检查：根配置泄漏、单 config 自包含、scripts 已删除、命名规范。"""

from __future__ import annotations

import json
import re
from pathlib import Path


def _iter_files(root: Path) -> list:
    """遍历仓库文件，跳过 .git/.venv/.futsalmot/__pycache__ 等。"""
    out = []
    skip_parts = {".git", ".venv", ".external", ".pytest_cache", "__pycache__",
                  ".futsalmot", ".claude", ".superpowers", "dist", "build",
                  "outputs"}  # outputs/ 为生成数据，不纳入卫生检查
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        parts = set(p.relative_to(root).parts)
        if parts & skip_parts:
            continue
        out.append(p)
    return out


class TestRootHygiene:
    def test_no_root_ue_import_config(self, repo_root):
        assert list(repo_root.glob("ue_import_config*.json")) == []

    def test_no_root_machine_config(self, repo_root):
        # 机器路径已直接写进 configs/，根目录不再有独立本机路径配置
        assert not (repo_root / "ue_import_config.json").exists()
        assert not (repo_root / ".futsalmot.local.json").exists()

    def test_no_mask_rendering_status(self, repo_root):
        assert not (repo_root / "MASK_RENDERING_STATUS.md").exists()

    def test_no_90fps_misnomer(self, repo_root):
        for p in _iter_files(repo_root):
            assert "90fps" not in p.name, f"错误命名 90fps: {p}"

    def test_gitignore_has_runtime(self, repo_root):
        gi = (repo_root / ".gitignore").read_text(encoding="utf-8")
        assert ".futsalmot/" in gi
        # 本地路径已入库，不再忽略本地配置
        assert ".futsalmot.local.json" not in gi
        assert "*.local.json" not in gi


class TestDatasetConfigs:
    def test_dataset_configs_self_contained(self, repo_root):
        """configs/*.json 必须自包含：无 profile 引用，含绝对机器路径。example.json 为占位符模板，跳过。"""
        for p in repo_root.glob("configs/*.json"):
            if p.name in ("example.json", "local.machine.example.json"):
                continue  # 模板用 <...> 占位符路径
            data = json.loads(p.read_text(encoding="utf-8"))
            assert data.get("schema") == "futsalmot_dataset_task"
            assert data.get("version") == 2
            # 不再引用独立 profile / paths 覆盖
            assert "export_profile" not in data
            assert "ue_profile" not in data
            assert "paths" not in data
            # 机器路径必填且为绝对路径（盘符/UNC）
            ds = data.get("dataset_root") or ""
            ue = data.get("ue_project_root") or ""
            assert re.match(r"^[A-Za-z]:[/\\\\]", ds), f"dataset_root 非绝对路径: {p}"
            assert re.match(r"^[A-Za-z]:[/\\\\]", ue), f"ue_project_root 非绝对路径: {p}"
            # 内联 export / ue 块
            assert "export" in data and data["export"].get("scenario")
            assert "ue" in data and data["ue"].get("annotation_export")

    def test_dataset_configs_parse(self, repo_root):
        from grf_ue_bridge.config import loader
        for tf in repo_root.glob("configs/*.json"):
            if tf.name == "local.machine.example.json":
                continue
            loader.load_task_config(tf)  # 可解析即可


class TestNoDeprecatedScripts:
    def test_scripts_dir_removed(self, repo_root):
        # scripts/ 薄包装已删除（正式实现在包内），不应再出现
        assert not list(repo_root.glob("scripts/*.py"))

    def test_formal_logic_in_package(self, repo_root):
        assert (repo_root / "src" / "grf_ue_bridge" / "workflows" / "task_audit.py").is_file()
        assert (repo_root / "src" / "grf_ue_bridge" / "tools" / "resource_monitor.py").is_file()
        assert (repo_root / "src" / "grf_ue_bridge" / "tools" / "process_measure.py").is_file()
