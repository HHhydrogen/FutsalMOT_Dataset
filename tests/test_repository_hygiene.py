"""仓库卫生检查：根配置泄漏、绝对路径、scripts 已删除、命名规范。"""

from __future__ import annotations

import re
from pathlib import Path


def _iter_files(root: Path) -> list:
    """遍历仓库文件，跳过 .git/.venv/.futsalmot/__pycache__ 等。"""
    out = []
    skip_parts = {".git", ".venv", ".external", ".pytest_cache", "__pycache__",
                  ".futsalmot", ".claude", ".superpowers", "dist", "build",
                  "outputs", "tasks"}  # tasks/ 为本地用户任务，不纳入卫生检查
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
        # 根目录不再有本机路径配置（.futsalmot.local.json 可能被忽略，但不应入库）
        assert not (repo_root / "ue_import_config.json").exists()

    def test_no_mask_rendering_status(self, repo_root):
        assert not (repo_root / "MASK_RENDERING_STATUS.md").exists()

    def test_no_90fps_misnomer(self, repo_root):
        for p in _iter_files(repo_root):
            assert "90fps" not in p.name, f"错误命名 90fps: {p}"

    def test_gitignore_has_local_and_runtime(self, repo_root):
        gi = (repo_root / ".gitignore").read_text(encoding="utf-8")
        assert ".futsalmot.local.json" in gi
        assert ".futsalmot/" in gi


class TestConfigNoAbsolutePaths:
    def test_configs_no_drive_paths(self, repo_root):
        # 正式配置（export/ue/tasks，不含 example/local）不得含盘符绝对路径
        pat = re.compile(r"[A-Za-z]:[/\\]")
        for p in repo_root.glob("configs/**/*.json"):
            if ".example." in p.name or "local" in p.name:
                continue
            text = p.read_text(encoding="utf-8")
            assert not pat.search(text), f"配置含盘符绝对路径: {p}"

    def test_example_task_no_absolute(self, repo_root):
        from grf_ue_bridge.config import loader
        for tf in repo_root.glob("configs/tasks/*.example.json"):
            loader.load_task_config(tf)  # 可解析即可


class TestNoDeprecatedScripts:
    def test_scripts_dir_removed(self, repo_root):
        # scripts/ 薄包装已删除（正式实现在包内），不应再出现
        assert not list(repo_root.glob("scripts/*.py"))

    def test_formal_logic_in_package(self, repo_root):
        assert (repo_root / "src" / "grf_ue_bridge" / "workflows" / "task_audit.py").is_file()
        assert (repo_root / "src" / "grf_ue_bridge" / "tools" / "resource_monitor.py").is_file()
        assert (repo_root / "src" / "grf_ue_bridge" / "tools" / "process_measure.py").is_file()
