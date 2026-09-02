"""公开 Config v3 文档示例与当前 schema 的一致性检查。"""

from __future__ import annotations

import json
import re
from pathlib import Path

from grf_ue_bridge.config.models import TaskConfigV3


DOCS = (
    "README.md",
    "configs/README.md",
    "CLAUDE.md",
    "docs/REPRODUCIBILITY_AND_MANIFEST.md",
)


def _v3_examples(text: str):
    for match in re.finditer(r"```json\s*(.*?)\s*```", text, re.DOTALL):
        snippet = match.group(1).strip()
        if '"schema": "futsalmot_task"' not in snippet:
            continue
        data = json.loads(snippet)
        if data.get("schema") == "futsalmot_task" and data.get("version") == 3:
            yield data


def test_documented_v3_examples_validate_and_use_explicit_camera_mapping(repo_root):
    examples = []
    for relative_path in DOCS:
        examples.extend(_v3_examples((repo_root / relative_path).read_text(encoding="utf-8")))

    assert examples
    for example in examples:
        task = TaskConfigV3.model_validate(example)
        assert task.cameras == {"C03": "FrontCamera", "C07": "RearCamera"}


def test_v3_documentation_does_not_show_legacy_postprocess_config(repo_root):
    for relative_path in DOCS:
        text = (repo_root / relative_path).read_text(encoding="utf-8")
        for example in _v3_examples(text):
            assert "postprocess" not in example

    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    pose_section = readme.split("### COCO 17 点定义", 1)[0]
    assert '"postprocess"' not in pose_section
