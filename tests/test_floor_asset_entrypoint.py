"""可分发地板入口的无UE加载和路径边界。"""

import runpy
from pathlib import Path


def load_entry():
    return runpy.run_path(str(Path(__file__).resolve().parents[1] / "ue/create_floor_assets_v1.py"))


def test_distributable_source_paths_do_not_depend_on_saved():
    entry = load_entry()
    image, receipt = entry["source_paths"](Path("portable_project"))
    assert image == Path("portable_project/SourceAssets/Floors/V1/T_FloorNeutral_Albedo_v1.png")
    assert receipt == Path("portable_project/SourceAssets/Floors/V1/source_receipt.json")


def test_target_paths_are_explicit_unique_and_owned():
    entry = load_entry()
    paths = entry["target_paths"]()
    assert len(paths) == len(set(paths)) == 6
    assert all(p.startswith("/Game/FutsalMOT/Environment/Floors/V1/") for p in paths)


def test_entry_does_not_import_unreal_until_main():
    # run_path能够在无UE的P1环境完整加载入口，验证两个Python环境隔离。
    entry = load_entry()
    assert callable(entry["main"])
    assert "unreal" not in entry
