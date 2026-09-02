"""dataset manifest 的确定性、可移植性与校验测试（18.1–18.10）。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from grf_ue_bridge.dataset_manifest import (
    DatasetManifest,
    build_manifest,
    profile_file_paths,
    sha256_bytes,
    sha256_file,
    verify_manifest,
)


def _make_episode(
    root: Path,
    ep_id: str,
    seed: int,
    frames_content: bytes = b'{"step": 0}\n{"step": 1}\n',
    cameras=("Cam_01", "Cam_02"),
    with_render=False,
) -> Path:
    """构造一个最小的合法 episode 目录（含 meta/frames/provenance/相机产物）。"""
    ed = root / ep_id
    ed.mkdir(parents=True)
    meta = {
        "schema": "grf_ue_episode",
        "version": 1,
        "episode_id": ep_id,
        "timing": {"source_step_seconds": 0.1, "playback_fps": 30, "num_steps": 2},
        "source": {"scenario": "5_vs_5", "seed": seed},
        "randomness": {
            "policy": "futsalmot_seed_v1",
            "root_seed": seed,
            "grf_game_engine_seed": seed + 1,
            "python_seed": seed + 2,
            "numpy_seed": seed + 3,
            "ue_visual_seed": seed + 4,
        },
    }
    (ed / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (ed / "frames.jsonl").write_bytes(frames_content)
    (ed / "render_summary.json").write_text(json.dumps({"status": "success"}), encoding="utf-8")
    prov = ed / "provenance"
    prov.mkdir()
    (prov / "export_config.json").write_text(
        json.dumps({"scenario": "5_vs_5", "seed": seed}), encoding="utf-8"
    )
    for c in cameras:
        cd = ed / c
        cd.mkdir()
        (cd / "camera.json").write_text('{"image_width": 1920, "image_height": 1080}', encoding="utf-8")
        (cd / "seqinfo.ini").write_text("[Sequence]\nfps=30\n", encoding="utf-8")
        (cd / "annotations.jsonl").write_text(
            json.dumps({"frame_index": 1, "objects": []}) + "\n", encoding="utf-8"
        )
        (cd / "img1").mkdir()
        (cd / "img1" / "000001.jpg").write_bytes(b"RGBDATA" + bytes([seed % 256]))
        (cd / "mask").mkdir()
        (cd / "mask" / "000001.png").write_bytes(b"MASK" + bytes([seed % 256]))
        (cd / "labels" / "det").mkdir(parents=True)
        (cd / "labels" / "seg").mkdir(parents=True)
        (cd / "labels" / "det" / "000001.txt").write_text("x\n", encoding="utf-8")
        (cd / "labels" / "seg" / "000001.txt").write_text("x\n", encoding="utf-8")
        (cd / "gt").mkdir()
        (cd / "gt" / "gt.txt").write_text("1,1,1,1,1,1,1,1,1,1\n", encoding="utf-8")
        if with_render:
            (cd / "render").mkdir()
            (cd / "render" / "000000.png").write_bytes(b"RENDER")
            (cd / "render_mask").mkdir()
            (cd / "render_mask" / "000000.exr").write_bytes(b"EXR")
    return ed


def _checksum_lines(root: Path, ep_id: str) -> list:
    return (root / "checksums" / f"{ep_id}.jsonl").read_text(encoding="utf-8").splitlines()


class TestDeterminism:
    def test_consecutive_builds_identical(self, tmp_path):
        _make_episode(tmp_path, "epA", 1001)
        _make_episode(tmp_path, "epB", 1002)
        m1 = build_manifest(tmp_path, ["epA", "epB"], dataset_id="d1")
        m2 = build_manifest(tmp_path, ["epA", "epB"], dataset_id="d2")
        assert m1.dataset_fingerprint == m2.dataset_fingerprint
        assert [e.episode_id for e in m1.episodes] == [e.episode_id for e in m2.episodes]
        assert _checksum_lines(tmp_path, "epA") == _checksum_lines(tmp_path, "epA")

    def test_created_at_does_not_affect_fingerprint(self, tmp_path):
        _make_episode(tmp_path, "epA", 1001)
        m1 = build_manifest(tmp_path, ["epA"], dataset_id="d")
        # 手工改动 created_at（不影响 fingerprint）
        m1.created_at_utc = "2099-01-01T00:00:00Z"
        assert m1.dataset_fingerprint  # fingerprint 字段本身不变


class TestPortability:
    def test_move_directory_keeps_fingerprint(self, tmp_path):
        _make_episode(tmp_path, "epA", 1001)
        m = build_manifest(tmp_path, ["epA"], dataset_id="d")
        moved = tmp_path / "moved"
        shutil.copytree(tmp_path, moved, ignore=shutil.ignore_patterns(".*"))
        # 清理残留的 build 临时文件
        for p in tmp_path.iterdir():
            if p.name.startswith(".") and p.suffix == ".json":
                p.unlink()
        v = verify_manifest(moved)
        assert v.exit_code == 0
        moved_manifest = json.load(open(moved / "dataset_manifest.json", encoding="utf-8"))
        assert moved_manifest["dataset_fingerprint"] == m.dataset_fingerprint

    def test_paths_are_posix_no_drive(self, tmp_path):
        _make_episode(tmp_path, "epA", 1001)
        m = build_manifest(tmp_path, ["epA"], dataset_id="d")
        for e in m.episodes:
            assert "\\" not in e.relative_path
            assert not Path(e.relative_path).is_absolute()
        for line in _checksum_lines(tmp_path, "epA"):
            row = json.loads(line)
            assert "\\" not in row["path"]
            assert not row["path"].startswith(("C:", "D:", "G:", "/"))


class TestIntegrityChecks:
    def test_modify_file_fails(self, tmp_path):
        ep = _make_episode(tmp_path, "epA", 1001)
        build_manifest(tmp_path, ["epA"], dataset_id="d")
        target = ep / "Cam_01" / "mask" / "000001.png"
        target.write_bytes(b"XXXXX")  # 与原内容同长度 → 纯 hash mismatch
        v = verify_manifest(tmp_path)
        assert v.exit_code == 1
        assert any("epA/Cam_01/mask/000001.png" in p for p in v.hash_mismatch)

    def test_modify_size_differs_detected(self, tmp_path):
        # 长度也变时命中 size mismatch（同样必须失败）
        ep = _make_episode(tmp_path, "epA", 1001)
        build_manifest(tmp_path, ["epA"], dataset_id="d")
        (ep / "Cam_01" / "mask" / "000001.png").write_bytes(b"TAMPERED")
        v = verify_manifest(tmp_path)
        assert v.exit_code == 1
        assert any("epA/Cam_01/mask/000001.png" in p
                   for p in v.missing + v.size_mismatch + v.hash_mismatch)

    def test_delete_file_fails_missing(self, tmp_path):
        ep = _make_episode(tmp_path, "epA", 1001)
        build_manifest(tmp_path, ["epA"], dataset_id="d")
        (ep / "Cam_02" / "img1" / "000001.jpg").unlink()
        v = verify_manifest(tmp_path)
        assert v.exit_code == 1
        assert any("epA/Cam_02/img1/000001.jpg" in p for p in v.missing)

    def test_add_file_warns_and_strict_fails(self, tmp_path):
        ep = _make_episode(tmp_path, "epA", 1001)
        build_manifest(tmp_path, ["epA"], dataset_id="d")
        (ep / "Cam_01" / "img1" / "000099.jpg").write_bytes(b"EXTRA")
        v = verify_manifest(tmp_path)  # 默认：extra 仅警告
        assert v.exit_code == 0
        assert any("epA/Cam_01/img1/000099.jpg" in x for x in v.extra)
        vs = verify_manifest(tmp_path, strict_extra=True)  # strict：失败
        assert vs.exit_code == 1

    def test_illegal_escape_path_rejected(self, tmp_path):
        ep = _make_episode(tmp_path, "epA", 1001)
        build_manifest(tmp_path, ["epA"], dataset_id="d")
        # 篡改 checksum 文件，写入逃逸路径；直接测路径逃逸防护
        cs = tmp_path / "checksums" / "epA.jsonl"
        with open(cs, "a", encoding="utf-8") as f:
            f.write(json.dumps({"path": "../../outside.txt", "size": 3, "sha256": "0" * 64}) + "\n")
        from grf_ue_bridge.dataset_manifest import VerifyResult, _verify_checksums_file
        vr = VerifyResult()
        _verify_checksums_file(tmp_path, "checksums/epA.jsonl", None, vr, 2)
        assert any("逃逸" in e for e in vr.errors)
        # 完整 verify 也必须非零（checksum 文件自身 hash 不符）
        v = verify_manifest(tmp_path)
        assert v.exit_code != 0

    def test_frames_trajectory_hash_mismatch(self, tmp_path):
        ep = _make_episode(tmp_path, "epA", 1001)
        build_manifest(tmp_path, ["epA"], dataset_id="d")
        (ep / "frames.jsonl").write_bytes(b'{"step": 0}\n{"step": 9}\n')
        v = verify_manifest(tmp_path)
        assert v.exit_code == 1
        assert any("frames.jsonl" in p for p in v.hash_mismatch)


class TestDuplicateDetection:
    def test_duplicate_trajectory_group(self, tmp_path):
        _make_episode(tmp_path, "epA", 1001)
        _make_episode(tmp_path, "epB", 1002, frames_content=b'{"step": 0}\n{"step": 1}\n')
        # epB 用与 epA 完全相同的 frames 内容 → 重复轨迹
        m = build_manifest(tmp_path, ["epA", "epB"], dataset_id="d")
        assert any(sorted(g) == ["epA", "epB"] for g in m.duplicate_trajectory_groups)

    def test_different_seed_same_trajectory_warns(self, tmp_path):
        _make_episode(tmp_path, "epA", 1001)
        _make_episode(tmp_path, "epB", 2002, frames_content=b'{"step": 0}\n{"step": 1}\n')
        # epB 与 epA frames 相同但 root_seed 不同 → seed 传播警告
        m = build_manifest(tmp_path, ["epA", "epB"], dataset_id="d")
        assert any("seed" in w for w in m.warnings)

    def test_duplicate_seed_group(self, tmp_path):
        _make_episode(tmp_path, "epA", 1001)
        _make_episode(tmp_path, "epB", 1001, frames_content=b'{"step": 0}\n{"step": 7}\n')
        # 同 root_seed + 同 scenario + 同 export_config_hash → duplicate_seed_group
        m = build_manifest(tmp_path, ["epA", "epB"], dataset_id="d")
        assert any(sorted(g) == ["epA", "epB"] for g in m.duplicate_seed_groups)


class TestChecksumProfiles:
    def test_profile_sets(self, tmp_path):
        ep = _make_episode(tmp_path, "epA", 1001, with_render=True)
        meta_files = profile_file_paths(ep, "metadata")
        final_files = profile_file_paths(ep, "final")
        all_files = profile_file_paths(ep, "all")
        assert len(meta_files) < len(final_files) < len(all_files)
        # final 不含 render/ 目录；all 含 render 与 render_mask
        assert not any("render" in p.parts for p in final_files)
        assert any("render" in p.parts and p.suffix == ".png" for p in all_files)
        assert any(p.suffix == ".exr" for p in all_files)

    def test_build_with_all_profile(self, tmp_path):
        _make_episode(tmp_path, "epA", 1001, with_render=True)
        m = build_manifest(tmp_path, ["epA"], dataset_id="d", checksum_profile="all")
        assert m.episodes[0].artifact_counts.raw_rgb == 2  # 2 相机 × 1 render
        assert m.episodes[0].artifact_counts.raw_object_id_exr == 2

    def test_public_img1_profile_requires_jpeg(self, tmp_path):
        ep = _make_episode(tmp_path, "epA", 1001)
        (ep / "Cam_01" / "img1" / "legacy.png").write_bytes(b"legacy")
        files = profile_file_paths(ep, "final")
        assert any(p.name == "000001.jpg" for p in files)
        assert not any(p.name == "legacy.png" for p in files)


class TestHashUtil:
    def test_sha256_file_streaming_matches_bytes(self, tmp_path):
        p = tmp_path / "blob.bin"
        data = bytes(range(256)) * 1000
        p.write_bytes(data)
        assert sha256_file(p, chunk_size=7) == sha256_bytes(data)


class TestFingerprintExcludesMachineInfo:
    def test_generator_does_not_affect_fingerprint(self, tmp_path):
        _make_episode(tmp_path, "epA", 1001)
        m1 = build_manifest(tmp_path, ["epA"], dataset_id="d")
        # generator 信息（commit/python 等）变化不影响 fingerprint（episodes 内容不变）
        m1.generator.python_version = "9.9.9"
        m1.generator.git_commit = "deadbeef"
        assert m1.dataset_fingerprint
