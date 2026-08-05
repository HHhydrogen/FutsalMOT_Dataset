"""数据集级 manifest：索引、校验和、去重检测与稳定 fingerprint。

只负责对**明确指定的 episode** 做只读汇总与校验：

  - 汇总每个 episode 的 metadata、相机、产物数量与字节数
  - 按 checksum profile 生成逐文件 SHA-256（流式，不整载内存）
  - 计算 trajectory hash（frames.jsonl 原始字节）与 canonical hash
  - 检测重复 root_seed/配置组合、重复轨迹、跨 seed 同轨迹（seed 传播失败线索）
  - 计算稳定 dataset fingerprint（只依赖内容，不含时间/绝对路径）
  - verify-manifest 手动校验文件完整性

不负责：生成 episode、调用 UE、重试、划分训练集、自动运行 validator。

路径约定：全部使用 POSIX 风格相对路径，不写盘符/用户名/绝对路径。
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from pydantic import BaseModel, Field

MANIFEST_SCHEMA = "futsalmot_dataset_manifest"
FINGERPRINT_SCHEMA = "futsalmot_dataset_fingerprint_v1"
CHECKSUM_PROFILES = ("metadata", "final", "all")

# 哈希 chunk 大小（1 MiB），避免大文件整读内存
_DEFAULT_CHUNK_MB = 1


# ── 数据模型 ──────────────────────────────────────────────────────────

class GeneratorInfo(BaseModel):
    """生成 manifest 的工具链信息（不进 fingerprint）。"""

    repository: str
    git_commit: Optional[str]
    branch: Optional[str]
    dirty_worktree: bool
    package_name: str
    package_version: str
    python_version: str
    platform: str
    unreal_engine_version: Optional[str] = None


class ExternalSourcesInfo(BaseModel):
    """外部仓库锁定版本。"""

    google_research_football_commit: Optional[str] = None
    grf_marl_commit: Optional[str] = None


class ArtifactCounts(BaseModel):
    """每个 episode 的产物数量统计。"""

    rgb_final: int = 0          # img1/*.png
    instance_mask: int = 0      # mask/*.png
    annotation_frames: int = 0  # annotations.jsonl 行数
    yolo_detect_files: int = 0  # labels/det/*.txt
    yolo_segment_files: int = 0  # labels/seg/*.txt
    mot_sequences: int = 0      # gt/gt.txt 数
    raw_rgb: int = 0            # render/*.png
    raw_object_id_exr: int = 0  # render_mask/*.exr


class ArtifactBytes(BaseModel):
    """每个 episode 的产物字节统计。"""

    rgb_final: int = 0
    instance_mask: int = 0
    annotations: int = 0
    labels: int = 0
    raw_rgb: int = 0
    raw_object_id_exr: int = 0


class EpisodeManifestEntry(BaseModel):
    """单个 episode 的 manifest 记录。"""

    episode_id: str
    relative_path: str  # 相对 dataset_root，POSIX

    root_seed: Optional[int] = None
    grf_game_engine_seed: Optional[int] = None
    ue_visual_seed: Optional[int] = None
    seed_policy: Optional[str] = None

    scenario: Optional[str] = None
    trajectory_fps: Optional[float] = None
    playback_fps: Optional[int] = None

    frames_per_camera: Optional[int] = None
    camera_count: int = 0
    camera_ids: List[str] = Field(default_factory=list)

    artifact_counts: ArtifactCounts = Field(default_factory=ArtifactCounts)
    artifact_bytes: ArtifactBytes = Field(default_factory=ArtifactBytes)

    # 各相机最终帧数不一致时置 True（不掩盖差异）
    frame_count_inconsistent: bool = False

    config_hashes: Dict[str, str] = Field(default_factory=dict)
    content_hashes: Dict[str, str] = Field(default_factory=dict)

    checksums_file: Optional[str] = None  # 相对 dataset_root，POSIX
    checksums_file_sha256: Optional[str] = None

    reports: Dict[str, str] = Field(default_factory=dict)


class DatasetManifest(BaseModel):
    """数据集根目录级 manifest。"""

    schema_: str = Field(MANIFEST_SCHEMA, alias="schema")
    version: int = 1
    dataset_id: str
    created_at_utc: str

    generator: GeneratorInfo
    external_sources: ExternalSourcesInfo

    episodes: List[EpisodeManifestEntry]
    totals: ArtifactCounts

    duplicate_seed_groups: List[List[str]] = Field(default_factory=list)
    duplicate_trajectory_groups: List[List[str]] = Field(default_factory=list)

    # 构建期检测到的警告（不进 fingerprint）：如不同 root seed 产生相同轨迹
    warnings: List[str] = Field(default_factory=list)

    dataset_fingerprint: str

    model_config = {"populate_by_name": True}


# ── 哈希与路径工具 ────────────────────────────────────────────────────

def sha256_bytes(data: bytes) -> str:
    """计算 bytes 的 SHA-256。"""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = _DEFAULT_CHUNK_MB * 1024 * 1024) -> str:
    """流式计算文件 SHA-256（不整载内存）。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_file_with_size(
    path: Path, chunk_size: int = _DEFAULT_CHUNK_MB * 1024 * 1024
) -> Tuple[str, int]:
    """流式计算文件 SHA-256 与字节数。"""
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
            size += len(block)
    return h.hexdigest(), size


def _posix(p: Path) -> str:
    """转 POSIX 相对路径（反斜杠 → /）。"""
    return p.as_posix()


def _relative_to_root(path: Path, root: Path) -> str:
    """返回相对 dataset_root 的 POSIX 路径。"""
    return path.relative_to(root).as_posix()


def _is_contained(candidate: Path, root: Path) -> bool:
    """candidate 是否严格位于 root 内（防路径逃逸）。"""
    try:
        candidate.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


# ── checksum profile 与忽略规则 ───────────────────────────────────────

_EPISODE_LEVEL_METADATA = ("meta.json", "frames.jsonl", "render_summary.json")
_CAMERA_METADATA = ("camera.json", "annotations.jsonl", "mask_config.json", "seqinfo.ini")
_PROVENANCE_FILES = ("provenance/export_config.json", "provenance/external_sources.lock.json")


def profile_file_paths(
    episode_dir: Path,
    profile: str,
) -> List[Path]:
    """按 checksum profile 列出 episode 内应校验的文件（相对 episode_dir 的相对 Path）。

    - metadata: meta/frames/render_summary + 每相机元数据 + gt/gt.txt + provenance
    - final:    metadata + img1/ + mask/ + labels/det/ + labels/seg/
    - all:      final + render/ + render_mask/
    """
    if profile not in CHECKSUM_PROFILES:
        raise ValueError(f"未知 checksum profile: {profile!r}（可选 {'/'.join(CHECKSUM_PROFILES)}）")

    out: List[Path] = []
    for name in _EPISODE_LEVEL_METADATA:
        p = episode_dir / name
        if p.is_file():
            out.append(p)
    for prov in _PROVENANCE_FILES:
        p = episode_dir / prov
        if p.is_file():
            out.append(p)

    cameras = sorted(d for d in episode_dir.iterdir() if d.is_dir() and (d / "camera.json").is_file())
    for cam in cameras:
        for name in _CAMERA_METADATA:
            p = cam / name
            if p.is_file():
                out.append(p)
        gt = cam / "gt" / "gt.txt"
        if gt.is_file():
            out.append(gt)

    if profile in ("final", "all"):
        for cam in cameras:
            for sub, suffix in (
                ("img1", ".png"),
                ("mask", ".png"),
                ("labels/det", ".txt"),
                ("labels/seg", ".txt"),
            ):
                d = cam / sub
                if d.is_dir():
                    out.extend(sorted(
                        p for p in d.iterdir() if p.is_file() and p.suffix == suffix
                    ))
    if profile == "all":
        for cam in cameras:
            for sub in ("render", "render_mask"):
                d = cam / sub
                if d.is_dir():
                    out.extend(sorted(p for p in d.rglob("*") if p.is_file()))
    # 去重（保留顺序）
    seen: Set[Path] = set()
    dedup: List[Path] = []
    for p in out:
        if p not in seen:
            seen.add(p)
            dedup.append(p)
    return dedup


# ── episode 信息收集 ──────────────────────────────────────────────────

def _load_meta(episode_dir: Path) -> Optional[dict]:
    p = episode_dir / "meta.json"
    if not p.exists():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def _load_external_sources(episode_dir: Path) -> dict:
    """从 episode 的 provenance 快照读取外部仓库锁定版本（无快照返回空）。"""
    cand = episode_dir / "provenance" / "external_sources.lock.json"
    if cand.is_file():
        try:
            with open(cand, encoding="utf-8") as f:
                return json.load(f).get("repositories", {})
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _discover_cameras(episode_dir: Path) -> List[Path]:
    return sorted(d for d in episode_dir.iterdir() if d.is_dir() and (d / "camera.json").is_file())


def _frames_jsonl_hashes(episode_dir: Path) -> Dict[str, str]:
    """frames.jsonl 的 raw + canonical hash（文件缺失时返回空 dict）。"""
    p = episode_dir / "frames.jsonl"
    if not p.exists():
        return {}
    raw = p.read_bytes()
    try:
        canonical = (
            "\n".join(
                json.dumps(json.loads(line), sort_keys=True, separators=(",", ":"))
                for line in raw.decode("utf-8").splitlines() if line.strip()
            )
            + "\n"
        )
        return {
            "trajectory_hash": sha256_bytes(raw),
            "trajectory_canonical_hash": sha256_bytes(canonical.encode("utf-8")),
        }
    except Exception:  # noqa: BLE001
        return {"trajectory_hash": sha256_bytes(raw)}


def _config_hashes_from_provenance(episode_dir: Path) -> Dict[str, str]:
    """从 provenance/ 快照计算配置 hash；无快照返回空 dict（不伪造 provenance）。"""
    out: Dict[str, str] = {}
    for name in ("export_config.json", "external_sources.lock.json"):
        p = episode_dir / "provenance" / name
        if p.is_file():
            out[name] = sha256_file(p)
    return out


def _count_annotation_frames(cam_dir: Path) -> int:
    p = cam_dir / "annotations.jsonl"
    if not p.exists():
        return 0
    try:
        with open(p, encoding="utf-8") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def _collect_artifact_stats(cam_dir: Path) -> Tuple[ArtifactCounts, ArtifactBytes]:
    """统计单个相机的产物数量与字节。"""
    counts = ArtifactCounts()
    bytes_ = ArtifactBytes()
    img1 = cam_dir / "img1"
    mask = cam_dir / "mask"
    det = cam_dir / "labels" / "det"
    seg = cam_dir / "labels" / "seg"
    render = cam_dir / "render"
    rmask = cam_dir / "render_mask"
    ann = cam_dir / "annotations.jsonl"

    def _add_dir(d: Path, suffix: str) -> Tuple[int, int]:
        n = s = 0
        if d.is_dir():
            for p in d.rglob(f"*{suffix}"):
                if p.is_file():
                    n += 1
                    s += p.stat().st_size
        return n, s

    n, s = _add_dir(img1, ".png"); counts.rgb_final = n; bytes_.rgb_final = s
    n, s = _add_dir(mask, ".png"); counts.instance_mask = n; bytes_.instance_mask = s
    n, s = _add_dir(det, ".txt"); counts.yolo_detect_files = n
    n, s2 = _add_dir(seg, ".txt"); counts.yolo_segment_files = n
    bytes_.labels = s + s2
    n, s = _add_dir(render, ".png"); counts.raw_rgb = n; bytes_.raw_rgb = s
    n, s = _add_dir(rmask, ".exr"); counts.raw_object_id_exr = n; bytes_.raw_object_id_exr = s
    if ann.is_file():
        counts.annotation_frames = _count_annotation_frames(cam_dir)
        bytes_.annotations = ann.stat().st_size
    if (cam_dir / "gt" / "gt.txt").is_file():
        counts.mot_sequences = 1
    return counts, bytes_


def _write_checksums_file(
    dataset_root: Path,
    episode_dir: Path,
    files: List[Path],
    workers: int,
    chunk_mb: int = _DEFAULT_CHUNK_MB,
) -> Tuple[Path, str]:
    """为 episode 计算逐文件校验和，原子写入 checksums/<episode_id>.jsonl。

    返回 (checksums_path, checksums_file_sha256)。行序按相对 dataset_root 的
    POSIX 路径排序，保证确定性。
    """
    episode_id = episode_dir.name
    rel_files = sorted(
        (_relative_to_root(p, dataset_root), p) for p in files
    )

    def _hash_one(item) -> Dict[str, object]:
        rel, path = item
        digest, size = sha256_file_with_size(path, chunk_mb * 1024 * 1024)
        return {"path": rel, "size": size, "sha256": digest}

    results: List[Dict[str, object]] = []
    if workers > 1 and len(rel_files) > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_hash_one, rel_files))
    else:
        results = [_hash_one(item) for item in rel_files]

    checksum_dir = dataset_root / "checksums"
    checksum_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{episode_id}.checksums.", dir=str(checksum_dir))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            for row in results:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        final = checksum_dir / f"{episode_id}.jsonl"
        os.replace(tmp, final)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return final, sha256_file(final)


def collect_episode(
    dataset_root: Path,
    episode_dir: Path,
    profile: str,
    workers: int,
    chunk_mb: int,
) -> EpisodeManifestEntry:
    """收集一个 episode 的统计、校验和与哈希，生成 EpisodeManifestEntry。"""
    episode_id = episode_dir.name
    rel_path = _relative_to_root(episode_dir, dataset_root)

    meta = _load_meta(episode_dir)
    timing = (meta or {}).get("timing", {})
    randomness = (meta or {}).get("randomness") or {}
    source = (meta or {}).get("source", {})

    # legacy（无 randomness）：以 source.seed 作 best-effort root_seed；
    # GRF 引擎 seed 与 policy 保持 None（旧 episode 未验证可复现）
    root_seed = randomness.get("root_seed")
    if root_seed is None and isinstance(source.get("seed"), int):
        root_seed = source.get("seed")

    cameras = _discover_cameras(episode_dir)
    per_cam_frames = [_count_annotation_frames(c) for c in cameras]

    counts = ArtifactCounts()
    bytes_ = ArtifactBytes()
    for cam in cameras:
        c, b = _collect_artifact_stats(cam)
        counts = ArtifactCounts(
            rgb_final=counts.rgb_final + c.rgb_final,
            instance_mask=counts.instance_mask + c.instance_mask,
            annotation_frames=counts.annotation_frames + c.annotation_frames,
            yolo_detect_files=counts.yolo_detect_files + c.yolo_detect_files,
            yolo_segment_files=counts.yolo_segment_files + c.yolo_segment_files,
            mot_sequences=counts.mot_sequences + c.mot_sequences,
            raw_rgb=counts.raw_rgb + c.raw_rgb,
            raw_object_id_exr=counts.raw_object_id_exr + c.raw_object_id_exr,
        )
        bytes_ = ArtifactBytes(
            rgb_final=bytes_.rgb_final + b.rgb_final,
            instance_mask=bytes_.instance_mask + b.instance_mask,
            annotations=bytes_.annotations + b.annotations,
            labels=bytes_.labels + b.labels,
            raw_rgb=bytes_.raw_rgb + b.raw_rgb,
            raw_object_id_exr=bytes_.raw_object_id_exr + b.raw_object_id_exr,
        )

    frames_per_camera = per_cam_frames[0] if per_cam_frames else None
    frame_count_inconsistent = len(set(per_cam_frames)) > 1 if per_cam_frames else False

    files = profile_file_paths(episode_dir, profile)
    checksums_path, checksums_sha = _write_checksums_file(
        dataset_root, episode_dir, files, workers, chunk_mb
    )

    content_hashes = _frames_jsonl_hashes(episode_dir)
    config_hashes = _config_hashes_from_provenance(episode_dir)

    trajectory_fps = None
    step_sec = timing.get("source_step_seconds")
    if isinstance(step_sec, (int, float)) and step_sec > 0:
        trajectory_fps = round(1.0 / step_sec, 6)

    reports = {}
    for name in ("audit/soak_audit_report.json", "audit/soak_audit_report.md",
                 "audit/spot_check_index.md"):
        p = episode_dir / name
        if p.is_file():
            reports[name] = _relative_to_root(p, dataset_root)

    return EpisodeManifestEntry(
        episode_id=episode_id,
        relative_path=rel_path,
        root_seed=root_seed,
        grf_game_engine_seed=randomness.get("grf_game_engine_seed"),
        ue_visual_seed=randomness.get("ue_visual_seed"),
        seed_policy=randomness.get("policy"),
        scenario=source.get("scenario") or (meta or {}).get("scenario"),
        trajectory_fps=trajectory_fps,
        playback_fps=timing.get("playback_fps"),
        frames_per_camera=frames_per_camera,
        camera_count=len(cameras),
        camera_ids=[c.name for c in cameras],
        artifact_counts=counts,
        artifact_bytes=bytes_,
        frame_count_inconsistent=frame_count_inconsistent,
        config_hashes=config_hashes,
        content_hashes=content_hashes,
        checksums_file=_relative_to_root(checksums_path, dataset_root),
        checksums_file_sha256=checksums_sha,
        reports=reports,
    )


# ── 去重检测 ──────────────────────────────────────────────────────────

def _duplicate_seed_groups(entries: List[EpisodeManifestEntry]) -> List[List[str]]:
    """按 (root_seed, scenario, export_config_hash) 分组，找出重复组。"""
    groups: Dict[Tuple, List[str]] = {}
    for e in entries:
        key = (e.root_seed, e.scenario, e.config_hashes.get("export_config.json"))
        groups.setdefault(key, []).append(e.episode_id)
    return [ids for ids in groups.values() if len(ids) > 1]


def _duplicate_trajectory_groups(entries: List[EpisodeManifestEntry]) -> List[List[str]]:
    """按 trajectory_hash 分组，找出重复轨迹组。"""
    groups: Dict[str, List[str]] = {}
    for e in entries:
        h = e.content_hashes.get("trajectory_hash")
        if h:
            groups.setdefault(h, []).append(e.episode_id)
    return [ids for ids in groups.values() if len(ids) > 1]


def _cross_seed_same_trajectory_warnings(entries: List[EpisodeManifestEntry]) -> List[str]:
    """不同 root_seed 但 trajectory_hash 相同的组合 → seed 传播失败线索。"""
    by_hash: Dict[str, List[EpisodeManifestEntry]] = {}
    for e in entries:
        h = e.content_hashes.get("trajectory_hash")
        if h:
            by_hash.setdefault(h, []).append(e)
    warns = []
    for h, group in by_hash.items():
        seeds = {e.root_seed for e in group}
        if len(seeds) > 1:
            ids = ", ".join(e.episode_id for e in group)
            warns.append(
                f"不同 root seed（{sorted(seeds)}）产生相同 trajectory hash：{ids} —— "
                "可能的 seed 传播失败"
            )
    return warns


# ── fingerprint ───────────────────────────────────────────────────────

def _compute_fingerprint(entries: List[EpisodeManifestEntry]) -> str:
    """基于稳定内容计算 dataset fingerprint（不含时间/绝对路径/机器信息）。"""
    ep_list = []
    for e in sorted(entries, key=lambda x: (x.episode_id, x.relative_path)):
        ep_list.append({
            "episode_id": e.episode_id,
            "relative_path": e.relative_path,
            "trajectory_hash": e.content_hashes.get("trajectory_hash"),
            "checksums_file_sha256": e.checksums_file_sha256,
            "config_hashes": {k: v for k, v in sorted(e.config_hashes.items())},
        })
    payload = json.dumps(
        {"schema": FINGERPRINT_SCHEMA, "episodes": ep_list},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(payload)


# ── 生成器信息 ────────────────────────────────────────────────────────

def _git_info() -> Tuple[Optional[str], Optional[str], bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip() or None
        branch = subprocess.run(
            ["git", "branch", "--show-current"], capture_output=True, text=True
        ).stdout.strip() or None
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True
        ).stdout.strip())
        return commit, branch, dirty
    except Exception:  # noqa: BLE001
        return None, None, False


def _generator_info() -> GeneratorInfo:
    commit, branch, dirty = _git_info()
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            pkg_ver = version("grf-ue-bridge")
        except PackageNotFoundError:  # pragma: no cover
            pkg_ver = "unknown"
    except Exception:  # noqa: BLE001
        pkg_ver = "unknown"
    return GeneratorInfo(
        repository="https://github.com/HHhydrogen/FutsalMOT_Dataset.git",
        git_commit=commit,
        branch=branch,
        dirty_worktree=dirty,
        package_name="grf-ue-bridge",
        package_version=pkg_ver,
        python_version=platform.python_version(),
        platform=platform.platform(),
        unreal_engine_version=None,
    )


def _external_sources_info(episode_dirs: Sequence[Path]) -> ExternalSourcesInfo:
    """外部仓库锁定版本：优先 provenance 快照；否则用 episode meta.source
    （导出时记录的提交号）。都缺失时返回空（不伪造 provenance）。"""
    for ep in episode_dirs:
        repos = _load_external_sources(ep)
        if repos:
            return ExternalSourcesInfo(
                google_research_football_commit=(
                    repos.get("google-research-football") or {}).get("commit"),
                grf_marl_commit=(repos.get("GRF_MARL") or {}).get("commit"),
            )
    for ep in episode_dirs:
        src = (_load_meta(ep) or {}).get("source", {})
        if src.get("football_commit") or src.get("grf_marl_commit"):
            return ExternalSourcesInfo(
                google_research_football_commit=src.get("football_commit") or None,
                grf_marl_commit=src.get("grf_marl_commit") or None,
            )
    return ExternalSourcesInfo()


# ── build / verify 入口 ───────────────────────────────────────────────

def build_manifest(
    dataset_root: Path,
    episode_ids: Optional[Sequence[str]],
    dataset_id: str,
    checksum_profile: str = "final",
    workers: int = 4,
    chunk_mb: int = _DEFAULT_CHUNK_MB,
) -> DatasetManifest:
    """构建 dataset manifest（原子写入 dataset_manifest.json）。

    episode_ids 为 None 时只纳入满足合法 episode 结构（含 camera.json）的目录；
    否则只纳入显式指定的 episode。不默认扫描任意子目录。
    """
    dataset_root = dataset_root.resolve()
    if not dataset_root.is_dir():
        raise ValueError(f"dataset root 不存在: {dataset_root}")

    candidate_dirs = []
    if episode_ids:
        for eid in episode_ids:
            d = dataset_root / eid
            if not d.is_dir():
                raise ValueError(f"episode 目录不存在: {d}")
            candidate_dirs.append(d)
    else:
        for d in sorted(dataset_root.iterdir()):
            if d.is_dir() and _discover_cameras(d):
                candidate_dirs.append(d)

    entries: List[EpisodeManifestEntry] = []
    for ep in candidate_dirs:
        entries.append(collect_episode(
            dataset_root, ep, checksum_profile, workers, chunk_mb
        ))
    entries.sort(key=lambda e: (e.episode_id, e.relative_path))

    totals = ArtifactCounts()
    for e in entries:
        totals = ArtifactCounts(
            rgb_final=totals.rgb_final + e.artifact_counts.rgb_final,
            instance_mask=totals.instance_mask + e.artifact_counts.instance_mask,
            annotation_frames=totals.annotation_frames + e.artifact_counts.annotation_frames,
            yolo_detect_files=totals.yolo_detect_files + e.artifact_counts.yolo_detect_files,
            yolo_segment_files=totals.yolo_segment_files + e.artifact_counts.yolo_segment_files,
            mot_sequences=totals.mot_sequences + e.artifact_counts.mot_sequences,
            raw_rgb=totals.raw_rgb + e.artifact_counts.raw_rgb,
            raw_object_id_exr=totals.raw_object_id_exr + e.artifact_counts.raw_object_id_exr,
        )

    dup_seed = _duplicate_seed_groups(entries)
    dup_traj = _duplicate_trajectory_groups(entries)
    warnings = list(_cross_seed_same_trajectory_warnings(entries))
    if dup_seed:
        warnings.append(
            f"可能的重复 seed/配置组合：{dup_seed}（有意固定轨迹时可忽略）"
        )

    manifest = DatasetManifest(
        schema=MANIFEST_SCHEMA,
        version=1,
        dataset_id=dataset_id,
        created_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        generator=_generator_info(),
        external_sources=_external_sources_info(candidate_dirs),
        episodes=entries,
        totals=totals,
        duplicate_seed_groups=dup_seed,
        duplicate_trajectory_groups=dup_traj,
        warnings=warnings,
        dataset_fingerprint=_compute_fingerprint(entries),
    )

    fd, tmp = tempfile.mkstemp(
        prefix=".dataset_manifest.", suffix=".json", dir=str(dataset_root)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(manifest.model_dump_json(by_alias=True, indent=2) + "\n")
        final = dataset_root / "dataset_manifest.json"
        os.replace(tmp, final)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return manifest


# ── verify ────────────────────────────────────────────────────────────

class VerifyResult:
    """verify-manifest 的结果汇总。"""

    def __init__(self) -> None:
        self.exit_code = 0
        self.checked = 0
        self.matched = 0
        self.missing: List[str] = []
        self.size_mismatch: List[str] = []
        self.hash_mismatch: List[str] = []
        self.extra: List[str] = []
        self.errors: List[str] = []
        self.warnings: List[str] = []


def _verify_checksums_file(
    dataset_root: Path,
    checksum_rel: str,
    expected_sha: Optional[str],
    result: VerifyResult,
    workers: int,
) -> None:
    """校验一个 checksum JSONL：文件自身 hash、逐行 path/size/sha256。"""
    path = dataset_root / checksum_rel
    if not path.is_file():
        result.missing.append(checksum_rel)
        return
    if expected_sha and sha256_file(path) != expected_sha:
        result.hash_mismatch.append(f"{checksum_rel}（checksum 文件自身 hash 不符）")
        return

    lines = path.read_text(encoding="utf-8").splitlines()
    jobs = []
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            result.errors.append(f"{checksum_rel}: 非法行: {e}")
            continue
        rel = str(row.get("path", ""))
        target = dataset_root / rel
        if not _is_contained(target, dataset_root):
            result.errors.append(f"路径逃逸 dataset root: {rel}")
            continue
        jobs.append((rel, target, int(row.get("size", -1)), str(row.get("sha256", ""))))

    def _check(item) -> Tuple[str, str, bool, bool]:
        rel, target, size, sha = item
        if not target.is_file():
            return rel, "missing", False, False
        digest, actual_size = sha256_file_with_size(target)
        if actual_size != size:
            return rel, "size", False, False
        if digest != sha:
            return rel, "hash", False, False
        return rel, "ok", True, True

    if workers > 1 and len(jobs) > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            outcomes = list(ex.map(_check, jobs))
    else:
        outcomes = [_check(j) for j in jobs]

    for rel, kind, ok, _ in outcomes:
        result.checked += 1
        if ok:
            result.matched += 1
        elif kind == "missing":
            result.missing.append(rel)
        elif kind == "size":
            result.size_mismatch.append(rel)
        else:
            result.hash_mismatch.append(rel)


def _ignore_for_extra(rel: str) -> bool:
    """extra 文件检查的忽略规则。

    忽略：checksum 自身、debug/audit 可视化、视频、临时文件，以及
    render/ 与 render_mask/（原始渲染中间产物——final/metadata profile
    刻意不校验；all profile 下它们在校验集内，不会出现在 extra）。
    """
    if rel.startswith("checksums/") or rel == "dataset_manifest.json":
        return True
    if "/debug/" in rel or rel.startswith("debug/"):
        return True
    if "/audit/" in rel or rel.startswith("audit/"):
        return True
    if "/render/" in rel or rel.startswith("render/"):
        return True
    if "/render_mask/" in rel or rel.startswith("render_mask/"):
        return True
    if rel.endswith(".tmp") or rel.endswith(".log"):
        return True
    if rel.rsplit("/", 1)[-1].startswith("video_"):
        return True
    if rel.endswith(".gitkeep"):
        return True
    return False


def verify_manifest(
    dataset_root: Path,
    manifest_path: Optional[Path] = None,
    workers: int = 4,
    chunk_mb: int = _DEFAULT_CHUNK_MB,
    strict_extra: bool = False,
) -> VerifyResult:
    """校验 dataset manifest 与实际内容。返回 VerifyResult（exit_code 见其字段）。

    exit_code：0=通过；1=内容不匹配/缺失；2=manifest/schema/参数错误。
    """
    dataset_root = dataset_root.resolve()
    result = VerifyResult()
    manifest_path = manifest_path or (dataset_root / "dataset_manifest.json")
    if not manifest_path.is_file():
        result.errors.append(f"manifest 不存在: {manifest_path}")
        result.exit_code = 2
        return result
    try:
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        result.errors.append(f"manifest 不是合法 JSON: {e}")
        result.exit_code = 2
        return result

    if data.get("schema") != MANIFEST_SCHEMA:
        result.errors.append(f"manifest.schema 非法: {data.get('schema')!r}")
        result.exit_code = 2
    if data.get("version") != 1:
        result.errors.append(f"manifest.version 非法: {data.get('version')!r}")
        result.exit_code = 2
    if result.exit_code == 2:
        return result

    # 反序列化为模型（同时校验 schema），供 fingerprint 重算与字段访问
    try:
        entries = [EpisodeManifestEntry(**ep) for ep in data.get("episodes", [])]
    except Exception as e:  # noqa: BLE001
        result.errors.append(f"manifest.episodes 不符合 schema: {e}")
        result.exit_code = 2
        return result

    seen_ids: Set[str] = set()
    seen_paths: Set[str] = set()
    for ep in entries:
        eid = ep.episode_id
        rel = ep.relative_path
        if eid in seen_ids:
            result.errors.append(f"episode_id 重复: {eid}")
        seen_ids.add(eid)
        if rel in seen_paths:
            result.errors.append(f"relative_path 重复: {rel}")
        seen_paths.add(rel)

        ep_dir = dataset_root / rel
        if not ep_dir.is_dir():
            result.missing.append(rel)
            continue
        if ep.checksums_file:
            _verify_checksums_file(
                dataset_root, ep.checksums_file, ep.checksums_file_sha256, result, workers
            )
        # trajectory hash 校验（若存在 frames.jsonl）
        traj_hash = (ep.content_hashes or {}).get("trajectory_hash")
        fp = ep_dir / "frames.jsonl"
        if traj_hash and fp.is_file():
            if sha256_file(fp, chunk_mb * 1024 * 1024) != traj_hash:
                result.hash_mismatch.append(f"{rel}/frames.jsonl（trajectory hash 不符）")

    # extra files：episode 目录中未在校验和集合里的文件
    checksummed: Set[Path] = set()
    for ep in entries:
        ep_dir = dataset_root / ep.relative_path
        if not ep.checksums_file:
            continue
        cpath = dataset_root / ep.checksums_file
        if not cpath.is_file():
            continue
        for line in cpath.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                checksummed.add(dataset_root / str(row["path"]))
            except (json.JSONDecodeError, KeyError):
                continue
    # 提前解析一次 checksummed 集合（避免对每个文件重复 resolve 的 O(n²)）
    resolved_checksummed = {x.resolve() for x in checksummed}
    for ep in entries:
        ep_dir = dataset_root / ep.relative_path
        if not ep_dir.is_dir():
            continue
        for p in ep_dir.rglob("*"):
            if not p.is_file():
                continue
            rel = _relative_to_root(p, dataset_root)
            if _ignore_for_extra(rel):
                continue
            if p.resolve() not in resolved_checksummed:
                result.extra.append(rel)

    # 重新计算 fingerprint
    try:
        recomputed = _compute_fingerprint([e for e in entries])
        if recomputed != data.get("dataset_fingerprint"):
            result.errors.append("dataset_fingerprint 与内容不符（可重新计算但结果不同）")
    except Exception as e:  # noqa: BLE001
        result.errors.append(f"fingerprint 重算失败: {e}")

    # 重复轨迹警告
    dup_groups = data.get("duplicate_trajectory_groups", [])
    if dup_groups:
        result.warnings.append(f"重复轨迹组: {dup_groups}")

    if result.errors:
        result.exit_code = max(result.exit_code, 2)
    elif result.missing or result.size_mismatch or result.hash_mismatch:
        result.exit_code = max(result.exit_code, 1)
    if strict_extra and result.extra:
        result.exit_code = max(result.exit_code, 1)
    return result
