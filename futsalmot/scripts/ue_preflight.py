# FutsalMOT A3.3b UE Preflight — 只读检查工具
# Version marker: A3_3B_UE_PREFLIGHT_READ_ONLY_8P_V5
# V2: 修复 SkeletalMesh/AnimSequence skeleton 访问 API (.skeleton 属性而非 .get_skeleton())
# V3: 修复 Character.get_components() → get_components_by_class() wrapped in try/except
# V4: 支持 pipeline_current.json、动态 contact frames、动态 Sequence 列表
#
# 用途：在 Unreal Editor 中运行，只读检查 A3.3b 构建所需的所有前提条件。
# 不创建、不修改、不保存任何 UE 资产。
#
# 运行方式（在 UE Python 控制台中）：
#   py "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/02_run_unreal.py"
#
# 输出：
#   Content/FutsalMOT/code/_agent_test_outputs/preflight_<seq_id>.json

import unreal
import os
import json
import sys

# ============================================================
# Constants
# ============================================================

SCRIPT_VERSION = "A3_3B_UE_PREFLIGHT_READ_ONLY_8P_V5"
DEFAULT_PROJECT_ROOT = "D:/projects/FustalMOT_UEDataset"
SCRIPT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
CODE_DIR = SCRIPT_DIR
AGENT_TEST_OUTPUTS = os.path.join(CODE_DIR, "_agent_test_outputs")
CURRENT_RUN_POINTER = os.path.join(CODE_DIR, "configs", "pipeline_current.json")
DEFAULT_ACTION_MAP_PATH = os.path.join(CODE_DIR, "configs", "action_animation_map.json")


def resolve_config_path():
    env_path = os.environ.get("FUTSALMOT_CONFIG_PATH")
    if env_path:
        return os.path.normpath(os.path.abspath(env_path)), "environment"
    if os.path.isfile(CURRENT_RUN_POINTER):
        try:
            with open(CURRENT_RUN_POINTER, "r", encoding="utf-8-sig") as f:
                pointer = json.load(f)
            raw = pointer.get("paths", {}).get("a3_3_config")
            if isinstance(raw, str) and raw.strip():
                path = raw.strip()
                if not os.path.isabs(path):
                    path = os.path.join(CODE_DIR, path)
                return os.path.normpath(os.path.abspath(path)), "pipeline_current.json"
        except Exception:
            pass
    fallback = os.path.join(
        CODE_DIR, "configs", "events", "generated", "episode_random_0001_t1_a33.json"
    )
    return os.path.normpath(os.path.abspath(fallback)), "legacy_default"


DEFAULT_CONFIG_PATH, CONFIG_PATH_SOURCE = resolve_config_path()
PROJECT_ROOT = DEFAULT_PROJECT_ROOT
V2_SCANNER_OUTPUT = os.path.join(
    PROJECT_ROOT, "Saved", "FutsalMOT", "animation_assets", "action_animation_candidates_v2.json"
)

EXPECTED_PLAYERS = ["Player_{:02d}".format(i) for i in range(1, 9)]
EXPECTED_BALLS = ["Ball_01"]
EXPECTED_CAMERAS = ["CineCam_01", "CineCam_02", "CineCam_03", "CineCam_04"]
EXPECTED_OBJECTS = EXPECTED_PLAYERS + EXPECTED_BALLS

# ============================================================
# Helper: safe skeleton access (UE API varies by version)
# ============================================================

def safe_get_skeleton(asset, label="asset", result=None):
    """Try multiple UE Python API patterns to get the skeleton."""
    # Pattern 1: .skeleton property (UE5 common API)
    try:
        skel = getattr(asset, "skeleton", None)
        if skel is not None:
            return skel
    except Exception:
        pass
    # Pattern 2: .get_skeleton() method (some UE builds)
    try:
        skel = asset.get_skeleton()
        if skel is not None:
            return skel
    except AttributeError:
        pass
    except Exception:
        pass
    # Pattern 3: try to get via skeletal mesh
    try:
        skel = getattr(asset, "get_skeleton", None)
        if skel is not None:
            return skel()
    except Exception:
        pass
    if result is not None:
        result.add_warning("skeleton", "{}.skeleton 不可访问".format(label))
    return None


# ============================================================
# Result accumulator
# ============================================================

class PreflightResult:
    def __init__(self, seq_id):
        self.seq_id = seq_id
        self.errors = []
        self.warnings = []
        self.checks = {
            "config_integrity": {},
            "timeline": {},
            "objects": {},
            "actors": {},
            "animation_assets": {},
            "sequence": {},
            "output_paths": {}
        }
        self.status = "PASS"

    def add_error(self, category, message):
        self.errors.append({"category": category, "message": message})
        self.status = "ERROR"

    def add_warning(self, category, message):
        self.warnings.append({"category": category, "message": message})
        if self.status == "PASS":
            self.status = "WARNING"

    def to_dict(self):
        return {
            "schema_version": "1.0",
            "tool_version": SCRIPT_VERSION,
            "read_only": True,
            "seq_id": self.seq_id,
            "status": self.status,
            "errors": self.errors,
            "warnings": self.warnings,
            "checks": self.checks,
            "summary": {
                "error_count": len(self.errors),
                "warning_count": len(self.warnings),
                "check_count": sum(1 for cat in self.checks.values() for _ in cat.values())
            }
        }


# ============================================================
# 1. Config loading
# ============================================================

def check_config_integrity(result):
    checks = result.checks["config_integrity"]

    # Main trajectory config
    if not os.path.exists(DEFAULT_CONFIG_PATH):
        result.add_error("config_integrity", "轨迹配置不存在: {}".format(DEFAULT_CONFIG_PATH))
        checks["config_exists"] = False
        return False
    checks["config_path"] = DEFAULT_CONFIG_PATH
    checks["config_path_source"] = CONFIG_PATH_SOURCE
    checks["config_exists"] = True

    try:
        with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        result.add_error("config_integrity", "轨迹配置无法解析: {}".format(str(e)))
        checks["config_parseable"] = False
        return False
    checks["config_parseable"] = True

    global PROJECT_ROOT, V2_SCANNER_OUTPUT
    raw_project_root = config.get("project_root", DEFAULT_PROJECT_ROOT)
    PROJECT_ROOT = os.path.normpath(os.path.abspath(str(raw_project_root)))
    V2_SCANNER_OUTPUT = os.path.join(
        PROJECT_ROOT, "Saved", "FutsalMOT", "animation_assets", "action_animation_candidates_v2.json"
    )
    checks["project_root"] = PROJECT_ROOT.replace("\\", "/")

    # Derive expected objects/cameras from the active config.  The 8-player
    # constants above remain a fallback for malformed or legacy configs.
    global EXPECTED_PLAYERS, EXPECTED_BALLS, EXPECTED_OBJECTS, EXPECTED_CAMERAS
    raw_objects = config.get("objects", {})
    if isinstance(raw_objects, dict) and raw_objects:
        def object_sort_key(item):
            object_id, raw = item
            if isinstance(raw, dict):
                try:
                    return (int(raw.get("track_id", 10 ** 9)), str(object_id))
                except Exception:
                    pass
            return (10 ** 9, str(object_id))

        player_items = [
            item
            for item in raw_objects.items()
            if isinstance(item[1], dict) and item[1].get("category") == "player"
        ]
        ball_items = [
            item
            for item in raw_objects.items()
            if isinstance(item[1], dict) and item[1].get("category") == "ball"
        ]
        if player_items:
            EXPECTED_PLAYERS = [item[0] for item in sorted(player_items, key=object_sort_key)]
        if ball_items:
            EXPECTED_BALLS = [item[0] for item in sorted(ball_items, key=object_sort_key)]
        EXPECTED_OBJECTS = EXPECTED_PLAYERS + EXPECTED_BALLS

    raw_cameras = config.get("cameras")
    if isinstance(raw_cameras, dict) and raw_cameras:
        EXPECTED_CAMERAS = [
            str(value).strip()
            for _, value in sorted(raw_cameras.items())
            if str(value).strip()
        ]

    checks["expected_players"] = list(EXPECTED_PLAYERS)
    checks["expected_balls"] = list(EXPECTED_BALLS)
    checks["expected_cameras"] = list(EXPECTED_CAMERAS)
    roster = config.get("roster")
    if isinstance(roster, dict):
        checks["roster"] = roster
        declared_count = roster.get("player_count")
        if declared_count is not None and int(declared_count) != len(EXPECTED_PLAYERS):
            result.add_error(
                "config_integrity",
                "roster.player_count={} 与配置 player objects={} 不一致".format(
                    declared_count, len(EXPECTED_PLAYERS)
                ),
            )
        goalkeepers = roster.get("goalkeepers", [])
        if isinstance(goalkeepers, list) and goalkeepers:
            result.add_error(
                "config_integrity",
                "当前 4v4 outfield 配置不应包含守门员: {}".format(goalkeepers),
            )
    if len(EXPECTED_PLAYERS) != 8:
        result.add_error(
            "config_integrity",
            "当前阶段要求 8 名场上球员，配置中检测到 {} 名".format(
                len(EXPECTED_PLAYERS)
            ),
        )

    # Action map
    if not os.path.exists(DEFAULT_ACTION_MAP_PATH):
        result.add_warning("config_integrity", "动画映射配置不存在: {}".format(DEFAULT_ACTION_MAP_PATH))
        checks["action_map_exists"] = False
        action_map = None
    else:
        try:
            with open(DEFAULT_ACTION_MAP_PATH, "r", encoding="utf-8") as f:
                action_map = json.load(f)
            checks["action_map_path"] = DEFAULT_ACTION_MAP_PATH
            checks["action_map_exists"] = True
        except Exception as e:
            result.add_warning("config_integrity", "动画映射配置无法解析: {}".format(str(e)))
            action_map = None
            checks["action_map_parseable"] = False

    # V2 scanner output (optional, read-only cross-reference)
    v2_data = None
    if os.path.exists(V2_SCANNER_OUTPUT):
        try:
            with open(V2_SCANNER_OUTPUT, "r", encoding="utf-8") as f:
                v2_data = json.load(f)
            checks["v2_scanner_output"] = V2_SCANNER_OUTPUT
            checks["v2_scanner_asset_count"] = v2_data.get("asset_count", len(v2_data.get("all_assets", [])))
        except Exception as e:
            result.add_warning("config_integrity", "V2 扫描结果存在但无法解析: {}".format(str(e)))
    else:
        result.add_warning("config_integrity", "V2 扫描结果文件不存在: {}。这不是硬阻塞，但建议先运行 V2 扫描器。".format(V2_SCANNER_OUTPUT))
        checks["v2_scanner_output"] = None

    return config, action_map, v2_data


# ============================================================
# 2. Timeline check
# ============================================================

def check_timeline(config, result):
    checks = result.checks["timeline"]
    raw_timeline = config.get("timeline", {})
    checks["raw_timeline_keys"] = list(raw_timeline.keys())

    seq_id = raw_timeline.get("seq_id", config.get("seq_id", None))
    if seq_id:
        result.seq_id = seq_id
    else:
        result.add_error("timeline", "seq_id 不存在")

    fps = raw_timeline.get("display_rate", None)
    checks["display_rate"] = fps
    if fps is None:
        result.add_error("timeline", "display_rate (fps) 缺失")
    elif fps != 30.0:
        result.add_warning("timeline", "fps={}，预期 30".format(fps))

    frame_start = raw_timeline.get("frame_start", None)
    frame_end = raw_timeline.get("frame_end", None)
    checks["frame_start"] = frame_start
    checks["frame_end"] = frame_end

    if frame_start is None:
        result.add_error("timeline", "frame_start 缺失")
    elif frame_start != 0:
        result.add_error("timeline", "frame_start={}，预期 0".format(frame_start))

    if frame_end is None:
        result.add_error("timeline", "frame_end 缺失")
    elif frame_end != 299:
        result.add_warning("timeline", "frame_end={}，预期 299".format(frame_end))

    if frame_start is not None and frame_end is not None:
        total = frame_end - frame_start + 1
        checks["implied_total_frames"] = total
        if total != 300:
            result.add_warning("timeline", "总帧数={}（frame_end-frame_start+1），预期 300".format(total))


# ============================================================
# 3. Objects check
# ============================================================

def check_objects(config, result):
    checks = result.checks["objects"]
    raw_objects = {k: v for k, v in config.get("objects", {}).items()}

    # Object presence
    found_objects = list(raw_objects.keys())
    checks["found_objects"] = found_objects
    for expected in EXPECTED_OBJECTS:
        if expected not in raw_objects:
            result.add_error("objects", "缺少对象: {}".format(expected))

    # Track ID uniqueness
    track_ids = {}
    class_ids_present = {}
    for obj_name, obj_data in raw_objects.items():
        tid = obj_data.get("track_id")
        cid = obj_data.get("class_id")
        if tid is not None:
            if tid in track_ids:
                result.add_error("objects", "Track ID {} 重复: {} 和 {}".format(tid, track_ids[tid], obj_name))
            else:
                track_ids[tid] = obj_name
        if cid is not None:
            class_ids_present[cid] = True
    checks["track_ids"] = track_ids
    checks["class_ids_present"] = sorted(class_ids_present.keys())

    # Per-object checks
    timeline = config.get("timeline", {})
    expected_start = int(timeline.get("frame_start", 0))
    expected_end = int(timeline.get("frame_end", 299))
    expected_count = expected_end - expected_start + 1
    checks["expected_player_count"] = len(EXPECTED_PLAYERS)
    checks["expected_object_count"] = len(EXPECTED_OBJECTS)
    checks["expected_keyframes_per_object"] = expected_count

    for obj_name in EXPECTED_OBJECTS:
        obj_data = raw_objects.get(obj_name)
        if obj_data is None:
            continue
        checks[obj_name] = {}
        kfs = obj_data.get("keyframes", [])
        n_kf = len(kfs)
        checks[obj_name]["keyframe_count"] = n_kf

        if n_kf != expected_count:
            result.add_error(
                "objects",
                "{} keyframes={}，预期 {}".format(obj_name, n_kf, expected_count),
            )

        # Frame continuity
        if n_kf > 0:
            frames = [kf.get("frame") for kf in kfs if "frame" in kf]
            checks[obj_name]["frame_range"] = [frames[0], frames[-1]] if frames else None
            if frames and frames[0] != expected_start:
                result.add_error(
                    "objects",
                    "{} 起始帧={}，预期 {}".format(
                        obj_name, frames[0], expected_start
                    ),
                )
            if frames and frames[-1] != expected_end:
                result.add_error(
                    "objects",
                    "{} 结束帧={}，预期 {}".format(
                        obj_name, frames[-1], expected_end
                    ),
                )
            if frames:
                for i in range(len(frames) - 1):
                    if frames[i+1] - frames[i] != 1:
                        result.add_error("objects", "{} 帧不连续: frame {} → {} 跳变".format(obj_name, frames[i], frames[i+1]))
                        break

        # Yaw check for players
        if obj_name in EXPECTED_PLAYERS:
            has_yaw = all(isinstance(kf, dict) and "yaw_deg" in kf for kf in kfs)
            checks[obj_name]["has_yaw_deg"] = has_yaw
            if not has_yaw:
                result.add_error("objects", "{} 缺少逐帧 yaw_deg".format(obj_name))

            # Action timeline (supports inclusive or half-open source schema)
            at = obj_data.get("action_timeline", [])
            checks[obj_name]["action_timeline_segments"] = len(at)
            if not at:
                result.add_error("objects", "{} 缺少 action_timeline".format(obj_name))
            else:
                normalized = []
                interval_modes = set()
                for idx, seg in enumerate(at):
                    if not isinstance(seg, dict):
                        result.add_error("objects", "{} action_timeline[{}] 不是 object".format(obj_name, idx))
                        continue
                    try:
                        seg_start = int(seg.get("start_frame"))
                        if "end_frame_exclusive" in seg:
                            seg_end_exc = int(seg.get("end_frame_exclusive"))
                            interval_modes.add("half_open")
                            if "end_frame" in seg and seg_end_exc != int(seg.get("end_frame")) + 1:
                                result.add_error("objects", "{} action_timeline[{}] 端帧字段冲突".format(obj_name, idx))
                        elif "end_frame" in seg:
                            seg_end_exc = int(seg.get("end_frame")) + 1
                            interval_modes.add("inclusive_source")
                        else:
                            raise ValueError("missing end field")
                        if seg_end_exc <= seg_start:
                            raise ValueError("non-positive interval")
                        normalized.append((seg_start, seg_end_exc))
                    except Exception as exc:
                        result.add_error("objects", "{} action_timeline[{}] 无效: {}".format(obj_name, idx, exc))

                normalized.sort()
                checks[obj_name]["action_timeline_interval_modes"] = sorted(interval_modes)
                checks[obj_name]["action_timeline_normalized"] = [list(x) for x in normalized]
                if normalized:
                    coverage_start = normalized[0][0]
                    coverage_end_exc = normalized[-1][1]
                    checks[obj_name]["action_coverage_half_open"] = [coverage_start, coverage_end_exc]
                    frame_start = int(config.get("timeline", {}).get("frame_start", 0))
                    frame_end_exc = int(config.get("timeline", {}).get("frame_end", 299)) + 1
                    if coverage_start != frame_start:
                        result.add_error("objects", "{} action_timeline 起始帧={}，预期 {}".format(obj_name, coverage_start, frame_start))
                    if coverage_end_exc != frame_end_exc:
                        result.add_error("objects", "{} action_timeline 结束={}，预期 {}（半开区间）".format(obj_name, coverage_end_exc, frame_end_exc))
                    for idx in range(len(normalized) - 1):
                        curr_end_exc = normalized[idx][1]
                        next_start = normalized[idx + 1][0]
                        if curr_end_exc > next_start:
                            result.add_error("objects", "{} action_timeline 重叠: {} > {}".format(obj_name, curr_end_exc, next_start))
                        elif curr_end_exc < next_start:
                            result.add_error("objects", "{} action_timeline 空洞: {}..{}".format(obj_name, curr_end_exc, next_start))

        else:
            checks[obj_name]["has_state_timeline"] = "state_timeline" in obj_data
            if "state_timeline" not in obj_data:
                result.add_error("objects", "Ball_01 缺少 state_timeline")

    # Top-level metadata
    checks["has_event_frame_map"] = "event_frame_map" in config
    checks["has_possession_timeline"] = "possession_timeline" in config
    checks["has_contact_frames"] = "contact_frames" in config

    if "event_frame_map" not in config:
        result.add_error("objects", "顶层缺少 event_frame_map")
    if "possession_timeline" not in config:
        result.add_error("objects", "顶层缺少 possession_timeline")
    if "contact_frames" not in config:
        result.add_error("objects", "顶层缺少 contact_frames")
    else:
        contact_frames = config.get("contact_frames", [])
        frames_found = []
        event_ids_found = []
        frame_start = int(config.get("timeline", {}).get("frame_start", 0))
        frame_end = int(config.get("timeline", {}).get("frame_end", 299))
        for idx, item in enumerate(contact_frames):
            if not isinstance(item, dict):
                result.add_error("objects", "contact_frames[{}] 不是 object".format(idx))
                continue
            event_id = item.get("event_id")
            frame = item.get("frame")
            if event_id is None or frame is None:
                result.add_error("objects", "contact_frames[{}] 缺少 event_id/frame".format(idx))
                continue
            frame = int(frame)
            frames_found.append(frame)
            event_ids_found.append(str(event_id))
            if frame < frame_start or frame > frame_end:
                result.add_error("objects", "contact frame {} 超出 {}..{}".format(frame, frame_start, frame_end))
        if len(event_ids_found) != len(set(event_ids_found)):
            result.add_error("objects", "contact_frames 存在重复 event_id")
        checks["contact_frames_found"] = frames_found
        checks["contact_event_ids"] = event_ids_found

    # Bbox config
    player_cfg = config.get("player_bbox", config.get("player", {}))
    bbox_keys = sorted(player_cfg.keys()) if isinstance(player_cfg, dict) else []
    checks["bbox_config_keys"] = bbox_keys
    if not bbox_keys:
        result.add_warning("objects", "配置中未发现 bbox 参数")


# ============================================================
# 4. Actor check (UE Editor only)
# ============================================================

def check_actors(result):
    """检查 UE Editor World 中的 Actor。仅在 UE 中有效。"""
    checks = result.checks["actors"]

    # Get editor world
    editor_subsys = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    if not editor_subsys:
        result.add_warning("actors", "无法获取 EditorActorSubsystem（不在 UE 编辑器中？）")
        checks["editor_available"] = False
        return
    checks["editor_available"] = True

    all_actors = editor_subsys.get_all_level_actors()

    def find_actor(label):
        for actor in all_actors:
            if actor.get_actor_label() == label:
                return actor
        return None

    # Check players
    for player_name in EXPECTED_PLAYERS:
        player_entry = {}
        actor = find_actor(player_name)
        if actor:
            player_entry["found"] = True
            player_entry["actor_label"] = actor.get_actor_label()
            player_entry["actor_class"] = actor.get_class().get_name()
            player_entry["object_path"] = actor.get_path_name()
            loc = actor.get_actor_location()
            rot = actor.get_actor_rotation()
            player_entry["location"] = {"x": loc.x, "y": loc.y, "z": loc.z}
            player_entry["rotation"] = {"pitch": rot.pitch, "yaw": rot.yaw, "roll": rot.roll}

            # Skeletal mesh component
            skeletal_comps = actor.get_components_by_class(unreal.SkeletalMeshComponent)
            if skeletal_comps:
                skel_comp = skeletal_comps[0]
                skel_mesh = skel_comp.get_skeletal_mesh_asset()
                if skel_mesh:
                    player_entry["skeletal_mesh_path"] = skel_mesh.get_path_name()
                    # Safe skeleton access (V2 fix: use .skeleton property)
                    skel = safe_get_skeleton(skel_mesh, "{} SkeletalMesh".format(player_name), result)
                    if skel is not None:
                        try:
                            player_entry["skeleton_path"] = skel.get_path_name()
                        except Exception as e:
                            player_entry["skeleton_path"] = None
                            result.add_warning("actors", "{} Skeleton 路径不可读: {}".format(player_name, str(e)))
                    else:
                        player_entry["skeleton_path"] = None
                        result.add_warning("actors", "{} SkeletalMesh 未设置 Skeleton".format(player_name))
                else:
                    player_entry["skeletal_mesh_path"] = None
                    result.add_warning("actors", "{} SkeletalMeshComponent 没有设置 Skeletal Mesh".format(player_name))

                # Anim instance
                anim_instance = skel_comp.get_anim_instance()
                if anim_instance:
                    try:
                        player_entry["anim_instance_class"] = anim_instance.get_class().get_name()
                    except Exception:
                        player_entry["anim_instance_class"] = None
                else:
                    player_entry["anim_instance_class"] = None

                # Control Rig check — enumerate components to find ControlRig patterns
                # V3 fix: use get_components_by_class() wrapped in try/except for UE API compatibility
                has_cr = False
                try:
                    for comp_class in [unreal.SkeletalMeshComponent, unreal.ActorComponent]:
                        try:
                            comps = actor.get_components_by_class(comp_class)
                            for comp in comps:
                                try:
                                    cname = comp.get_class().get_name()
                                    if "control" in cname.lower() or "rig" in cname.lower():
                                        has_cr = True
                                        break
                                except Exception:
                                    pass
                        except Exception:
                            pass
                except Exception:
                    # Cannot enumerate components in this UE build; skip CR check
                    pass
                player_entry["has_control_rig_conflict"] = has_cr
                if has_cr:
                    result.add_warning("actors", "{} 可能存在 Control Rig 组件。如果在 A3.3b Sequencer 中产生冲突，需在动画轨道构建时跳过 Control Rig 轨道。".format(player_name))
            else:
                player_entry["skeletal_mesh_component"] = False
                result.add_error("actors", "{} 未找到 SkeletalMeshComponent".format(player_name))
        else:
            player_entry["found"] = False
            result.add_error("actors", "Actor 不存在: {}".format(player_name))

        checks[player_name] = player_entry

    # Check ball actors
    for ball_name in EXPECTED_BALLS:
        ball_entry = {}
        ball_actor = find_actor(ball_name)
        if ball_actor:
            ball_entry["found"] = True
            ball_entry["actor_label"] = ball_actor.get_actor_label()
            ball_entry["actor_class"] = ball_actor.get_class().get_name()
            ball_entry["object_path"] = ball_actor.get_path_name()
            loc = ball_actor.get_actor_location()
            ball_entry["location"] = {"x": loc.x, "y": loc.y, "z": loc.z}
        else:
            ball_entry["found"] = False
            result.add_error("actors", "Actor 不存在: {}".format(ball_name))
        checks[ball_name] = ball_entry

    # Check cameras
    for cam_name in EXPECTED_CAMERAS:
        cam_entry = {}
        cam_actor = find_actor(cam_name)
        if cam_actor:
            cam_entry["found"] = True
            cam_entry["actor_class"] = cam_actor.get_class().get_name()
            cam_entry["object_path"] = cam_actor.get_path_name()
        else:
            cam_entry["found"] = False
            result.add_error("actors", "CineCameraActor 不存在: {}".format(cam_name))
        checks[cam_name] = cam_entry


# ============================================================
# 5. Animation asset check
# ============================================================

def check_animation_assets(config, action_map, v2_data, result):
    checks = result.checks["animation_assets"]

    # Parse action map
    if action_map is None:
        result.add_warning("animation_assets", "action_animation_map.json 未加载，跳过动画检查")
        return

    anim_section = action_map.get("animation", {})
    strict = anim_section.get("strict_action_assets", False)
    fallback = anim_section.get("fallback_action", "jog")
    action_assets = anim_section.get("action_assets", {})
    checks["strict_action_assets"] = strict
    checks["fallback_action"] = fallback

    # Build V2 cross-reference skeleton lookup
    v2_skeletons = {}
    if v2_data:
        for asset in v2_data.get("all_assets", []):
            name = asset.get("asset_name", "")
            pkg = asset.get("package_path", "")
            full_path = pkg + "/" + name
            v2_skeletons[full_path] = asset.get("skeleton_path")
            v2_skeletons[name] = asset.get("skeleton_path")

    ACTIONS_TO_CHECK = ["idle", "jog", "dribble", "pass", "receive", "shot", "defend"]
    for action_name in ACTIONS_TO_CHECK:
        entry = {}
        path = action_assets.get(action_name, None)
        entry["configured_path"] = path
        entry["is_null"] = path is None

        if path is None:
            if action_name in ["idle", "jog"]:
                result.add_warning("animation_assets", "{} 动画路径为 null（建议为 Idle 和 Jog 配置有效动画）".format(action_name))
            else:
                result.add_warning("animation_assets", "{} 动画路径为 null（fallback 预期，Fab 采购 HOLD）".format(action_name))
            checks[action_name] = entry
            continue

        # Try to load in UE
        try:
            asset = unreal.load_asset(path)
            if asset is None:
                asset = unreal.load_object(name=path, outer=None)
        except Exception as e:
            entry["load_error"] = str(e)
            asset = None

        if asset is None:
            result.add_error("animation_assets", "{} 资产无法加载: {}".format(action_name, path))
            entry["loadable"] = False
            checks[action_name] = entry
            continue

        entry["loadable"] = True
        entry["loaded_class"] = asset.get_class().get_name()
        entry["path"] = path

        # Check type compatibility
        is_seq = isinstance(asset, unreal.AnimSequence)
        is_montage = isinstance(asset, unreal.AnimMontage)
        is_composite = isinstance(asset, unreal.AnimComposite)
        is_compatible = is_seq or is_montage or is_composite
        entry["is_sequencer_section_compatible"] = is_compatible
        if not is_compatible:
            result.add_warning("animation_assets", "{} 资产类型 {} 可能不兼容 Sequencer 动画 Section".format(action_name, asset.get_class().get_name()))

        # Skeleton info (V2 fix: safe access)
        if is_seq or is_montage:
            skeleton = safe_get_skeleton(asset, "{} asset".format(action_name), result)
            if skeleton is not None:
                try:
                    entry["skeleton_path"] = skeleton.get_path_name()
                except Exception as e:
                    entry["skeleton_path"] = None
                    result.add_warning("animation_assets", "{} Skeleton 路径不可读: {}".format(action_name, str(e)))
            else:
                entry["skeleton_path"] = None
                result.add_warning("animation_assets", "{} 资产未设置 Skeleton".format(action_name))

        # Play length
        if hasattr(asset, "get_play_length"):
            try:
                entry["play_length"] = asset.get_play_length()
            except Exception:
                entry["play_length"] = None

        # Cross-reference with V2 scanner
        canonical = path
        if canonical in v2_skeletons:
            entry["v2_skeleton_crossref"] = v2_skeletons[canonical]
            if entry.get("skeleton_path"):
                entry["v2_skeleton_match"] = entry["skeleton_path"] == v2_skeletons[canonical]

        checks[action_name] = entry

    # Skeleton consistency check across assets
    skeleton_paths = set()
    for action_name in ACTIONS_TO_CHECK:
        entry = checks.get(action_name, {})
        sk = entry.get("skeleton_path", None)
        if sk:
            skeleton_paths.add(sk)
    if len(skeleton_paths) > 1:
        result.add_warning("animation_assets", "动画资产使用的 Skeleton 不统一: {}".format(skeleton_paths))
    elif len(skeleton_paths) == 1:
        checks["unified_skeleton"] = list(skeleton_paths)[0]


# ============================================================
# 6. Sequence check (read-only)
# ============================================================

def _asset_data_text(asset_data, name):
    try:
        return str(getattr(asset_data, name))
    except Exception:
        try:
            return str(asset_data.get_editor_property(name))
        except Exception:
            return ""


def _query_level_sequence_assets(registry):
    try:
        class_path = unreal.TopLevelAssetPath("/Script/LevelSequence", "LevelSequence")
        return list(registry.get_assets_by_class(class_path, True) or [])
    except Exception:
        try:
            return list(registry.get_assets_by_class("LevelSequence", True) or [])
        except Exception:
            return []


def check_sequence(config, result):
    checks = result.checks["sequence"]
    sequence_names = config.get("sequences", config.get("sequence_names", []))
    if not isinstance(sequence_names, list) or not sequence_names:
        result.add_error("sequence", "配置缺少 sequences/sequence_names")
        return
    sequence_names = [str(x).strip() for x in sequence_names if str(x).strip()]
    checks["expected_sequence_names"] = sequence_names

    registry = unreal.AssetRegistryHelpers.get_asset_registry()
    if not registry:
        result.add_warning("sequence", "AssetRegistry 不可用")
        return
    assets = _query_level_sequence_assets(registry)
    by_name = {}
    for asset_data in assets:
        asset_name = _asset_data_text(asset_data, "asset_name")
        package_name = _asset_data_text(asset_data, "package_name")
        if asset_name:
            by_name[asset_name] = (asset_data, package_name)

    timeline = config.get("timeline", {})
    expected_start = int(timeline.get("frame_start", 0))
    expected_end_exc = int(timeline.get("frame_end", 299)) + 1
    entries = {}

    for sequence_name in sequence_names:
        entry = {"sequence_name": sequence_name}
        match = by_name.get(sequence_name)
        if match is None:
            entry["exists"] = False
            result.add_warning("sequence", "Level Sequence 尚不存在（首次构建前允许）: {}".format(sequence_name))
            entries[sequence_name] = entry
            continue
        asset_data, package_name = match
        entry["exists"] = True
        entry["package_name"] = package_name
        try:
            seq = asset_data.get_asset()
        except Exception as exc:
            seq = None
            entry["load_error"] = str(exc)
        if seq is None:
            result.add_error("sequence", "Sequence 无法加载: {}".format(sequence_name))
            entries[sequence_name] = entry
            continue

        try:
            entry["display_rate"] = str(seq.get_display_rate())
        except Exception:
            pass
        try:
            entry["playback_start"] = int(seq.get_playback_start())
            entry["playback_end"] = int(seq.get_playback_end())
            if entry["playback_start"] != expected_start or entry["playback_end"] != expected_end_exc:
                result.add_warning(
                    "sequence",
                    "{} playback=[{}, {})，目标=[{}, {})".format(
                        sequence_name,
                        entry["playback_start"],
                        entry["playback_end"],
                        expected_start,
                        expected_end_exc,
                    ),
                )
        except Exception:
            pass

        tracks = []
        for method_name in ("get_tracks", "get_master_tracks"):
            try:
                if hasattr(seq, method_name):
                    tracks.extend(list(getattr(seq, method_name)() or []))
            except Exception:
                pass
        unique = []
        seen = set()
        for track in tracks:
            key = str(track)
            if key not in seen:
                seen.add(key)
                unique.append(track)
        entry["track_count"] = len(unique)
        section_count = 0
        camera_cut_ranges = []
        control_rig_tracks = []
        for track in unique:
            try:
                class_name = track.get_class().get_name()
            except Exception:
                class_name = ""
            if "ControlRig" in class_name:
                control_rig_tracks.append(class_name)
            try:
                sections = list(track.get_sections() or [])
            except Exception:
                sections = []
            section_count += len(sections)
            if "CameraCut" in class_name:
                for section in sections:
                    try:
                        start = int(section.get_start_frame().value)
                        end = int(section.get_end_frame().value)
                        camera_cut_ranges.append([start, end])
                        if start != expected_start or end != expected_end_exc:
                            result.add_warning(
                                "sequence",
                                "{} Camera Cut=[{}, {})，目标=[{}, {})".format(
                                    sequence_name, start, end, expected_start, expected_end_exc
                                ),
                            )
                    except Exception:
                        pass
        entry["section_count"] = section_count
        entry["camera_cut_ranges"] = camera_cut_ranges
        entry["control_rig_tracks"] = control_rig_tracks
        if not camera_cut_ranges:
            result.add_warning("sequence", "{} 未读取到 Camera Cut Section".format(sequence_name))
        entries[sequence_name] = entry
    checks["sequences"] = entries


# ============================================================
# 7. Output path check (read-only)
# ============================================================

def check_output_paths(result):
    checks = result.checks["output_paths"]

    annotations_dir = os.path.join(PROJECT_ROOT, "Saved", "FutsalMOT", "annotations")
    images_clean_dir = os.path.join(PROJECT_ROOT, "Saved", "FutsalMOT", "images_clean")

    for label, d in [("annotations", annotations_dir), ("images_clean", images_clean_dir)]:
        entry = {"resolved_path": d.replace("\\", "/")}
        if os.path.exists(d):
            entry["exists"] = True
        else:
            entry["exists"] = False
            result.add_warning("output_paths", "生产目录尚不存在: {}。首次构建时由 UE/MRQ 创建。".format(d))
        checks[label] = entry

    # Agent test output directory
    agent_output = AGENT_TEST_OUTPUTS
    checks["agent_test_outputs"] = {
        "resolved_path": agent_output.replace("\\", "/"),
        "exists": os.path.exists(agent_output)
    }


# ============================================================
# 8. Write report
# ============================================================

def write_report(result):
    if not os.path.exists(AGENT_TEST_OUTPUTS):
        os.makedirs(AGENT_TEST_OUTPUTS)

    safe_seq_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in result.seq_id)
    report_path = os.path.join(AGENT_TEST_OUTPUTS, "preflight_{}.json".format(safe_seq_id or "unknown"))
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
    unreal.log("[PREFLIGHT] 报告已写入: {}".format(report_path))
    print("[PREFLIGHT] 报告已写入: {}".format(report_path))


# ============================================================
# Main
# ============================================================

def main():
    unreal.log("[PREFLIGHT] === 开始 A3.3b Preflight (8P V5) ===")
    unreal.log("[PREFLIGHT] 版本: {}".format(SCRIPT_VERSION))
    unreal.log("[PREFLIGHT] 只读: True — 不会创建或修改任何 UE 资产")

    result = PreflightResult("unknown")

    # Step 1: Config loading
    unreal.log("[PREFLIGHT] Step 1/7: 加载配置...")
    print("[PREFLIGHT] Step 1/7: 加载配置...")
    loaded = check_config_integrity(result)
    if loaded is False:
        write_report(result)
        unreal.log("[PREFLIGHT] 配置加载失败，终止")
        return

    config, action_map, v2_data = loaded
    result.checks["config_integrity"]["status"] = "loaded"

    # Step 2: Timeline
    unreal.log("[PREFLIGHT] Step 2/7: 检查时间轴...")
    check_timeline(config, result)

    # Step 3: Objects
    unreal.log("[PREFLIGHT] Step 3/7: 检查对象...")
    check_objects(config, result)

    # Step 4: Actors (UE Editor)
    unreal.log("[PREFLIGHT] Step 4/7: 检查 Actor...")
    check_actors(result)

    # Step 5: Animation assets
    unreal.log("[PREFLIGHT] Step 5/7: 检查动画资产...")
    check_animation_assets(config, action_map, v2_data, result)

    # Step 6: Sequence (read-only)
    unreal.log("[PREFLIGHT] Step 6/7: 检查 Sequence...")
    check_sequence(config, result)

    # Step 7: Output paths
    unreal.log("[PREFLIGHT] Step 7/7: 检查输出路径...")
    check_output_paths(result)

    # Summary
    unreal.log("[PREFLIGHT] Status: {}".format(result.status))
    unreal.log("[PREFLIGHT] Errors: {}  Warnings: {}".format(
        len(result.errors), len(result.warnings)))
    for err in result.errors:
        unreal.log("[PREFLIGHT] ERROR   [{}] {}".format(err["category"], err["message"]))
    for warn in result.warnings:
        unreal.log("[PREFLIGHT] WARNING [{}] {}".format(warn["category"], warn["message"]))

    write_report(result)

    unreal.log("[PREFLIGHT] === Preflight 完成 ===")
    print("[PREFLIGHT] 完成。Status: {}".format(result.status))
    print("[PREFLIGHT] 请在 code/_agent_test_outputs/ 下查看完整报告。")


if __name__ == "__main__":
    main()
