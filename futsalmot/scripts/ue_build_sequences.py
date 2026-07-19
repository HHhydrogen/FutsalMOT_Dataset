# FutsalMOT A3.3b episode build script
# Version marker: A3_3B_ACTION_TIMELINE_ANIMATION_SECTIONS_8P_V3
# Based on the stable A2.2 multi-keyframe / camera-cut / safe-bbox pipeline.
# Adds per-action Skeletal Animation sections driven by objects.<id>.action_timeline.
# Missing action assets fall back safely to Jog unless strict_action_assets=true.

import unreal
import os
import json
import math

# ============================================================
# External JSON config
# ============================================================

# 默认项目路径仅用于定位配置文件。
# 正式项目根目录仍优先读取 JSON 中的 project_root。
DEFAULT_PROJECT_ROOT = "D:/projects/FustalMOT_UEDataset"

# 在 UE 的“执行 Python 脚本”方式下，通常可以取得 __file__。
# 如果某些 UE 版本没有提供 __file__，则回退到当前项目的 code 目录。
try:
    SCRIPT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
except Exception:
    SCRIPT_DIR = os.path.join(
        DEFAULT_PROJECT_ROOT,
        "Content",
        "FutsalMOT",
        "code"
    )

CURRENT_RUN_POINTER = os.path.join(
    SCRIPT_DIR,
    "configs",
    "pipeline_current.json",
)


def resolve_pipeline_config_path():
    """Resolve A3.3 config: env override -> current-run pointer -> legacy default."""
    env_path = os.environ.get("FUTSALMOT_CONFIG_PATH")
    if env_path:
        return os.path.normpath(os.path.abspath(env_path)), "environment"

    if os.path.isfile(CURRENT_RUN_POINTER):
        try:
            with open(CURRENT_RUN_POINTER, "r", encoding="utf-8-sig") as f:
                pointer = json.load(f)
            raw_path = pointer.get("paths", {}).get("a3_3_config")
            if isinstance(raw_path, str) and raw_path.strip():
                path = raw_path.strip()
                if not os.path.isabs(path):
                    path = os.path.join(SCRIPT_DIR, path)
                return os.path.normpath(os.path.abspath(path)), "pipeline_current.json"
        except Exception as exc:
            unreal.log_warning(
                "无法读取 pipeline_current.json，将使用 legacy default: {}".format(exc)
            )

    legacy = os.path.join(
        SCRIPT_DIR,
        "configs",
        "events",
        "generated",
        "episode_random_0001_t1_a33.json",
    )
    return os.path.normpath(os.path.abspath(legacy)), "legacy_default"


CONFIG_PATH, CONFIG_PATH_SOURCE = resolve_pipeline_config_path()

# 可选动画映射覆盖文件。用于把 action_timeline 中的动作名映射到
# 当前 UE 项目内真实存在的 AnimSequence 资源。
ACTION_MAP_OVERRIDE_PATH = os.environ.get(
    "FUTSALMOT_ACTION_MAP_PATH",
    os.path.join(SCRIPT_DIR, "configs", "action_animation_map.json")
)


DEFAULT_PLAYER_BBOX_BONES = [
    "head",
    "neck_01",
    "spine_05",
    "spine_04",
    "spine_03",
    "spine_02",
    "spine_01",
    "pelvis",

    "clavicle_l",
    "upperarm_l",
    "lowerarm_l",
    "hand_l",
    "index_03_l",
    "middle_03_l",

    "clavicle_r",
    "upperarm_r",
    "lowerarm_r",
    "hand_r",
    "index_03_r",
    "middle_03_r",

    "thigh_l",
    "calf_l",
    "foot_l",
    "ball_l",

    "thigh_r",
    "calf_r",
    "foot_r",
    "ball_r",
]

# 各骨骼中心到人体轮廓的近似半径，单位 cm。
# 旧算法只投影骨骼中心线，因此手臂、腿、头顶和鞋会落在框外。
# 新算法把每个骨骼视为一个面向相机的圆形包络，再投影求最小外接矩形。
DEFAULT_PLAYER_BONE_RADIUS_CM = {
    "head": 15.0,
    "neck_01": 10.0,
    "spine_05": 16.0,
    "spine_04": 17.0,
    "spine_03": 18.0,
    "spine_02": 18.0,
    "spine_01": 17.0,
    "pelvis": 18.0,

    "clavicle_l": 11.0,
    "upperarm_l": 9.0,
    "lowerarm_l": 7.5,
    "hand_l": 8.0,
    "index_03_l": 4.0,
    "middle_03_l": 4.0,

    "clavicle_r": 11.0,
    "upperarm_r": 9.0,
    "lowerarm_r": 7.5,
    "hand_r": 8.0,
    "index_03_r": 4.0,
    "middle_03_r": 4.0,

    "thigh_l": 11.0,
    "calf_l": 9.0,
    "foot_l": 10.0,
    "ball_l": 8.0,

    "thigh_r": 11.0,
    "calf_r": 9.0,
    "foot_r": 10.0,
    "ball_r": 8.0,
}


def fail_config(message):
    raise RuntimeError(
        "配置文件错误：{}\nCONFIG_PATH={}".format(message, CONFIG_PATH)
    )


def load_json_config(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "找不到序列配置文件：{}\n"
            "请先运行 00_run_pipeline.py，或设置 FUTSALMOT_CONFIG_PATH。".format(path)
        )

    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            "JSON 格式错误：{}，line={}，column={}".format(
                e.msg,
                e.lineno,
                e.colno
            )
        )

    if not isinstance(data, dict):
        fail_config("JSON 顶层必须是 object/dict")

    return data


def require_dict(parent, key):
    value = parent.get(key)

    if not isinstance(value, dict):
        fail_config("字段 '{}' 必须是 JSON object".format(key))

    return value


def require_list(parent, key):
    value = parent.get(key)

    if not isinstance(value, list):
        fail_config("字段 '{}' 必须是 JSON array".format(key))

    return value


def require_nonempty_string(parent, key):
    value = parent.get(key)

    if not isinstance(value, str) or not value.strip():
        fail_config("字段 '{}' 必须是非空字符串".format(key))

    return value.strip()


def to_int(value, field_name):
    try:
        return int(value)
    except Exception:
        fail_config("字段 '{}' 必须可以转换为整数，当前值={}".format(field_name, value))


def to_float(value, field_name):
    try:
        return float(value)
    except Exception:
        fail_config("字段 '{}' 必须可以转换为浮点数，当前值={}".format(field_name, value))


def to_bool(value, field_name):
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(value)

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ["true", "1", "yes", "on"]:
            return True
        if normalized in ["false", "0", "no", "off"]:
            return False

    fail_config("字段 '{}' 必须是布尔值，当前值={}".format(field_name, value))


def vector_from_json(value, field_name):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        fail_config("字段 '{}' 必须是长度为 3 的数组 [x, y, z]".format(field_name))

    return unreal.Vector(
        to_float(value[0], field_name + "[0]"),
        to_float(value[1], field_name + "[1]"),
        to_float(value[2], field_name + "[2]")
    )


def optional_vector_from_json(parent, key, default_value):
    value = parent.get(key, default_value)
    return vector_from_json(value, key)


def normalize_path(path):
    return os.path.normpath(os.path.abspath(path))


def deep_merge_dict(base, override):
    result = dict(base)

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge_dict(result[key], value)
        else:
            result[key] = value

    return result


CONFIG = load_json_config(CONFIG_PATH)

# action_animation_map.json 是可选覆盖，不改变轨迹生成器输出文件。
# 用户只需要维护一份动画路径映射，后续所有 episode 均可复用。
if os.path.isfile(ACTION_MAP_OVERRIDE_PATH):
    try:
        with open(ACTION_MAP_OVERRIDE_PATH, "r", encoding="utf-8-sig") as f:
            action_override = json.load(f)

        if not isinstance(action_override, dict):
            fail_config("action_animation_map.json 顶层必须是 JSON object")

        override_animation = action_override.get(
            "animation",
            action_override
        )

        if not isinstance(override_animation, dict):
            fail_config(
                "action_animation_map.json 的 animation 必须是 JSON object"
            )

        CONFIG["animation"] = deep_merge_dict(
            CONFIG.get("animation", {}),
            override_animation
        )
    except json.JSONDecodeError as e:
        raise RuntimeError(
            "动作动画映射 JSON 格式错误：{} line={} column={}\nPATH={}".format(
                e.msg,
                e.lineno,
                e.colno,
                ACTION_MAP_OVERRIDE_PATH
            )
        )

# ============================================================
# Basic config loaded from JSON
# ============================================================

PROJECT_ROOT = normalize_path(
    str(CONFIG.get("project_root", DEFAULT_PROJECT_ROOT))
)

SEQ_ID = require_nonempty_string(CONFIG, "seq_id")

IMAGE_CONFIG = require_dict(CONFIG, "image")
IMAGE_WIDTH = to_int(IMAGE_CONFIG.get("width"), "image.width")
IMAGE_HEIGHT = to_int(IMAGE_CONFIG.get("height"), "image.height")

if IMAGE_WIDTH <= 0 or IMAGE_HEIGHT <= 0:
    fail_config("image.width 和 image.height 必须大于 0")

TIMELINE_CONFIG = require_dict(CONFIG, "timeline")
FRAME_START = to_int(TIMELINE_CONFIG.get("frame_start"), "timeline.frame_start")
FRAME_END = to_int(TIMELINE_CONFIG.get("frame_end"), "timeline.frame_end")
DISPLAY_RATE = to_int(TIMELINE_CONFIG.get("display_rate", 30), "timeline.display_rate")

if FRAME_END < FRAME_START:
    fail_config("timeline.frame_end 不能小于 timeline.frame_start")

if DISPLAY_RATE <= 0:
    fail_config("timeline.display_rate 必须大于 0")

# 不允许单独配置，始终由 frame_end 自动计算，避免不同步。
FRAME_END_EXCLUSIVE = FRAME_END + 1

SEQUENCE_NAMES = CONFIG.get("sequences", CONFIG.get("sequence_names"))

if not isinstance(SEQUENCE_NAMES, list) or len(SEQUENCE_NAMES) == 0:
    fail_config("字段 'sequences' 必须是非空数组")

for i, seq_name in enumerate(SEQUENCE_NAMES):
    if not isinstance(seq_name, str) or not seq_name.strip():
        fail_config("sequences[{}] 必须是非空字符串".format(i))

SEQUENCE_NAMES = [str(x).strip() for x in SEQUENCE_NAMES]

CAMERAS_RAW = require_dict(CONFIG, "cameras")

if len(CAMERAS_RAW) == 0:
    fail_config("字段 'cameras' 不能为空")

CAMERAS = {}

for cam_id, actor_label in CAMERAS_RAW.items():
    if not isinstance(cam_id, str) or not cam_id.strip():
        fail_config("cameras 中的 camera_id 必须是非空字符串")

    if not isinstance(actor_label, str) or not actor_label.strip():
        fail_config("cameras.{} 的 Actor Label 必须是非空字符串".format(cam_id))

    CAMERAS[cam_id.strip()] = actor_label.strip()

ANIMATION_CONFIG = CONFIG.get("animation", {})

if not isinstance(ANIMATION_CONFIG, dict):
    fail_config("字段 'animation' 必须是 JSON object")

ANIM_PATH = str(
    ANIMATION_CONFIG.get(
        "jog_asset",
        ANIMATION_CONFIG.get(
            "asset_path",
            "/Game/Characters/Mannequins/Anims/Unarmed/Jog/MF_Unarmed_Jog_Fwd"
        )
    )
).strip()

PLAY_RATE = to_float(
    ANIMATION_CONFIG.get("play_rate", 1.0),
    "animation.play_rate"
)

if PLAY_RATE <= 0.0:
    fail_config("animation.play_rate 必须大于 0")

SUPPORTED_ACTIONS = [
    "idle",
    "jog",
    "dribble",
    "pass",
    "receive",
    "shot",
    "defend",
]

FALLBACK_ACTION = str(
    ANIMATION_CONFIG.get("fallback_action", "jog")
).strip().lower()

if FALLBACK_ACTION not in SUPPORTED_ACTIONS:
    fail_config(
        "animation.fallback_action 必须是 {} 中之一".format(
            SUPPORTED_ACTIONS
        )
    )

STRICT_ACTION_ASSETS = to_bool(
    ANIMATION_CONFIG.get("strict_action_assets", False),
    "animation.strict_action_assets"
)

RAW_ACTION_ASSETS = ANIMATION_CONFIG.get(
    "action_assets",
    ANIMATION_CONFIG.get("animation_map", {})
)

if RAW_ACTION_ASSETS is None:
    RAW_ACTION_ASSETS = {}

if not isinstance(RAW_ACTION_ASSETS, dict):
    fail_config("animation.action_assets 必须是 JSON object")

ACTION_ASSET_PATHS = {}

for action_name in SUPPORTED_ACTIONS:
    raw_path = RAW_ACTION_ASSETS.get(action_name)

    if raw_path is None and action_name == "jog":
        raw_path = ANIM_PATH

    if raw_path is None:
        ACTION_ASSET_PATHS[action_name] = None
    else:
        path_text = str(raw_path).strip()
        ACTION_ASSET_PATHS[action_name] = path_text if path_text else None

RAW_ACTION_PLAY_RATES = ANIMATION_CONFIG.get(
    "action_play_rates",
    {}
)

if RAW_ACTION_PLAY_RATES is None:
    RAW_ACTION_PLAY_RATES = {}

if not isinstance(RAW_ACTION_PLAY_RATES, dict):
    fail_config("animation.action_play_rates 必须是 JSON object")

ACTION_PLAY_RATES = {}

for action_name in SUPPORTED_ACTIONS:
    action_rate = to_float(
        RAW_ACTION_PLAY_RATES.get(action_name, PLAY_RATE),
        "animation.action_play_rates.{}".format(action_name)
    )

    if action_rate <= 0.0:
        fail_config(
            "animation.action_play_rates.{} 必须大于 0".format(
                action_name
            )
        )

    ACTION_PLAY_RATES[action_name] = action_rate

# 默认回退链。显式 action 资源不存在时，依次尝试这些动作的资源。
DEFAULT_ACTION_FALLBACK_CHAIN = {
    "idle": ["idle", "jog"],
    "jog": ["jog"],
    "dribble": ["dribble", "jog"],
    "pass": ["pass", "shot", "jog"],
    "receive": ["receive", "idle", "jog"],
    "shot": ["shot", "pass", "jog"],
    "defend": ["defend", "jog"],
}

PLAYER_CONFIG = CONFIG.get("player", {})

if not isinstance(PLAYER_CONFIG, dict):
    fail_config("字段 'player' 必须是 JSON object")

PLAYER_FORWARD_CORRECTION_DEG = to_float(
    PLAYER_CONFIG.get("forward_correction_deg", 0.0),
    "player.forward_correction_deg"
)

PLAYER_GROUND_Z_CM = to_float(
    PLAYER_CONFIG.get("ground_z_cm", 90.0),
    "player.ground_z_cm"
)

PLAYER_BBOX_CENTER_Z_OFFSET_CM = to_float(
    PLAYER_CONFIG.get("bbox_center_z_offset_cm", 0.0),
    "player.bbox_center_z_offset_cm"
)

PLAYER_BBOX_EXTENT_CM = optional_vector_from_json(
    PLAYER_CONFIG,
    "bbox_extent_cm",
    [38.0, 38.0, 92.0]
)

PLAYER_2D_BBOX_PADDING_PX = to_float(
    PLAYER_CONFIG.get("bbox_padding_px", 3.0),
    "player.bbox_padding_px"
)

if PLAYER_2D_BBOX_PADDING_PX < 0.0:
    fail_config("player.bbox_padding_px 不能小于 0")

# 在骨骼包络之外，再按人物投影高度增加少量自适应边距。
# 该边距用于吸收动画求值差异、运动模糊和骨骼中心到网格表面的残差。
PLAYER_2D_BBOX_ADAPTIVE_PADDING_RATIO = to_float(
    PLAYER_CONFIG.get("bbox_adaptive_padding_ratio", 0.025),
    "player.bbox_adaptive_padding_ratio"
)

if PLAYER_2D_BBOX_ADAPTIVE_PADDING_RATIO < 0.0:
    fail_config("player.bbox_adaptive_padding_ratio 不能小于 0")

PLAYER_BBOX_BONE_RADIUS_SCALE = to_float(
    PLAYER_CONFIG.get("bbox_bone_radius_scale", 1.10),
    "player.bbox_bone_radius_scale"
)

if PLAYER_BBOX_BONE_RADIUS_SCALE <= 0.0:
    fail_config("player.bbox_bone_radius_scale 必须大于 0")

PLAYER_BBOX_DEFAULT_BONE_RADIUS_CM = to_float(
    PLAYER_CONFIG.get("bbox_default_bone_radius_cm", 7.0),
    "player.bbox_default_bone_radius_cm"
)

if PLAYER_BBOX_DEFAULT_BONE_RADIUS_CM <= 0.0:
    fail_config("player.bbox_default_bone_radius_cm 必须大于 0")

PLAYER_BBOX_BONE_RADIUS_CM = dict(DEFAULT_PLAYER_BONE_RADIUS_CM)
PLAYER_BBOX_BONE_RADIUS_OVERRIDES = PLAYER_CONFIG.get("bbox_bone_radius_cm", {})

if PLAYER_BBOX_BONE_RADIUS_OVERRIDES is None:
    PLAYER_BBOX_BONE_RADIUS_OVERRIDES = {}

if not isinstance(PLAYER_BBOX_BONE_RADIUS_OVERRIDES, dict):
    fail_config("player.bbox_bone_radius_cm 必须是 JSON object")

for bone_name, radius_value in PLAYER_BBOX_BONE_RADIUS_OVERRIDES.items():
    bone_name = str(bone_name).strip()

    if not bone_name:
        fail_config("player.bbox_bone_radius_cm 中的骨骼名不能为空")

    radius_cm = to_float(
        radius_value,
        "player.bbox_bone_radius_cm.{}".format(bone_name)
    )

    if radius_cm <= 0.0:
        fail_config("player.bbox_bone_radius_cm.{} 必须大于 0".format(bone_name))

    PLAYER_BBOX_BONE_RADIUS_CM[bone_name] = radius_cm

PLAYER_BBOX_EVALUATE_SEQUENCE_POSE = to_bool(
    PLAYER_CONFIG.get("bbox_evaluate_sequence_pose", True),
    "player.bbox_evaluate_sequence_pose"
)

PLAYER_BBOX_BONES = PLAYER_CONFIG.get(
    "bbox_bones",
    DEFAULT_PLAYER_BBOX_BONES
)

if not isinstance(PLAYER_BBOX_BONES, list) or len(PLAYER_BBOX_BONES) == 0:
    fail_config("player.bbox_bones 必须是非空数组")

PLAYER_BBOX_BONES = [str(x).strip() for x in PLAYER_BBOX_BONES if str(x).strip()]

PLAYER_KEYPOINT_BONES = PLAYER_CONFIG.get(
    "keypoint_bones",
    PLAYER_BBOX_BONES
)

if not isinstance(PLAYER_KEYPOINT_BONES, list) or len(PLAYER_KEYPOINT_BONES) == 0:
    fail_config("player.keypoint_bones 必须是非空数组")

PLAYER_KEYPOINT_BONES = [str(x).strip() for x in PLAYER_KEYPOINT_BONES if str(x).strip()]

if len(PLAYER_KEYPOINT_BONES) == 0:
    fail_config("player.keypoint_bones 不能全为空字符串")

BALL_CONFIG = CONFIG.get("ball", {})

if not isinstance(BALL_CONFIG, dict):
    fail_config("字段 'ball' 必须是 JSON object")

BALL_RADIUS_CM = to_float(
    BALL_CONFIG.get("radius_cm", 11.0),
    "ball.radius_cm"
)

if BALL_RADIUS_CM <= 0.0:
    fail_config("ball.radius_cm 必须大于 0")

CLASS_ID_MAP_RAW = CONFIG.get(
    "class_id_map",
    {
        "player": 0,
        "ball": 1,
    }
)

if not isinstance(CLASS_ID_MAP_RAW, dict) or len(CLASS_ID_MAP_RAW) == 0:
    fail_config("class_id_map 必须是非空 JSON object")

CLASS_ID_MAP = {}

for category_name, class_id in CLASS_ID_MAP_RAW.items():
    category_name = str(category_name).strip()

    if not category_name:
        fail_config("class_id_map 中的类别名不能为空")

    CLASS_ID_MAP[category_name] = to_int(
        class_id,
        "class_id_map.{}".format(category_name)
    )

RAW_TRACK_ID_MAP = CONFIG.get("track_id_map", {})

if RAW_TRACK_ID_MAP is None:
    RAW_TRACK_ID_MAP = {}

if not isinstance(RAW_TRACK_ID_MAP, dict):
    fail_config("track_id_map 必须是 JSON object")

RAW_OBJECTS = require_dict(CONFIG, "objects")

if len(RAW_OBJECTS) == 0:
    fail_config("字段 'objects' 不能为空")


def parse_object_keyframes(obj_label, raw_cfg):
    """
    A2 支持两种格式：

    1. 新格式：keyframes=[{"frame": 0, "loc": [...]}, ...]
    2. 旧格式：start/end（自动转换为首尾两个关键帧）

    当前只支持 linear 分段插值。
    """
    interpolation = str(raw_cfg.get("interpolation", "linear")).strip().lower()

    if interpolation != "linear":
        fail_config(
            "objects.{}.interpolation 当前只支持 'linear'，当前值={}".format(
                obj_label,
                interpolation
            )
        )

    raw_keyframes = raw_cfg.get("keyframes")
    trajectory_source = "keyframes"

    if raw_keyframes is None:
        if "start" not in raw_cfg or "end" not in raw_cfg:
            fail_config(
                "objects.{} 必须提供 keyframes，或同时提供旧格式 start/end".format(
                    obj_label
                )
            )

        raw_keyframes = [
            {
                "frame": FRAME_START,
                "loc": raw_cfg.get("start"),
            },
            {
                "frame": FRAME_END,
                "loc": raw_cfg.get("end"),
            },
        ]
        trajectory_source = "legacy_start_end"

    if not isinstance(raw_keyframes, list) or len(raw_keyframes) == 0:
        fail_config("objects.{}.keyframes 必须是非空数组".format(obj_label))

    keyframes = []
    seen_frames = set()

    for i, raw_kf in enumerate(raw_keyframes):
        field_prefix = "objects.{}.keyframes[{}]".format(obj_label, i)

        if not isinstance(raw_kf, dict):
            fail_config("{} 必须是 JSON object".format(field_prefix))

        if "frame" not in raw_kf:
            fail_config("{}.frame 缺失".format(field_prefix))

        frame = to_int(raw_kf.get("frame"), field_prefix + ".frame")

        if frame in seen_frames:
            fail_config(
                "objects.{} 的 frame={} 出现重复关键帧".format(
                    obj_label,
                    frame
                )
            )

        seen_frames.add(frame)

        loc = vector_from_json(
            raw_kf.get("loc"),
            field_prefix + ".loc"
        )

        explicit_yaw = None

        if "yaw_deg" in raw_kf and raw_kf.get("yaw_deg") is not None:
            explicit_yaw = to_float(
                raw_kf.get("yaw_deg"),
                field_prefix + ".yaw_deg"
            )

        keyframes.append({
            "frame": frame,
            "loc": loc,
            "explicit_yaw_deg": explicit_yaw,
        })

    keyframes.sort(key=lambda k: k["frame"])

    if keyframes[0]["frame"] != FRAME_START:
        fail_config(
            "objects.{} 第一个关键帧必须等于 timeline.frame_start={}，当前={}".format(
                obj_label,
                FRAME_START,
                keyframes[0]["frame"]
            )
        )

    if keyframes[-1]["frame"] != FRAME_END:
        fail_config(
            "objects.{} 最后一个关键帧必须等于 timeline.frame_end={}，当前={}".format(
                obj_label,
                FRAME_END,
                keyframes[-1]["frame"]
            )
        )

    return keyframes, interpolation, trajectory_source



def normalize_source_events(raw_value):
    if raw_value is None:
        return []

    if isinstance(raw_value, str):
        raw_value = [raw_value]

    if not isinstance(raw_value, list):
        fail_config("action_timeline.source_events 必须是数组或字符串")

    result = []

    for value in raw_value:
        value = str(value).strip()
        if value and value not in result:
            result.append(value)

    return result


def merge_adjacent_action_segments(segments):
    merged = []

    for segment in segments:
        if (
            merged
            and merged[-1]["action"] == segment["action"]
            and merged[-1]["end_frame"] + 1 == segment["start_frame"]
        ):
            merged[-1]["end_frame"] = segment["end_frame"]

            for event_id in segment["source_events"]:
                if event_id not in merged[-1]["source_events"]:
                    merged[-1]["source_events"].append(event_id)
        else:
            merged.append({
                "start_frame": int(segment["start_frame"]),
                "end_frame": int(segment["end_frame"]),
                "action": str(segment["action"]),
                "source_events": list(segment["source_events"]),
            })

    return merged


def parse_action_timeline(obj_label, raw_cfg, use_animation):
    if not use_animation:
        return []

    raw_timeline = raw_cfg.get("action_timeline")

    if raw_timeline is None:
        return [{
            "start_frame": FRAME_START,
            "end_frame": FRAME_END,
            "action": FALLBACK_ACTION,
            "source_events": [],
        }]

    if not isinstance(raw_timeline, list) or len(raw_timeline) == 0:
        fail_config(
            "objects.{}.action_timeline 必须是非空数组".format(
                obj_label
            )
        )

    parsed = []

    for index, raw_segment in enumerate(raw_timeline):
        prefix = "objects.{}.action_timeline[{}]".format(
            obj_label,
            index
        )

        if not isinstance(raw_segment, dict):
            fail_config("{} 必须是 JSON object".format(prefix))

        start_frame = to_int(
            raw_segment.get("start_frame"),
            prefix + ".start_frame"
        )
        end_frame = to_int(
            raw_segment.get("end_frame"),
            prefix + ".end_frame"
        )
        action = str(
            raw_segment.get("action", "")
        ).strip().lower()

        if action not in SUPPORTED_ACTIONS:
            fail_config(
                "{}.action={} 不受支持；允许值={}".format(
                    prefix,
                    action,
                    SUPPORTED_ACTIONS
                )
            )

        if start_frame < FRAME_START or end_frame > FRAME_END:
            fail_config(
                "{} 范围 [{}, {}] 超出时间轴 [{}, {}]".format(
                    prefix,
                    start_frame,
                    end_frame,
                    FRAME_START,
                    FRAME_END
                )
            )

        if end_frame < start_frame:
            fail_config(
                "{}.end_frame 不能小于 start_frame".format(prefix)
            )

        parsed.append({
            "start_frame": start_frame,
            "end_frame": end_frame,
            "action": action,
            "source_events": normalize_source_events(
                raw_segment.get("source_events")
            ),
        })

    parsed.sort(
        key=lambda item: (
            item["start_frame"],
            item["end_frame"],
            item["action"]
        )
    )

    completed = []
    cursor = FRAME_START

    for segment in parsed:
        if segment["start_frame"] < cursor:
            fail_config(
                "objects.{} 的 action_timeline 在 frame={} 发生重叠".format(
                    obj_label,
                    segment["start_frame"]
                )
            )

        if segment["start_frame"] > cursor:
            completed.append({
                "start_frame": cursor,
                "end_frame": segment["start_frame"] - 1,
                "action": FALLBACK_ACTION,
                "source_events": [],
            })

        completed.append(segment)
        cursor = segment["end_frame"] + 1

    if cursor <= FRAME_END:
        completed.append({
            "start_frame": cursor,
            "end_frame": FRAME_END,
            "action": FALLBACK_ACTION,
            "source_events": [],
        })

    return merge_adjacent_action_segments(completed)



OBJECT_TRAJECTORIES = {}
TRACK_ID_MAP = {}
USED_TRACK_IDS = set()

for obj_label, raw_cfg in RAW_OBJECTS.items():
    if not isinstance(obj_label, str) or not obj_label.strip():
        fail_config("objects 中的 object label 必须是非空字符串")

    obj_label = obj_label.strip()

    if not isinstance(raw_cfg, dict):
        fail_config("objects.{} 必须是 JSON object".format(obj_label))

    category = require_nonempty_string(raw_cfg, "category")

    if category not in CLASS_ID_MAP:
        if "class_id" in raw_cfg:
            CLASS_ID_MAP[category] = to_int(
                raw_cfg["class_id"],
                "objects.{}.class_id".format(obj_label)
            )
        else:
            fail_config(
                "objects.{}.category='{}' 不在 class_id_map 中".format(
                    obj_label,
                    category
                )
            )

    if "class_id" in raw_cfg:
        object_class_id = to_int(
            raw_cfg["class_id"],
            "objects.{}.class_id".format(obj_label)
        )

        if object_class_id != CLASS_ID_MAP[category]:
            fail_config(
                "objects.{}.class_id={} 与 class_id_map.{}={} 不一致".format(
                    obj_label,
                    object_class_id,
                    category,
                    CLASS_ID_MAP[category]
                )
            )

    track_id_value = raw_cfg.get(
        "track_id",
        RAW_TRACK_ID_MAP.get(obj_label)
    )

    if track_id_value is None:
        fail_config(
            "objects.{} 缺少 track_id，且 track_id_map 中也未定义".format(obj_label)
        )

    track_id = to_int(
        track_id_value,
        "objects.{}.track_id".format(obj_label)
    )

    if track_id < 0:
        fail_config("objects.{}.track_id 不能小于 0".format(obj_label))

    if track_id in USED_TRACK_IDS:
        fail_config("track_id={} 被多个对象重复使用".format(track_id))

    USED_TRACK_IDS.add(track_id)
    TRACK_ID_MAP[obj_label] = track_id

    scale_vec = vector_from_json(
        raw_cfg.get("scale", [1.0, 1.0, 1.0]),
        "objects.{}.scale".format(obj_label)
    )

    if scale_vec.x <= 0.0 or scale_vec.y <= 0.0 or scale_vec.z <= 0.0:
        fail_config("objects.{}.scale 的三个值都必须大于 0".format(obj_label))

    use_animation = bool(
        raw_cfg.get("use_animation", category == "player")
    )

    keyframes, interpolation, trajectory_source = parse_object_keyframes(
        obj_label,
        raw_cfg
    )

    action_timeline = parse_action_timeline(
        obj_label,
        raw_cfg,
        use_animation
    )

    OBJECT_TRAJECTORIES[obj_label] = {
        "category": category,
        "team": raw_cfg.get("team"),
        "role": raw_cfg.get("role"),
        "keyframes": keyframes,
        "scale": scale_vec,
        "use_animation": use_animation,
        "interpolation": interpolation,
        "trajectory_source": trajectory_source,
        "action_timeline": action_timeline,
        # 在找到 Actor 后，根据模型 Mesh 相对旋转计算。
        "yaw_keyframes_unwrapped": None,
    }

ANNOTATION_DIR = os.path.join(
    PROJECT_ROOT,
    "Saved",
    "FutsalMOT",
    "annotations"
)
os.makedirs(ANNOTATION_DIR, exist_ok=True)

OUT_JSON = os.path.join(
    ANNOTATION_DIR,
    "objects_bbox_2d_clean_{}.json".format(SEQ_ID)
)

OUT_JSONL = os.path.join(
    ANNOTATION_DIR,
    "objects_bbox_2d_clean_{}.jsonl".format(SEQ_ID)
)

# ============================================================
# General helpers
# ============================================================

def find_actor_by_label(label):
    actors = unreal.EditorLevelLibrary.get_all_level_actors()

    for actor in actors:
        if actor.get_actor_label() == label:
            return actor

    return None


def load_asset(path):
    asset = unreal.EditorAssetLibrary.load_asset(path)

    if asset is not None:
        return asset

    asset_name = path.split("/")[-1]
    return unreal.EditorAssetLibrary.load_asset(path + "." + asset_name)



def load_action_animation_assets():
    loaded_by_path = {}
    resolved_assets = {}
    resolution_report = {}

    def load_cached(path):
        if not path:
            return None

        if path not in loaded_by_path:
            loaded_by_path[path] = load_asset(path)

        return loaded_by_path[path]

    jog_path = ACTION_ASSET_PATHS.get("jog") or ANIM_PATH
    jog_asset = load_cached(jog_path)

    if jog_asset is None:
        raise RuntimeError(
            "无法加载 Jog/fallback 动画资源: {}".format(jog_path)
        )

    for action_name in SUPPORTED_ACTIONS:
        requested_path = ACTION_ASSET_PATHS.get(action_name)
        resolved_asset = None
        resolved_action = None
        resolved_path = None

        for candidate_action in DEFAULT_ACTION_FALLBACK_CHAIN[action_name]:
            candidate_path = ACTION_ASSET_PATHS.get(candidate_action)

            if candidate_action == "jog" and not candidate_path:
                candidate_path = jog_path

            candidate_asset = load_cached(candidate_path)

            if candidate_asset is not None:
                resolved_asset = candidate_asset
                resolved_action = candidate_action
                resolved_path = candidate_path
                break

        if resolved_asset is None:
            if STRICT_ACTION_ASSETS:
                raise RuntimeError(
                    "动作 {} 没有可加载动画资源，且 strict_action_assets=true".format(
                        action_name
                    )
                )

            resolved_asset = jog_asset
            resolved_action = "jog"
            resolved_path = jog_path

        fallback_used = (
            requested_path is None
            or requested_path != resolved_path
            or resolved_action != action_name
        )

        if STRICT_ACTION_ASSETS and fallback_used:
            raise RuntimeError(
                "动作 {} 未解析到自身动画资源；requested={} resolved={}".format(
                    action_name,
                    requested_path,
                    resolved_path
                )
            )

        resolved_assets[action_name] = resolved_asset
        resolution_report[action_name] = {
            "requested_path": requested_path,
            "resolved_action": resolved_action,
            "resolved_path": resolved_path,
            "fallback_used": bool(fallback_used),
            "play_rate": float(ACTION_PLAY_RATES[action_name]),
        }

        if fallback_used:
            unreal.log_warning(
                "[ANIM FALLBACK] action={} requested={} -> {} ({})".format(
                    action_name,
                    requested_path,
                    resolved_action,
                    resolved_path
                )
            )
        else:
            unreal.log(
                "[ANIM OK] action={} asset={} play_rate={}".format(
                    action_name,
                    resolved_path,
                    ACTION_PLAY_RATES[action_name]
                )
            )

    return resolved_assets, resolution_report


def find_level_sequence_by_name(seq_name):
    asset_registry = unreal.AssetRegistryHelpers.get_asset_registry()

    try:
        class_path = unreal.TopLevelAssetPath("/Script/LevelSequence", "LevelSequence")
        assets = asset_registry.get_assets_by_class(class_path, True)
    except Exception:
        assets = asset_registry.get_assets_by_class("LevelSequence", True)

    for asset_data in assets:
        try:
            asset_name = str(asset_data.asset_name)
        except Exception:
            asset_name = ""

        if asset_name == seq_name:
            try:
                return asset_data.get_asset()
            except Exception:
                return unreal.EditorAssetLibrary.load_asset(
                    asset_data.get_soft_object_path().to_string()
                )

    return None


def safe_get(obj, prop_name, default=None):
    try:
        return obj.get_editor_property(prop_name)
    except Exception:
        try:
            return getattr(obj, prop_name)
        except Exception:
            return default


def normalize_yaw_deg(yaw):
    while yaw > 180.0:
        yaw -= 360.0

    while yaw < -180.0:
        yaw += 360.0

    return yaw


def movement_yaw_from_start_end(start, end):
    dx = float(end.x - start.x)
    dy = float(end.y - start.y)

    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0.0

    yaw = math.degrees(math.atan2(dy, dx))
    return normalize_yaw_deg(yaw)


def get_mesh_relative_yaw(actor):
    skel_comp = actor.get_component_by_class(unreal.SkeletalMeshComponent)

    if skel_comp is None:
        return 0.0

    try:
        rel_rot = skel_comp.get_relative_rotation()
        return float(rel_rot.yaw)
    except Exception:
        return 0.0


def compute_player_yaw(actor, start, end):
    move_yaw = movement_yaw_from_start_end(start, end)
    mesh_rel_yaw = get_mesh_relative_yaw(actor)

    final_yaw = move_yaw - mesh_rel_yaw + PLAYER_FORWARD_CORRECTION_DEG

    return normalize_yaw_deg(final_yaw)



def vector_xy_length(v):
    return math.sqrt(float(v.x) * float(v.x) + float(v.y) * float(v.y))


def find_nonzero_direction(keyframes, index):
    """
    为轨迹关键点估计运动切线方向：
    - 首点优先看向后续第一个不同位置；
    - 末点优先沿用前面最后一个移动方向；
    - 中间点优先使用前后非重复位置形成的中心切线。
    """
    current = keyframes[index]["loc"]

    prev_loc = None
    for j in range(index - 1, -1, -1):
        candidate = keyframes[j]["loc"]
        delta = current - candidate
        if vector_xy_length(delta) > 1e-6:
            prev_loc = candidate
            break

    next_loc = None
    for j in range(index + 1, len(keyframes)):
        candidate = keyframes[j]["loc"]
        delta = candidate - current
        if vector_xy_length(delta) > 1e-6:
            next_loc = candidate
            break

    if prev_loc is not None and next_loc is not None:
        direction = next_loc - prev_loc
        if vector_xy_length(direction) > 1e-6:
            return direction

    if next_loc is not None:
        direction = next_loc - current
        if vector_xy_length(direction) > 1e-6:
            return direction

    if prev_loc is not None:
        direction = current - prev_loc
        if vector_xy_length(direction) > 1e-6:
            return direction

    return unreal.Vector(1.0, 0.0, 0.0)


def unwrap_yaw_values(normalized_values):
    if not normalized_values:
        return []

    result = [float(normalized_values[0])]

    for value in normalized_values[1:]:
        candidate = float(value)
        previous = result[-1]

        while candidate - previous > 180.0:
            candidate -= 360.0

        while candidate - previous < -180.0:
            candidate += 360.0

        result.append(candidate)

    return result


def build_yaw_keyframes_for_actor(actor, category, keyframes):
    if category != "player":
        values = []
        for kf in keyframes:
            explicit = kf.get("explicit_yaw_deg")
            values.append(float(explicit) if explicit is not None else 0.0)
        return unwrap_yaw_values([normalize_yaw_deg(v) for v in values])

    mesh_rel_yaw = get_mesh_relative_yaw(actor)
    normalized = []

    for i, kf in enumerate(keyframes):
        explicit = kf.get("explicit_yaw_deg")

        if explicit is not None:
            final_yaw = float(explicit)
        else:
            direction = find_nonzero_direction(keyframes, i)
            move_yaw = math.degrees(math.atan2(direction.y, direction.x))
            final_yaw = move_yaw - mesh_rel_yaw + PLAYER_FORWARD_CORRECTION_DEG

        normalized.append(normalize_yaw_deg(final_yaw))

    return unwrap_yaw_values(normalized)


def find_trajectory_segment(keyframes, frame_id):
    if frame_id <= keyframes[0]["frame"]:
        return 0, 0.0

    if frame_id >= keyframes[-1]["frame"]:
        return max(0, len(keyframes) - 2), 1.0

    for i in range(len(keyframes) - 1):
        f0 = keyframes[i]["frame"]
        f1 = keyframes[i + 1]["frame"]

        if f0 <= frame_id <= f1:
            if f1 == f0:
                return i, 0.0

            alpha = float(frame_id - f0) / float(f1 - f0)
            return i, alpha

    return max(0, len(keyframes) - 2), 1.0


def sample_object_trajectory(cfg, frame_id):
    keyframes = cfg["keyframes"]
    yaw_values = cfg["yaw_keyframes_unwrapped"]

    if len(keyframes) == 1:
        loc = keyframes[0]["loc"]
        yaw_unwrapped = yaw_values[0] if yaw_values else 0.0
        return loc, normalize_yaw_deg(yaw_unwrapped), 0, 0.0

    segment_index, alpha = find_trajectory_segment(keyframes, frame_id)
    k0 = keyframes[segment_index]
    k1 = keyframes[segment_index + 1]

    loc = lerp_vector(k0["loc"], k1["loc"], alpha)

    yaw0 = yaw_values[segment_index]
    yaw1 = yaw_values[segment_index + 1]
    yaw_unwrapped = yaw0 + (yaw1 - yaw0) * alpha

    return loc, normalize_yaw_deg(yaw_unwrapped), segment_index, alpha



def sample_action_at_frame(cfg, frame_id):
    timeline = cfg.get("action_timeline") or []

    for segment in timeline:
        if segment["start_frame"] <= frame_id <= segment["end_frame"]:
            return segment["action"], list(segment.get("source_events", []))

    return None, []


def action_timeline_for_json(cfg):
    result = []

    for segment in cfg.get("action_timeline") or []:
        result.append({
            "start_frame": int(segment["start_frame"]),
            "end_frame": int(segment["end_frame"]),
            "action": str(segment["action"]),
            "source_events": list(segment.get("source_events", [])),
        })

    return result


def keyframes_for_json(cfg):
    result = []
    yaw_values = cfg.get("yaw_keyframes_unwrapped") or []

    for i, kf in enumerate(cfg["keyframes"]):
        yaw_value = yaw_values[i] if i < len(yaw_values) else 0.0
        result.append({
            "frame": int(kf["frame"]),
            "loc": [
                float(kf["loc"].x),
                float(kf["loc"].y),
                float(kf["loc"].z),
            ],
            "yaw_deg": float(normalize_yaw_deg(yaw_value)),
        })

    return result


def make_rotator_yaw(yaw_deg):
    rot = unreal.Rotator()

    try:
        rot.set_editor_property("pitch", 0.0)
        rot.set_editor_property("yaw", float(yaw_deg))
        rot.set_editor_property("roll", 0.0)
    except Exception:
        rot = unreal.Rotator(0.0, float(yaw_deg), 0.0)

    return rot


def lerp_vector(a, b, alpha):
    return unreal.Vector(
        a.x + (b.x - a.x) * alpha,
        a.y + (b.y - a.y) * alpha,
        a.z + (b.z - a.z) * alpha,
    )


def set_actor_transform_direct(actor, loc, yaw_deg, scale_vec):
    try:
        actor.set_actor_location(loc, False, False)
    except Exception:
        pass

    try:
        actor.set_actor_rotation(make_rotator_yaw(yaw_deg), False)
    except Exception:
        pass

    try:
        actor.set_actor_scale3d(scale_vec)
    except Exception:
        pass


# ============================================================
# Sequencer helpers
# ============================================================

def binding_name(binding):
    try:
        return str(binding.get_display_name())
    except Exception:
        return ""


def get_child_bindings(binding):
    children = []

    try:
        children.extend(binding.get_child_possessables())
    except Exception:
        pass

    try:
        children.extend(binding.get_child_spawnables())
    except Exception:
        pass

    return children


def collect_binding_tree(binding):
    result = [binding]

    for child in get_child_bindings(binding):
        result.extend(collect_binding_tree(child))

    return result


def find_root_binding(sequence, actor_label):
    for binding in sequence.get_bindings():
        if binding_name(binding) == actor_label:
            return binding

    return None


def get_or_create_root_binding(sequence, actor, actor_label):
    binding = find_root_binding(sequence, actor_label)

    if binding is not None:
        return binding

    binding = sequence.add_possessable(actor)

    try:
        binding.set_display_name(actor_label)
    except Exception:
        pass

    return binding


def find_mesh_binding_under_root(root_binding, actor_label):
    all_bindings = collect_binding_tree(root_binding)
    candidates = []

    for b in all_bindings:
        name = binding_name(b)

        if name == actor_label + "_Mesh":
            return b

        if "Mesh" in name or "SkeletalMesh" in name or "Mannequin" in name:
            candidates.append(b)

    if candidates:
        return candidates[0]

    return None


def get_or_create_mesh_binding(sequence, actor, actor_label, root_binding):
    mesh_binding = find_mesh_binding_under_root(root_binding, actor_label)

    if mesh_binding is not None:
        try:
            mesh_binding.set_display_name(actor_label + "_Mesh")
        except Exception:
            pass
        return mesh_binding

    skel_comp = actor.get_component_by_class(unreal.SkeletalMeshComponent)

    if skel_comp is None:
        unreal.log_error("{} 找不到 SkeletalMeshComponent".format(actor_label))
        return None

    try:
        mesh_binding = root_binding.add_child_possessable(skel_comp)
    except Exception:
        mesh_binding = sequence.add_possessable(skel_comp)

    try:
        mesh_binding.set_display_name(actor_label + "_Mesh")
    except Exception:
        pass

    return mesh_binding


def remove_tracks_by_keywords(binding, keywords):
    removed = 0

    try:
        tracks = list(binding.get_tracks())
    except Exception:
        return removed

    for track in tracks:
        try:
            cname = track.get_class().get_name()
        except Exception:
            cname = ""

        need_remove = False

        for kw in keywords:
            if kw in cname:
                need_remove = True
                break

        if need_remove:
            try:
                binding.remove_track(track)
                removed += 1
            except Exception as e:
                unreal.log_warning(
                    "删除轨道失败 binding={} track={} error={}".format(
                        binding_name(binding), cname, e
                    )
                )

    return removed


def clean_animation_and_controlrig(binding):
    return remove_tracks_by_keywords(
        binding,
        [
            "ControlRig",
            "ControlRigParameter",
            "MovieSceneControlRig",
            "MovieSceneSkeletalAnimationTrack",
        ]
    )


def remove_transform_tracks(binding):
    return remove_tracks_by_keywords(
        binding,
        [
            "MovieScene3DTransformTrack",
        ]
    )


def add_key_linear(channel, frame, value):
    key = None

    try:
        key = channel.add_key(
            unreal.FrameNumber(frame),
            float(value),
            0.0,
            unreal.SequenceTimeUnit.DISPLAY_RATE
        )
    except Exception:
        try:
            key = channel.add_key(unreal.FrameNumber(frame), float(value))
        except Exception as e:
            unreal.log_warning(
                "添加关键帧失败 frame={} value={} error={}".format(
                    frame, value, e
                )
            )
            return None

    try:
        key.set_interpolation_mode(unreal.RichCurveInterpMode.RCIM_LINEAR)
    except Exception:
        pass

    return key


def get_section_channels(section):
    for method_name in ["get_channels", "get_all_channels"]:
        try:
            if hasattr(section, method_name):
                channels = getattr(section, method_name)()
                if channels:
                    return list(channels)
        except Exception:
            pass

    try:
        ext = unreal.MovieSceneSectionExtensions
    except Exception:
        ext = None

    if ext is not None:
        for method_name in ["get_channels", "get_all_channels"]:
            try:
                if hasattr(ext, method_name):
                    channels = getattr(ext, method_name)(section)
                    if channels:
                        return list(channels)
            except Exception:
                pass

        for channel_class_name in [
            "MovieSceneScriptingDoubleChannel",
            "MovieSceneScriptingFloatChannel",
        ]:
            try:
                channel_class = getattr(unreal, channel_class_name)
            except Exception:
                channel_class = None

            if channel_class is None:
                continue

            try:
                if hasattr(ext, "get_channels_by_type"):
                    channels = ext.get_channels_by_type(section, channel_class)
                    if channels:
                        return list(channels)
            except Exception:
                pass

    for channel_class_name in [
        "MovieSceneScriptingDoubleChannel",
        "MovieSceneScriptingFloatChannel",
    ]:
        try:
            channel_class = getattr(unreal, channel_class_name)
        except Exception:
            channel_class = None

        if channel_class is None:
            continue

        try:
            if hasattr(section, "get_channels_by_type"):
                channels = section.get_channels_by_type(channel_class)
                if channels:
                    return list(channels)
        except Exception:
            pass

    available_methods = []

    try:
        available_methods = [m for m in dir(section) if "channel" in m.lower()]
    except Exception:
        pass

    unreal.log_error("无法获取 Transform Section 的 channels。")
    unreal.log_error("Section class: {}".format(section.get_class().get_name()))
    unreal.log_error("Available channel-like methods: {}".format(available_methods))

    raise RuntimeError("Cannot get channels from MovieScene3DTransformSection")


def get_channel_name(channel):
    try:
        return str(channel.get_name())
    except Exception:
        return ""


def normalized_channel_name(channel):
    name = get_channel_name(channel).lower()
    name = name.replace(".", "")
    name = name.replace(" ", "")
    name = name.replace("_", "")
    name = name.replace("-", "")

    return name


def classify_transform_channel(channel, fallback_index):
    n = normalized_channel_name(channel)

    aliases = {
        "Location.X": ["locationx", "translationx", "xlocation", "xtranslation"],
        "Location.Y": ["locationy", "translationy", "ylocation", "ytranslation"],
        "Location.Z": ["locationz", "translationz", "zlocation", "ztranslation"],

        "Rotation.X": ["rotationx", "rotx", "xrotation", "xrot"],
        "Rotation.Y": ["rotationy", "roty", "yrotation", "yrot"],
        "Rotation.Z": ["rotationz", "rotz", "zrotation", "zrot"],

        "Scale.X": ["scalex", "xscale"],
        "Scale.Y": ["scaley", "yscale"],
        "Scale.Z": ["scalez", "zscale"],
    }

    for canonical, names in aliases.items():
        if n in names:
            return canonical

    fallback_order = [
        "Location.X",
        "Location.Y",
        "Location.Z",
        "Rotation.X",
        "Rotation.Y",
        "Rotation.Z",
        "Scale.X",
        "Scale.Y",
        "Scale.Z",
    ]

    if fallback_index < len(fallback_order):
        return fallback_order[fallback_index]

    return None


def add_transform_track_force(binding, keyframes, yaw_keyframes_unwrapped, scale_vec):
    removed = remove_transform_tracks(binding)

    track = binding.add_track(unreal.MovieScene3DTransformTrack)
    section = track.add_section()
    section.set_range(FRAME_START, FRAME_END_EXCLUSIVE)

    channels = get_section_channels(section)
    channel_report = []

    for i, ch in enumerate(channels):
        canonical = classify_transform_channel(ch, i)
        raw_name = get_channel_name(ch)

        if canonical is None:
            continue

        for kf_index, kf in enumerate(keyframes):
            frame = int(kf["frame"])
            loc = kf["loc"]

            if canonical == "Location.X":
                value = loc.x
            elif canonical == "Location.Y":
                value = loc.y
            elif canonical == "Location.Z":
                value = loc.z
            elif canonical == "Rotation.X":
                value = 0.0
            elif canonical == "Rotation.Y":
                value = 0.0
            elif canonical == "Rotation.Z":
                value = yaw_keyframes_unwrapped[kf_index]
            elif canonical == "Scale.X":
                value = scale_vec.x
            elif canonical == "Scale.Y":
                value = scale_vec.y
            elif canonical == "Scale.Z":
                value = scale_vec.z
            else:
                continue

            add_key_linear(ch, frame, value)

        channel_report.append("{}=>{}".format(raw_name, canonical))

    return removed, channel_report

def add_action_animations_to_mesh_binding(
    mesh_binding,
    action_timeline,
    action_assets
):
    clean_animation_and_controlrig(mesh_binding)

    track = mesh_binding.add_track(
        unreal.MovieSceneSkeletalAnimationTrack
    )
    sections = []

    if not action_timeline:
        action_timeline = [{
            "start_frame": FRAME_START,
            "end_frame": FRAME_END,
            "action": FALLBACK_ACTION,
            "source_events": [],
        }]

    for segment in action_timeline:
        action_name = segment["action"]
        anim_asset = action_assets.get(action_name)

        if anim_asset is None:
            raise RuntimeError(
                "动作 {} 没有已解析动画资产".format(action_name)
            )

        section = track.add_section()
        section.set_range(
            int(segment["start_frame"]),
            int(segment["end_frame"]) + 1
        )

        params = section.get_editor_property("params")
        params.set_editor_property("animation", anim_asset)

        try:
            params.set_editor_property(
                "play_rate",
                float(ACTION_PLAY_RATES[action_name])
            )
        except Exception:
            pass

        section.set_editor_property("params", params)

        try:
            section.set_row_index(0)
        except Exception:
            pass

        sections.append({
            "action": action_name,
            "start_frame": int(segment["start_frame"]),
            "end_frame": int(segment["end_frame"]),
            "section": section,
        })

    return track, sections


def set_sequence_range(sequence):
    try:
        sequence.set_display_rate(unreal.FrameRate(DISPLAY_RATE, 1))
    except Exception:
        try:
            unreal.MovieSceneSequenceExtensions.set_display_rate(
                sequence,
                unreal.FrameRate(DISPLAY_RATE, 1)
            )
        except Exception:
            pass

    try:
        sequence.set_playback_start(FRAME_START)
    except Exception:
        try:
            unreal.MovieSceneSequenceExtensions.set_playback_start(
                sequence,
                FRAME_START
            )
        except Exception:
            pass

    try:
        sequence.set_playback_end(FRAME_END_EXCLUSIVE)
    except Exception:
        try:
            unreal.MovieSceneSequenceExtensions.set_playback_end(
                sequence,
                FRAME_END_EXCLUSIVE
            )
        except Exception:
            pass


def find_camera_cut_tracks(sequence):
    """跨 UE 版本查找 Camera Cut Track。"""
    tracks = []

    try:
        ext = unreal.MovieSceneSequenceExtensions
    except Exception:
        ext = None

    if ext is not None:
        for method_name in [
            "find_tracks_by_type",
            "find_master_tracks_by_type",
            "find_tracks_by_exact_type",
            "find_master_tracks_by_exact_type",
        ]:
            try:
                if hasattr(ext, method_name):
                    found = getattr(ext, method_name)(
                        sequence,
                        unreal.MovieSceneCameraCutTrack
                    )
                    if found:
                        tracks.extend(list(found))
            except Exception:
                pass

    for method_name in [
        "find_tracks_by_type",
        "find_master_tracks_by_type",
        "find_tracks_by_exact_type",
        "find_master_tracks_by_exact_type",
    ]:
        try:
            if hasattr(sequence, method_name):
                found = getattr(sequence, method_name)(
                    unreal.MovieSceneCameraCutTrack
                )
                if found:
                    tracks.extend(list(found))
        except Exception:
            pass

    if not tracks:
        candidates = []

        for method_name in ["get_tracks", "get_master_tracks"]:
            try:
                if hasattr(sequence, method_name):
                    found = getattr(sequence, method_name)()
                    if found:
                        candidates.extend(list(found))
            except Exception:
                pass

        if ext is not None:
            for method_name in ["get_tracks", "get_master_tracks"]:
                try:
                    if hasattr(ext, method_name):
                        found = getattr(ext, method_name)(sequence)
                        if found:
                            candidates.extend(list(found))
                except Exception:
                    pass

        for track in candidates:
            try:
                class_name = track.get_class().get_name()
            except Exception:
                class_name = ""

            if "CameraCut" in class_name:
                tracks.append(track)

    unique_tracks = []
    seen = set()

    for track in tracks:
        key = str(track)
        if key not in seen:
            seen.add(key)
            unique_tracks.append(track)

    return unique_tracks


def extend_camera_cut_sections(sequence):
    """
    自动把所有 Camera Cut Section 扩展到 [FRAME_START, FRAME_END_EXCLUSIVE)。
    MRQ 实际可渲染范围会受 Camera Cut Section 限制，因此必须同步更新。
    """
    tracks = find_camera_cut_tracks(sequence)
    section_count = 0

    for track in tracks:
        try:
            sections = list(track.get_sections())
        except Exception:
            sections = []

        for section in sections:
            try:
                section.set_range(FRAME_START, FRAME_END_EXCLUSIVE)
                section_count += 1
            except Exception:
                try:
                    unreal.MovieSceneSectionExtensions.set_range(
                        section,
                        FRAME_START,
                        FRAME_END_EXCLUSIVE
                    )
                    section_count += 1
                except Exception as e:
                    unreal.log_warning(
                        "Camera Cut Section 范围设置失败: {}".format(e)
                    )

    if section_count == 0:
        unreal.log_warning(
            "未找到可扩展的 Camera Cut Section；请检查该 Sequence 是否存在 Camera Cuts。"
        )

    return len(tracks), section_count


def open_sequence_for_pose_evaluation(sequence):
    """
    打开一个已写入动画轨道的 Sequence，供逐帧求值骨骼姿态。
    如果当前 UE 版本不支持该编辑器 API，则返回 False，bbox 仍使用安全包络。
    """
    if not PLAYER_BBOX_EVALUATE_SEQUENCE_POSE:
        return False

    try:
        lib = unreal.LevelSequenceEditorBlueprintLibrary
    except Exception:
        unreal.log_warning("当前 UE 未提供 LevelSequenceEditorBlueprintLibrary，跳过逐帧姿态求值。")
        return False

    try:
        opened = lib.open_level_sequence(sequence)
        if opened is False:
            unreal.log_warning("无法打开用于 bbox 姿态求值的 Level Sequence。")
            return False

        try:
            lib.pause()
        except Exception:
            pass

        return True
    except Exception as e:
        unreal.log_warning("打开姿态求值 Sequence 失败: {}".format(e))
        return False


def evaluate_open_sequence_at_frame(frame_id):
    """让 Sequencer 在指定显示帧求值 Transform 与 Skeletal Animation。"""
    try:
        lib = unreal.LevelSequenceEditorBlueprintLibrary

        # 旧版接口直接接收 int；该接口会更新当前 Sequencer 的求值结果。
        if hasattr(lib, "set_current_time"):
            lib.set_current_time(int(frame_id))
        elif hasattr(lib, "set_current_local_time"):
            lib.set_current_local_time(int(frame_id))
        else:
            return False

        try:
            lib.refresh_current_level_sequence()
        except Exception:
            pass

        return True
    except Exception as e:
        unreal.log_warning(
            "Frame {} 的 Sequencer 姿态求值失败，将继续使用安全包络: {}".format(
                frame_id,
                e
            )
        )
        return False


# ============================================================
# Camera projection
# ============================================================

def project_world_to_camera_uv(world_location, camera_actor):
    cine_comp = camera_actor.get_component_by_class(unreal.CineCameraComponent)

    if cine_comp is None:
        return {
            "uv": [None, None],
            "depth_cm": None,
            "in_image": False,
            "in_front": False,
        }

    focal_length = safe_get(cine_comp, "current_focal_length", None)
    filmback = safe_get(cine_comp, "filmback", None)

    if focal_length is None or filmback is None:
        return {
            "uv": [None, None],
            "depth_cm": None,
            "in_image": False,
            "in_front": False,
        }

    sensor_width = safe_get(filmback, "sensor_width", None)
    sensor_height = safe_get(filmback, "sensor_height", None)

    if sensor_width is None or sensor_height is None:
        return {
            "uv": [None, None],
            "depth_cm": None,
            "in_image": False,
            "in_front": False,
        }

    fx = focal_length / sensor_width * IMAGE_WIDTH
    fy = focal_length / sensor_height * IMAGE_HEIGHT
    cx = IMAGE_WIDTH / 2.0
    cy = IMAGE_HEIGHT / 2.0

    cam_loc = camera_actor.get_actor_location()
    forward = camera_actor.get_actor_forward_vector()
    right = camera_actor.get_actor_right_vector()
    up = camera_actor.get_actor_up_vector()

    rel = world_location - cam_loc

    depth = rel.x * forward.x + rel.y * forward.y + rel.z * forward.z
    x_right = rel.x * right.x + rel.y * right.y + rel.z * right.z
    z_up = rel.x * up.x + rel.y * up.y + rel.z * up.z

    if depth <= 1e-6:
        return {
            "uv": [None, None],
            "depth_cm": float(depth),
            "in_image": False,
            "in_front": False,
        }

    u = cx + fx * (x_right / depth)
    v = cy - fy * (z_up / depth)

    in_image = (0.0 <= u < IMAGE_WIDTH) and (0.0 <= v < IMAGE_HEIGHT)

    return {
        "uv": [float(u), float(v)],
        "depth_cm": float(depth),
        "in_image": bool(in_image),
        "in_front": True,
    }


def make_box_corners(origin, extent):
    corners = []

    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                corners.append(
                    unreal.Vector(
                        origin.x + sx * extent.x,
                        origin.y + sy * extent.y,
                        origin.z + sz * extent.z,
                    )
                )

    return corners


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ============================================================
# Player tight 2D bbox from bones
# ============================================================

def get_skeletal_mesh_component(actor):
    return actor.get_component_by_class(unreal.SkeletalMeshComponent)


def get_socket_or_bone_world_location(skel_comp, name):
    # 避免无效骨骼名返回组件原点并污染 bbox。
    try:
        if hasattr(skel_comp, "get_bone_index"):
            bone_index = int(skel_comp.get_bone_index(name))
            if bone_index < 0:
                return None
    except Exception:
        pass

    try:
        return skel_comp.get_socket_location(name)
    except Exception:
        return None


def get_player_bone_world_samples(actor):
    skel_comp = get_skeletal_mesh_component(actor)

    if skel_comp is None:
        return []

    samples = []

    for bone_name in PLAYER_BBOX_BONES:
        p = get_socket_or_bone_world_location(skel_comp, bone_name)

        if p is None:
            continue

        radius_cm = PLAYER_BBOX_BONE_RADIUS_CM.get(
            bone_name,
            PLAYER_BBOX_DEFAULT_BONE_RADIUS_CM
        ) * PLAYER_BBOX_BONE_RADIUS_SCALE

        samples.append({
            "bone_name": bone_name,
            "location": p,
            "radius_cm": float(radius_cm),
        })

    return samples


def get_player_keypoint_world_samples(actor):
    skel_comp = get_skeletal_mesh_component(actor)

    if skel_comp is None:
        return []

    samples = []

    for index, bone_name in enumerate(PLAYER_KEYPOINT_BONES):
        p = get_socket_or_bone_world_location(skel_comp, bone_name)

        if p is None:
            samples.append({
                "index": int(index),
                "bone_name": bone_name,
                "location": None,
            })
            continue

        samples.append({
            "index": int(index),
            "bone_name": bone_name,
            "location": p,
        })

    return samples


def project_player_keypoints_2d(actor, category, camera_actor):
    if category != "player":
        return None

    keypoints = []
    visible_count = 0

    for sample in get_player_keypoint_world_samples(actor):
        location = sample["location"]

        if location is None:
            proj = {
                "uv": [None, None],
                "depth_cm": None,
                "in_image": False,
                "in_front": False,
            }
        else:
            proj = project_world_to_camera_uv(location, camera_actor)

        visible = bool(proj["in_front"] and proj["in_image"])
        if visible:
            visible_count += 1

        uv = proj["uv"]
        keypoints.append({
            "index": int(sample["index"]),
            "name": sample["bone_name"],
            "x": float(uv[0]) if uv[0] is not None else None,
            "y": float(uv[1]) if uv[1] is not None else None,
            "visibility": 2 if visible else 1 if proj["in_front"] else 0,
            "in_image": bool(proj["in_image"]),
            "in_front": bool(proj["in_front"]),
            "depth_cm": float(proj["depth_cm"]) if proj["depth_cm"] is not None else None,
        })

    yolo_values = []

    for kp in keypoints:
        if kp["x"] is None or kp["y"] is None:
            yolo_values.extend([0.0, 0.0, 0])
        else:
            yolo_values.extend([
                float(kp["x"]) / float(IMAGE_WIDTH),
                float(kp["y"]) / float(IMAGE_HEIGHT),
                int(kp["visibility"]),
            ])

    return {
        "keypoints_2d": keypoints,
        "keypoints_2d_yolo": yolo_values,
        "keypoints_2d_visible_count": int(visible_count),
        "keypoints_2d_total": int(len(keypoints)),
    }


def vector_add_scaled(origin, direction, scalar):
    return unreal.Vector(
        origin.x + direction.x * scalar,
        origin.y + direction.y * scalar,
        origin.z + direction.z * scalar,
    )


def make_bone_envelope_world_points(center, radius_cm, camera_actor):
    """
    在相机成像平面内，用 8 个方向近似投影一个骨骼圆形包络。
    与固定像素 padding 相比，该方法会随人物深度自动缩放。
    """
    right = camera_actor.get_actor_right_vector()
    up = camera_actor.get_actor_up_vector()
    diagonal_scale = radius_cm / math.sqrt(2.0)

    points = [center]
    points.append(vector_add_scaled(center, right, radius_cm))
    points.append(vector_add_scaled(center, right, -radius_cm))
    points.append(vector_add_scaled(center, up, radius_cm))
    points.append(vector_add_scaled(center, up, -radius_cm))

    p = vector_add_scaled(center, right, diagonal_scale)
    points.append(vector_add_scaled(p, up, diagonal_scale))
    p = vector_add_scaled(center, right, diagonal_scale)
    points.append(vector_add_scaled(p, up, -diagonal_scale))
    p = vector_add_scaled(center, right, -diagonal_scale)
    points.append(vector_add_scaled(p, up, diagonal_scale))
    p = vector_add_scaled(center, right, -diagonal_scale)
    points.append(vector_add_scaled(p, up, -diagonal_scale))

    return points


def bbox_from_player_bones(actor, current_loc, camera_actor):
    """
    使用“骨骼中心 + 每骨骼物理半径”的二维包络生成 player tight bbox。

    相比旧版只投影骨骼中心线：
    - 头顶、肩部、手臂、腿部和鞋不会轻易落在框外；
    - 仍明显比 Actor Bounds 紧；
    - 包络半径以 cm 定义，会随相机距离和焦距自然缩放。
    """
    bone_samples = get_player_bone_world_samples(actor)

    if len(bone_samples) == 0:
        return None

    envelope_uv_points = []
    center_point_records = []

    for sample in bone_samples:
        center = sample["location"]
        radius_cm = sample["radius_cm"]

        center_proj = project_world_to_camera_uv(center, camera_actor)
        center_point_records.append(center_proj)

        for p in make_bone_envelope_world_points(center, radius_cm, camera_actor):
            proj = project_world_to_camera_uv(p, camera_actor)

            if proj["in_front"] and proj["uv"][0] is not None:
                envelope_uv_points.append(proj["uv"])

    if len(envelope_uv_points) < 4:
        return None

    xs = [p[0] for p in envelope_uv_points]
    ys = [p[1] for p in envelope_uv_points]

    raw_x1 = min(xs)
    raw_y1 = min(ys)
    raw_x2 = max(xs)
    raw_y2 = max(ys)
    raw_h = max(1.0, raw_y2 - raw_y1)

    adaptive_padding_px = (
        PLAYER_2D_BBOX_PADDING_PX
        + raw_h * PLAYER_2D_BBOX_ADAPTIVE_PADDING_RATIO
    )

    x1 = raw_x1 - adaptive_padding_px
    y1 = raw_y1 - adaptive_padding_px
    x2 = raw_x2 + adaptive_padding_px
    y2 = raw_y2 + adaptive_padding_px

    intersects = not (
        x2 < 0 or
        y2 < 0 or
        x1 >= IMAGE_WIDTH or
        y1 >= IMAGE_HEIGHT
    )

    if not intersects:
        return {
            "visible": False,
            "center_uv_clean": [None, None],
            "bbox_2d_clean": None,
            "bbox_xyxy_clean": None,
            "bbox_corners_uv": center_point_records,
            "bounds_source": "player_bone_envelope_2d",
            "bounds_origin_cm": [
                float(current_loc.x),
                float(current_loc.y),
                float(current_loc.z)
            ],
            "bounds_extent_cm": [0.0, 0.0, 0.0],
            "bbox_bone_points_used": int(len(bone_samples)),
            "bbox_adaptive_padding_px": float(adaptive_padding_px),
        }

    x1c = clamp(x1, 0.0, IMAGE_WIDTH - 1.0)
    y1c = clamp(y1, 0.0, IMAGE_HEIGHT - 1.0)
    x2c = clamp(x2, 0.0, IMAGE_WIDTH - 1.0)
    y2c = clamp(y2, 0.0, IMAGE_HEIGHT - 1.0)

    w = x2c - x1c
    h = y2c - y1c

    visible = w > 1.0 and h > 1.0

    center_u = (x1c + x2c) * 0.5
    center_v = (y1c + y2c) * 0.5

    return {
        "visible": bool(visible),
        "center_uv_clean": [float(center_u), float(center_v)] if visible else [None, None],
        "bbox_2d_clean": [float(x1c), float(y1c), float(w), float(h)] if visible else None,
        "bbox_xyxy_clean": [float(x1c), float(y1c), float(x2c), float(y2c)] if visible else None,
        "bbox_corners_uv": center_point_records,
        "bounds_source": "player_bone_envelope_2d",
        "bounds_origin_cm": [
            float(current_loc.x),
            float(current_loc.y),
            float(current_loc.z)
        ],
        "bounds_extent_cm": [0.0, 0.0, 0.0],
        "bbox_bone_points_used": int(len(bone_samples)),
        "bbox_adaptive_padding_px": float(adaptive_padding_px),
    }


# ============================================================
# Fallback object bbox
# ============================================================

def get_player_bounds(actor, current_loc):
    """
    fallback 版 player bbox。
    只有骨骼点 2D bbox 失败时才使用。
    """
    origin = unreal.Vector(
        current_loc.x,
        current_loc.y,
        current_loc.z + PLAYER_BBOX_CENTER_Z_OFFSET_CM
    )

    extent = unreal.Vector(
        PLAYER_BBOX_EXTENT_CM.x,
        PLAYER_BBOX_EXTENT_CM.y,
        PLAYER_BBOX_EXTENT_CM.z
    )

    return origin, extent, "manual_player_fallback_bbox"


def get_object_bounds(actor, obj_label, category, current_loc):
    """
    ball 使用稳定半径；
    player 优先骨骼点 bbox，失败才走这里的 fallback。
    """
    if category == "ball":
        origin = unreal.Vector(
            current_loc.x,
            current_loc.y,
            current_loc.z
        )

        extent = unreal.Vector(
            BALL_RADIUS_CM,
            BALL_RADIUS_CM,
            BALL_RADIUS_CM
        )

        return origin, extent, "manual_ball_radius"

    if category == "player":
        return get_player_bounds(actor, current_loc)

    origin, extent = actor.get_actor_bounds(False)
    return origin, extent, "actor_bounds_fallback"


def bbox_from_bounds(actor, obj_label, category, current_loc, camera_actor):
    # player 优先使用骨骼点投影生成紧 bbox
    if category == "player":
        player_bbox = bbox_from_player_bones(actor, current_loc, camera_actor)

        if player_bbox is not None:
            return player_bbox

    # fallback：3D bbox 投影
    origin, extent, bounds_source = get_object_bounds(
        actor,
        obj_label,
        category,
        current_loc
    )

    corners = make_box_corners(origin, extent)

    uv_list = []
    corner_records = []

    for corner in corners:
        proj = project_world_to_camera_uv(corner, camera_actor)
        corner_records.append(proj)

        if proj["in_front"] and proj["uv"][0] is not None:
            uv_list.append(proj["uv"])

    center_proj = project_world_to_camera_uv(origin, camera_actor)

    if len(uv_list) == 0:
        return {
            "visible": False,
            "center_uv_clean": center_proj["uv"],
            "bbox_2d_clean": None,
            "bbox_xyxy_clean": None,
            "bbox_corners_uv": corner_records,
            "bounds_source": bounds_source,
            "bounds_origin_cm": [float(origin.x), float(origin.y), float(origin.z)],
            "bounds_extent_cm": [float(extent.x), float(extent.y), float(extent.z)],
        }

    xs = [p[0] for p in uv_list]
    ys = [p[1] for p in uv_list]

    x1 = min(xs)
    y1 = min(ys)
    x2 = max(xs)
    y2 = max(ys)

    intersects = not (
        x2 < 0 or
        y2 < 0 or
        x1 >= IMAGE_WIDTH or
        y1 >= IMAGE_HEIGHT
    )

    if not intersects:
        return {
            "visible": False,
            "center_uv_clean": center_proj["uv"],
            "bbox_2d_clean": None,
            "bbox_xyxy_clean": None,
            "bbox_corners_uv": corner_records,
            "bounds_source": bounds_source,
            "bounds_origin_cm": [float(origin.x), float(origin.y), float(origin.z)],
            "bounds_extent_cm": [float(extent.x), float(extent.y), float(extent.z)],
        }

    x1c = clamp(x1, 0.0, IMAGE_WIDTH - 1.0)
    y1c = clamp(y1, 0.0, IMAGE_HEIGHT - 1.0)
    x2c = clamp(x2, 0.0, IMAGE_WIDTH - 1.0)
    y2c = clamp(y2, 0.0, IMAGE_HEIGHT - 1.0)

    w = x2c - x1c
    h = y2c - y1c

    visible = w > 1.0 and h > 1.0

    return {
        "visible": bool(visible),
        "center_uv_clean": center_proj["uv"],
        "bbox_2d_clean": [float(x1c), float(y1c), float(w), float(h)] if visible else None,
        "bbox_xyxy_clean": [float(x1c), float(y1c), float(x2c), float(y2c)] if visible else None,
        "bbox_corners_uv": corner_records,
        "bounds_source": bounds_source,
        "bounds_origin_cm": [float(origin.x), float(origin.y), float(origin.z)],
        "bounds_extent_cm": [float(extent.x), float(extent.y), float(extent.z)],
    }


# ============================================================
# Main: prepare actors and runtime trajectories
# ============================================================

unreal.log("===================================")
unreal.log("01 UE Build episode - A3.3b action timeline animation sections")
unreal.log("CONFIG_PATH = {}".format(CONFIG_PATH))
unreal.log("CONFIG_PATH_SOURCE = {}".format(CONFIG_PATH_SOURCE))
unreal.log("PROJECT_ROOT = {}".format(PROJECT_ROOT))
unreal.log("SEQ_ID = {}".format(SEQ_ID))
unreal.log("FRAME_RANGE = {}..{}".format(FRAME_START, FRAME_END))
unreal.log("DISPLAY_RATE = {}".format(DISPLAY_RATE))
unreal.log("OBJECT_COUNT = {}".format(len(OBJECT_TRAJECTORIES)))
unreal.log("PLAYER_2D_BBOX_PADDING_PX = {}".format(PLAYER_2D_BBOX_PADDING_PX))
unreal.log("PLAYER_2D_BBOX_ADAPTIVE_PADDING_RATIO = {}".format(PLAYER_2D_BBOX_ADAPTIVE_PADDING_RATIO))
unreal.log("PLAYER_BBOX_BONE_RADIUS_SCALE = {}".format(PLAYER_BBOX_BONE_RADIUS_SCALE))
unreal.log("PLAYER_BBOX_EVALUATE_SEQUENCE_POSE = {}".format(PLAYER_BBOX_EVALUATE_SEQUENCE_POSE))
unreal.log("PLAYER_KEYPOINT_BONES = {}".format(PLAYER_KEYPOINT_BONES))
unreal.log("ACTION_MAP_OVERRIDE_PATH = {}".format(ACTION_MAP_OVERRIDE_PATH))
unreal.log("STRICT_ACTION_ASSETS = {}".format(STRICT_ACTION_ASSETS))
unreal.log("===================================")

actors_by_label = {}

for obj_label, cfg in OBJECT_TRAJECTORIES.items():
    actor = find_actor_by_label(obj_label)

    if actor is None:
        raise RuntimeError("找不到 Actor: {}".format(obj_label))

    actors_by_label[obj_label] = actor
    cfg["yaw_keyframes_unwrapped"] = build_yaw_keyframes_for_actor(
        actor,
        cfg["category"],
        cfg["keyframes"]
    )

    if len(cfg["yaw_keyframes_unwrapped"]) != len(cfg["keyframes"]):
        raise RuntimeError("{} 的 yaw 关键帧数量与位置关键帧数量不一致".format(obj_label))

needs_animation = any(
    cfg["use_animation"] for cfg in OBJECT_TRAJECTORIES.values()
)

ACTION_ANIMATION_ASSETS = {}
ACTION_ANIMATION_RESOLUTION = {}

if needs_animation:
    (
        ACTION_ANIMATION_ASSETS,
        ACTION_ANIMATION_RESOLUTION,
    ) = load_action_animation_assets()


# ============================================================
# Main: build sequencer
# ============================================================

for seq_name in SEQUENCE_NAMES:
    sequence = find_level_sequence_by_name(seq_name)

    if sequence is None:
        unreal.log_error("找不到 Level Sequence: {}".format(seq_name))
        continue

    set_sequence_range(sequence)
    camera_cut_track_count, camera_cut_section_count = extend_camera_cut_sections(sequence)

    unreal.log("-----------------------------------")
    unreal.log("Sequence: {}".format(seq_name))
    unreal.log(
        "[OK] Camera Cuts tracks={} sections_extended={} range=[{}, {})".format(
            camera_cut_track_count,
            camera_cut_section_count,
            FRAME_START,
            FRAME_END_EXCLUSIVE
        )
    )

    for obj_label, cfg in OBJECT_TRAJECTORIES.items():
        actor = actors_by_label[obj_label]
        category = cfg["category"]
        scale_vec = cfg["scale"]
        keyframes = cfg["keyframes"]
        yaw_keyframes = cfg["yaw_keyframes_unwrapped"]

        start_loc, start_yaw, _, _ = sample_object_trajectory(cfg, FRAME_START)
        set_actor_transform_direct(actor, start_loc, start_yaw, scale_vec)

        root_binding = get_or_create_root_binding(sequence, actor, obj_label)

        removed_transform, channel_report = add_transform_track_force(
            root_binding,
            keyframes,
            yaw_keyframes,
            scale_vec
        )

        removed_anim_control = 0

        if cfg["use_animation"]:
            related_bindings = collect_binding_tree(root_binding)

            for b in related_bindings:
                removed_anim_control += clean_animation_and_controlrig(b)

            mesh_binding = get_or_create_mesh_binding(
                sequence,
                actor,
                obj_label,
                root_binding
            )

            animation_sections = []

            if mesh_binding is not None:
                clean_animation_and_controlrig(mesh_binding)
                _, animation_sections = add_action_animations_to_mesh_binding(
                    mesh_binding,
                    cfg.get("action_timeline") or [],
                    ACTION_ANIMATION_ASSETS
                )
        else:
            animation_sections = []

        unreal.log(
            "[OK] {} category={} keyframes={} transform_removed={} "
            "anim_control_removed={} animation_sections={} actions={} "
            "start_yaw={:.2f}".format(
                obj_label,
                category,
                len(keyframes),
                removed_transform,
                removed_anim_control,
                len(animation_sections),
                [
                    item["action"]
                    for item in animation_sections
                ],
                start_yaw
            )
        )

    try:
        unreal.EditorAssetLibrary.save_loaded_asset(sequence)
        unreal.log("Saved sequence: {}".format(seq_name))
    except Exception as e:
        unreal.log_warning("保存 Sequence 失败 {}: {}".format(seq_name, e))


# ============================================================
# Main: export bbox annotations
# ============================================================

pose_evaluation_sequence = None

for seq_name in SEQUENCE_NAMES:
    pose_evaluation_sequence = find_level_sequence_by_name(seq_name)
    if pose_evaluation_sequence is not None:
        break

pose_evaluation_active = False

if pose_evaluation_sequence is not None:
    pose_evaluation_active = open_sequence_for_pose_evaluation(
        pose_evaluation_sequence
    )

unreal.log(
    "Pose evaluation for bbox: {}".format(
        "ENABLED" if pose_evaluation_active else "FALLBACK_ENVELOPE_ONLY"
    )
)

camera_actors = {}

for cam_id, cam_label in CAMERAS.items():
    cam_actor = find_actor_by_label(cam_label)

    if cam_actor is None:
        raise RuntimeError("找不到相机: {} / {}".format(cam_id, cam_label))

    camera_actors[cam_id] = cam_actor

records = []

for frame_id in range(FRAME_START, FRAME_END + 1):
    current_states = {}

    for obj_label, cfg in OBJECT_TRAJECTORIES.items():
        actor = actors_by_label[obj_label]
        category = cfg["category"]
        scale_vec = cfg["scale"]

        current_loc, yaw_deg, segment_index, segment_alpha = sample_object_trajectory(
            cfg,
            frame_id
        )

        set_actor_transform_direct(actor, current_loc, yaw_deg, scale_vec)

        current_action, source_events = sample_action_at_frame(
            cfg,
            frame_id
        )

        current_states[obj_label] = {
            "actor": actor,
            "category": category,
            "team": cfg.get("team"),
            "role": cfg.get("role"),
            "loc": current_loc,
            "yaw_deg": yaw_deg,
            "scale": scale_vec,
            "segment_index": segment_index,
            "segment_alpha": segment_alpha,
            "action": current_action,
            "action_source_events": source_events,
        }

    if pose_evaluation_active:
        if not evaluate_open_sequence_at_frame(frame_id):
            pose_evaluation_active = False

    for cam_id, cam_actor in camera_actors.items():
        objects = []

        for obj_label, state in current_states.items():
            actor = state["actor"]
            category = state["category"]
            current_loc = state["loc"]
            yaw_deg = state["yaw_deg"]
            scale_vec = state["scale"]
            current_action = state.get("action")
            action_source_events = state.get(
                "action_source_events",
                []
            )

            bbox_info = bbox_from_bounds(
                actor,
                obj_label,
                category,
                current_loc,
                cam_actor
            )

            keypoint_info = project_player_keypoints_2d(
                actor,
                category,
                cam_actor
            )

            obj_record = {
                "object_id": obj_label,
                "category": category,
                "class_id": CLASS_ID_MAP.get(category, -1),
                "track_id": TRACK_ID_MAP.get(obj_label, -1),
                "team": state.get("team"),
                "role": state.get("role"),

                "world_cm": [
                    float(current_loc.x),
                    float(current_loc.y),
                    float(current_loc.z),
                ],
                "world_m": [
                    float(current_loc.x) / 100.0,
                    float(current_loc.y) / 100.0,
                    float(current_loc.z) / 100.0,
                ],
                "yaw_deg": float(yaw_deg),
                "scale": [
                    float(scale_vec.x),
                    float(scale_vec.y),
                    float(scale_vec.z),
                ],
                "trajectory_segment_index": int(state["segment_index"]),
                "trajectory_segment_alpha": float(state["segment_alpha"]),
                "action": current_action,
                "action_source_events": list(action_source_events),

                "visible": bbox_info["visible"],
                "center_uv_clean": bbox_info["center_uv_clean"],
                "bbox_2d_clean": bbox_info["bbox_2d_clean"],
                "bbox_xyxy_clean": bbox_info["bbox_xyxy_clean"],
                "bbox_corners_uv": bbox_info["bbox_corners_uv"],
                "bounds_source": bbox_info["bounds_source"],
                "bounds_origin_cm": bbox_info["bounds_origin_cm"],
                "bounds_extent_cm": bbox_info["bounds_extent_cm"],
                "bbox_bone_points_used": bbox_info.get("bbox_bone_points_used", None),
                "bbox_adaptive_padding_px": bbox_info.get("bbox_adaptive_padding_px", None),
            }

            if keypoint_info is not None:
                obj_record.update(keypoint_info)

            objects.append(obj_record)

        record = {
            "seq_id": SEQ_ID,
            "frame_id": frame_id,
            "camera_id": cam_id,
            "image_width": IMAGE_WIDTH,
            "image_height": IMAGE_HEIGHT,
            "rgb_path": "images_clean/{}/{}/{:06d}.png".format(
                SEQ_ID,
                cam_id,
                frame_id
            ),
            "objects": objects,
        }

        records.append(record)

# 导出后把 Actor 放回第一帧状态
for obj_label, cfg in OBJECT_TRAJECTORIES.items():
    actor = actors_by_label[obj_label]
    start_loc, start_yaw, _, _ = sample_object_trajectory(cfg, FRAME_START)
    set_actor_transform_direct(actor, start_loc, start_yaw, cfg["scale"])

trajectory_metadata = {}

for obj_label, cfg in OBJECT_TRAJECTORIES.items():
    trajectory_metadata[obj_label] = {
        "category": cfg["category"],
        "team": cfg.get("team"),
        "role": cfg.get("role"),
        "track_id": TRACK_ID_MAP[obj_label],
        "interpolation": cfg["interpolation"],
        "trajectory_source": cfg["trajectory_source"],
        "keyframes": keyframes_for_json(cfg),
        "action_timeline": action_timeline_for_json(cfg),
    }

output_data = {
    "schema_version": str(CONFIG.get("schema_version", "2.0")),
    "trajectory_schema": "multi_keyframe_linear_v1",
    "bbox_schema": "player_bone_envelope_2d_v1",
    "keypoint_schema": "player_skeleton_2d_v1",
    "keypoint_convention": "ue_skeletal_bone_centers",
    "keypoint_visibility_definition": {
        "0": "bone missing or behind camera",
        "1": "projected in front of camera but outside image",
        "2": "projected inside image",
    },
    "keypoint_names": list(PLAYER_KEYPOINT_BONES),
    "action_schema": "per_object_action_timeline_v1",
    "animation_section_schema": "sequencer_skeletal_animation_sections_v1",
    "action_animation_resolution": ACTION_ANIMATION_RESOLUTION,
    "event_frame_map": CONFIG.get("event_frame_map", {}),
    "contact_frames": CONFIG.get("contact_frames", []),
    "possession_timeline": CONFIG.get("possession_timeline", []),
    "ball_state_timeline": CONFIG.get(
        "ball_state_timeline",
        CONFIG.get("objects", {}).get("Ball_01", {}).get(
            "state_timeline",
            []
        )
    ),
    "player_bbox_config": {
        "fixed_padding_px": PLAYER_2D_BBOX_PADDING_PX,
        "adaptive_padding_ratio": PLAYER_2D_BBOX_ADAPTIVE_PADDING_RATIO,
        "bone_radius_scale": PLAYER_BBOX_BONE_RADIUS_SCALE,
        "default_bone_radius_cm": PLAYER_BBOX_DEFAULT_BONE_RADIUS_CM,
        "evaluate_sequence_pose": PLAYER_BBOX_EVALUATE_SEQUENCE_POSE,
    },
    "config_path": CONFIG_PATH.replace("\\", "/"),
    "seq_id": SEQ_ID,
    "frame_start": FRAME_START,
    "frame_end": FRAME_END,
    "display_rate": DISPLAY_RATE,
    "image_width": IMAGE_WIDTH,
    "image_height": IMAGE_HEIGHT,
    "camera_ids": list(CAMERAS.keys()),
    "object_ids": list(OBJECT_TRAJECTORIES.keys()),
    "class_id_map": CLASS_ID_MAP,
    "track_id_map": TRACK_ID_MAP,
    "roster": CONFIG.get("roster"),
    "movement_optimization": CONFIG.get("movement_optimization"),
    "trajectories": trajectory_metadata,
    "records": records,
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

with open(OUT_JSONL, "w", encoding="utf-8") as f:
    for record in records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

unreal.log("===================================")
unreal.log("[DONE] A3.3b episode built, action animation sections added, annotations exported")
unreal.log("JSON : {}".format(OUT_JSON))
unreal.log("JSONL: {}".format(OUT_JSONL))
unreal.log("Records: {}".format(len(records)))
unreal.log("Expected records: {}".format(
    len(CAMERAS) * (FRAME_END - FRAME_START + 1)
))
unreal.log("Trajectory interpolation: piecewise linear")
unreal.log("Yaw mode: explicit A3.3 yaw + continuous unwrap fallback")
unreal.log("Animation mode: action_timeline -> multiple Skeletal Animation sections")
unreal.log("Player bbox: player_bone_envelope_2d first, fallback 3D bbox")
unreal.log("Player keypoints: player_skeleton_2d_v1 count={}".format(len(PLAYER_KEYPOINT_BONES)))
unreal.log("===================================")
unreal.log("下一步：MRQ 渲染 frame {}..{} 到 images_clean/{}/cam_XX。".format(
    FRAME_START,
    FRAME_END,
    SEQ_ID
))
unreal.log("然后运行 03_check_labels.py，并指定本序列 annotation JSON。")
unreal.log("===================================")
