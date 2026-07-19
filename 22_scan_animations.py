# FutsalMOT UE animation asset scanner
# Version marker: A3_3B_ACTION_ANIMATION_ASSET_SCANNER_V3
#
# Run inside Unreal Engine:
# py "D:/projects/FustalMOT_UEDataset/Content/FutsalMOT/code/22_scan_animations.py"
#
# Output:
# Saved/FutsalMOT/animation_assets/action_animation_candidates_v2.json
#
# V3: derive project root from Unreal project_dir; fall back to the known project path.
#
# V2 fixes:
# 1. Avoids printing Python SoftObjectPath struct representations.
# 2. Builds canonical paths from AssetData.package_name + asset_name.
# 3. Writes a directly loadable "/Game/..." path for action_animation_map.json.
# 4. Reports skeleton, duration, class, and compatibility with
#    MovieSceneSkeletalAnimationTrack.
# 5. Writes all scanned animation assets, not only keyword matches.

import unreal
import os
import json
import traceback


DEFAULT_PROJECT_ROOT = "D:/projects/FustalMOT_UEDataset"


def resolve_project_root():
    """Resolve the active Unreal project directory without assuming drive/path."""
    try:
        raw = unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_dir())
        if raw:
            return os.path.normpath(str(raw))
    except Exception:
        pass
    return os.path.normpath(DEFAULT_PROJECT_ROOT)


PROJECT_ROOT = resolve_project_root()

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "Saved",
    "FutsalMOT",
    "animation_assets",
)

OUTPUT_JSON = os.path.join(
    OUTPUT_DIR,
    "action_animation_candidates_v2.json",
)


ACTION_KEYWORDS = {
    "idle": [
        "idle",
        "stand",
        "standing",
        "breath",
        "breathe",
    ],
    "jog": [
        "jog",
        "run",
        "running",
        "locomotion",
        "forward",
        "sprint",
    ],
    "dribble": [
        "dribble",
        "dribbling",
        "ball_control",
        "ballcontrol",
        "soccer_dribble",
        "football_dribble",
    ],
    "pass": [
        "pass",
        "passing",
        "kick_pass",
        "soccer_pass",
        "football_pass",
        "short_kick",
        "kick",
    ],
    "receive": [
        "receive",
        "receiving",
        "trap",
        "trapping",
        "control",
        "first_touch",
        "firsttouch",
        "catch_ball",
    ],
    "shot": [
        "shot",
        "shoot",
        "shooting",
        "kick_shot",
        "soccer_kick",
        "football_kick",
        "volley",
        "kick",
    ],
    "defend": [
        "defend",
        "defending",
        "defensive",
        "strafe",
        "shuffle",
        "guard",
        "block",
    ],
}


EXCLUDE_KEYWORDS = [
    "additive",
    "pose",
    "preview",
    "test",
    "deprecated",
]


# MovieSceneSkeletalAnimationSection.params.animation expects an
# AnimSequenceBase-compatible asset. BlendSpace assets are useful for discovery,
# but should not be placed directly into action_animation_map.json.
DIRECT_SECTION_CLASSES = {
    "AnimSequence",
    "AnimMontage",
    "AnimComposite",
}


def safe_name(value):
    if value is None:
        return ""
    try:
        return str(value)
    except Exception:
        return ""


def get_asset_data_property(asset_data, property_name):
    try:
        return getattr(asset_data, property_name)
    except Exception:
        pass

    try:
        return asset_data.get_editor_property(property_name)
    except Exception:
        return None


def canonical_paths(asset_data):
    """
    Return paths without relying on SoftObjectPath.__str__.

    package_name:
        /Game/Folder/AS_Run

    asset_name:
        AS_Run

    load_path:
        /Game/Folder/AS_Run

    object_path:
        /Game/Folder/AS_Run.AS_Run
    """
    package_name = safe_name(
        get_asset_data_property(asset_data, "package_name")
    )
    asset_name = safe_name(
        get_asset_data_property(asset_data, "asset_name")
    )
    package_path = safe_name(
        get_asset_data_property(asset_data, "package_path")
    )

    if package_name and asset_name:
        object_path = "{}.{}".format(package_name, asset_name)
        load_path = package_name
    else:
        object_path = package_name or asset_name
        load_path = package_name or object_path

    return {
        "asset_name": asset_name,
        "package_name": package_name,
        "package_path": package_path,
        "load_path": load_path,
        "object_path": object_path,
    }


def get_path_name(value):
    if value is None:
        return None

    for method_name in [
        "get_path_name",
        "get_full_name",
    ]:
        try:
            method = getattr(value, method_name)
            result = method()
            if result:
                return str(result)
        except Exception:
            pass

    try:
        return str(value)
    except Exception:
        return None


def inspect_loaded_asset(asset_data, class_name):
    result = {
        "skeleton_path": None,
        "play_length_sec": None,
        "rate_scale": None,
        "loaded_class": None,
        "load_error": None,
    }

    try:
        asset = asset_data.get_asset()
    except Exception as exc:
        result["load_error"] = "{}: {}".format(
            type(exc).__name__,
            exc,
        )
        return result

    if asset is None:
        result["load_error"] = "asset_data.get_asset() returned None"
        return result

    try:
        result["loaded_class"] = asset.get_class().get_name()
    except Exception:
        result["loaded_class"] = class_name

    try:
        skeleton = asset.get_editor_property("skeleton")
        result["skeleton_path"] = get_path_name(skeleton)
    except Exception:
        pass

    for method_name in [
        "get_play_length",
        "get_play_length_seconds",
    ]:
        try:
            method = getattr(asset, method_name)
            value = float(method())
            result["play_length_sec"] = value
            break
        except Exception:
            pass

    if result["play_length_sec"] is None:
        for property_name in [
            "sequence_length",
            "play_length",
        ]:
            try:
                result["play_length_sec"] = float(
                    asset.get_editor_property(property_name)
                )
                break
            except Exception:
                pass

    try:
        result["rate_scale"] = float(
            asset.get_editor_property("rate_scale")
        )
    except Exception:
        pass

    return result


def normalized_search_text(record):
    values = [
        record.get("asset_name", ""),
        record.get("package_name", ""),
        record.get("package_path", ""),
        record.get("load_path", ""),
        record.get("object_path", ""),
        record.get("asset_class", ""),
        record.get("skeleton_path", "") or "",
    ]
    return " ".join(values).lower()


def keyword_score(search_text, keywords):
    score = 0
    matched = []

    for keyword in keywords:
        keyword_lower = keyword.lower()

        if keyword_lower in search_text:
            # Longer and more specific names rank above generic terms.
            score += max(1, len(keyword_lower))
            matched.append(keyword)

    for keyword in EXCLUDE_KEYWORDS:
        if keyword in search_text:
            score -= 5

    return score, matched


def query_assets_by_class(registry, package_path, class_name):
    try:
        class_path = unreal.TopLevelAssetPath(
            package_path,
            class_name,
        )
        result = registry.get_assets_by_class(
            class_path,
            True,
        )
        return list(result or [])
    except Exception:
        pass

    try:
        result = registry.get_assets_by_class(
            class_name,
            True,
        )
        return list(result or [])
    except Exception:
        return []


def get_animation_assets():
    registry = unreal.AssetRegistryHelpers.get_asset_registry()
    records = []
    seen = set()

    class_specs = [
        ("/Script/Engine", "AnimSequence"),
        ("/Script/Engine", "AnimMontage"),
        ("/Script/Engine", "AnimComposite"),
        ("/Script/Engine", "BlendSpace"),
        ("/Script/Engine", "BlendSpace1D"),
    ]

    for class_package, class_name in class_specs:
        asset_data_list = query_assets_by_class(
            registry,
            class_package,
            class_name,
        )

        for asset_data in asset_data_list:
            paths = canonical_paths(asset_data)
            dedupe_key = paths["object_path"] or paths["load_path"]

            if not dedupe_key or dedupe_key in seen:
                continue

            seen.add(dedupe_key)

            inspection = inspect_loaded_asset(
                asset_data,
                class_name,
            )

            record = {
                **paths,
                "asset_class": class_name,
                "direct_section_compatible": (
                    class_name in DIRECT_SECTION_CLASSES
                ),
                **inspection,
            }
            record["_search_text"] = normalized_search_text(record)
            records.append(record)

    records.sort(
        key=lambda item: (
            item["asset_class"].lower(),
            item["load_path"].lower(),
        )
    )
    return records


def build_candidates(assets):
    candidates = {}

    for action, keywords in ACTION_KEYWORDS.items():
        scored = []

        for asset in assets:
            score, matched = keyword_score(
                asset["_search_text"],
                keywords,
            )

            if score <= 0:
                continue

            record = {
                key: value
                for key, value in asset.items()
                if not key.startswith("_")
            }
            record["score"] = score
            record["matched_keywords"] = matched
            scored.append(record)

        scored.sort(
            key=lambda item: (
                not item["direct_section_compatible"],
                -item["score"],
                item["load_path"].lower(),
            )
        )
        candidates[action] = scored[:100]

    return candidates


def json_safe_assets(assets):
    return [
        {
            key: value
            for key, value in asset.items()
            if not key.startswith("_")
        }
        for asset in assets
    ]


def log_candidate(action, item):
    unreal.log(
        "  score={} class={} compatible={} name={} path={} "
        "skeleton={} length={}".format(
            item["score"],
            item["asset_class"],
            item["direct_section_compatible"],
            item["asset_name"],
            item["load_path"],
            item.get("skeleton_path"),
            item.get("play_length_sec"),
        )
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    unreal.log("===================================")
    unreal.log("FutsalMOT action animation asset scanner V2")
    unreal.log("Scanning animation assets...")

    assets = get_animation_assets()
    candidates = build_candidates(assets)

    output = {
        "schema_version": "2.0",
        "scanner_version": (
            "A3_3B_ACTION_ANIMATION_ASSET_SCANNER_V2"
        ),
        "asset_count": len(assets),
        "direct_section_compatible_count": sum(
            1
            for asset in assets
            if asset["direct_section_compatible"]
        ),
        "keywords": ACTION_KEYWORDS,
        "notes": {
            "map_path_field": (
                "Copy candidate.load_path into "
                "action_animation_map.json"
            ),
            "compatible_classes": sorted(DIRECT_SECTION_CLASSES),
            "blendspace_warning": (
                "BlendSpace/BlendSpace1D are listed for discovery but "
                "must not be assigned directly to a Sequencer skeletal "
                "animation section."
            ),
        },
        "candidates": candidates,
        "all_assets": json_safe_assets(assets),
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as file:
        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    unreal.log("All animation-like assets: {}".format(len(assets)))
    unreal.log(
        "Direct section compatible assets: {}".format(
            output["direct_section_compatible_count"]
        )
    )
    unreal.log("Output: {}".format(OUTPUT_JSON))

    for action in ACTION_KEYWORDS:
        action_candidates = candidates[action]
        compatible_count = sum(
            1
            for item in action_candidates
            if item["direct_section_compatible"]
        )

        unreal.log(
            "{} candidates: {} (compatible={})".format(
                action,
                len(action_candidates),
                compatible_count,
            )
        )

        for item in action_candidates[:10]:
            log_candidate(action, item)

    unreal.log("===================================")


try:
    main()
except Exception as exc:
    unreal.log_error(
        "Animation scanner V2 failed: {}: {}".format(
            type(exc).__name__,
            exc,
        )
    )
    unreal.log_error(traceback.format_exc())
    raise
