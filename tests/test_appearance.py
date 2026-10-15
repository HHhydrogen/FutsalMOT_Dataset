"""MPFB episode appearance generator 的纯 Python 契约测试。"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from grf_ue_bridge.appearance import (
    APPEARANCE_POLICY,
    build_appearance_manifest,
    catalog_fingerprint,
    derive_appearance_seeds,
    validate_appearance_manifest,
    write_appearance_manifest,
)


ENTITY_IDS = ("L0", "L1", "L2", "L3", "L4", "R0", "R1", "R2", "R3", "R4")


def appearance_catalog() -> dict:
    return {
        "mode": "seeded",
        "policy": APPEARANCE_POLICY,
        "catalog_id": "mpfb_players_003_009_v1",
        "models": [
            {
                "player_id": player_id,
                "mesh": f"/Game/FutsalMOT/Characters/MPFB/Players/Player_{player_id:03d}/MPFB_Player_{player_id:03d}",
                "anim_class": "/Game/FutsalMOT/Characters/MPFB/Core/Retarget/ABP_MPFB_RuntimeRetarget",
                "skeleton": "/Game/FutsalMOT/Characters/MPFB/Core/Skeleton/SKEL_MPFB_Base",
            }
            for player_id in range(3, 10)
        ],
        "skins": [
            {
                "skin_id": f"skin-{index:02d}",
                "texture": f"/Game/FutsalMOT/Characters/MPFB/Textures/Skins/Male/T_Skin_{index:02d}",
            }
            for index in range(1, 10)
        ],
        "kits": [
            {
                "set_id": f"{index:02d}",
                "jersey_material": f"/Game/FutsalMOT/Characters/MPFB/Materials/M_MPFB_Jersey_Set_{index:02d}",
                "shorts_material": f"/Game/FutsalMOT/Characters/MPFB/Materials/M_MPFB_Shorts_Set_{index:02d}",
            }
            for index in range(1, 6)
        ],
        "body_texture_parameter": "SkinTexture",
    }


def appearance_meta(*, ue_visual_seed: int = 1001) -> dict:
    entities = []
    for entity_id in ENTITY_IDS:
        entities.append(
            {
                "id": entity_id,
                "team": "left" if entity_id.startswith("L") else "right",
                "is_goalkeeper": entity_id in {"L2", "R1"},
            }
        )
    return {
        "schema": "grf_ue_episode",
        "version": 1,
        "episode_id": "appearance-test-001",
        "source": {"seed": 42},
        "randomness": {
            "policy": "futsalmot_seed_v1",
            "root_seed": 42,
            "ue_visual_seed": ue_visual_seed,
        },
        "entities": entities,
    }


def write_frames(tmp_path: Path, content: bytes = b'{"step":0}\n') -> Path:
    path = tmp_path / "frames.jsonl"
    path.write_bytes(content)
    return path


def test_appearance_seeds_follow_the_approved_byte_protocol():
    seeds = derive_appearance_seeds(1001)

    assert set(seeds) == {"model", "skin", "kit"}
    for namespace, value in seeds.items():
        payload = f"{APPEARANCE_POLICY}:1001:{namespace}".encode("utf-8")
        expected = int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF
        assert value == expected

    assert len(set(seeds.values())) == 3


def test_appearance_seed_protocol_is_stable_in_a_subprocess():
    code = (
        "import json; "
        "from grf_ue_bridge.appearance import derive_appearance_seeds; "
        "print(json.dumps(derive_appearance_seeds(1001), sort_keys=True))"
    )
    process = subprocess.run(
        [sys.executable, "-c", code], cwd=".", capture_output=True, text=True
    )

    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout) == derive_appearance_seeds(1001)


def test_manifest_is_deterministic_and_binds_exact_trajectory_bytes(tmp_path):
    catalog = appearance_catalog()
    meta = appearance_meta()
    frames_path = write_frames(tmp_path, b'{"step":0,"value":"exact"}\n')

    first = build_appearance_manifest(meta, frames_path, catalog)
    second = build_appearance_manifest(meta, frames_path, copy.deepcopy(catalog))

    assert first == second
    assert first["schema"] == "futsalmot_episode_appearance"
    assert first["version"] == 1
    assert first["episode_id"] == meta["episode_id"]
    assert first["trajectory_sha256"] == hashlib.sha256(frames_path.read_bytes()).hexdigest()
    assert first["catalog_fingerprint"] == catalog_fingerprint(catalog)
    assert first["fingerprint"]


@pytest.mark.parametrize("seed_source", ["randomness", "source"])
@pytest.mark.parametrize("root_seed", [0x80000000 + 123, -0x80000000 - 123])
def test_manifest_accepts_large_root_seed_and_keeps_derived_seeds_bounded(
    tmp_path, seed_source, root_seed
):
    catalog = appearance_catalog()
    meta = appearance_meta()
    meta["source"]["seed"] = root_seed
    meta["randomness"]["root_seed"] = (
        root_seed if seed_source == "randomness" else None
    )
    frames_path = write_frames(tmp_path)

    manifest = build_appearance_manifest(meta, frames_path, catalog)

    assert manifest["root_seed"] == root_seed
    assert all(0 <= seed <= 0x7FFFFFFF for seed in manifest["namespace_seeds"].values())
    assert validate_appearance_manifest(
        manifest, meta=meta, frames_path=frames_path, catalog=catalog
    ) == []


def test_manifest_uses_metadata_goalkeepers_and_fixed_kit_classes(tmp_path):
    manifest = build_appearance_manifest(
        appearance_meta(), write_frames(tmp_path), appearance_catalog()
    )

    assert {record["entity_id"] for record in manifest["entities"]} == set(ENTITY_IDS)
    by_id = {record["entity_id"]: record for record in manifest["entities"]}
    assert by_id["L2"]["is_goalkeeper"] is True
    assert by_id["R1"]["is_goalkeeper"] is True
    assert by_id["L0"]["kit_class"] == "left_outfield"
    assert by_id["R0"]["kit_class"] == "right_outfield"
    assert by_id["L2"]["kit_class"] == "left_goalkeeper"
    assert by_id["R1"]["kit_class"] == "right_goalkeeper"

    kit_classes = manifest["kit_classes"]
    assert list(kit_classes) == [
        "left_outfield",
        "right_outfield",
        "left_goalkeeper",
        "right_goalkeeper",
    ]
    assert len(set(kit_classes.values())) == 4
    assert all(record["kit_set_id"] == kit_classes[record["kit_class"]] for record in by_id.values())
    kits_by_id = {kit["set_id"]: kit for kit in appearance_catalog()["kits"]}
    assert all(
        record["jersey_material"] == kits_by_id[record["kit_set_id"]]["jersey_material"]
        and record["shorts_material"] == kits_by_id[record["kit_set_id"]]["shorts_material"]
        and record["jersey_material"].endswith(f"_Set_{record['kit_set_id']}")
        and record["shorts_material"].endswith(f"_Set_{record['kit_set_id']}")
        for record in by_id.values()
    )


def test_manifest_selection_matches_independent_protocol_rankings(tmp_path):
    catalog = appearance_catalog()
    meta = appearance_meta(ue_visual_seed=1001)
    manifest = build_appearance_manifest(meta, write_frames(tmp_path), catalog)
    by_entity = {record["entity_id"]: record for record in manifest["entities"]}
    policy = "futsalmot_appearance_v1"

    namespace_seeds = {}
    for namespace in ("model", "skin", "kit"):
        seed_bytes = hashlib.sha256(
            f"{policy}:1001:{namespace}".encode("utf-8")
        ).digest()[:4]
        namespace_seeds[namespace] = int.from_bytes(seed_bytes, "big") & 0x7FFFFFFF

    def ranked_id(namespace, entity_id, candidates, id_key):
        def rank(candidate):
            candidate_id = str(candidate[id_key])
            payload = (
                f"{policy}:rank:{namespace}:"
                f"{namespace_seeds[namespace]}:{entity_id}:{candidate_id}"
            ).encode("utf-8")
            return hashlib.sha256(payload).digest(), candidate_id

        return min(candidates, key=rank)[id_key]

    for entity_id in ENTITY_IDS:
        assert by_entity[entity_id]["player_id"] == ranked_id(
            "model", entity_id, catalog["models"], "player_id"
        )
        assert by_entity[entity_id]["skin_id"] == ranked_id(
            "skin", entity_id, catalog["skins"], "skin_id"
        )

    expected_kit_ids = sorted(
        (
            (
                hashlib.sha256(
                    (
                        f"{policy}:rank:kit:{namespace_seeds['kit']}:"
                        f"kit:{kit['set_id']}"
                    ).encode("utf-8")
                ).digest(),
                kit["set_id"],
            )
            for kit in catalog["kits"]
        )
    )[:4]
    expected_kit_classes = {
        kit_class: kit_id
        for kit_class, (_, kit_id) in zip(
            ("left_outfield", "right_outfield", "left_goalkeeper", "right_goalkeeper"),
            expected_kit_ids,
        )
    }
    assert manifest["namespace_seeds"] == namespace_seeds
    assert {kit["set_id"] for kit in catalog["kits"]} == {"01", "02", "03", "04", "05"}
    assert manifest["kit_classes"] == expected_kit_classes
    assert all(
        record["kit_set_id"] == expected_kit_classes[record["kit_class"]]
        for record in by_entity.values()
    )
    kits_by_id = {kit["set_id"]: kit for kit in catalog["kits"]}
    assert all(
        record["jersey_material"] == kits_by_id[record["kit_set_id"]]["jersey_material"]
        and record["shorts_material"] == kits_by_id[record["kit_set_id"]]["shorts_material"]
        for record in by_entity.values()
    )


def test_manifest_validation_accepts_current_inputs_and_rejects_bad_fingerprint(tmp_path):
    catalog = appearance_catalog()
    meta = appearance_meta()
    frames_path = write_frames(tmp_path)
    manifest = build_appearance_manifest(meta, frames_path, catalog)

    assert validate_appearance_manifest(
        manifest, meta=meta, frames_path=frames_path, catalog=catalog
    ) == []

    bad = copy.deepcopy(manifest)
    bad["fingerprint"] = "0" * 64
    errors = validate_appearance_manifest(
        bad, meta=meta, frames_path=frames_path, catalog=catalog
    )
    assert any("fingerprint" in error for error in errors)


def test_manifest_validation_rejects_boolean_version_with_valid_fingerprint(tmp_path):
    catalog = appearance_catalog()
    meta = appearance_meta()
    frames_path = write_frames(tmp_path)
    manifest = build_appearance_manifest(meta, frames_path, catalog)
    manifest["version"] = True
    fingerprint_payload = {
        key: value for key, value in manifest.items() if key != "fingerprint"
    }
    manifest["fingerprint"] = hashlib.sha256(
        json.dumps(
            fingerprint_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    errors = validate_appearance_manifest(
        manifest, meta=meta, frames_path=frames_path, catalog=catalog
    )

    assert any("version" in error for error in errors)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("randomness", "ue_visual_seed", float("inf")),
        ("randomness", "root_seed", float("inf")),
        ("source", "seed", float("inf")),
        ("randomness", "ue_visual_seed", 1001.5),
        ("randomness", "ue_visual_seed", True),
        ("randomness", "root_seed", True),
    ],
)
def test_manifest_validation_reports_non_integer_seed_without_raising(
    tmp_path, section, field, value
):
    catalog = appearance_catalog()
    meta = appearance_meta()
    if section == "source":
        meta["randomness"]["root_seed"] = None
    meta[section][field] = value
    frames_path = write_frames(tmp_path)
    manifest = build_appearance_manifest(
        appearance_meta(), frames_path, catalog
    )

    errors = validate_appearance_manifest(
        manifest, meta=meta, frames_path=frames_path, catalog=catalog
    )

    assert errors
    assert any("seed" in error for error in errors)


def test_valid_ball_does_not_change_complete_appearance_manifest(tmp_path):
    catalog = appearance_catalog()
    meta_without_ball = appearance_meta()
    meta_with_ball = copy.deepcopy(meta_without_ball)
    meta_with_ball["entities"].append({"id": "BALL", "team": None, "is_goalkeeper": False})
    frames_path = write_frames(tmp_path)

    manifest_without_ball = build_appearance_manifest(meta_without_ball, frames_path, catalog)
    manifest_with_ball = build_appearance_manifest(meta_with_ball, frames_path, catalog)

    assert manifest_with_ball == manifest_without_ball
    assert manifest_with_ball["fingerprint"] == manifest_without_ball["fingerprint"]
    assert manifest_with_ball["root_seed"] == manifest_without_ball["root_seed"]
    assert manifest_with_ball["ue_visual_seed"] == manifest_without_ball["ue_visual_seed"]
    assert manifest_with_ball["namespace_seeds"] == manifest_without_ball["namespace_seeds"]
    assert manifest_with_ball["kit_classes"] == manifest_without_ball["kit_classes"]
    assert manifest_with_ball["entities"] == manifest_without_ball["entities"]
    assert [
        (record["entity_id"], record["player_id"], record["skin_id"], record["kit_set_id"])
        for record in manifest_with_ball["entities"]
    ] == [
        (record["entity_id"], record["player_id"], record["skin_id"], record["kit_set_id"])
        for record in manifest_without_ball["entities"]
    ]
    assert len(manifest_with_ball["entities"]) == 10
    assert {record["entity_id"] for record in manifest_with_ball["entities"]} == set(ENTITY_IDS)
    assert validate_appearance_manifest(
        manifest_with_ball, meta=meta_with_ball, frames_path=frames_path, catalog=catalog
    ) == []


def test_ball_missing_team_key_is_rejected(tmp_path):
    meta = appearance_meta()
    meta["entities"].append({"id": "BALL", "is_goalkeeper": False})

    with pytest.raises(ValueError, match="BALL 的 team 必须为 null"):
        build_appearance_manifest(meta, write_frames(tmp_path), appearance_catalog())


@pytest.mark.parametrize(
    ("ball_metadata", "message"),
    [
        ({"id": "BALL", "team": "left", "is_goalkeeper": False}, "BALL 的 team 必须为 null"),
        ({"id": "BALL", "team": None, "is_goalkeeper": True}, "BALL.is_goalkeeper 必须为 false"),
    ],
)
def test_ball_with_wrong_team_or_goalkeeper_is_rejected(tmp_path, ball_metadata, message):
    meta = appearance_meta()
    meta["entities"].append(ball_metadata)

    with pytest.raises(ValueError, match=message):
        build_appearance_manifest(meta, write_frames(tmp_path), appearance_catalog())


def test_duplicate_ball_entity_is_rejected(tmp_path):
    meta = appearance_meta()
    ball = {"id": "BALL", "team": None, "is_goalkeeper": False}
    meta["entities"].extend([ball, copy.deepcopy(ball)])

    with pytest.raises(ValueError, match="BALL 不能重复"):
        build_appearance_manifest(meta, write_frames(tmp_path), appearance_catalog())


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda catalog, meta: catalog["models"].pop(), "models"),
        (lambda catalog, meta: catalog["skins"][1].update({"skin_id": catalog["skins"][0]["skin_id"]}), "skins"),
        (lambda catalog, meta: catalog["kits"].pop(), "kits"),
        (lambda catalog, meta: catalog["kits"][0].update({"set_id": "06"}), "kits"),
        (lambda catalog, meta: catalog["skins"][0].update({"texture": "Textures/Skin"}), "/Game/"),
        (lambda catalog, meta: catalog.update({"policy": "futsalmot_appearance_v9"}), "policy"),
    ],
)
def test_invalid_catalog_is_rejected(tmp_path, change, message):
    catalog = appearance_catalog()
    meta = appearance_meta()
    change(catalog, meta)

    with pytest.raises(ValueError, match=message):
        build_appearance_manifest(meta, write_frames(tmp_path), catalog)


def test_game_root_without_asset_path_is_rejected(tmp_path):
    catalog = appearance_catalog()
    catalog["skins"][0]["texture"] = "/Game/"

    with pytest.raises(ValueError, match="/Game/"):
        build_appearance_manifest(
            appearance_meta(), write_frames(tmp_path), catalog
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda meta: meta["entities"].pop(),
        lambda meta: meta["entities"].append(copy.deepcopy(meta["entities"][0])),
        lambda meta: meta["entities"][0].update({"team": "right"}),
        lambda meta: meta["entities"][2].update({"is_goalkeeper": False}),
        lambda meta: meta["entities"][0].update({"is_goalkeeper": True}),
    ],
)
def test_invalid_roster_or_goalkeeper_metadata_is_rejected(tmp_path, mutate):
    meta = appearance_meta()
    mutate(meta)

    with pytest.raises(ValueError):
        build_appearance_manifest(meta, write_frames(tmp_path), appearance_catalog())


def test_changed_trajectory_bytes_are_rejected(tmp_path):
    catalog = appearance_catalog()
    meta = appearance_meta()
    frames_path = write_frames(tmp_path, b"original\n")
    manifest = build_appearance_manifest(meta, frames_path, catalog)
    frames_path.write_bytes(b"changed\n")

    errors = validate_appearance_manifest(
        manifest, meta=meta, frames_path=frames_path, catalog=catalog
    )
    assert any("trajectory" in error for error in errors)


def test_write_manifest_is_atomic_and_portable(tmp_path):
    catalog = appearance_catalog()
    manifest = build_appearance_manifest(
        appearance_meta(), write_frames(tmp_path), catalog
    )

    output = write_appearance_manifest(tmp_path, manifest)

    assert output == tmp_path / "provenance" / "appearance.json"
    assert json.loads(output.read_text(encoding="utf-8")) == manifest
    assert output.read_bytes().endswith(b"\n")
    assert not list((tmp_path / "provenance").glob("*.tmp"))
    assert str(tmp_path) not in output.read_text(encoding="utf-8")
