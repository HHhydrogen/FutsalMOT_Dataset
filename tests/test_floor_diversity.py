"""验证地板因素的重现性、输入边界及配对语义。"""

import random
from copy import deepcopy

import pytest

from grf_ue_bridge.floor_diversity import CATALOG, build_floor_manifest, paired_floor_manifests, select_floor_id

INPUTS = dict(trajectory_sha256="a" * 64, frozen_context_sha256="b" * 64,
              texture_sha256="c" * 64)


def test_reproducible_and_rng_independent():
    before = random.getstate()
    result = build_floor_manifest(1021, **INPUTS)
    assert random.getstate() == before
    random.Random(42).random()
    assert result == build_floor_manifest(1021, **INPUTS)
    assert result["application_status"] == "planned_not_applied"


def test_context_does_not_change_floor_selection():
    a = build_floor_manifest(1021, **INPUTS)
    b = build_floor_manifest(1021, **{**INPUTS, "frozen_context_sha256": "d" * 64})
    assert a["variant"] == b["variant"]
    assert a["pairing_key"] != b["pairing_key"]


def test_pairs_only_change_floor():
    pairs = paired_floor_manifests(1021, **INPUTS)
    assert len({p["pairing_key"] for p in pairs}) == 1
    assert len({p["variant"]["id"] for p in pairs}) == 4
    assert len({p["fingerprint"] for p in pairs}) == 4
    for key in ("trajectory_sha256", "frozen_context_sha256", "material_parameters"):
        assert all(p[key] == pairs[0][key] for p in pairs)


def test_returns_no_shared_mutable_catalog():
    before = deepcopy(CATALOG)
    output = build_floor_manifest(42, **INPUTS)
    output["variant"]["base_color_linear"][0] = 99
    assert CATALOG == before


@pytest.mark.parametrize("seed", [True, -1, 2**31, 1.5, "1"])
def test_bad_seed(seed):
    with pytest.raises(ValueError):
        build_floor_manifest(seed, **INPUTS)


@pytest.mark.parametrize("key", list(INPUTS))
def test_bad_sha(key):
    with pytest.raises(ValueError):
        build_floor_manifest(1, **{**INPUTS, key: "missing"})


@pytest.mark.parametrize("floor_id", ["not_a_floor", ""])
def test_bad_floor_id(floor_id):
    with pytest.raises(ValueError):
        build_floor_manifest(1, floor_id=floor_id, **INPUTS)


def test_seed_pool_covers_catalog():
    selected = {build_floor_manifest(seed, **INPUTS)["variant"]["id"] for seed in range(100)}
    assert selected == {p["id"] for p in CATALOG}


def test_ue_preview_seed_selector_matches_manifest():
    for seed in (0, 1021, 1025, 1007, 1042, 1044, 1048, 1102, 1131, 2**31 - 1):
        assert select_floor_id(seed) == build_floor_manifest(seed, **INPUTS)["variant"]["id"]
