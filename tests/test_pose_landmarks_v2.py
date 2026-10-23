"""独立 COCO17 表面锚点契约的拒绝与形变不变量测试。"""
import copy

import pytest

from pose_landmarks_v2 import (
    COCO17_NAMES, COCO17_FLIP_IDX, HEAD_NAMES, evaluate_surface_anchor,
    mesh_fingerprints, validate_landmark_manifest, mpfb_openpose_head_candidates,
)

VERTICES = [[0, 0, 0], [2, 0, 0], [0, 2, 0]]
TRIANGLES = [[0, 1, 2]]


def fixture_manifest():
    identity = {"asset_path": "/Game/Test/Head", "lod": 0, **mesh_fingerprints(VERTICES, TRIANGLES)}
    return {
        "schema": "futsalmot_head_surface_landmarks", "version": 1,
        "mesh_identity": identity, "coordinate_frame": "ue_component_cm",
        "source_method": "deformed_surface_barycentric",
        "calibration": {"status": "accepted", "evidence_ref": "test-only.json",
                        "max_reprojection_error_px": 0.2, "acceptance_threshold_px": 1},
        "anchors": {name: {"triangle_vertices": [0, 1, 2], "barycentric": [.2, .3, .5],
                    "semantic_review": {"reviewed": True, "method": "test_fixture_only",
                                        "evidence_ref": "not-a-production-calibration"}} for name in HEAD_NAMES},
    }


def test_coco_names_and_flip_are_involution():
    assert len(set(COCO17_NAMES)) == 17
    assert COCO17_NAMES[:5] == HEAD_NAMES
    assert tuple(COCO17_FLIP_IDX[i] for i in COCO17_FLIP_IDX) == tuple(range(17))


def test_actual_deformed_vertices_and_affine_transform_commute():
    point = evaluate_surface_anchor(VERTICES, [0, 1, 2], [.2, .3, .5])
    assert point == pytest.approx([.6, 1, 0])
    transformed = [[-v[1]+7, v[0]*2-4, v[2]+9] for v in VERTICES]
    assert evaluate_surface_anchor(transformed, [0, 1, 2], [.2, .3, .5]) == pytest.approx([6, -2.8, 9])


@pytest.mark.parametrize("indices,weights", [([0, 0, 1], [1, 0, 0]), ([0, 1, 3], [1, 0, 0]),
    ([0, 1, 2], [-.1, .5, .6]), ([0, 1, 2], [.2, .2, .2]),
    ([0, 1, 2], [float('nan'), .5, .5]), ([False, 1, 2], [1, 0, 0])])
def test_reject_invalid_anchor(indices, weights):
    with pytest.raises(ValueError):
        evaluate_surface_anchor(VERTICES, indices, weights)


def test_reject_degenerate_surface():
    with pytest.raises(ValueError, match="退化"):
        evaluate_surface_anchor([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [0, 1, 2], [1, 0, 0])


def test_accept_complete_fixture_but_reject_mesh_reorder():
    manifest = fixture_manifest()
    identity = manifest['mesh_identity']
    validate_landmark_manifest(manifest, identity, VERTICES, TRIANGLES)
    with pytest.raises(ValueError, match="geometry_sha256"):
        validate_landmark_manifest(manifest, identity, VERTICES[::-1], TRIANGLES)


@pytest.mark.parametrize("change", ["missing", "unknown", "unreviewed", "error", "offset", "identity", "nan"])
def test_reject_incomplete_or_unproven_calibration(change):
    manifest = copy.deepcopy(fixture_manifest())
    if change == 'missing': del manifest['anchors']['nose']
    if change == 'unknown': manifest['calibration']['status'] = 'unreviewed'
    if change == 'unreviewed': manifest['anchors']['nose']['semantic_review']['reviewed'] = False
    if change == 'error': manifest['calibration']['max_reprojection_error_px'] = 2
    if change == 'offset': manifest['source_method'] = 'head_bone_fixed_offset'
    if change == 'identity': manifest['mesh_identity']['asset_path'] = '/Game/Other'
    if change == 'nan': manifest['calibration']['max_reprojection_error_px'] = float('nan')
    with pytest.raises(ValueError):
        validate_landmark_manifest(manifest, fixture_manifest()['mesh_identity'], VERTICES, TRIANGLES)


def test_openpose_semantics_use_tail_not_head():
    rig = {"bones": {bone: {"head": {"vertex_indices": [9]},
           "tail": {"strategy": "MEAN", "vertex_indices": [1, 2]}}
           for bone in ('Neck', 'LEye', 'REye', 'LEar', 'REar')}}
    result = mpfb_openpose_head_candidates(rig)
    assert all(v['vertex_indices'] == [1, 2] for v in result.values())
    assert result['nose']['status'] == 'candidate_unreviewed'
