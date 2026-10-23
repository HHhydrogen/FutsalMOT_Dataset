"""离线f32矩阵/LBS的数值顺序和严格SHA拒绝；全部为CPU假例。"""

import base64
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from grf_ue_bridge.pose17_offline_lbs_v2 import (
    candidate_ref_to_local, component_points_to_world_m, decode_reference_geometry, decode_skin_snapshot, matrix_bytes, multiply_f32_sse_order, reconstruct_surface, transform_matrix_f32,
)


def transform(translation=(0., 0., 0.)):
    return dict(translation_cm=list(translation), rotation_xyzw=[0., 0., 0., 1.], scale_xyz=[1., 1., 1.])


def toy():
    identity = dict(asset_path='/Game/Toy.Mesh', signature_scheme='ue_lod_le_u32_f32_v1', lod=0, vertex_count=1,
                    topology_sha256='b'*64, geometry_sha256='c'*64)
    inverses = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
    inverses[0, 3, 0] = -1
    inverses[1, 3, 0] = -20
    expected = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
    expected[0, 3, 0] = 2
    skin = dict(identity=identity, weights_sha256='d'*64, inverse_matrices=inverses, bone_names=['root', 'unused'], active_bone_indices=[0],
                global_bone_indices=np.array([[0, 1]]), normalized_weights=np.array([[1., 0.]], dtype=np.float32))
    row = dict(ref_to_local_signature_scheme='ue_ref_to_local_rowmajor_f32_le_v1',
               mesh_identity=identity, skin_weight_signature_scheme='ue_component_lod0_fixed_skinweight_raw_v1', skin_weight_data_sha256='d'*64,
               ref_to_local_matrices_sha256=hashlib.sha256(matrix_bytes(expected)).hexdigest(),
               component_path='Toy.MPFBVisual', bone_space_base_component_path='Toy.MPFBVisual',
               bone_pose_transform_encoding='ue_transform_translation_cm_quat_xyzw_scale_xyz_v1',
               bone_space_base_names=['root', 'unused'],
               bone_space_base_transforms_ue_cm=[transform((3., 0., 0.)), transform((90., 0., 0.))])
    return row, skin


def test_f32_matrix_product_uses_sequential_sse_order_not_double_dot():
    left = np.zeros((4, 4), dtype=np.float32)
    left[0] = [1e8, 1, -1e8, 1]
    right = np.ones((4, 4), dtype=np.float32)
    assert multiply_f32_sse_order(left, right)[0, 0] == 1
    assert np.dot(left.astype(np.float64), right.astype(np.float64))[0, 0] == 2


def test_transform_is_row_vector_translation_and_f64_quaternion_first():
    value = transform((10., 20., 30.))
    value['scale_xyz'] = [2., 3., 4.]
    value['rotation_xyzw'] = [0., 0., 2**-.5, 2**-.5]
    matrix = transform_matrix_f32(value)
    assert matrix[3].tolist() == [10., 20., 30., 1.]
    assert matrix[0, 1] == 2 and matrix[1, 0] == -3 and matrix[2, 2] == 4
    # 先在double构造再cast，不能先将quat降f32抹掉该微小非零对角。
    assert 0 < abs(matrix[0, 0]) < 1e-12


def test_exact_raw_sha_unlocks_surface_and_inactive_bones_stay_identity():
    row, skin = toy()
    report, matrices = candidate_ref_to_local(row, skin)
    assert report['ref_to_local_sha256_matched'] and not report['render_surface_equivalence_verified']
    assert report['visibility_state'] == 'unknown'
    assert matrices[0, 3, 0] == 2
    assert np.array_equal(matrices[1], np.eye(4, dtype=np.float32))
    _, positions = reconstruct_surface(row, skin, [[1., 0., 0.]])
    assert positions.tolist() == [[3., 0., 0.]]


def test_sha_mismatch_returns_no_matrices_and_refuses_surface_even_when_close():
    row, skin = toy()
    skin['inverse_matrices'][0, 3, 0] = np.nextafter(np.float32(-1), np.float32(-2))
    report, matrices = candidate_ref_to_local(row, skin)
    assert not report['ref_to_local_sha256_matched'] and matrices is None
    with pytest.raises(ValueError, match='SHA不一致'):
        reconstruct_surface(row, skin, [[1., 0., 0.]])


@pytest.mark.parametrize('case', ['leader', 'order', 'encoding', 'missing_sha', 'unknown_scheme', 'bad_active', 'duplicate_active'])
def test_unsupported_or_missing_historical_pose_evidence_is_rejected(case):
    row, skin = toy()
    if case == 'leader': row['bone_space_base_component_path'] = 'Other.Driver'
    if case == 'order': row['bone_space_base_names'] = ['unused', 'root']
    if case == 'encoding': row['bone_pose_transform_encoding'] = 'guessed_text'
    if case == 'missing_sha': row.pop('ref_to_local_matrices_sha256')
    if case == 'unknown_scheme': row['ref_to_local_signature_scheme'] = 'other'
    if case == 'bad_active': skin['active_bone_indices'] = [2]
    if case == 'duplicate_active': skin['active_bone_indices'] = [0, 0]
    with pytest.raises(ValueError): candidate_ref_to_local(row, skin)


def test_matrix_raw_encoding_preserves_signed_zero():
    plus = np.eye(4, dtype=np.float32)[None]
    minus = plus.copy(); minus[0, 0, 1] = -0.
    assert np.array_equal(plus, minus)
    assert matrix_bytes(plus) != matrix_bytes(minus)


def skin_snapshot_toy():
    row, skin = toy()
    raw = bytes([0, 1, 255, 0])
    inverse = matrix_bytes(skin['inverse_matrices'])
    bones = [dict(name=name, parent_index=index-1, inverse_reference_row_major_f32=skin['inverse_matrices'][index].ravel().tolist())
             for index,name in enumerate(skin['bone_names'])]
    return dict(schema='futsalmot_pose17_asset_base_skin_data_v1',read_succeeded=True,
        historical_native_weight_sha256_matched=True,local_read_stable=True,scene_or_asset_mutation_performed=False,
        component_effective_weights_read=False,dynamic_surface_reconstruction_verified=False,
        weight_source='asset_lod0_imported_base_skin_weight_buffer',skin_weight_signature_scheme='ue_component_lod0_fixed_skinweight_raw_v1',
        engine_frame_before=8,engine_frame_after=8,mesh_identity=skin['identity'],raw_weight_data_base64=base64.b64encode(raw).decode(),
        skin_weight_data_sha256=hashlib.sha256(raw).hexdigest(),expected_historical_skin_weight_sha256=hashlib.sha256(raw).hexdigest(),
        raw_weight_data_byte_count=4,bone_index_byte_size=1,bone_weight_byte_size=1,max_bone_influences=2,
        constant_vertex_stride_bytes=4,constant_weights_offset_bytes=2,max_raw_bone_weight=65535,
        raw_weight_to_float_multiplier=float(np.float32(1/65535)),bones=bones,active_bone_indices=[0],
        inverse_reference_matrices_base64=base64.b64encode(inverse).decode(),inverse_reference_matrices_sha256=hashlib.sha256(inverse).hexdigest(),
        inverse_reference_signature_scheme='ue_inverse_ref_pose_rowmajor_f32_le_v1',
        sections=[dict(base_vertex_index=0,vertex_count=1,section_index=0,disabled=False,bone_map=[0,1])],
        vertices=[dict(vertex_index=0,section_index=0,section_local_bone_indices=[0,1],global_bone_indices=[0,1],
                       raw_weights_u16=[65535,0],normalized_weights=[1.,0.])])


def test_skin_snapshot_base64_raw_layout_and_u8_to_u16_matches_declared_slots():
    snapshot=skin_snapshot_toy()
    decoded=decode_skin_snapshot(snapshot)
    assert decoded['normalized_weights'].tolist()==[[1.,0.]]
    assert decoded['global_bone_indices'].tolist()==[[0,1]]
    assert decoded['bone_names']==['root','unused']
    assert decoded['weights_sha256']==snapshot['expected_historical_skin_weight_sha256']
    # 实际UStruct b前缀/camel键也必须按同契约解析，不填任何缺省字段。
    snapshot['bReadSucceeded']=snapshot.pop('read_succeeded')
    assert decode_skin_snapshot(snapshot)['weights_sha256']==decoded['weights_sha256']


@pytest.mark.parametrize('case',['weight_sha','inverse_sha','decoded_weight','local_bone','global_bone','section_overlap','inverse_json','effective_claim','frame_changed'])
def test_skin_snapshot_rejects_signature_layout_and_scope_mismatch(case):
    snapshot=skin_snapshot_toy()
    if case=='weight_sha':snapshot['expected_historical_skin_weight_sha256']='a'*64
    if case=='inverse_sha':snapshot['inverse_reference_matrices_sha256']='a'*64
    if case=='decoded_weight':snapshot['vertices'][0]['raw_weights_u16'][0]=255
    if case=='local_bone':snapshot['vertices'][0]['section_local_bone_indices'][0]=1
    if case=='global_bone':snapshot['vertices'][0]['global_bone_indices'][0]=1
    if case=='section_overlap':snapshot['sections'].append(copy.deepcopy(snapshot['sections'][0]))
    if case=='inverse_json':snapshot['bones'][0]['inverse_reference_row_major_f32'][0]=2.
    if case=='effective_claim':snapshot['component_effective_weights_read']=True
    if case=='frame_changed':snapshot['engine_frame_after']=9
    with pytest.raises(ValueError):decode_skin_snapshot(snapshot)


def test_reference_geometry_matches_exact_bytes_and_explicit_legacy_integer_strings():
    _,skin=toy()
    positions=np.array([[0,0,0],[1,0,0],[0,1,0]],dtype='<f4')
    triangles=np.array([[0,1,2]],dtype='<u4')
    skin['identity'].update(vertex_count=3,geometry_sha256=hashlib.sha256(positions.tobytes()).hexdigest(),
                            topology_sha256=hashlib.sha256(triangles.tobytes()).hexdigest())
    old_identity=dict(skin['identity'],lod='0',vertex_count='3')
    geometry=dict(identity=old_identity,positions_cm=positions.tolist(),triangles=triangles.tolist())
    rebuilt,_=decode_reference_geometry(geometry,skin)
    assert rebuilt.tobytes()==positions.tobytes()
    geometry['positions_cm'][1][0]=float(np.nextafter(np.float32(1),np.float32(2)))
    with pytest.raises(ValueError,match='原始SHA'):decode_reference_geometry(geometry,skin)


@pytest.mark.parametrize('case',['mesh','weights'])
def test_historical_actual_mesh_and_effective_weights_are_required_before_matrix_hash(case):
    row,skin=toy();row=copy.deepcopy(row)
    if case=='mesh':row['mesh_identity']['asset_path']='/Game/Other.Mesh'
    if case=='weights':row['skin_weight_data_sha256']='a'*64
    with pytest.raises(ValueError):candidate_ref_to_local(row,skin)


@pytest.mark.parametrize('mismatch,surface_frames',[(False,None),(True,None),(False,[1])])
def test_cli_cpu_toy_full_native_gates_and_batch_surface_refusal(tmp_path,mismatch,surface_frames):
    """复用现有GPU schema toy，实际normalizer/门禁/磁盘输出，不模拟真实渲染。"""
    from test_inspect_pose17_smoke import gpu_toy_case, LIMB_BONES
    from grf_ue_bridge.tools.reconstruct_pose17_lbs import inspect
    plan,receipt,sample=gpu_toy_case(tmp_path)
    # 现有通用toy允许DriverMesh来源；本离线路径只支持实际joint的同MPFBVisual来源。
    sample['bone_space_base_component_path']=sample['component_path']
    count=len(LIMB_BONES)
    positions=np.array([[0,0,0],[1,0,0],[0,1,0]],dtype='<f4')
    triangles=np.array([[0,1,2]],dtype='<u4')
    identity=copy.deepcopy(sample['mesh_identity']);identity['lod']=identity.pop('l_od')
    identity.update(geometry_sha256=hashlib.sha256(positions.tobytes()).hexdigest(),
                    topology_sha256=hashlib.sha256(triangles.tobytes()).hexdigest())
    candidate=Path(plan['targets'][0]['landmark_manifest'])
    candidate.write_text(json.dumps({'ue_identity':identity}),encoding='utf-8')
    sample['manifest_sha256']=hashlib.sha256(candidate.read_bytes()).hexdigest()
    sample['mesh_identity']={**identity,'l_od':identity['lod']};sample['mesh_identity'].pop('lod')
    raw=bytes([0,1,255,0])*3
    raw_sha=hashlib.sha256(raw).hexdigest()
    inverse=np.repeat(np.eye(4,dtype=np.float32)[None],count,axis=0)
    inverse_bytes=matrix_bytes(inverse)
    snapshot=skin_snapshot_toy()
    snapshot.update(mesh_identity=identity,raw_weight_data_base64=base64.b64encode(raw).decode(),raw_weight_data_byte_count=len(raw),
        skin_weight_data_sha256=raw_sha,expected_historical_skin_weight_sha256=raw_sha,
        inverse_reference_matrices_base64=base64.b64encode(inverse_bytes).decode(),inverse_reference_matrices_sha256=hashlib.sha256(inverse_bytes).hexdigest(),
        bones=[dict(name=name,parent_index=index-1,inverse_reference_row_major_f32=inverse[index].ravel().tolist()) for index,name in enumerate(LIMB_BONES)],
        active_bone_indices=list(range(count)),sections=[dict(base_vertex_index=0,vertex_count=3,section_index=0,disabled=False,bone_map=[0,1])],
        vertices=[dict(vertex_index=i,section_index=0,section_local_bone_indices=[0,1],global_bone_indices=[0,1],raw_weights_u16=[65535,0],normalized_weights=[1.,0.]) for i in range(3)])
    sample['skin_weight_data_sha256']=raw_sha
    sample['ref_to_local_matrices_sha256']=hashlib.sha256(inverse_bytes).hexdigest()
    attempt=tmp_path/'attempt';attempt.mkdir();plan['frame_count']=2
    (attempt/'plan.json').write_text(json.dumps(plan),encoding='utf-8')
    records=[]
    for frame in range(2):
        value=copy.deepcopy(sample);value.update(output_frame=frame,sequence_source_frame=frame)
        if mismatch and frame==1:value['ref_to_local_matrices_sha256']='a'*64
        value['world_points_m']=value.pop('world_m');records.append({'sample':value})
    native=attempt/'CineCam_01_native_pose_restore_receipt.json'
    native.write_text(json.dumps(dict(receipt,schema='futsalmot_pose17_native_receipt_v1',setup_error='',record_limit_reached=False,records=records)),encoding='utf-8')
    skin_path=tmp_path/'skin.json';skin_path.write_text(json.dumps({'actual_skin':snapshot}),encoding='utf-8')
    reference=tmp_path/'reference.json';reference.write_text(json.dumps(dict(identity=identity,positions_cm=positions.tolist(),triangles=triangles.tolist())),encoding='utf-8')
    output=tmp_path/'result'
    report=inspect([native],[skin_path],output,'actual_skin',[reference],surface_frames)
    assert not report['errors'],report['errors']
    assert report['matched_rows']==(1 if mismatch else 2)
    assert report['all_requested_native_matrix_sha_matched'] is (not mismatch)
    assert len(list(output.glob('*.npz')))==(0 if mismatch else 1 if surface_frames else 2)
    if mismatch:
        assert all(row['component_surface_file'] is None for row in report['rows'])
    assert report['visibility_state']=='unknown' and not report['render_surface_equivalence_verified']
    for path in output.glob('*.npz'):
        with np.load(path,allow_pickle=False) as surface:
            assert surface['positions_cm'].dtype==np.float32
            assert str(surface['unit'])=='cm' and str(surface['coordinate_space'])=='mesh_component'
            assert json.loads(str(surface['mesh_identity_json']))==identity
            if surface_frames:assert int(surface['output_frame'])==1
    with pytest.raises(ValueError,match='已存在'):inspect([native],[skin_path],output,'actual_skin',[reference])


def test_world_transform_uses_native_scale_quaternion_translation_and_cm_to_m():
    value=transform((100.,200.,300.));value['scale_xyz']=[2.,3.,4.]
    value['rotation_xyzw']=[0.,0.,2**-.5,2**-.5]
    point=component_points_to_world_m([[10.,0.,0.]],value)[0]
    assert point==pytest.approx([1.,2.2,3.],abs=1e-14)
