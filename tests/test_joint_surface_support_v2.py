"""九像素真实射线与局部支持候选；toy不等于真实COCO可见性。"""

import copy
import json

import numpy as np
import pytest

from grf_ue_bridge.joint_surface_support_v2 import JOINT_CHAINS, build_regions, intersect_region, observe_joint_support, pixel_ray, prepare_region


TARGET,OTHER=0x3f800001,0x3f800002


def fixture():
    camera={'intrinsics':dict(width=7,height=7,fx=100.,fy=100.,cx=3.,cy=3.),
            'extrinsics':dict(world_location_m=[0,0,0],forward=[0,0,1],right=[1,0,0],up=[0,1,0])}
    vertices=np.array([[-2,-2,9.5],[2,-2,9.5],[0,2,9.5]],dtype=np.float64)
    triangles=np.array([[0,1,2]],dtype=np.int32)
    region=dict(region_sha256='a'*64,status='candidate_unreviewed',triangle_indices=[0])
    depth=np.full((7,7),9.5,dtype=np.float32)
    layers=[np.zeros((7,7,4),dtype=np.float32) for _ in range(3)]
    layers[0][:,:,0]=np.asarray(TARGET,dtype=np.uint32).view(np.float32);layers[0][:,:,1]=1
    return camera,vertices,triangles,region,depth,layers


def observe(data,uv=(3.,3.),actors=None):
    return observe_joint_support(uv,*data,TARGET,actors if actors is not None else {TARGET:'Player_L0',OTHER:'Player_R0'})


def test_vectorized_region_matches_existing_weight_and_radius_definition():
    from joint_visibility_support_v2 import prepare_support_region
    vertices=[[-1,-1,9.5],[1,-1,9.5],[0,1,9.5],[-1,-1,9],[1,-1,9],[0,1,9]]
    triangles=[[0,1,2],[3,4,5]];weights=[1,1,1,0,0,0]
    expected=prepare_support_region(vertices,triangles,weights,[0,0,10],2,.5)
    assert prepare_region(vertices,triangles,weights,[0,0,10],2,.5).tolist()==expected['triangle_indices']==[0]


def test_reference_regions_bind_side_chain_actual_identity_and_candidate_parameters():
    names=sorted({name for _,chain in JOINT_CHAINS.values() for name in chain})
    index=names.index('hand_l')
    skin=dict(bone_names=names,inverse_matrices=np.repeat(np.eye(4,dtype=np.float32)[None],len(names),axis=0),
              global_bone_indices=np.full((3,1),index,dtype=np.int32),raw_weights_u16=np.full((3,1),65535,dtype=np.int32),
              identity={'asset_path':'/Game/Toy.Mesh'},weights_sha256='b'*64)
    vertices=np.array([[-1,-1,0],[1,-1,0],[0,1,0]],dtype=np.float32)
    triangles=np.array([[0,1,2]],dtype=np.int32)
    regions=build_regions(skin,vertices,triangles,.08,.25)
    assert regions['left_wrist']['triangle_indices']==[0]
    assert regions['right_wrist']['triangle_indices']==[]
    assert regions['left_wrist']['status']=='candidate_unreviewed'
    assert regions['left_wrist']['mesh_identity']==skin['identity']
    other=build_regions(skin,vertices,triangles,.09,.25)
    assert regions['left_wrist']['region_sha256']!=other['left_wrist']['region_sha256']


def test_joint_candidate_uses_local_surface_and_never_promotes_visibility():
    data=fixture();result=observe(data)
    assert result['candidate_state']=='visible' and result['visibility_state']=='unknown'
    assert result['support_ray_hit_count']==9
    assert all(abs(s['support_forward_depth_m']-9.5)<1e-12 for s in result['pixel_samples'])
    assert result['visibility_method']=='not_measured'
    assert len({tuple(s['pixel_xy']) for s in result['pixel_samples']})==9


@pytest.mark.parametrize('external',[False,True])
def test_nearer_own_and_other_surfaces_are_distinct_candidates(external):
    data=fixture();data[4][:]=9
    if external:data[5][0][:,:,0]=np.asarray(OTHER,dtype=np.uint32).view(np.float32)
    result=observe(data)
    assert result['candidate_state']=='occluded' and result['occlusion_kind']==('external' if external else 'self')
    assert result['visibility_state']=='unknown'


def test_pixel_center_rays_have_independent_analytic_sloped_surface_depths():
    data=fixture();camera,vertices,_,_,depth,_=data
    vertices[:,2]=9.5+.03*vertices[:,0]
    for y in range(7):
        for x in range(7):depth[y,x]=9.5/(1-.03*(x-3)/100.)
    result=observe(data,uv=(3.4,3.3))
    assert result['candidate_state']=='visible'
    values=[s['support_forward_depth_m'] for s in result['pixel_samples']]
    assert max(values)-min(values)>0.005
    assert max(abs(s['depth_delta_m']) for s in result['pixel_samples'])<5e-7
    origin,direction=pixel_ray(camera,3,3)
    assert origin.tolist()==[0,0,0] and direction.tolist()==[0,0,1]


def test_ray_only_intersects_own_region_not_all_body_triangles():
    _,vertices,triangles,_,_,_=fixture()
    vertices=np.concatenate([vertices,vertices-np.array([0,0,.5])])
    triangles=np.concatenate([triangles,[[3,4,5]]])
    assert intersect_region([0,0,0],[0,0,1],vertices,triangles,[0])['ray_distance_m']==9.5
    assert intersect_region([0,0,0],[0,0,1],vertices,triangles,[0,1])['ray_distance_m']==9


@pytest.mark.parametrize('case',['empty','partial_rays','coverage','ids','invalid_depth','spread','unmapped','in_front','image_edge','nonfinite_weight'])
def test_unknown_reasons_remain_unknown_without_string_upgrade(case):
    data=list(fixture());uv=(3.,3.);actors=None
    if case=='empty':data[3]['triangle_indices']=[]
    if case=='partial_rays':data[1]*=np.array([.01,.01,1])
    if case=='coverage':data[5][0][3,3,1]=.5
    if case=='ids':data[5][0][3,3,0]=np.asarray(OTHER,dtype=np.uint32).view(np.float32)
    if case=='invalid_depth':data[4][3,3]=np.nan
    if case=='spread':data[4][3,3]=9.48
    if case=='unmapped':
        data[4][:]=9;data[5][0][:,:,0]=np.asarray(OTHER,dtype=np.uint32).view(np.float32);actors={TARGET:'Player_L0'}
    if case=='in_front':data[4][:]=10
    if case=='image_edge':uv=(0.,0.)
    if case=='nonfinite_weight':data[5][0][3,3,1]=np.nan
    result=observe(data,uv,actors)
    assert result['candidate_state']=='unknown' and result['visibility_state']=='unknown'
    assert result['reason']!='not_evaluated'
    json.dumps(result,allow_nan=False)
