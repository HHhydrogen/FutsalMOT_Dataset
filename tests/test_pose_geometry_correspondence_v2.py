"""参考空间显式转换、拆点及三角拓扑拒绝策略测试。"""
import pytest
from pose_geometry_correspondence_v2 import transform_points, match_reference_vertices, map_anchor_to_target_triangle
from pose_geometry_correspondence_v2 import map_point_within_source_polygon

IDENTITY=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
SOURCE=[[0,0,0],[1,0,0],[0,1,0]]


def test_explicit_units_axes_and_translation():
    matrix=[[0,100,0,3],[-100,0,0,4],[0,0,100,5],[0,0,0,1]]
    assert transform_points(SOURCE,matrix)==[[3,4,5],[3,-96,5],[103,4,5]]


def test_split_vertices_reordered_triangle_and_weights():
    target=[[0,1,0],[0,0,0],[1,0,0],[0,0,0]]
    mapping=match_reference_vertices(SOURCE,target,IDENTITY,1e-6)
    assert mapping['source_to_target_candidates']==[[1,3],[2],[0]]
    anchor=map_anchor_to_target_triangle([0,1,2],[.2,.3,.5],mapping,[[0,2,3]])
    assert anchor=={'triangle_index':0,'triangle_vertices':[0,2,3],'barycentric':[.5,.3,.2]}


def test_missing_reference_and_ambiguous_surface_rejected():
    mapping=match_reference_vertices(SOURCE,SOURCE,IDENTITY,.001)
    for triangles in ([],[[0,1,2],[2,1,0]]):
        with pytest.raises(ValueError,match='唯一'):
            map_anchor_to_target_triangle([0,1,2],[.2,.3,.5],mapping,triangles)


def test_no_implicit_scale_or_alignment():
    mapping=match_reference_vertices(SOURCE,[[0,0,0],[100,0,0],[0,100,0]],IDENTITY,.001)
    assert mapping['source_coverage']==pytest.approx(1/3)
    with pytest.raises(ValueError):
        map_anchor_to_target_triangle([0,1,2],[1,0,0],mapping,[[0,1,2]])


@pytest.mark.parametrize('tolerance',[0,-1,float('nan'),float('inf')])
def test_invalid_tolerance(tolerance):
    with pytest.raises(ValueError):
        match_reference_vertices(SOURCE,SOURCE,IDENTITY,tolerance)


def test_not_nearest_triangle_fallback():
    mapping=match_reference_vertices(SOURCE,[[0,0,0],[1,0,0],[0,.99,0]],IDENTITY,.001)
    assert mapping['source_coverage']==pytest.approx(2/3)
    with pytest.raises(ValueError):
        map_anchor_to_target_triangle([0,1,2],[.2,.3,.5],mapping,[[0,1,2]])


def test_retriangulated_same_polygon_with_explicit_error_bound():
    points=[[0,0,0],[1,0,0],[1,1,0],[0,1,0]]
    mapping=match_reference_vertices(points,points,IDENTITY,1e-6)
    result=map_point_within_source_polygon([.2,.3,0],[0,1,2,3],mapping,points,[[0,1,3],[1,2,3]],1e-6)
    assert result['triangle_vertices']==[0,1,3]
    assert result['projection_error_target_units']==pytest.approx(0)
    with pytest.raises(ValueError):
        map_point_within_source_polygon([.2,.3,.01],[0,1,2,3],mapping,points,[[0,1,3],[1,2,3]],1e-6)
