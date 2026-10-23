"""参考几何的严格位置/三角对应；支持UE导入拆点、重排，不猜坐标轴或偏移。"""
import math
from itertools import product


def transform_points(points, matrix):
    """显式4x4仿射矩阵；米→厘米和轴转换须由外部证据确定。"""
    if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
        raise ValueError("需要4x4矩阵")
    if any(not math.isfinite(v) for row in matrix for v in row):
        raise ValueError("矩阵必须有限")
    if list(matrix[3]) != [0, 0, 0, 1]:
        raise ValueError("仅支持仿射参考空间转换")
    output = []
    for point in points:
        if len(point) != 3 or any(not math.isfinite(v) for v in point):
            raise ValueError("参考顶点必须为有限三维坐标")
        output.append([sum(matrix[j][i]*point[i] for i in range(3))+matrix[j][3] for j in range(3)])
    return output


def match_reference_vertices(source_points, target_points, source_to_target, tolerance):
    """返回source→所有几何一致target候选；容差使用target单位，不自动放大。"""
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("容差必须为有限正数")
    source = transform_points(source_points, source_to_target)
    target = transform_points(target_points, [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    grid = {}
    def cell(point):
        return tuple(math.floor(v/tolerance) for v in point)
    for index, point in enumerate(target):
        grid.setdefault(cell(point), []).append(index)
    matches = []
    errors = []
    for point in source:
        base = cell(point)
        found = []
        for delta in product((-1,0,1), repeat=3):
            for index in grid.get(tuple(base[i]+delta[i] for i in range(3)), []):
                error = math.dist(point, target[index])
                if error <= tolerance:
                    found.append((index,error))
        found.sort()
        matches.append([index for index,_ in found])
        if found:
            errors.append(min(error for _,error in found))
    return {"source_to_target_candidates": matches,
            "source_vertex_count": len(source), "target_vertex_count": len(target),
            "source_coverage": sum(bool(m) for m in matches)/len(matches) if matches else 0,
            "max_best_match_error": max(errors) if errors else None,
            "tolerance_target_units": tolerance,
            "source_to_target_matrix": source_to_target}


def map_anchor_to_target_triangle(source_triangle, barycentric, correspondence, target_triangles):
    """在真实目标三角拓扑上寻找唯一面，重排权重；无对应/歧义均拒绝。"""
    if len(source_triangle) != 3 or len(set(source_triangle)) != 3:
        raise ValueError("源锚点三角索引错误")
    if len(barycentric) != 3 or any(not math.isfinite(v) or v < 0 or v > 1 for v in barycentric) or abs(sum(barycentric)-1)>1e-8:
        raise ValueError("源锚点重心权重错误")
    matches = correspondence['source_to_target_candidates']
    if any(type(i) is not int or i < 0 or i >= len(matches) for i in source_triangle):
        raise ValueError("源锚点索引越界")
    candidates = [set(matches[i]) for i in source_triangle]
    found = []
    for triangle_index, triangle in enumerate(target_triangles):
        if len(triangle) != 3 or len(set(triangle)) != 3:
            raise ValueError("目标三角拓扑错误")
        slots = []
        for target_index in triangle:
            possible = [i for i in range(3) if target_index in candidates[i]]
            if len(possible) != 1:
                break
            slots.append(possible[0])
        if len(slots) == 3 and len(set(slots)) == 3:
            found.append({"triangle_index": triangle_index, "triangle_vertices": list(triangle),
                          "barycentric": [barycentric[i] for i in slots]})
    if len(found) != 1:
        raise ValueError("锚点未找到唯一目标三角，候选面数="+str(len(found)))
    return found[0]


def map_point_within_source_polygon(point, source_polygon, correspondence, target_points, target_triangles, tolerance):
    """只在同一个原始面的目标重三角化片内映射，超固定误差或歧义则拒绝。"""
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("容差必须为有限正数")
    matches=correspondence['source_to_target_candidates']
    allowed=set(i for source in source_polygon for i in matches[source])
    found=[]
    for index,triangle in enumerate(target_triangles):
        if not set(triangle).issubset(allowed):continue
        a,b,c=[target_points[i] for i in triangle]
        u=[b[i]-a[i] for i in range(3)];v=[c[i]-a[i] for i in range(3)];p=[point[i]-a[i] for i in range(3)]
        dot=lambda x,y:sum(x[i]*y[i] for i in range(3))
        uu,uv,vv,pu,pv=dot(u,u),dot(u,v),dot(v,v),dot(p,u),dot(p,v)
        den=uu*vv-uv*uv
        if den<=0:continue
        beta,gamma=(vv*pu-uv*pv)/den,(uu*pv-uv*pu)/den
        weights=[1-beta-gamma,beta,gamma]
        if min(weights)<-1e-8 or max(weights)>1+1e-8:continue
        projected=[sum(weights[j]*[a,b,c][j][i] for j in range(3)) for i in range(3)]
        error=math.dist(point,projected)
        if error<=tolerance:
            found.append({'triangle_index':index,'triangle_vertices':list(triangle),
                          'barycentric':weights,'projection_error_target_units':error})
    if len(found)!=1:
        raise ValueError('同原始面重三角化映射不唯一/超容差，候选面数='+str(len(found)))
    return found[0]
