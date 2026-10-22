"""真实 UE Quaternion/Euler 回读舍入不应误判事务恢复失败。"""
import copy
import pytest
import episode_appearance as module


def state():
    payload={field:None for field in module._SNAPSHOT_FIELDS}
    for field in ('actor_transform','relative_transform'):
        payload[field]={'location':[202.20588147640228,-228.66026312112808,90.0],
                        'rotation':[0.0,-102.43160247802749,0.0],'scale':[1.0,1.0,1.0]}
    return payload


def test_actual_ue_rotation_roundoff_is_accepted():
    expected=state();actual=copy.deepcopy(expected)
    actual['actor_transform']['rotation'][1]=-102.43160247802743
    assert module._snapshot_matches_state(expected,actual)


@pytest.mark.parametrize('field', ['actor_transform','relative_transform'])
@pytest.mark.parametrize('component', ['location','rotation','scale'])
def test_material_transform_drift_is_rejected(field,component):
    expected=state();actual=copy.deepcopy(expected)
    actual[field][component][0]+=0.001
    assert not module._snapshot_matches_state(expected,actual)


def test_equivalent_euler_wrap_is_accepted():
    expected=state();actual=copy.deepcopy(expected)
    expected['actor_transform']['rotation'][1]=180.0
    actual['actor_transform']['rotation'][1]=-180.0
    assert module._snapshot_matches_state(expected,actual)


def test_nontransform_identity_and_material_slots_remain_exact():
    expected=state();actual=copy.deepcopy(expected)
    expected['override_materials']=['/Game/Original']
    actual['override_materials']=['/Game/Different']
    assert not module._snapshot_matches_state(expected,actual)


@pytest.mark.parametrize('field', ['actor_transform', 'relative_transform'])
def test_actual_ue_float_position_readback_is_same_quantized_coordinate(field):
    expected = state(); actual = copy.deepcopy(expected)
    expected[field]['location'][0] = -202.2058868408203
    actual[field]['location'][0] = -202.20588147640228
    assert module._snapshot_matches_state(expected, actual)


def test_adjacent_float_position_is_rejected():
    expected = state(); actual = copy.deepcopy(expected)
    expected['actor_transform']['location'][0] = 202.2058868408203
    actual['actor_transform']['location'][0] = 202.20590209960938
    assert not module._snapshot_matches_state(expected, actual)


def test_large_coordinate_cannot_expand_position_tolerance():
    expected = state(); actual = copy.deepcopy(expected)
    expected['actor_transform']['location'][0] = 100000000.0
    actual['actor_transform']['location'][0] = 100000000.01
    assert not module._snapshot_matches_state(expected, actual)


def test_finite_coordinate_outside_float32_does_not_raise():
    expected = state(); actual = copy.deepcopy(expected)
    expected['actor_transform']['location'][0] = 3.5e38
    actual['actor_transform']['location'][0] = 3.5e38 + 1e30
    assert not module._snapshot_matches_state(expected, actual)
