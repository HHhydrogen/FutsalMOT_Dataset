"""exporter._build_entities 的 GRF role 数据链测试。

验证：真实 left_team_roles / right_team_roles 进入 meta.entities；
GK 的 role / is_goalkeeper 正确；roles 缺失/异常时回退默认阵容；
entity ID 恒为 L0-L4 / R0-R4 / BALL。
"""

from grf_ue_bridge.exporter import _DEFAULT_ROLES, _build_entities, _observed_roles


def _obs(left_roles=None, right_roles=None):
    o = {}
    if left_roles is not None:
        o["left_team_roles"] = left_roles
    if right_roles is not None:
        o["right_team_roles"] = right_roles
    return o


class TestObservedRoles:
    def test_real_roles_passthrough(self):
        roles = _observed_roles(_obs([0, 7, 9, 2, 1], [0, 7, 9, 2, 1]), "left_team_roles")
        assert roles == [0, 7, 9, 2, 1]

    def test_fallback_when_missing(self):
        assert _observed_roles({}, "left_team_roles") == _DEFAULT_ROLES
        assert _observed_roles(_obs(), "right_team_roles") == _DEFAULT_ROLES

    def test_fallback_when_malformed(self):
        # 非 list、长度不足、非法 role id、非 int → 回退
        assert _observed_roles(_obs(left_roles="x"), "left_team_roles") == _DEFAULT_ROLES
        assert _observed_roles(_obs(left_roles=[0, 7]), "left_team_roles") == _DEFAULT_ROLES
        assert _observed_roles(_obs(left_roles=[0, 99, 9, 2, 1]), "left_team_roles") == _DEFAULT_ROLES
        assert _observed_roles(_obs(left_roles=[0, "7", 9, 2, 1]), "left_team_roles") == _DEFAULT_ROLES

    def test_truncates_to_players(self):
        roles = _observed_roles(_obs(left_roles=[0, 7, 9, 2, 1, 3]), "left_team_roles")
        assert roles == [0, 7, 9, 2, 1]


class TestBuildEntities:
    def _ids_and_roles(self, entities):
        return {
            e.id: (e.role, e.is_goalkeeper)
            for e in entities if e.id != "BALL"
        }

    def test_real_roles_into_entities(self):
        entities = _build_entities(_obs([0, 7, 9, 2, 1], [0, 7, 9, 2, 1]))
        m = self._ids_and_roles(entities)
        assert m["L0"] == ("goalkeeper", True)
        assert m["L1"] == ("right_midfielder", False)
        assert m["L2"] == ("center_forward", False)
        assert m["L3"] == ("left_back", False)
        assert m["L4"] == ("center_back", False)
        assert m["R0"] == ("goalkeeper", True)
        assert m["R1"] == ("right_midfielder", False)
        # entity ID 顺序与数量不变
        assert list(e.id for e in entities) == (
            ["L0", "L1", "L2", "L3", "L4", "R0", "R1", "R2", "R3", "R4", "BALL"]
        )

    def test_roles_that_differ_from_default(self):
        # 若真实阵容不同（如 GK 不在 index 0），也如实进入 meta（不改 ID 语义）
        entities = _build_entities(_obs([7, 0, 9, 2, 1], [9, 7, 0, 2, 1]))
        m = self._ids_and_roles(entities)
        assert m["L0"] == ("right_midfielder", False)
        assert m["L1"] == ("goalkeeper", True)   # 真实 GK 在 index 1，如实记录
        assert m["R0"] == ("center_forward", False)
        assert m["R2"] == ("goalkeeper", True)

    def test_fallback_when_roles_missing(self):
        entities = _build_entities({})
        m = self._ids_and_roles(entities)
        assert m["L0"] == ("goalkeeper", True)
        assert m["L1"] == ("right_midfielder", False)
        assert len(entities) == 11

    def test_fallback_when_roles_malformed(self):
        entities = _build_entities(_obs(left_roles=[0, 99], right_roles=None))
        m = self._ids_and_roles(entities)
        assert m["L0"] == ("goalkeeper", True)
        assert m["R4"] == ("center_back", False)

    def test_empty_observation_snapshot(self):
        # result.snapshots 为空时传 {}，走 fallback
        entities = _build_entities({})
        assert [e.id for e in entities] == \
            ["L0", "L1", "L2", "L3", "L4", "R0", "R1", "R2", "R3", "R4", "BALL"]