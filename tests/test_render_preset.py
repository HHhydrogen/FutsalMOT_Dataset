"""render_preset.py 的纯 Python 测试：cv_gt deterministic render preset 契约。

不接触 unreal——只验证配置解析与派生出的属性覆盖 dict（enum 以 (类名, 成员名) 标记）。
"""

import pytest

from render_preset import (
    AA_NONE,
    AA_TAA,
    AA_TSR,
    DEFAULT_CV_GT,
    PRESET_CINEMATIC,
    PRESET_CV_GT,
    camera_post_process_overrides,
    mrq_aa_overrides,
    mrq_temporal_overrides,
    normalize_cv_gt,
    post_process_console_vars,
    resolve_preset,
)


class TestResolvePreset:
    def test_default_is_cv_gt(self):
        preset, cv_gt = resolve_preset({})
        assert preset == PRESET_CV_GT
        assert cv_gt == {}

    def test_explicit_cv_gt(self):
        preset, cv_gt = resolve_preset({"preset": "cv_gt", "cv_gt": {"motion_blur": False}})
        assert preset == PRESET_CV_GT
        assert cv_gt == {"motion_blur": False}

    def test_null_preset_no_override(self):
        preset, cv_gt = resolve_preset({"preset": None})
        assert preset is None

    def test_cinematic_reserved(self):
        preset, _ = resolve_preset({"preset": "cinematic"})
        assert preset == PRESET_CINEMATIC

    def test_invalid_preset_raises(self):
        with pytest.raises(ValueError):
            resolve_preset({"preset": "bogus"})


class TestNormalizeCvGt:
    def test_defaults(self):
        out = normalize_cv_gt({})
        assert out == DEFAULT_CV_GT
        assert out["motion_blur"] is False
        assert out["depth_of_field"] is False
        assert out["chromatic_aberration"] is False
        assert out["lens_distortion"] is False
        assert out["anti_aliasing"] == AA_TAA
        assert out["temporal_sampling"] is False

    def test_partial_override(self):
        out = normalize_cv_gt({"anti_aliasing": "tsr"})
        assert out["anti_aliasing"] == AA_TSR
        assert out["motion_blur"] is False  # 其余保持默认

    def test_invalid_aa_raises(self):
        with pytest.raises(ValueError):
            normalize_cv_gt({"anti_aliasing": "bogus"})

    def test_non_bool_flag_raises(self):
        with pytest.raises(ValueError):
            normalize_cv_gt({"motion_blur": "yes"})


class TestCameraPostProcessOverrides:
    def test_cv_gt_disables_spatial_effects(self):
        o = camera_post_process_overrides(PRESET_CV_GT, {})
        assert o["post_process_blend_weight"] == 1.0
        assert o["motion_blur_amount"] == 0.0
        assert o["scene_fringe_intensity"] == 0.0  # UE 色差属性名
        assert o["depth_of_field_method"] == ("EDepthOfFieldMethod", "DOFM_None")
        assert o["depth_of_field_fstop"] == 32.0

    def test_motion_blur_enabled_keeps_amount(self):
        o = camera_post_process_overrides(PRESET_CV_GT, {"motion_blur": True})
        assert "motion_blur_amount" not in o
        assert o["scene_fringe_intensity"] == 0.0

    def test_dof_enabled_skips_method(self):
        o = camera_post_process_overrides(PRESET_CV_GT, {"depth_of_field": True})
        assert "depth_of_field_method" not in o
        assert "depth_of_field_fstop" not in o

    def test_null_preset_no_override(self):
        assert camera_post_process_overrides(None, {}) == {}

    def test_cinematic_no_override(self):
        assert camera_post_process_overrides(PRESET_CINEMATIC, {}) == {}


class TestMrqTemporalOverrides:
    def test_cv_gt_single_instant(self):
        o = mrq_temporal_overrides(PRESET_CV_GT, {})
        assert o["temporal_accumulation_method"] == ("MoviePipelineTemporalAccumulationMethod", "NONE")
        assert o["temporal_sample_count"] == 1

    def test_temporal_sampling_true_allows_accumulation(self):
        # 显式开启时间采样 → 未来模式，不强制 NONE（此时才允许 temporal integration）
        assert mrq_temporal_overrides(PRESET_CV_GT, {"temporal_sampling": True}) == {}

    def test_null_no_override(self):
        assert mrq_temporal_overrides(None, {}) == {}


class TestMrqAaOverrides:
    def test_default_taa(self):
        o = mrq_aa_overrides(PRESET_CV_GT, {})
        assert o["anti_aliasing_method"] == ("MoviePipelineAntiAliasingMethod", "TAA")
        assert o["render_warm_up_count"] == 0

    def test_aa_none(self):
        o = mrq_aa_overrides(PRESET_CV_GT, {"anti_aliasing": AA_NONE})
        assert o["anti_aliasing_method"] == ("MoviePipelineAntiAliasingMethod", "None")

    def test_null_no_override(self):
        assert mrq_aa_overrides(None, {}) == {}

    def test_cinematic_no_override(self):
        assert mrq_aa_overrides(PRESET_CINEMATIC, {}) == {}


from render_episode import _resolve_enum_member  # noqa: E402


class _FakeEnum:
    """模拟 UE 枚举（成员命名可能为 TAA / TA_A、NONE / None 等变体）。"""
    TAA = 2
    NONE = 0
    AMORTIZED = 1
    None_ = 0
    BokehDOF = 4


class TestResolveEnumMember:
    def test_exact(self):
        assert _resolve_enum_member(_FakeEnum, "TAA") == 2

    def test_case_insensitive(self):
        assert _resolve_enum_member(_FakeEnum, "none") == 0
        assert _resolve_enum_member(_FakeEnum, "amortized") == 1

    def test_normalized(self):
        # TA_A → 归一化 taa → 命中 TAA
        assert _resolve_enum_member(_FakeEnum, "TA_A") == 2
        assert _resolve_enum_member(_FakeEnum, "None") == 0

    def test_no_match(self):
        assert _resolve_enum_member(_FakeEnum, "DOFM_None") is None  # 无对应成员
        assert _resolve_enum_member(_FakeEnum, "BOGUS") is None


class TestPostProcessConsoleVars:
    def test_cv_gt_disables_effects(self):
        cv = post_process_console_vars(PRESET_CV_GT, {})
        assert cv["r.MotionBlur.Amount"] == 0.0
        assert cv["r.DepthOfFieldQuality"] == 0.0
        assert cv["r.SceneColorFringeQuality"] == 0.0

    def test_motion_blur_enabled_keeps_cvar(self):
        cv = post_process_console_vars(PRESET_CV_GT, {"motion_blur": True})
        assert "r.MotionBlur.Amount" not in cv
        assert "r.DepthOfFieldQuality" in cv

    def test_null_no_override(self):
        assert post_process_console_vars(None, {}) == {}

    def test_cinematic_no_override(self):
        assert post_process_console_vars(PRESET_CINEMATIC, {}) == {}
