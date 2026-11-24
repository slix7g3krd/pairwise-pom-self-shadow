"""六项修复要求的属性测试，与 `test_pom_self_shadow.py` 的契约测试并存。

覆盖：
1. 自遮挡检测：起始纹素不判遮挡；掠射角容差；
2. mip 与步进联动：同一足迹得同一 mip、迟滞只切一级、步数与 mip 同步；
3. 深度偏移：随纹素/斜率缩放、层间错开、深度与阴影深度一致；
4. 镜像方向校正：u/v 分量独立翻转；
5. 降级过渡：两端连续、无跳变；
6. 采样次数上限：掠射角不迭代爆炸。
"""

import math

import pytest

from graphics import pom_self_shadow as mod


# ---------------------------------------------------------------------------
# 1. 自遮挡检测
# ---------------------------------------------------------------------------

def test_origin_sample_is_never_occluded():
    # 起点纹素与射线同高，步进尚未离开起点。
    assert mod.self_occlusion(0.5, 1.0, step=0.0, origin=0.0) is False
    # 掠射角下严重量化噪声也不应把起点判成遮挡：第一次采样的步进
    # 距离恰好等于起点本身。
    assert mod.self_occlusion(0.5, 1.0, step=1e-6, origin=1e-6,
                              cos_theta=0.0) is False


def test_occlusion_needs_surface_above_ray():
    assert mod.self_occlusion(0.5, 0.9, step=1.0) is True
    assert mod.self_occlusion(0.9, 0.5, step=1.0) is False


def test_grazing_tolerance_absorbs_quantisation():
    # 高出量小于基础容差：不遮挡。掠射时容差只放大，判定只会更宽松。
    assert mod.self_occlusion(0.5, 0.5 + 0.5 * mod.OCCLUSION_EPS,
                              step=1.0, cos_theta=1.0) is False
    assert mod.self_occlusion(0.5, 0.5 + 0.5 * mod.OCCLUSION_EPS,
                              step=1.0, cos_theta=0.0) is False


def test_grazing_angle_monotonic():
    assert mod.grazing_angle(1.0) == pytest.approx(0.0)
    assert mod.grazing_angle(0.0) == pytest.approx(90.0)
    assert mod.grazing_angle(-0.5) == pytest.approx(60.0)


# ---------------------------------------------------------------------------
# 2. mip 与步进联动
# ---------------------------------------------------------------------------

def test_mip_is_stable_for_equal_footprints():
    # 同一平面（相同 ray_length）任何一次调用都得到同一个 mip，
    # 不依赖屏幕导数，因此远处不会逐像素抖动。
    a = mod.mip_step(0.05)
    b = mod.mip_step(0.05)
    assert a["mip"] == b["mip"]
    assert a["steps"] == b["steps"]


def test_mip_matches_step_footprint():
    for ray in (0.001, 0.01, 0.05, 0.2, 1.0):
        info = mod.mip_step(ray)
        # 一步不超过一个该级纹素，mip 与步进严格同频。
        assert info["step_uv"] <= info["texel_uv"] + 1e-12
        # 高度幅值随实际采样足迹衰减（texels_per_uv 默认 256）。
        assert info["height_scale"] == pytest.approx(
            (1.0 / 256.0) / info["texel_uv"], rel=1e-9
        )


def test_mip_increases_with_distance():
    near = mod.mip_step(0.01)["mip"]
    far = mod.mip_step(1.0)["mip"]
    assert far > near


def test_mip_hysteresis_limits_single_step_jump():
    # 足迹突然变大，mip 也只能升一级，不会整段跳变。
    info = mod.mip_step(4.0, previous_mip=0, hysteresis=1)
    assert info["mip"] == 1
    # 连续多帧才爬到目标层级。
    mip = 0
    for _ in range(16):
        mip = mod.mip_step(4.0, previous_mip=mip)["mip"]
    assert mip == mod.mip_step(4.0)["mip"]


def test_step_budget_shrinks_height_scale():
    # 预算把步长撑到一个纹素以上时，等效 mip 变粗、高度幅值同步衰减。
    info = mod.mip_step(1.0, base_steps=4096)
    assert info["steps"] == mod.MAX_STEPS
    assert info["height_scale"] < 1.0


# ---------------------------------------------------------------------------
# 3. 深度偏移
# ---------------------------------------------------------------------------

def test_depth_bias_scales_with_texel_and_slope():
    flat = mod.depth_bias(0.5, texel_uv=0.001, slope=0.0)
    steep = mod.depth_bias(0.5, texel_uv=0.001, slope=1.0)
    assert steep["bias"] > flat["bias"]


def test_depth_bias_is_clamped():
    info = mod.depth_bias(0.5, texel_uv=10.0, slope=10.0, max_bias=0.02)
    assert info["bias"] == pytest.approx(0.02)
    assert 0.0 <= info["depth"] <= 1.0


def test_layers_do_not_share_depth():
    first = mod.depth_bias(0.5, texel_uv=0.001, layer=0)
    second = mod.depth_bias(0.5, texel_uv=0.001, layer=1)
    assert second["separation"] > first["separation"]
    assert second["depth"] != first["depth"]


def test_shadow_uses_same_bias_as_depth():
    for layer in range(3):
        info = mod.depth_bias(0.5, texel_uv=0.01, slope=0.3, layer=layer)
        assert info["shadow_depth"] == info["depth"]


# ---------------------------------------------------------------------------
# 4. 镜像方向校正
# ---------------------------------------------------------------------------

def test_mirror_flips_matching_component_only():
    assert mod.mirror_fix((0.5, 0.25), mirror_u=True) == (-0.5, 0.25)
    assert mod.mirror_fix((0.5, 0.25), mirror_v=True) == (0.5, -0.25)
    assert mod.mirror_fix((0.5, 0.25), mirror_u=True, mirror_v=True) == (-0.5, -0.25)


def test_mirror_accepts_negative_scale():
    assert mod.mirror_fix(0.5, mirror_u=-1.0) == (-0.5, 0.0)
    assert mod.mirror_fix(0.5, mirror_u=2.0) == (0.5, 0.0)


def test_no_mirror_leaves_step_untouched():
    assert mod.mirror_fix((0.5, -0.25)) == (0.5, -0.25)


# ---------------------------------------------------------------------------
# 5. 降级过渡
# ---------------------------------------------------------------------------

def test_degrade_endpoints():
    near = mod.degrade_blend(0.0, height_offset=1.0)
    far = mod.degrade_blend(1.0, height_offset=1.0)
    assert near["weight"] == 0.0 and near["height_offset"] == pytest.approx(1.0)
    assert far["weight"] == 1.0 and far["height_offset"] == pytest.approx(0.0)
    assert far["normal_mix"] == pytest.approx(1.0)


def test_degrade_is_continuous():
    # 过渡区间内高度偏移单调、无跳变：相邻采样差远小于总幅值。
    prev = mod.degrade_blend(0.0, height_offset=1.0)["height_offset"]
    for i in range(1, 101):
        cur = mod.degrade_blend(i / 100.0, height_offset=1.0)["height_offset"]
        assert cur <= prev + 1e-12
        assert prev - cur < 0.1
        prev = cur
    assert prev == pytest.approx(0.0, abs=1e-12)


def test_degrade_smoothstep_flat_at_edges():
    # smoothstep 两端导数为零：刚进入/离开过渡区时变化远小于线性过渡
    # 在同一步长内的变化（区间宽 0.3，线性时每 1e-3 约 3.3e-3）。
    eps = 1e-3
    low = mod.degrade_blend(mod.DEGRADE_LOW, height_offset=1.0)["height_offset"]
    low_next = mod.degrade_blend(mod.DEGRADE_LOW + eps,
                                 height_offset=1.0)["height_offset"]
    high = mod.degrade_blend(mod.DEGRADE_HIGH, height_offset=1.0)["height_offset"]
    high_prev = mod.degrade_blend(mod.DEGRADE_HIGH - eps,
                                  height_offset=1.0)["height_offset"]
    assert abs(low - low_next) < 1e-4
    assert abs(high - high_prev) < 1e-4


# ---------------------------------------------------------------------------
# 6. 采样次数上限
# ---------------------------------------------------------------------------

def test_step_cap_respects_budget():
    assert mod.step_cap(10_000) == mod.MAX_STEPS
    assert mod.step_cap(10_000, budget=8) == 8
    assert mod.step_cap(0) == 1
    assert mod.step_cap(-5) == 1


def test_grazing_does_not_explode_step_count():
    # 极端掠射（路径极长）也要停在预算内。
    info = mod.mip_step(1e6, base_steps=1)
    assert info["steps"] <= mod.MAX_STEPS
    assert math.isfinite(info["step_uv"])


def test_step_count_is_bounded_everywhere():
    for ray in (0.0, 1e-9, 0.001, 0.1, 1.0, 100.0, 1e9):
        info = mod.mip_step(ray)
        assert 1 <= info["steps"] <= mod.MAX_STEPS
        assert info["mip"] <= mod.MAX_MIP


# ---------------------------------------------------------------------------
# 运维辅助
# ---------------------------------------------------------------------------

def test_validation_flags_bad_config():
    bad = mod.validation({"base_steps": 0, "degrade_low": 0.9,
                          "degrade_high": 0.6, "texels_per_uv": 0})
    assert bad["ok"] is False
    assert len(bad["errors"]) >= 3
    assert mod.validation({"base_steps": 32, "max_steps": 64})["ok"] is True


def test_height_range_clamps_and_swaps():
    assert mod.height_range(1.5) == pytest.approx(1.0)
    assert mod.height_range(-0.5) == pytest.approx(0.0)
    assert mod.height_range(0.3, low=0.4, high=0.2) == pytest.approx(0.3)


def test_uv_wrap_is_continuous_across_seam():
    assert mod.uv_wrap(1.25) == pytest.approx(0.25)
    assert mod.uv_wrap(-0.25) == pytest.approx(0.75)
    assert mod.uv_wrap((1.0, -1.0)) == (0.0, 0.0)


def test_precision_quantises_deterministically():
    assert mod.precision(3.0) == pytest.approx(1.0)
    assert mod.precision(-1.0) == pytest.approx(0.0)
    assert mod.precision(0.123456789, 8) == pytest.approx(0.12156862745098039)


def test_metrics_and_audit_accept_single_value():
    m = mod.metrics(1)
    assert m["samples"] == 1
    assert mod.audit(1)["occluded"] is False


def test_fallback_disables_parallax():
    info = mod.fallback("missing heightmap")
    assert info["parallax_scale"] == 0.0
    assert info["normal_mix"] == pytest.approx(1.0)
    assert info["reason"] == "missing heightmap"


def test_determinism_is_stable_and_order_sensitive():
    assert mod.determinism({"b": 1, "a": 2}) == mod.determinism({"a": 2, "b": 1})
    assert mod.determinism(1) != mod.determinism(2)
