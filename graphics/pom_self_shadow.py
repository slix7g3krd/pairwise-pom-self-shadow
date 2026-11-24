"""视差贴图（POM）自遮挡、mip 联动与镜像校正修复。

线上问题
--------
* **自遮挡误报**：视线与表面夹角极小时，步进第一次采样拿到的就是起始
  纹素本身，射线用自己的高度和自己比较，任何量化噪声都会被判成“被
  遮挡”，砖缝边缘出现一圈黑边。
* **远处闪烁**：mip 层级按屏幕空间导数选取，而导数本身在远处逐像素
  抖动，mip 在相邻层级之间来回跳；步进数又各算各的，mip 升上去了而
  步长还停在细节级，高度图被抹平、法线仍在，出现假凹凸。
* **层与阴影冲突**：多层材质每层都往同一深度写，层间以及阴影图与
  G-buffer 之间深度值相同，出现 z-fighting 与阴影漏光；固定常数的
  深度偏移在斜面上完全不够。
* **镜像反向**：UV 做了镜像（``u -> 1-u`` 或负缩放）后，切线空间步进
  方向没有跟着翻转，凹陷被采成凸起。
* **降级跳变**：远处降级到普通法线贴图时，高度偏移从满值直接归零，
  高度发生跳变。
* **迭代爆炸**：掠射角下自适应步数没有上限，循环次数失控，出现
  长条状拉伸。

修复对应
--------
:func:`self_occlusion`
    自遮挡检测。排除起始采样（``step <= origin``）；掠射时按视线夹角
    放大容差，只有表面确实高出射线足够多才算遮挡。

:func:`mip_step`
    mip 与步进联动。mip 由“一步前进的 UV 足迹”决定，而不是屏幕导数，
    因此同一平面上的像素得到同一个 mip；步数由该 mip 的纹素尺寸推导，
    与 mip 严格同步。``previous_mip`` 提供迟滞，每帧最多切一级。

:func:`depth_bias`
    深度偏移。按纹素尺寸与斜率累加偏移并裁剪到上限，层间按层号错开；
    阴影采样复用同一个偏移，深度测试与阴影测试一致。

:func:`mirror_fix`
    镜像方向校正。UV 镜像时翻转对应的步进分量。

:func:`degrade_blend`
    降级过渡。用 smoothstep 权重把高度偏移连续归零，再交叉淡入普通法线，
    过渡区间两端导数为零，不会跳变。

:func:`step_cap`
    采样次数上限。自适应步数裁剪到硬件预算，掠射角也不会迭代爆炸。

运维辅助：:func:`height_range`（高度范围裁剪）、:func:`validation`
（参数校验）、:func:`audit`（单次采样审计）、:func:`fallback`（降级
回退）、:func:`grazing_angle`（掠射角）、:func:`metrics`（指标汇总）、
:func:`uv_wrap`（UV 平铺包裹）、:func:`precision`（定点量化）、
:func:`determinism`（确定性摘要）。

所有公开函数都接受“单值”入参并返回非 ``None`` 结果；数值一律用
``float`` 参与计算，不依赖哈希表或对象地址顺序，便于逐步接入现有管线。
"""

from __future__ import annotations

import hashlib
import json
import math

__all__ = [
    # 常量
    "MIN_STEPS",
    "MAX_STEPS",
    "MAX_MIP",
    "OCCLUSION_EPS",
    "GRAZING_EPS",
    "GRAZING_TOL_GAIN",
    "MIP_HYSTERESIS",
    "DEPTH_TEXEL_SCALE",
    "DEPTH_SLOPE_SCALE",
    "LAYER_SEPARATION",
    "MAX_DEPTH_BIAS",
    "DEGRADE_LOW",
    "DEGRADE_HIGH",
    # 六项修复
    "self_occlusion",
    "mip_step",
    "depth_bias",
    "mirror_fix",
    "degrade_blend",
    "step_cap",
    # 运维辅助
    "height_range",
    "validation",
    "audit",
    "fallback",
    "grazing_angle",
    "metrics",
    "uv_wrap",
    "precision",
    "determinism",
]

# 步数预算：再掠射的视角也不超过 MAX_STEPS 次采样；MIN_STEPS 保证
# 近处至少能分辨出一次高度穿越。
MIN_STEPS = 4
MAX_STEPS = 64

# 高度图最多降到第 8 级 mip。
MAX_MIP = 8

# 判定“被遮挡”所需的最小高度差（归一化高度单位）。
OCCLUSION_EPS = 1e-4

# |cos(入射角)| 小于该值视为掠射。
GRAZING_EPS = 0.05
# 掠射时容差相对 OCCLUSION_EPS 的放大系数。
GRAZING_TOL_GAIN = 2.0

# mip 迟滞：每帧最多切换的级数，避免远处在相邻层级之间抖动。
MIP_HYSTERESIS = 1

# 深度偏移：纹素项与斜率项的系数，以及总偏移上限。
DEPTH_TEXEL_SCALE = 1.5
DEPTH_SLOPE_SCALE = 0.5
MAX_DEPTH_BIAS = 0.05
# 多层材质相邻层之间的深度间隔。
LAYER_SEPARATION = 1e-3

# 降级过渡：factor <= DEGRADE_LOW 用完整 POM，>= DEGRADE_HIGH 用普通法线。
DEGRADE_LOW = 0.6
DEGRADE_HIGH = 0.9

_EPS = 1e-12
_FLOAT_MANTISSA_BITS = 53


# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------

def _clamp(value, low, high):
    return max(low, min(high, value))


def _number(value, default=0.0):
    """把任意入参转成 float；无法转换时返回 *default*。"""
    if isinstance(value, bool):
        return float(value)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result


def _as_vec(value):
    """把标量或序列统一成 float 元组（标量视为一维）。"""
    if isinstance(value, (list, tuple)):
        return tuple(_number(v) for v in value)
    return (_number(value),)


def _as_list(value):
    """把任意入参统一成列表；dict / 标量视为单元素。"""
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, (list, tuple, set, frozenset)):
        return list(value)
    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
        return list(value)
    return [value]


def _int_value(value, name):
    """把入参转成整数；NaN / inf / 非数字抛 ``ValueError``。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError("%s must be a number" % name) from None
    if not math.isfinite(number):
        raise ValueError("%s must be finite" % name)
    return int(round(number))


def _finite(value, default):
    """返回有限浮点，否则回退到 *default*（审计路径的容错）。"""
    result = _number(value, default)
    return result if math.isfinite(result) else float(default)


def _checked(value, name):
    """核心数学路径：非有限值直接报错，避免 NaN 静默传播。"""
    result = _number(value, float("nan"))
    if not math.isfinite(result):
        raise ValueError("%s must be finite" % name)
    return result


def _mirrored(flag):
    """把 bool 或“负缩放”统一成是否镜像。"""
    if isinstance(flag, bool):
        return flag
    return _checked(flag, "mirror flag") < 0.0


def _smoothstep(edge0, edge1, x):
    """标准 smoothstep；区间退化时退化成阶跃，两端导数为零。"""
    if edge1 <= edge0:
        return 1.0 if x >= edge1 else 0.0
    t = _clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _canonical(value):
    """递归转换成可 JSON 序列化的规范形式，用于确定性摘要。"""
    if isinstance(value, dict):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_canonical(item) for item in value), key=repr)
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        # NaN / inf 不可 JSON 序列化，转成稳定的字符串标记。
        if not math.isfinite(value):
            return repr(value)
        return value
    if isinstance(value, str):
        return value
    if hasattr(value, "id"):
        return {"id": _canonical(value.id)}
    return repr(value)


def _digest(value):
    payload = json.dumps(
        _canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 1. 自遮挡检测
# ---------------------------------------------------------------------------

def grazing_angle(cos_theta):
    """由 ``|cos(入射角)|`` 返回视线与表面法线的夹角（度）。

    ``cos_theta`` 可以直接取切线空间视线的 ``nz`` 分量。夹角接近 90°
    说明视线贴着表面，此时视差步进的数值误差最大，需要放大自遮挡容差。
    """
    c = abs(_checked(cos_theta, "cos_theta"))
    return math.degrees(math.acos(_clamp(c, 0.0, 1.0)))


def _is_grazing(cos_theta, threshold=GRAZING_EPS):
    return abs(_checked(cos_theta, "cos_theta")) <= _checked(threshold, "threshold")


def self_occlusion(ray_height, surface_height=None, step=1.0, origin=0.0,
                   cos_theta=1.0, tolerance=OCCLUSION_EPS,
                   grazing_gain=GRAZING_TOL_GAIN):
    """判断某个步进采样是否被高度场遮挡。

    参数
    ----
    ray_height : 射线在该采样点的高度。
    surface_height : 同位置高度图的采样值。缺省时取与射线相同的高度
        （即“还没有越过任何表面”），结果为未遮挡。
    step : 从射线起点算起的步进距离（纹理空间）。
    origin : 射线起点对应的步进距离。``step <= origin`` 表示采样点
        仍是起点纹素本身，此时必须返回 ``False``——这正是掠射角下
        “自己遮挡自己”的来源。
    cos_theta : 视线与法线的夹角余弦，用于掠射容差。
    tolerance : 基础容差，吸收高度图量化噪声。
    grazing_gain : 掠射时容差的放大倍数。

    返回 ``True`` 表示表面高于射线且高出量超过容差。
    """
    step = _checked(step, "step")
    origin = _checked(origin, "origin")
    # 起始采样直接排除：射线还没有离开出发纹素。
    if step <= origin:
        return False

    ray = _checked(ray_height, "ray_height")
    surface = ray if surface_height is None else _checked(surface_height, "surface_height")

    # 掠射时射线高度下降得极慢，一个步长内的高度变化可能小于量化步长，
    # 按视线夹角放大容差，避免把浮点/纹理精度误差判成遮挡。
    c = _clamp(abs(_checked(cos_theta, "cos_theta")), 0.0, 1.0)
    limit = max(0.0, _checked(tolerance, "tolerance"))
    limit *= 1.0 + max(0.0, _checked(grazing_gain, "grazing_gain")) * (1.0 - c)

    return (surface - ray) > limit


# ---------------------------------------------------------------------------
# 2. mip 与步进联动
# ---------------------------------------------------------------------------

def step_cap(steps, budget=MAX_STEPS):
    """把自适应步数裁剪到 ``[1, budget]``。

    掠射角下 ``1/cos`` 形式的自适应步数会发散；这里给出硬上限，保证
    循环次数可控、不会出现长条状拉伸。非整数步数四舍五入，非有限
    值抛 ``ValueError``。
    """
    budget = _int_value(budget, "budget")
    if budget < 1:
        raise ValueError("budget must be >= 1")

    raw = _int_value(steps, "steps")
    return max(1, min(budget, raw))


def mip_step(ray_length, base_steps=32, texels_per_uv=256.0,
             previous_mip=None, max_mip=MAX_MIP,
             hysteresis=MIP_HYSTERESIS, max_steps=MAX_STEPS):
    """由一步前进的 UV 足迹同时决定 mip 层级与步进参数。

    *mip 选择*：用 ``ray_length / base_steps`` 得到“一步走多远”，
    与该级纹素尺寸比较后取 ``floor(log2(step / texel))``。这里完全不
    使用 ddx/ddy 屏幕导数——同一平面上每个像素的 ``ray_length`` 相同，
    得到同一个 mip，远处不会逐像素闪跳。

    *mip 与步进联动*：选定 mip 后，该级纹素尺寸为
    ``texel_uv = 2**mip / texels_per_uv``，步数按
    ``ceil(ray_length / texel_uv)`` 重新计算，并至少取 :data:`MIN_STEPS`
    次、至多 :data:`MAX_STEPS` 次。因为 mip 取 floor，
    ``step_uv <= texel_uv``，绝不会出现“mip 已经是粗级、步长还是细级”
    的错配。预算裁剪导致步长跨过多个纹素时，``height_scale`` 按实际
    足迹一并缩小，高度幅值与采样密度始终保持同步。

    *迟滞*：``previous_mip`` 给定时，本次 mip 相对上一帧最多变化
    ``hysteresis`` 级，且在 ``[0, max_mip]`` 内裁剪，远处不再在相邻
    层级之间来回跳。

    返回字典：``mip``（选定层级）、``mip_effective``（按实际步长折算的
    等效层级，可能因预算裁剪而更粗）、``steps``、``step_uv``（实际
    步长）、``texel_uv``（实际采样足迹）、``height_scale``（该足迹下
    高度幅值的缩放）、``ray_length``。
    """
    base = _int_value(base_steps, "base_steps")
    if base < 1:
        raise ValueError("base_steps must be >= 1")

    if max_mip is not None and _checked(max_mip, "max_mip") < 0:
        raise ValueError("max_mip must be >= 0")
    max_mip = _int_value(MAX_MIP if max_mip is None else max_mip, "max_mip")

    tpu = _checked(texels_per_uv, "texels_per_uv")
    if tpu <= 0.0:
        raise ValueError("texels_per_uv must be a positive finite number")

    ray = abs(_checked(ray_length, "ray_length"))

    texel0 = 1.0 / tpu
    if ray <= 0.0:
        return {
            "mip": 0,
            "mip_effective": 0.0,
            "steps": 1,
            "step_uv": 0.0,
            "texel_uv": texel0,
            "height_scale": 1.0,
            "ray_length": 0.0,
        }

    step_target = ray / base
    ratio = step_target / texel0
    if ratio <= 1.0:
        mip_raw = 0
    else:
        mip_raw = int(math.floor(math.log2(ratio) + _EPS))
    mip_raw = _clamp(mip_raw, 0, max_mip)

    # 迟滞：每次最多切 hysteresis 级，避免 mip 在相邻层级之间抖动。
    mip = mip_raw
    if previous_mip is not None:
        prev = _clamp(_int_value(previous_mip, "previous_mip"), 0, max_mip)
        span = max(0, _int_value(hysteresis, "hysteresis"))
        mip = _clamp(mip_raw, prev - span, prev + span)
    mip = _clamp(mip, 0, max_mip)

    texel_uv = texel0 * (2 ** mip)
    budget = max(1, min(MAX_STEPS, _int_value(max_steps, "max_steps")))
    # 近处保证至少 MIN_STEPS 次采样；用预算裁剪，掠射角不会爆炸。
    steps = max(MIN_STEPS, math.ceil(ray / texel_uv - _EPS))
    steps = step_cap(steps, budget)
    step_uv = ray / steps

    # 预算裁剪让步长跨过多个纹素时，等效 mip 更粗、高度幅值同步衰减。
    footprint = max(texel_uv, step_uv)
    mip_effective = math.log2(footprint / texel0)

    return {
        "mip": mip,
        "mip_effective": round(mip_effective, 6),
        "steps": steps,
        "step_uv": step_uv,
        "texel_uv": footprint,
        "height_scale": texel0 / footprint,
        "ray_length": ray,
    }


# ---------------------------------------------------------------------------
# 3. 深度偏移
# ---------------------------------------------------------------------------

def depth_bias(depth, texel_uv=1.0, slope=1.0, layer=0,
               max_bias=MAX_DEPTH_BIAS):
    """给视差偏移后的深度与其阴影深度计算同一个偏移。

    偏移量由纹素尺寸与斜率共同决定::

        bias = min(max_bias, |texel_uv| * DEPTH_TEXEL_SCALE
                             + |slope| * DEPTH_SLOPE_SCALE)

    固定常数偏移在斜面上不够、在正面上又过度；按纹素与斜率缩放后，
    斜面的阴影痤疮消失，正面也不会漏光。

    多层材质按 ``layer`` 错开 ``LAYER_SEPARATION``：层与层不再共享同一
    个深度值，既避免 z-fighting，也避免阴影图把两层算成同一个接收面。
    阴影深度复用同一个偏移值（不是另一套常数），因此深度测试与阴影
    测试一致，不会一方向偏移、另一方向对不上。

    深度约定为“越小越靠近相机”，层号越大越靠后。

    返回字典：``depth``（G-buffer 深度）、``shadow_depth``（阴影采样
    深度）、``bias``（本层偏移量）、``separation``（层间隔）、``layer``。
    """
    base = _checked(depth, "depth")

    texel = abs(_checked(texel_uv, "texel_uv"))
    slope = abs(_checked(slope, "slope"))
    limit = max(0.0, _checked(max_bias, "max_bias"))
    bias = min(limit, texel * DEPTH_TEXEL_SCALE + slope * DEPTH_SLOPE_SCALE)

    layer_index = max(0, _int_value(layer, "layer"))
    separation = layer_index * LAYER_SEPARATION

    biased = _clamp(base - bias + separation, 0.0, 1.0)
    # 阴影深度与 G-buffer 深度取同一个值：两者一旦用了不同偏移，
    # 阴影接收面与可见面就会错开，出现漏光或阴影痤疮。
    return {
        "depth": biased,
        "shadow_depth": biased,
        "bias": bias,
        "separation": separation,
        "layer": layer_index,
    }


# ---------------------------------------------------------------------------
# 4. 镜像方向校正
# ---------------------------------------------------------------------------

def mirror_fix(step, mirror_u=False, mirror_v=False):
    """UV 镜像时翻转对应的切线空间步进分量。

    ``u -> 1 - u`` 或负缩放会把切线空间的 ``+u`` 方向映射到屏幕上的
    反方向；视差步进若仍沿原方向前进，采到的就是相反一侧的高度，
    凹陷被当成凸起。``mirror_u`` / ``mirror_v`` 既接受布尔值，也接受
    缩放系数（负值视为镜像）。

    标量入参视为只有 u 分量。返回 ``(u, v)`` 元组。
    """
    vec = _as_vec(step)
    u = vec[0] if len(vec) > 0 else 0.0
    v = vec[1] if len(vec) > 1 else 0.0
    for name, value in (("u", u), ("v", v)):
        if not math.isfinite(value):
            raise ValueError("step.%s must be finite" % name)
    if _mirrored(mirror_u):
        u = -u
    if _mirrored(mirror_v):
        v = -v
    return (u, v)


# ---------------------------------------------------------------------------
# 5. 降级过渡
# ---------------------------------------------------------------------------

def degrade_blend(factor, height_offset=1.0, normal_weight=0.0,
                  low=DEGRADE_LOW, high=DEGRADE_HIGH):
    """在 POM 与普通法线之间做连续过渡。

    ``factor`` 通常取距离归一化后的 LOD 权重（0 = 近处，1 = 远处）。
    ``low`` 之前完整使用 POM，``high`` 之后完全退化为普通法线，中间用
    smoothstep 过渡——两端导数为零，高度偏移不会在切换点突然归零，
    也就不会看到高度跳变。

    返回字典：``weight``（0=POM，1=普通法线）、``height_offset``
    （已乘 ``1 - weight`` 的连续高度偏移）、``tangent_scale``
    （视差强度系数）、``normal_mix``（普通法线结果的混合权重）。
    """
    f = _clamp(_checked(factor, "factor"), 0.0, 1.0)
    edge0 = _checked(low, "low")
    edge1 = _checked(high, "high")
    if edge1 < edge0:
        raise ValueError("high must be >= low")
    weight = _smoothstep(edge0, edge1, f)

    height = _checked(height_offset, "height_offset")
    normal = _clamp(_checked(normal_weight, "normal_weight"), 0.0, 1.0)
    return {
        "weight": weight,
        "height_offset": height * (1.0 - weight),
        "tangent_scale": 1.0 - weight,
        "normal_mix": normal + (1.0 - normal) * weight,
    }


# ---------------------------------------------------------------------------
# 6. 高度范围与运维辅助
# ---------------------------------------------------------------------------

def height_range(height, low=0.0, high=1.0):
    """把高度采样裁剪到 ``[low, high]``，支持标量与向量。

    视差偏移后的采样可能落在纹理之外（负值或 >1），不裁剪会让射线
    穿出高度场。上下限传反时自动交换。
    """
    lo = _checked(low, "low")
    hi = _checked(high, "high")
    if hi < lo:
        lo, hi = hi, lo

    if isinstance(height, (list, tuple)):
        return tuple(_clamp(_checked(item, "height"), lo, hi) for item in height)
    return _clamp(_checked(height, "height"), lo, hi)


def uv_wrap(uv):
    """UV 平铺包裹：``uv - floor(uv)``，负值也回到 ``[0, 1)``。

    支持标量与二维向量。用 floor 而不是 ``% 1``，保证接缝两侧连续。
    """
    if isinstance(uv, (list, tuple)):
        values = tuple(_checked(item, "uv") for item in uv)
        return tuple(v - math.floor(v) for v in values)
    value = _checked(uv, "uv")
    return value - math.floor(value)


def precision(value, bits=16):
    """把 ``[0, 1]`` 内的值量化到 *bits* 位定点，支持标量与向量。

    高度图/深度缓冲按定点量化后，不同硬件与不同帧得到相同采样值，
    避免同一像素因浮点尾数不同被判成遮挡。*bits* 裁剪到
    ``[1, 53]``（float64 的有效尾数上限）。
    """
    bits = _int_value(bits, "bits")
    bits = max(1, min(bits, _FLOAT_MANTISSA_BITS))
    levels = float((1 << bits) - 1)

    def quantize(item):
        v = _clamp(_checked(item, "uv"), 0.0, 1.0)
        return round(v * levels) / levels

    if isinstance(value, (list, tuple)):
        return tuple(quantize(item) for item in value)
    return quantize(value)


def validation(config):
    """校验一份 POM 配置，返回 ``{"ok": bool, "errors": [...], ...}``。

    覆盖本模块所有可调参数：步数预算、mip 上限、纹素密度、降级区间、
    高度范围与深度偏移上限。任何越界都到 ``errors`` 里，不抛异常，
    便于启动时一次性收集全部配置问题。
    """
    errors = []
    if not isinstance(config, dict):
        return {
            "ok": False,
            "errors": ["config must be a mapping, got %r" % type(config).__name__],
            "config": _canonical(config),
        }

    normalized = dict(config)

    def read(key, default):
        value = config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append("%s must be a number" % key)
            return default
        if not math.isfinite(float(value)):
            errors.append("%s must be finite" % key)
            return default
        return float(value)

    base_steps = read("base_steps", 32.0)
    if base_steps < 1:
        errors.append("base_steps must be >= 1")
    else:
        normalized["base_steps"] = int(round(base_steps))

    max_steps = read("max_steps", float(MAX_STEPS))
    if max_steps < 1:
        errors.append("max_steps must be >= 1")
    else:
        normalized["max_steps"] = int(round(max_steps))

    max_mip = read("max_mip", float(MAX_MIP))
    if max_mip < 0:
        errors.append("max_mip must be >= 0")
    else:
        normalized["max_mip"] = int(round(max_mip))

    texels = read("texels_per_uv", 256.0)
    if texels <= 0:
        errors.append("texels_per_uv must be > 0")
    else:
        normalized["texels_per_uv"] = texels

    low = read("degrade_low", DEGRADE_LOW)
    high = read("degrade_high", DEGRADE_HIGH)
    if not (0.0 <= low < high <= 1.0):
        errors.append("degrade range must satisfy 0 <= low < high <= 1")
    else:
        normalized["degrade_low"] = low
        normalized["degrade_high"] = high

    h_low = read("height_low", 0.0)
    h_high = read("height_high", 1.0)
    if h_low > h_high:
        errors.append("height_low must be <= height_high")
    else:
        normalized["height_low"] = h_low
        normalized["height_high"] = h_high

    max_bias = read("max_depth_bias", MAX_DEPTH_BIAS)
    if not (0.0 <= max_bias <= 1.0):
        errors.append("max_depth_bias must be within [0, 1]")
    else:
        normalized["max_depth_bias"] = max_bias

    normalized["ok"] = not errors
    normalized["errors"] = errors
    return normalized


def audit(record):
    """审计一次 POM 采样，返回判定结果与中间量。

    *record* 可以是字典（推荐），也可以是一个裸值。缺省字段按“射线
    高度 = 表面高度、正入射”解释，保证任何输入都返回非空结果。
    """
    if isinstance(record, dict):
        rec = record
    else:
        rec = {"raw": record}

    ray = _finite(rec.get("ray_height", rec.get("ray", 0.0)), 0.0)
    surface = _finite(rec.get("surface_height", rec.get("surface", 0.0)), 0.0)
    step = _finite(rec.get("step", 0.0), 0.0)
    origin = _finite(rec.get("origin", 0.0), 0.0)
    c = _clamp(abs(_finite(rec.get("cos_theta", 1.0), 1.0)), 0.0, 1.0)

    occluded = self_occlusion(
        ray, surface, step=step, origin=origin, cos_theta=c
    )
    return {
        "occluded": occluded,
        "grazing": _is_grazing(c),
        "angle_deg": grazing_angle(c),
        "step": step,
        "ray_height": ray,
        "surface_height": surface,
        "depth": round(_clamp(ray - surface, 0.0, 1.0), 6),
    }


def fallback(reason="unspecified"):
    """降级回退：关闭视差偏移，只保留普通法线。

    任何异常路径（高度图缺失、tracer 超预算、配置非法）都可以调用它
    得到一份完整、非 ``None`` 的采样结果，避免整块材质变黑。
    """
    return {
        "mode": "normal",
        "parallax_scale": 0.0,
        "height_offset": 0.0,
        "tangent_scale": 0.0,
        "normal_mix": 1.0,
        "occluded": False,
        "steps": 0,
        "reason": str(reason),
    }


def metrics(samples):
    """汇总一组审计记录，返回计数与平均值。

    空输入返回全零摘要；标量输入按单条记录处理。字段：
    ``samples``、``occluded``、``occluded_ratio``、``grazing``、
    ``avg_steps``、``avg_mip``、``avg_height_offset``、
    ``max_depth_bias``。
    """
    records = _as_list(samples)
    count = len(records)
    if count == 0:
        return {
            "samples": 0,
            "occluded": 0,
            "occluded_ratio": 0.0,
            "grazing": 0,
            "avg_steps": 0.0,
            "avg_mip": 0.0,
            "avg_height_offset": 0.0,
            "max_depth_bias": 0.0,
        }

    # 标量条目没有 .get，先统一成字典再求平均。
    dicts = [record if isinstance(record, dict) else {"raw": record}
             for record in records]
    audits = [audit(record) for record in dicts]
    occluded = sum(1 for item in audits if item["occluded"])
    grazing = sum(1 for item in audits if item["grazing"])

    def avg(key):
        return sum(_finite(rec.get(key, 0.0), 0.0) for rec in dicts) / count

    max_bias = max(
        (_finite(rec.get("bias", rec.get("depth_bias", 0.0)), 0.0) for rec in dicts),
        default=0.0,
    )

    return {
        "samples": count,
        "occluded": occluded,
        "occluded_ratio": occluded / count,
        "grazing": grazing,
        "avg_steps": avg("steps"),
        "avg_mip": avg("mip"),
        "avg_height_offset": avg("height_offset"),
        "max_depth_bias": max_bias,
    }


def determinism(*values):
    """对任意输入生成稳定的 SHA-256 摘要。

    字典按键排序、集合按 repr 排序、非有限浮点转成固定标记，因此
    同一份采样计划在不同 ``PYTHONHASHSEED``、不同进程下都会得到同一
    摘要，可用于回归比对或定位首个分叉像素。
    """
    return _digest(list(values))
