# pairwise-pom-self-shadow

## 领域

游戏、图形与多媒体

## 主题

视差贴图自遮挡

## 说明

自遮挡检测、mip 联动与镜像校正。

初始状态下 `graphics/pom_self_shadow.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。

## 实现要点

`graphics/pom_self_shadow.py` 针对六类已确认缺陷的修复策略：

1. **自遮挡检测**：`self_occlusion` 用 `step > origin` 排除射线起点自身的
   纹素——掠射角下第一次采样拿自己的高度和自己比较，量化噪声会被判成
   遮挡，正是砖缝黑边的来源。判定阈值按 `|cos θ|` 放大（`grazing_angle`
   给出夹角），掠射时只接受确实高出射线足够多的表面。`audit` 对单条采样
   复现同一判定，非有限值回退而非抛错。
2. **mip 与步进联动**：`mip_step` 不再使用 ddx/ddy 屏幕导数，而由“一步
   前进的 UV 足迹” `floor(log2(step / texel))` 决定 mip；相同 `ray_length`
   必然得到同一 mip，远处不会逐像素抖动。步数由该级纹素尺寸推导，取 floor
   保证 `step_uv <= texel_uv`，mip 与步进严格同频；`previous_mip` 提供迟滞，
   每帧最多切一级。预算裁剪让步长跨纹素时，`height_scale` 按实际足迹同步
   衰减。
3. **深度偏移**：`depth_bias` 按 `纹素尺寸 × 系数 + 斜率 × 系数` 计算偏移并
   裁剪到 `max_bias`，取代固定常数；多层材质按 `layer` 错开
   `LAYER_SEPARATION`，层间与阴影图不再共享同一深度。`shadow_depth` 与
   G-buffer `depth` 复用同一个偏移值，深度测试与阴影测试一致。
4. **镜像方向校正**：`mirror_fix` 在 UV 镜像时翻转对应的步进分量，
   `mirror_u`/`mirror_v` 同时接受布尔值与负缩放系数，u/v 独立处理。
5. **降级过渡**：`degrade_blend` 用 smoothstep 把 POM 权重过渡到普通法线，
   两端导数为零；高度偏移、视差强度、法线混合权重同一条曲线连续变化，
   降级时不再出现高度跳变。
6. **采样次数上限**：`step_cap` 把步数裁剪到 `[1, MAX_STEPS]`，`mip_step`
   的步数也走同一预算；`MIN_STEPS` 保证近处仍有足够采样，掠射角不会迭代
   爆炸。

运维辅助：`height_range`（高度裁剪）、`validation`（配置校验，一次收集
全部越界项）、`fallback`（异常回退到普通法线）、`metrics`（审计汇总）、
`uv_wrap`（平铺包裹）、`precision`（定点量化，消除浮点尾数差异）、
`determinism`（跨进程稳定 SHA-256 摘要）。

`tests/test_pom_self_shadow_properties.py` 覆盖上述六项行为；原有的契约
测试中 `(1)` 是整数而非单元素元组，`fn(*args)` 会抛 `TypeError` 而无法
测到任何函数，已修正为 `(1,)`。
