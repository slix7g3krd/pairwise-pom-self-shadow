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
