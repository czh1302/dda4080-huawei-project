# Normal Core Definition for Static Driving Frames

本文档定义本项目中用于单帧多视角自动驾驶场景分析的 **normal core**。该定义用于筛选普通场景，而不是穷举所有长尾场景。

## Motivation

在单帧设置下，许多动态风险无法可靠判断，例如急刹、cut-in、突然横穿等。因此，本项目只关注当前多视角画面中可见的静态或弱动态驾驶条件。

我们不直接用频率定义 normal，也不把 normal 简单等同于数据集中最常见的场景。normal 应该对应自动驾驶语境下的常规、清晰、低歧义场景。

## Definition

**Normal core** 指：从当前单帧多视角画面可见信息判断，车辆不需要采取超出常规驾驶策略的额外防御性行为的简单驾驶场景。

换句话说，如果该场景不明显要求车辆因为环境、可见性、道路结构、交通参与者或静态障碍而额外减速、增大车距、增加横向安全距离、提高让行警惕，或采取更保守的驾驶策略，则可视为 normal core。

## Inclusion Criteria

一个样本可被认为属于 normal core，通常应满足以下条件：

- 画面整体清晰，主要道路区域和交通参与者可辨认。
- 光照和天气条件不会显著降低感知可靠性。
- ego vehicle 周围有合理安全空间，与前方和侧向车辆保持常规距离。
- 前方道路通行状态清楚，没有明显阻塞、施工、封闭或异常静态障碍。
- 行人、骑行者、车辆等交通参与者处于常规位置和状态。
- 没有明显遮挡导致关键风险区域不可见。
- 不需要因为当前可见信息而采取额外减速、避让、让行或保守跟车策略。

## Exclusion Criteria

如果样本出现以下任一明显情况，则不应放入 normal core：

- 低光照、黑夜、强眩光、雨天、湿滑强反光、雾状低可见度等导致感知不确定性增加的条件。
- 前车距离过近、刹车灯明显亮起、前方拥堵或存在需要提前减速的迹象。
- 行人、骑行者或其他弱势交通参与者与 ego path 距离较近，或被遮挡但可能影响驾驶。
- 大车、停靠车辆、建筑物等造成关键区域严重遮挡。
- 施工区、锥桶、路障、封路、异常停放车辆或车道内障碍物。
- 道路结构复杂或不清楚，例如复杂路口、车道边界不明确、异常道路拓扑。
- 相机严重模糊、遮挡、过曝、欠曝或画面质量明显异常。
- 任何从单帧可见信息看需要额外防御性驾驶的情况。

## Annotation Protocol

标注时建议采用三类判断，而不是强行二分类：

- `normal_core`: 高置信普通场景，可用于学习 normal manifold。
- `not_normal_core`: 明显需要额外防御性驾驶或存在静态困难因素的场景。
- `uncertain`: 单帧信息不足、标注员无法稳定判断，暂不用于 normal 训练。

训练 SAE 或 one-class 模型时，建议只使用 `normal_core` 样本。`not_normal_core` 和 `uncertain` 不直接等同于长尾标签，而是留作后续发现、排序和人工分析。

## Relation to Long-Tail Discovery

本项目不试图预先穷举所有长尾类型。我们只定义一个高置信 normal core，让模型学习普通驾驶场景的表示分布。偏离该 normal core 的样本可能对应长尾、困难样本或异常采集条件，需要通过后续可解释特征、top activating samples 和人工复核进一步确认。

## Prompt Principle

使用 Cosmos Reason1 提取表示时，prompt 不应直接提供 normal/tail 定义，也不应要求模型判断是否长尾。推荐使用中性驾驶场景分析 prompt，使模型关注可见的静态驾驶相关因素，而不是直接执行分类任务。

Example:

```text
Analyze this multi-camera autonomous driving frame.
Describe visible static driving-relevant factors, including lighting, weather,
visibility, road layout, traffic participants, obstacles, construction elements,
and unusual static object states. Do not infer temporal events.
```
