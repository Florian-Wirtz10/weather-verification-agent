# 老师 Word 图表修改说明

依据：`需要修改的图.docx`。本次修改图表生成代码，并导出真实资料结果。

- 图 1：CSI 保留 10、20、35、45 dBZ 四面板；去掉指标箭头，刻度 13 pt、
  坐标名称 15 pt、子图标题 16 pt、总标题 18 pt、图例 14 pt。
- 图 2：POD 按上述四个阈值分别导出四张独立箱线图，每张比较 PredRNN、NowCastNet、StormDiT。
  箱体表示四分位距，横线表示中位数，菱形表示均值；散点是起报×时效样本。
  无观测事件导致的未定义 POD 不当作零，图中 n 是有效评分数。
- 图 3：按 Word 配图实现 FAR 分组柱状图，横轴为预报时效；按四个阈值分面。
  柱高采用公共样本列联表汇总后的 FAR，不是将各样本 FAR 简单平均。
  Word 称其为“直方图”，此处按配图含义实现，并非频数分布直方图。
- 图 4：保留示例的 20、35、45 dBZ × 1、3、5 像元九面板 FSS，去掉箭头并放大字体。
- 模型配色：PredRNN 绿色、NowCastNet 黄色、StormDiT 红色；网页交互图同步使用这些颜色。

在网页“检验功能与提示词”→ Benchmark 中选“老师修改版四类图”，点击“填入工作台”后运行。
此模板确定性生成请求，不需要 DeepSeek 解析；最终结果解读仍按原有配置执行。
修改已有任务不会自动重绘，需重新运行任务。

本次导出位置：`outputs/teacher_revision/2897214ed049/`。
资料为 2025-04-11 Wuhan Benchmark，三个模型共 24 个起报，6–72 分钟、12 个原生时效，
864 个模型×起报×时效配对。每张提供 PNG 和 SVG，ZIP 包含全部图片和评分 CSV。
Word 未指明原截图起报范围，本次使用全部公共起报；数值不应被当成对截图原始样本的复刻。

核心实现：`visualization/benchmark_plots.py`；执行接入：`webapi/execution.py`、
`benchmark_ui.py`；请求校验及模板：`agent/benchmark_request.py`；
网页模板：`frontend/src/components/ModelPromptView.tsx`。
