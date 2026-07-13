# Weather Verification Agent

面向已接入风雷预报个例的受限检验助手：自然语言只负责生成可审计的检验参数，确定性代码
负责读取资料、计算指标和绘图，LLM 可以在严格证据边界内解释计算结果。

它不是实时天气问答、任意数据分析或自由代码/制图工具。

## 阶段状态

- Phase 1（tag `v1.0-phase1`）：合成降水场、基础指标、绘图和 Streamlit demo。
- Phase 2（tag `v2.0-phase2`）：接入风雷 V1.0 REFMOSAIC、雷达 CREF、DeepSeek 单轮选图。
- Phase 3（当前）：可信的单轮检验助手。

Phase 3 的主要能力：

- 将自然语言解析成版本化、白名单化的请求 Schema；
- 支持反射率阈值、预报时效范围和精确到单张 Figure 的视图请求；
- 一次运行强制只使用一个风雷起报，预报和观测时间必须完全一致；
- 在解压大文件前按起报与时效过滤，逐对流式读取，避免内存随历史帧数累积；
- 显式区分 `ready / partial / unsupported / out_of_scope / no_data / parser_error`；
- 不支持、无数据或 API 失败时不计算、不乱出默认图；
- 保留 hits、misses、false alarms、correct negatives 和有效/无效格点数；
- 缺测值不再被当作正确否定，空间范围外不再复制最近边界值；
- 先由代码生成事实摘要，再可选地把白名单汇总发送给 DeepSeek 组织成文字解读；
- 结果解读只讨论用户明确请求的评分指标，不主动引入未请求的 POD/FAR/Bias 等指标；
- DeepSeek 解读失败时，确定性摘要、指标表和图仍然正常展示。
- 每张图可单独下载 PNG，并可下载请求指标 CSV、解读 Markdown 或完整 ZIP 结果包；
- Figure 转成 PNG 后立即释放，重复运行不会在 Session 中持续积累 Matplotlib 图对象；
- 检验完成后可以单独授权或重试 AI 解读，不会重新读取和计算风雷/雷达资料。

## 支持的自然语言能力

支持的评分指标：`CSI / POD / FAR / bias / ETS`。

支持的视图：

- 单项评分随预报时效变化曲线；
- 指定时效的风雷预报与雷达观测对比图；
- 指定时效的反射率误差图。

示例：

```text
使用35 dBZ阈值，只看前1小时，生成CSI和ETS两张曲线，并简要解读结果。
```

规范化后只会选择 6—60 分钟的严格匹配资料，并生成两张 Figure。

以下请求不会被偷偷替换成全部图表：

- “画 Taylor 图” → `unsupported`，不出图；
- “画 Taylor 图和 CSI” → `partial`，只出明确要求且已支持的 CSI；
- “检验 2025 年 4 月 10 日” → `no_data`，列出可用日期；
- “今天天气怎么样” → `out_of_scope`；
- DeepSeek/API/JSON 失败 → `parser_error`，可重试或主动点击明确默认方案。

## 数据与计算口径

当前真实个例链路为：

```text
风雷 V1.0 REFMOSAIC 雷达组合反射率预报（dBZ）
                    vs
雷达 CREF 组合反射率观测（dBZ）
```

- 风雷头时间按 UTC；雷达文件头北京时统一转换为 UTC；
- 只接受有效时间完全相等的预报—观测对；
- 风雷场通过最近邻重采样到雷达网格；完整风雷范围外填为缺测；
- 指标只使用预报和观测均为有限值的公共格点；
- 空间图与空间误差图仅在显示层将低于 0 dBZ 的值按 0 dBZ 呈现，以统一风雷的负值无回波编码与 CREF 的 0 dBZ 无回波编码；原始数据和检验评分不受影响；
- 时间序列保留逐 6 分钟数据，横轴主刻度每 30 分钟；原始缺帧处断线显示；
- 单个例结果不能推广为模型整体能力结论。

如果未来切换到风雷 V1.1 `V11-PRE`，必须使用同时段的定量降水实况和 mm 阈值，不能复用
当前 REFMOSAIC/CREF 的 dBZ 检验口径。

## 运行方式

项目使用 VS Code Dev Container。进入容器后：

```bash
streamlit run app.py
```

在 `.env` 中配置：

```dotenv
DEEPSEEK_API_KEY=...
FENGLEI_CASE_DIR=external/smoke_data/fenglei
RADAR_OBS_DIR=external/smoke_data/radar
```

页面默认不把指标发送给 DeepSeek。勾选“允许将汇总指标发送给 DeepSeek”后，只发送白名单事实
摘要，不发送路径、文件名、原始网格、图片或代码。

## 验证

```bash
python -m pytest -q
ruff check .
npx --yes pyright agent data_io reports tests verification visualization app.py --level error
```

`external/` 和 `.env` 均被 `.gitignore` 排除，不会进入仓库。

## 结构

```text
agent/
├── request_schema.py       # 不可信 JSON → 严格的领域请求 Schema
├── request_resolution.py   # 能力、日期、个例和状态的确定性闸门
├── nl_parser.py            # DeepSeek 只负责提出请求 JSON
├── orchestrator.py         # 单起报、按时效流式计算、按请求渲染
├── insight.py              # 确定性事实摘要 + 可选 LLM 文字组织
└── errors.py               # no-data / ambiguous-case / data-contract 错误

data_io/                    # 风雷、雷达读取与网格对齐
verification/               # 列联表和检验指标纯函数
visualization/              # 确定性 Matplotlib 图表
reports/                    # 报告输出
tests/                      # Schema、拒绝状态、科学计算、端到端测试
app.py                      # Streamlit UI
```

## 下一阶段候选

Phase 4 再考虑受限多轮修改、多个历史个例、持续性基线、FSS/邻域检验和 V1.1 降水适配；
通用声明式制图、跨个例批处理与上游任务调度放到更后续阶段。
