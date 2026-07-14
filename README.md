# Weather Verification Agent

面向已接入风雷预报个例的受限检验助手：自然语言只负责生成可审计的检验参数，确定性代码
负责读取资料、计算指标和绘图，LLM 可以在严格证据边界内解释计算结果。

它不是实时天气问答、任意数据分析或自由代码/制图工具。

## 阶段状态

- Phase 1（tag `v1.0-phase1`）：合成降水场、基础指标、绘图和 Streamlit demo。
- Phase 2（tag `v2.0-phase2`）：接入风雷 V1.0 REFMOSAIC、雷达 CREF、DeepSeek 单轮选图。
- Phase 3（tag `v3.0-phase3`）：可信的单轮检验助手。
- Phase 4（tag `v4.0-phase4`）：受控呈现与专业解读。
- Phase 5（当前）：全天公平检验、业务二进制 CREF、快速索引与断点指标缓存。

Phase 5 在不改变 Phase 4 单起报链路的前提下新增：

- 直接解析天气雷达拼图系统 V3.0 的 256 字节 `MOC/CREF` 头和 BZ2 压缩 `int16` 网格；按头字段校验文件长度、产品、坐标、时间、网格、压缩方式与缩放因子；
- 支持 2026-04-07 全天 7200 个风雷 REFMOSAIC 文件和 249 个 CREF `.bin` 交付文件；快速索引得到 240 个唯一观测时刻及 6735 对严格匹配样本；
- 同一有效时刻有多个观测版本时，默认选择生成时间最新的订正版，并在页面保留被替代版本审计；
- 文件名快速索引带指纹缓存，只抽查文件头；真实目录第二次索引无需逐个解压；
- 全天计算逐预报流式解压、逐有效时刻复用一帧观测，并把每个完成样本立即提交到 SQLite；中断后同配置可续算，完成后可直接读取指标缓存；
- 全天汇总按每个时效、阈值先合并 H/M/F/CN，再重新计算评分，而不是简单平均单样本分数；
- 新增全天评分曲线、起报时刻×时效热力图、跨起报样本分布图，以及只读取结构化事实的全天 AI 诊断；
- Phase 5 平台固定双方共同检验口径，不预设风雷或未来上游模型谁好谁坏。

Phase 4 在 Phase 3 基础上新增：

- 自然语言可在白名单内选择页面排版（单列、双列、2×2、报告式、动画优先），但不能生成或执行绘图代码；
- 支持同一起报、指定时效窗的“预报—观测反射率对比”GIF 动画（最多 30 帧、1—4 fps）；
- “解析并运行”一步完成；运行后仍可展开查看可审计的规范化请求与执行口径；
- 右侧固定、独立滚动的“AI 解读”栏。无论是否配置 DeepSeek 都有确定性摘要；配置 API 后自动生成四段式 AI 解读：图表导读、主要发现、指标分析、口径与边界；
- 不必手写排版或动画指令：空间过程/演变类请求自动提出受限 GIF 动画；最终页面按实际结果自动选择动画优先、空间图网格或报告式布局；
- 当用户目标是解释“空间回波演变与 CSI/ETS 等评分的关系”时，LLM 可选择联动评分动画：每帧把预报—观测空间场与完整评分曲线、当前时效指示线合成为**一个 GIF 文件**；
- AI 只接收结构化评分事实和图表目录，不接收原始网格、图片、路径或代码；因此可写“如图 1 所示”，但不会声称看到了图像；
- PNG、GIF、CSV、Markdown 和 ZIP 均可下载，ZIP 中含审计元数据与动画文件。

Phase 3 的主要能力：

- 将自然语言解析成版本化、白名单化的请求 Schema；
- 支持反射率阈值、预报时效范围和精确到单张 Figure 的视图请求；
- 一次运行强制只使用一个风雷起报，预报和观测时间必须完全一致；
- 在解压大文件前按起报与时效过滤，逐对流式读取，避免内存随历史帧数累积；
- 显式区分 `ready / partial / unsupported / out_of_scope / no_data / parser_error`；
- 不支持、无数据或 API 失败时不计算、不乱出默认图；
- 保留 hits、misses、false alarms、correct negatives 和有效/无效格点数；
- 缺测值不再被当作正确否定，空间范围外不再复制最近边界值；
- 先由代码生成事实摘要，再由 DeepSeek（若 API 可用）组织成文字解读；
- 结果解读只讨论用户明确请求的评分指标，不主动引入未请求的 POD/FAR/Bias 等指标；
- DeepSeek 解读失败时，确定性摘要、指标表和图仍然正常展示。
- 每张图可单独下载 PNG，并可下载请求指标 CSV、解读 Markdown 或完整 ZIP 结果包；
- Figure 转成 PNG 后立即释放，重复运行不会在 Session 中持续积累 Matplotlib 图对象；
- 检验完成后可重试 AI 解读，不会重新读取和计算风雷/雷达资料。

## 支持的自然语言能力

支持的评分指标：`CSI / POD / FAR / bias / ETS`。

支持的视图：

- 单项评分随预报时效变化曲线；
- 指定时效的风雷预报与雷达观测对比图；
- 指定时效的反射率误差图。
- 指定时效窗的预报—观测反射率对比动画。

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
- 2026-04-07 CREF 二进制交付的 `TimeZone` 字段实际为 0，但文件名有效时刻、头中观测时刻与风雷有效时刻共同表明其业务时钟为北京时间；解析器要求文件名和头时间严格一致后再减 8 小时，未知命名或不一致会拒绝；
- CREF 二进制原始值除以头中缩放因子 10；`int16` 最小值作为缺测，其它负值保留为低于正阈值的无回波编码；
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
FENGLEI_CASE_DIR=external/20250411风雷预测资料
RADAR_OBS_DIR=external/20250411观测资料
FENGLEI_DAILY_DIR=external/20260407风雷预测资料
RADAR_DAILY_DIR=external/20260407观测资料
```

页面始终显示确定性摘要；配置 `DEEPSEEK_API_KEY` 后，会自动把白名单事实摘要发送给 DeepSeek 生成专业解读。不会发送路径、文件名、原始网格、图片或代码；API 不可用时自动回退到确定性摘要。

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
├── daily_request.py        # 全天曲线/热力图/分布图的受限请求 Schema
├── daily_insight.py        # 全天结构化诊断事实 + 可选 LLM 解读
├── presentation.py         # 按实际媒体确定自动、安全的页面布局
└── errors.py               # no-data / ambiguous-case / data-contract 错误

data_io/                    # 风雷、NetCDF/BIN 雷达读取、快速目录索引与网格对齐
verification/               # 列联表、指标纯函数、全天流式评分与 SQLite 缓存
visualization/              # 单起报和全天确定性 Matplotlib 图表
reports/                    # 报告输出
tests/                      # Schema、拒绝状态、科学计算、端到端测试
app.py                      # Streamlit UI
```

## 下一阶段候选

后续接入上游模型时，应复用 Phase 5 已固定的观测选择、时间匹配、区域、网格配准、有效掩膜、
阈值和汇总口径，再并排展示两套结果。之后再考虑 2025-04-11 全天观测、跨多日统计、
持续性基线、FSS/邻域检验和 V1.1 降水适配。
