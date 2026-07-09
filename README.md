# Weather Verification Agent

自然语言驱动的 AI 模型降水预报检验智能体原型。

## 当前阶段目标

第一阶段实现的最小可行版本（已冻结，见 tag `v1.0-phase1`）：

- 使用随机二维数组模拟预报降水和实况降水；
- 计算降水预报检验指标；
- 自动绘制预报图、实况图、误差图；
- 使用 Streamlit 展示结果。

第二阶段（当前分支）在此之上接入真实数据和自然语言层：

- 读取风雷 REFMOSAIC 反射率预报（SWAN/diamond-131 二进制 + bz2）和雷达 CREF 观测（NetCDF），
  在 2025-04-11 湖北大范围强对流个例上做真实检验（`data_io/fenglei_mosaic.py`、
  `data_io/radar_obs.py`、`data_io/grid.py`）；
- `visualization/plots.py` 提供一份固定的图表菜单（反射率场、误差场、CSI/POD/FAR/ETS/
  频率偏差随预报时效变化曲线）；
- `agent/nl_parser.py` 用 DeepSeek 把自然语言请求解析成"选哪些阈值、调用菜单里的哪些图"，
  模型只能从固定菜单中选择，不允许生成或执行任意绘图代码；
- 私有的项目需求文档和风雷/雷达样例数据放在 `external/`（已被 `.gitignore` 排除，不随仓库
  分发）——`data_io` 里的读取器只接受路径参数，不硬编码依赖 `external/`。

## 技术栈

- Python 3.11
- NumPy / Pandas
- Matplotlib
- netCDF4
- Streamlit
- DeepSeek（OpenAI 兼容接口，经 `openai` SDK 调用）
- Docker Dev Container
- Claude Code

## 开发环境

本项目要求使用 VS Code Dev Containers（`.devcontainer/`），不支持也不建议在宿主机 Python 环境中直接运行。

容器隔离与安全设计：

- 唯一挂载进容器的宿主机路径就是本项目文件夹本身（`workspaceMount`），不挂载 `~/.ssh`、`~/.aws`、Docker socket 或任何其他宿主机目录/凭据。
- 容器内以非 root 用户 `dev`（UID/GID 1000，与宿主机常见用户 UID 对齐）运行，而非 root。
- Node.js（供 Claude Code 使用）通过官方维护的 [devcontainer Feature](https://containers.dev/features) 安装，而非从网络下载脚本直接 `curl | bash` 执行。
- 所有工具版本（Node、uv、pip、Claude Code）均在 `Dockerfile` / `devcontainer.json` 中锁定版本号，保证任何人在任何时间构建出的环境一致。
- 密钥只通过 `.env`（已被 `.gitignore` 排除，不会被提交或分发）注入，仓库中只保留 `.env.example` 占位模板。首次使用请自行复制：`cp .env.example .env` 并填入真实 key。
- `.claude/settings.local.json` 等会话本地状态已加入 `.gitignore`，不会随仓库分享出去。

## 运行方式

进入 Dev Container 后运行：

```bash
streamlit run app.py
```

`app.py` 有两个标签页：phase 1 的合成数据 demo，和 phase 2 的真实个例检验。真实个例检验需要
在 `.env` 里配置 `DEEPSEEK_API_KEY`、`FENGLEI_CASE_DIR`、`RADAR_OBS_DIR`（指向你本地的
`external/` 数据目录）。

## 项目结构

```text
weather_verification_agent/
├── agent/           # 自然语言编排层：解析用户请求，驱动下面各层的流水线
│   ├── nl_parser.py     # DeepSeek 请求解析：只能从固定图表菜单里选图，不生成代码
│   └── orchestrator.py  # 合成数据 pipeline + 真实个例 pipeline
├── verification/    # 检验指标：纯函数，不做 I/O、不画图
├── data_io/         # 数据读写层：合成场 + 风雷 mosaic / 雷达 CREF 真实数据读取
├── visualization/   # 绘图层：只接收数组/指标结果，不计算指标
├── reports/         # 报告生成：汇总指标与图表，写入 outputs/reports/
├── outputs/         # 运行产物（figures/tables/reports），已被 gitignore
├── external/        # 私有项目需求文档 + 风雷/雷达样例数据，已被 gitignore，不随仓库分发
├── tests/
├── app.py           # Streamlit 入口
└── pyproject.toml
```
