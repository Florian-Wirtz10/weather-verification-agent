# Weather Verification Agent

自然语言驱动的 AI 模型降水预报检验智能体原型。

## 当前阶段目标

第一阶段先实现一个最小可行版本：

- 使用随机二维数组模拟预报降水和实况降水；
- 计算降水预报检验指标；
- 自动绘制预报图、实况图、误差图；
- 使用 Streamlit 展示结果；
- 后续再接入真实气象数据和智能体框架。

## 技术栈

- Python 3.11
- NumPy
- Pandas
- Matplotlib
- Streamlit
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

当前 `app.py` 与各功能包（见下）仍是骨架阶段，尚未接入真实的检验流程。

## 项目结构

```text
weather_verification_agent/
├── agent/           # 自然语言编排层：解析用户请求，驱动下面各层的流水线
├── verification/    # 检验指标：纯函数，不做 I/O、不画图
├── data_io/         # 数据读写层：加载/生成预报与实况场
├── visualization/   # 绘图层：只接收数组/指标结果，不计算指标
├── reports/         # 报告生成：汇总指标与图表，写入 outputs/reports/
├── outputs/         # 运行产物（figures/tables/reports），已被 gitignore
├── tests/
├── app.py           # Streamlit 入口
├── template/        # 可复用的项目骨架（容器配置等），复制出去改个名字即可用于新项目
└── pyproject.toml
```

## 项目模板

`template/` 目录是从本项目提炼出的、不含任何天气检验业务代码的空骨架（容器配置、`.gitignore`、`pyproject.toml` 骨架等），用来快速创建新项目。用法见 `template/TEMPLATE_USAGE.md`。
