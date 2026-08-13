# 强对流预报智能检验平台

面向强对流个例的多模型、全时效检验与证据约束型 AI 解读平台。系统将
StormDiT、风雷、Zero-shot Benchmark 与 CREF 雷达观测放到统一检验流程中：
自然语言只负责选择受支持的参数和图表，资料读取、时空匹配、指标计算与制图均由确定性代码完成。

## 主要功能

- 支持 StormDiT、风雷单产品检验与严格公共样本下的双模型对比；
- 支持 ConvLSTM、PredRNN、Earthformer、NowCastNet、SimVP、DiffCast、
  Persistence 等 Zero-shot Benchmark；
- 支持单一起报、自定义 2—8 个起报、全部起报三种检验范围；
- 支持 CSI、POD、FAR、Bias、MAE、RMSE 和多尺度 FSS；
- 支持空间分布、GIF 动画、时效曲线、起报—时效矩阵、评分分布、
  相关性热力图和预报率—观测率图；
- 支持 PNG、SVG、GIF、CSV、Markdown 与完整 ZIP 下载；
- DeepSeek 可解析白名单内的自然语言要求并解读结构化结果；API 不可用时自动使用确定性摘要；
- React + FastAPI 工作台提供任务配置、执行进度、交互式图表、AI 解读和任务归档。

系统不提供实时天气查询、任意日期预报、自由代码执行或自由制图。

## 环境要求

- Python 3.11
- Node.js 20 或更高版本、npm（仅在修改或重新构建前端时需要）
- 建议 16 GB 内存和 SSD
- 完整业务资料约 51 GB，建议预留至少 80 GB 可用空间

## Windows PowerShell 操作文档

以下步骤适用于 Windows 10/11。请先安装 Python 3.11、Node.js 20 或更高版本和 Git，
然后在项目根目录打开 **PowerShell**。命令行提示符所在目录应能看到 `pyproject.toml`
和 `frontend` 文件夹。

### 1. 创建并激活 Python 环境

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

`Set-ExecutionPolicy` 只对当前 PowerShell 窗口生效，不会永久修改计算机策略。激活成功后，
命令行开头会出现 `(.venv)`。

### 2. 安装后端依赖

```powershell
python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 120 --retries 10
```

### 3. 安装并构建前端

```powershell
npm --prefix frontend ci --registry=https://registry.npmmirror.com
npm --prefix frontend run build
Test-Path .\frontend\dist\index.html
```

最后一条命令应输出 `True`。如果输出 `False`，说明前端尚未成功构建，不要继续启动服务，
应先查看上一条命令末尾的报错。

### 4. 配置资料和可选的 DeepSeek

首次运行时复制配置模板：

```powershell
Copy-Item .env.example .env
notepad .env
```

至少确认 `.env` 中的 `WEATHER_DATA_ROOT` 指向实际资料目录。需要使用自然语言解析和 AI
解读时填写 `DEEPSEEK_API_KEY`；不填写时，系统仍可执行默认计划并生成确定性摘要。真实资料的
推荐目录结构见下文“数据目录”。

### 5. 启动并访问系统

```powershell
.\.venv\Scripts\python.exe -m uvicorn webapi.main:app --host 127.0.0.1 --port 8000
```

保持这个 PowerShell 窗口运行，在浏览器打开：

- 系统首页：`http://127.0.0.1:8000`
- API 文档：`http://127.0.0.1:8000/api/docs`
- 健康检查：`http://127.0.0.1:8000/api/health`

终端出现 `Application startup complete` 表示服务已经启动。按 `Ctrl+C` 停止服务。
以后再次运行时，只需进入项目目录后执行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m uvicorn webapi.main:app --host 127.0.0.1 --port 8000
```

`127.0.0.1` 只允许本机访问。如果要供内网其他电脑访问，应在完成单位网络与访问控制配置后，
改用 `--host 0.0.0.0`，并通过防火墙、Nginx/Caddy 和单位认证入口限制访问。

## Linux 安装

```bash
git clone https://github.com/Florian-Wirtz10/weather-verification-agent.git
cd weather-verification-agent

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .

npm --prefix frontend ci
npm --prefix frontend run build
```

仓库同时包含已经构建好的 `frontend/dist`。只负责运行服务的 Linux 服务器不需要
安装 Node.js 或 npm，可以跳过上面两条 npm 命令；只有修改前端源码后才需要重新构建。

复制环境变量模板：

```bash
cp .env.example .env
```

至少配置资料根目录；DeepSeek API Key 为可选项：

```dotenv
DEEPSEEK_API_KEY=
WEATHER_DATA_ROOT=external/datasets

WEB_JOB_OUTPUT_DIR=outputs/web_jobs
WEB_JOB_RETENTION_DAYS=7
WEB_JOB_MAX_STORAGE_GB=20
WEB_JOB_CLEANUP_INTERVAL_MINUTES=60
```

相对路径始终以项目根目录解析。若单项资料存放在其它位置，可参考 `.env.example`
分别覆盖风雷、CREF、StormDiT 或 Benchmark 路径。

## 数据目录

真实资料不会进入 Git 仓库，需要单独复制为以下结构：

```text
external/datasets/
├── cases/
│   ├── 2025-04-11/
│   │   ├── forecasts/fenglei/
│   │   └── observations/cref/
│   └── 2026-04-08/
│       ├── forecasts/fenglei/
│       └── observations/cref/
├── models/stormdit/
└── benchmarks/2025-04-11/wuhan-zero-shot/
```

StormDiT、风雷与 CREF 按 UTC 有效时间严格匹配，不进行时间插值。双模型比较只使用
StormDiT、风雷与 CREF 三者共有的起报和有效时刻；空间配准、有效掩膜、区域和阈值口径保持一致。

## Linux 启动

仓库已包含生产前端，安装 Python 依赖后可直接启动统一服务：

```bash
weather-agent-web
```

浏览器打开：

- 工作台：`http://localhost:8000`
- API 文档：`http://localhost:8000/api/docs`
- 健康检查：`http://localhost:8000/api/health`

前端开发模式：

```bash
# 终端 1
uvicorn webapi.main:app --host 0.0.0.0 --port 8000 --reload

# 终端 2
npm --prefix frontend run dev
```

兼容的 Streamlit 入口：

```bash
streamlit run app.py
```

## 任务清理

服务启动时会立即清理过期任务，此后按配置周期执行：

- 超过 `WEB_JOB_RETENTION_DAYS` 的已完成任务会被删除；
- 总容量超过 `WEB_JOB_MAX_STORAGE_GB` 时，从最旧任务开始清理；
- 正在排队或执行的任务不会被删除；
- 保留天数或容量设为 `0` 可关闭对应规则；
- 清理周期设为 `0` 时只在服务启动时执行一次。

## 验证

```bash
pytest -q
ruff check .
npm --prefix frontend run lint
npm --prefix frontend run build
```

## 部署提示

- 当前任务队列和任务状态保存在单个进程内，服务器必须使用一个 Uvicorn worker；
- 生产环境建议绑定 `127.0.0.1:8000`，再通过 Nginx/Caddy 和单位认证入口访问；
- `.env`、API Key、真实资料和任务输出不得提交到 Git；
- `frontend/dist` 已作为部署资产纳入仓库；服务器无需安装 Node.js。前端源码发生变化时，
  应在兼容的构建环境中重新执行 `npm --prefix frontend run build` 并提交新的构建产物。

提示词示例见 [`prompts.md`](prompts.md)。
