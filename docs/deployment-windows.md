# 老师电脑部署清单（Windows 10/11）

仓库含前端构建产物，正常部署只需 Python 3.11，无需 Node.js。
真实资料、API 密钥、虚拟环境和任务输出均不在仓库中，需要单独配置。

## 1. 获取项目

安装 Python 3.11（勾选添加到 PATH），然后下载仓库 main 分支 ZIP 并解压，
或安装 Git 后执行：

```powershell
git clone https://github.com/Florian-Wirtz10/weather-verification-agent.git
cd weather-verification-agent
```

以下命令在项目根目录的 PowerShell 执行，应能看到 `pyproject.toml`。

## 2. 安装依赖

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e . -c constraints-security.txt
.\.venv\Scripts\python.exe -m pip check
Test-Path .\frontend\dist\index.html
```

最后应输出 `True`。这里直接调用虚拟环境 Python，不需要修改 PowerShell 执行策略。
首次安装需联网；若镜像尚未同步要求的新版依赖，请使用默认 PyPI 源重试，不要随意降低版本。

## 3. 复制资料并配置

完整资料约 51 GB，建议可用磁盘至少 80 GB、内存 16 GB。将原始资料单独复制到目标电脑，
保持根目录下的 `cases/`、`models/`、`benchmarks/` 层级，详见项目 README 的“数据目录”。

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
notepad .env
```

例如资料放在 D 盘，可设置：

```dotenv
WEATHER_DATA_ROOT=D:/weather-data/datasets
DEEPSEEK_API_KEY=
```

没有 DeepSeek 密钥也能使用默认计划及“老师修改版四类图”模板，解读使用确定性摘要。
自由输入自然语言要求需要有效密钥。不要把原开发电脑的 `.venv` 复制过来使用。

## 4. 启动与验收

```powershell
.\.venv\Scripts\python.exe -m uvicorn webapi.main:app --host 127.0.0.1 --port 8000
```

浏览器访问 <http://127.0.0.1:8000>；健康检查 <http://127.0.0.1:8000/api/health>
应返回 `status: ready`。健康检查只证明服务启动，不证明资料配置正确。

在“检验功能与提示词”中选“老师修改版四类图”→“填入工作台”，确认能列出起报，
运行后检查 CSI、四张 POD 箱线图、FAR 柱状图和 FSS，尝试下载 PNG/SVG/ZIP。
首次读资料可能较慢。按 Ctrl+C 停止服务，以后只需重新执行启动命令。

## 常见问题

- 没有可选起报：核对 `.env` 路径和数据目录层级，修改后重启。
- 8000 端口占用：改用 `--port 8001`，访问地址也改为 8001。
- 页面不是新版：重启后按 Ctrl+F5 刷新；旧任务不会自动重绘，需要重新执行。
- 仅用于本机时保留 `127.0.0.1`。本服务没有内置身份认证，不应直接开放到公网。
- 当前任务记录在进程内，重启后不会恢复；所需结果请及时下载。

本版本已在 Linux 环境完成测试和构建，Windows 目标电脑仍需按以上步骤实际验收。
