# 项目名称（改成你的项目名）

一句话说明这个项目是干什么的。

## 开发环境

本项目要求使用 VS Code Dev Containers（`.devcontainer/`）。

容器隔离与安全设计：

- 唯一挂载进容器的宿主机路径就是本项目文件夹本身（`workspaceMount`），不挂载 `~/.ssh`、`~/.aws`、Docker socket 或任何其他宿主机目录/凭据。
- 容器内以非 root 用户 `dev`（UID/GID 1000，与宿主机常见用户 UID 对齐）运行，而非 root。
- Node.js（供 Claude Code 使用）通过官方维护的 [devcontainer Feature](https://containers.dev/features) 安装，而非从网络下载脚本直接 `curl | bash` 执行。
- 所有工具版本（Node、uv、pip、Claude Code）均在 `Dockerfile` / `devcontainer.json` 中锁定版本号，保证任何人在任何时间构建出的环境一致。
- 密钥只通过 `.env`（已被 `.gitignore` 排除，不会被提交或分发）注入，仓库中只保留 `.env.example` 占位模板。首次使用请自行复制：`cp .env.example .env` 并填入真实 key。
- `.claude/settings.local.json` 等会话本地状态已加入 `.gitignore`，不会随仓库分享出去。

## 项目结构

业务代码目录请按需自己设计，建议遵循一个原则：**一个文件夹只负责一件事**，并在该文件夹的 `__init__.py` 里用一两句话写清楚它的职责（不该做什么也可以写）。这样任何人打开目录结构，不用问人就知道每部分的边界。

```text
your-project/
├── .devcontainer/     # 容器配置，一般不用改
├── outputs/           # 运行产生的文件，已 gitignore，不进版本库
├── tests/
├── .env.example
└── pyproject.toml
```

## 运行方式

进入 Dev Container 后，按你项目的实际入口命令填写，例如：

```bash
python -m your_package
```
