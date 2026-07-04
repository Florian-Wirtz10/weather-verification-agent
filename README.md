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

本项目推荐使用 VS Code Dev Containers。

## 运行方式

进入 Dev Container 后运行：

```bash
streamlit run app.py
```

## 项目结构

```text
weather_verification_agent/
├── agent/
├── verification/
├── data_io/
├── visualization/
├── reports/
├── outputs/
├── tests/
├── app.py
└── pyproject.toml
```
