# The Straight and Narrow: Do LLMs Possess an Internal Moral Path?

[English](README.md) | 简体中文

Moral Vector 与 Adaptive Moral Fusion（AMF）的实验代码和数据集，支持双语道德表征探测、跨语言迁移、激活引导及安全评测。

## 安装

建议使用 Python 3.11，并安装适合设备的 PyTorch/CUDA。

```bash
python -m pip install -r requirements.txt
python run.py check-data
```

模型支持 Llama-3.1 和 Qwen3，可通过 `--model` 指定 Hugging Face 模型 ID 或本地路径。Llama 权重需自行申请访问权限；每个模型需单独提取特征、训练探针和生成向量。

## 数据集

仓库包含中英文 MFD 词典、MVV 句对、HarmBench、XSTest，以及 ARC-Challenge、SciQ、MATH-500 的评测子集。MVV 使用每条记录的第一组句对，展开为英文 4,208 条、中文 11,014 条句子。

格式、来源与使用条款见 [datasets/README.md](datasets/README.zh-CN.md)。

## 运行实验

在仓库根目录执行，输出默认保存到 `runs/`。以下示例使用默认模型 `meta-llama/Llama-3.1-8B-Instruct`；可通过环境变量 `MODEL_PATH` 或 `--model` 替换模型。

```bash
# 双语特征提取
python run.py features --language en --layers all --output runs/en.npz
python run.py features --language zh --layers all --output runs/zh.npz

# 逐层十分类与双向跨语言迁移
python run.py probe --features runs/en.npz --output runs/probe_en.json
python run.py probe --features runs/zh.npz --output runs/probe_zh.json
python run.py transfer --en runs/en.npz --zh runs/zh.npz --layer 17

# 训练探针、提取向量与引导生成
python run.py fit --features runs/en.npz --layer 17 --output runs/bundle.pt
python run.py internal --features runs/en.npz --bundle runs/bundle.pt
python run.py steer --bundle runs/bundle.pt

# 安全评测、阈值消融与通用能力评测
python run.py safety --bundle runs/bundle.pt --thresholds 0 0.1 0.2 0.3 0.4
python run.py utility --bundle runs/bundle.pt

# 激活语义解码
python run.py oracle --bundle runs/bundle.pt --injection-layer 1
```

层编号为零起始 decoder block 索引，默认层为 17。仅运行引导实验时，特征提取可改用 `--layers 17`，减少内存与磁盘占用。`probe` 默认报告每折训练期间的最高准确率；使用 `--score-mode final` 可报告最后一轮准确率。完整选项见 `python run.py <command> --help`。

## API 评判与两轮反驳

设置环境变量 `OPENAI_API_KEY`，兼容服务可另设 `OPENAI_BASE_URL`。配置示例见 [.env.example](.env.example)。模型名称按服务商支持的标识填写。

```bash
python run.py judge --task steering --input runs/steering_results.json --judge-model gpt-5 --output runs/steering_judged.json
python run.py judge --task harmbench --input runs/safety/harmbench_amf_tau_0.2.csv --judge-model gpt-4o --output runs/harmbench_judged.json
python run.py judge --task xstest --input runs/safety/xstest_amf_tau_0.2.csv --judge-model gpt-4o --output runs/xstest_judged.json
python run.py debate --bundle runs/bundle.pt --input runs/steering_results.json --attacker-model YOUR_CLAUDE_MODEL_ID --judge-model gpt-5
```

重新生成 MVV 数据：

```bash
python run.py generate-data --language en --generator-model YOUR_GEMINI_MODEL_ID --output runs/new_en.jsonl
```

## 实现与验证

统一入口位于 `run.py`，实现位于 `moral_vector/`；独立实验脚本及其用法见 [legacy/](legacy/README.zh-CN.md)。

```bash
python -B -m unittest discover -s tests -v
```

## 许可证

作者代码采用 [MIT License](LICENSE)。第三方数据保留各自上游许可，详见 [数据来源与许可](datasets/README.zh-CN.md#数据来源和许可)。
