# 独立实验脚本

[English](README.md) | 简体中文

在仓库根目录运行 `python legacy/文件名.py`。输入数据位于 `datasets/`；模型通过 `MODEL_PATH` 设置，API 密钥通过 `OPENAI_API_KEY` 设置，兼容服务端点通过 `OPENAI_BASE_URL` 设置。可在脚本配置区修改输入路径和 API 模型名称。

需要探针与向量的脚本，可先使用统一入口提取特征、训练并导出：

```bash
python run.py features --language en --layers 17 --output runs/en.npz
python run.py fit --features runs/en.npz --layer 17 --output runs/bundle.pt
python run.py export-legacy --bundle runs/bundle.pt --output-dir .
python legacy/safety_generate.py
```

下游脚本默认读取英文 `probe_MFD_layer_17`。使用中文、其他模型或其他层时，相应调整模型、探针及向量路径。

| 文件 | 功能 |
|---|---|
| `dictionary_probe.py` | 词典提示词逐层十分类，保存探针 |
| `sentence_probe.py` | MVV 句子逐层十分类 |
| `dictionary_transfer.py` | 词典双向跨语言迁移 |
| `dictionary_geometry.py` | 表征离散度、方向与跨数据集相似度 |
| `safety_generate.py` | AMF 安全基准回答生成 |
| `utility_evaluate.py` | ARC / SciQ / MATH 评测 |
| `activation_observer.py` | HarmBench 提示词最后 token 的 vice 激活统计 |
| `judge_harmbench.py`, `judge_xstest.py` | 安全评判 |
| `judge_steering.py` | 五系数道德极性评判与绘图 |
| `1_...py` 到 `6_...py` | 两轮反驳实验 |
| `generate_mvv.py` | 中文句对生成 |
| `annotate_cmfd.py` | 中文词典极性标注 |

脚本输出默认由根目录 `.gitignore` 排除。
