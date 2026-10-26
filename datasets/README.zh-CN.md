# 输入数据集

[English](README.md) | 简体中文

MVV JSONL 用于构造探针样本和提取向量；`generated_results` 为数据生成阶段的文件名。Utility 数据集的来源与校验值见 [utility/SOURCES.json](utility/SOURCES.json)。

| 文件 | 内容 | 数量 | 来源 |
|---|---|---:|---|
| `MFD.csv` | 英文词典；`word,foundation` | 2,104 行 | MFD 2.0，Frimer et al. (2019) |
| `C-MFD.csv` | 中文词典；`chinese,foundation_labeled`，UTF-8 | 5,507 行 | C-MFD 2.0，Cheng & Zhang (2023)，含极性标注 |
| `mfd_generated_results.jsonl` | 英文 MVV 种子记录 | 2,104 记录 → 4,208 条句子 | MVV 生成语料 |
| `cmfd_generated_results.jsonl` | 中文 MVV 种子记录 | 5,507 记录 → 11,014 条句子 | MVV 生成语料 |
| `harmbench_behaviors_text_all.csv` | 行为及元信息 | 400 行 | [HarmBench](https://github.com/centerforaisafety/HarmBench) |
| `xstest_prompts.csv` | 问题及 safe/unsafe 标签 | 450 行 | [XSTest](https://github.com/paul-rottger/exaggerated-safety) |
| `utility/arc.jsonl` | ARC-Challenge Dev / validation 子集 | 299 行 | AI2 官方压缩包 |
| `utility/sciq.jsonl` | SciQ validation 子集 | 1,000 行 | AI2 官方压缩包 `valid.json` |
| `utility/math.jsonl` | MATH-500 测试题及标准答案 | 500 行 | OpenAI simple-evals 公开镜像 |

Utility 文件包含标准答案，供评测时使用。MATH-500 的 `answer` 从解答中最后一个完整的 `\boxed{...}` 提取。下载 URL、处理方式与 SHA-256 记录在 `utility/SOURCES.json`。

## MVV 格式

```json
{
  "target_word": "compassion",
  "category": "care.virtue",
  "pairs": [{
    "context": "At the animal shelter,",
    "virtue_suffix": "...",
    "vice_suffix": "..."
  }]
}
```

使用 `pairs[0]`：将 `context + virtue_suffix` 标为 `care.virtue`，将 `context + vice_suffix` 标为 `care.vice`。`category` 的后缀表示种子词极性；句子标签按对应 suffix 的极性构造。

| Foundation | 英文每种极性 | 中文每种极性 |
|---|---:|---:|
| Authority | 431 | 1,415 |
| Care | 470 | 1,477 |
| Fairness | 351 | 1,023 |
| Loyalty | 192 | 698 |
| Sanctity | 660 | 894 |

每个 foundation 内两种极性数量相等，五个 foundation 之间数量不同。

## 基准使用

HarmBench 默认使用 `Behavior` 作为提示。可用 `--functional-category` 筛选行为类型，用 `--include-context` 加入 `ContextString`。

XSTest 包含 250 条 safe 和 200 条 unsafe。错误拒答率以 safe 子集为分母。

在仓库根目录校验数据或重新下载 utility 数据：

```bash
python run.py check-data
python run.py prepare-utility --benchmarks arc sciq math
```

`prepare-utility` 默认保留已有文件。用 `--force` 重新下载，用 `--math-source huggingface` 或 `--math-source simple-evals` 指定 MATH 来源。

## 数据来源和许可

根目录的 [MIT 许可证](../LICENSE) 适用于作者代码。第三方数据沿用上游权利与使用条款：[MFD 2.0](https://osf.io/ezn37/)、Cheng & Zhang (2023) 的 C-MFD 2.0 论文、[HarmBench](https://github.com/centerforaisafety/HarmBench)、[XSTest](https://github.com/paul-rottger/exaggerated-safety)、[ARC](https://allenai.org/data/arc)、[SciQ](https://allenai.org/data/sciq)、[MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)、[OpenAI simple-evals](https://github.com/openai/simple-evals)。
