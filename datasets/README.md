# Input datasets

English | [简体中文](README.zh-CN.md)

MVV JSONL files provide inputs for probe training and vector extraction; `generated_results` refers to corpus generation. Utility dataset sources and checksums are recorded in [utility/SOURCES.json](utility/SOURCES.json).

| File | Contents | Count | Source |
|---|---|---:|---|
| `MFD.csv` | English dictionary; `word,foundation` | 2,104 rows | MFD 2.0, Frimer et al. (2019) |
| `C-MFD.csv` | Chinese dictionary; `chinese,foundation_labeled`, UTF-8 | 5,507 rows | C-MFD 2.0, Cheng & Zhang (2023), with polarity annotations |
| `mfd_generated_results.jsonl` | English MVV seed records | 2,104 records → 4,208 sentences | Generated MVV corpus |
| `cmfd_generated_results.jsonl` | Chinese MVV seed records | 5,507 records → 11,014 sentences | Generated MVV corpus |
| `harmbench_behaviors_text_all.csv` | Behaviors and metadata | 400 rows | [HarmBench](https://github.com/centerforaisafety/HarmBench) |
| `xstest_prompts.csv` | Prompts and safe/unsafe labels | 450 rows | [XSTest](https://github.com/paul-rottger/exaggerated-safety) |
| `utility/arc.jsonl` | ARC-Challenge Dev / validation subset | 299 rows | Official AI2 archive |
| `utility/sciq.jsonl` | SciQ validation subset | 1,000 rows | `valid.json` from the official AI2 archive |
| `utility/math.jsonl` | MATH-500 test problems and reference answers | 500 rows | OpenAI simple-evals public mirror |

Utility files include reference answers for evaluation. MATH-500's `answer` is extracted from the last balanced `\boxed{...}` expression in each solution. Download URLs, transformations, and SHA-256 checksums are recorded in `utility/SOURCES.json`.

## MVV format

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

Use `pairs[0]`: label `context + virtue_suffix` as `care.virtue` and `context + vice_suffix` as `care.vice`. The suffix in `category` describes the seed word's polarity; sentence labels follow the polarity of their corresponding suffix.

| Foundation | English per polarity | Chinese per polarity |
|---|---:|---:|
| Authority | 431 | 1,415 |
| Care | 470 | 1,477 |
| Fairness | 351 | 1,023 |
| Loyalty | 192 | 698 |
| Sanctity | 660 | 894 |

The two polarities have equal counts within each foundation. Counts differ across foundations.

## Benchmark usage

HarmBench uses `Behavior` as the prompt by default. Use `--functional-category` to filter behavior types and `--include-context` to append `ContextString`.

XSTest includes 250 safe and 200 unsafe prompts. Incorrect refusal rates use the safe subset as the denominator.

Validate inputs or download utility data from the repository root:

```bash
python run.py check-data
python run.py prepare-utility --benchmarks arc sciq math
```

`prepare-utility` preserves existing files by default. Use `--force` to download again and `--math-source huggingface` or `--math-source simple-evals` to choose the MATH source.

## Data sources and licenses

The root [MIT License](../LICENSE) applies to author code. Third-party datasets retain their upstream rights and terms: [MFD 2.0](https://osf.io/ezn37/), the C-MFD 2.0 paper by Cheng & Zhang (2023), [HarmBench](https://github.com/centerforaisafety/HarmBench), [XSTest](https://github.com/paul-rottger/exaggerated-safety), [ARC](https://allenai.org/data/arc), [SciQ](https://allenai.org/data/sciq), [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500), and [OpenAI simple-evals](https://github.com/openai/simple-evals).
