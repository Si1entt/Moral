# The Straight and Narrow: Do LLMs Possess an Internal Moral Path?

English | [简体中文](README.zh-CN.md)

Code and datasets for Moral Vectors and Adaptive Moral Fusion (AMF), including bilingual moral representation probing, cross-lingual transfer, activation steering, and safety evaluation.

## Installation

Python 3.11 is recommended. Install a PyTorch/CUDA build suitable for your device.

```bash
python -m pip install -r requirements.txt
python run.py check-data
```

Llama-3.1 and Qwen3 are supported. Set `--model` to a Hugging Face model ID or local path. Obtain access to Llama weights separately. Extract features, train probes, and generate vectors independently for each model.

## Datasets

The repository includes English and Chinese MFD dictionaries, MVV sentence pairs, HarmBench, XSTest, and evaluation subsets of ARC-Challenge, SciQ, and MATH-500. MVV uses the first pair in each record, yielding 4,208 English and 11,014 Chinese sentences.

See [datasets/README.md](datasets/README.md) for formats, sources, and terms of use.

## Experiments

Run commands from the repository root. Outputs are saved under `runs/` by default. These examples use the default model, `meta-llama/Llama-3.1-8B-Instruct`; select another model with the `MODEL_PATH` environment variable or `--model`.

```bash
# Extract bilingual features
python run.py features --language en --layers all --output runs/en.npz
python run.py features --language zh --layers all --output runs/zh.npz

# Layerwise ten-class probing and bidirectional cross-lingual transfer
python run.py probe --features runs/en.npz --output runs/probe_en.json
python run.py probe --features runs/zh.npz --output runs/probe_zh.json
python run.py transfer --en runs/en.npz --zh runs/zh.npz --layer 17

# Train a probe, extract vectors, and generate steered continuations
python run.py fit --features runs/en.npz --layer 17 --output runs/bundle.pt
python run.py internal --features runs/en.npz --bundle runs/bundle.pt
python run.py steer --bundle runs/bundle.pt

# Safety evaluation, threshold ablation, and utility evaluation
python run.py safety --bundle runs/bundle.pt --thresholds 0 0.1 0.2 0.3 0.4
python run.py utility --bundle runs/bundle.pt

# Decode activation semantics
python run.py oracle --bundle runs/bundle.pt --injection-layer 1
```

Layers use zero-based decoder block indices; the default is 17. Use `--layers 17` during feature extraction if you only need steering experiments, reducing memory and storage use. By default, `probe` reports each fold's highest accuracy during training; use `--score-mode final` for final-epoch accuracy. See `python run.py <command> --help` for all options.

## API evaluation and two-round debate

Set `OPENAI_API_KEY` and, for a compatible provider, `OPENAI_BASE_URL`. See [.env.example](.env.example). Use model identifiers supported by your provider.

```bash
python run.py judge --task steering --input runs/steering_results.json --judge-model gpt-5 --output runs/steering_judged.json
python run.py judge --task harmbench --input runs/safety/harmbench_amf_tau_0.2.csv --judge-model gpt-4o --output runs/harmbench_judged.json
python run.py judge --task xstest --input runs/safety/xstest_amf_tau_0.2.csv --judge-model gpt-4o --output runs/xstest_judged.json
python run.py debate --bundle runs/bundle.pt --input runs/steering_results.json --attacker-model YOUR_CLAUDE_MODEL_ID --judge-model gpt-5
```

To generate new MVV records:

```bash
python run.py generate-data --language en --generator-model YOUR_GEMINI_MODEL_ID --output runs/new_en.jsonl
```

## Implementation and checks

The unified entry point is `run.py`, with implementations in `moral_vector/`. See [legacy/](legacy/README.md) for standalone experiment scripts and usage.

```bash
python -B -m unittest discover -s tests -v
```

## License

Author code is licensed under the [MIT License](LICENSE). Third-party datasets retain their upstream licenses; see [data sources and licenses](datasets/README.md#data-sources-and-licenses).
