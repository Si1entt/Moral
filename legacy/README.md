# Standalone experiment scripts

English | [简体中文](README.zh-CN.md)

Run `python legacy/<script>.py` from the repository root. Inputs are under `datasets/`. Set `MODEL_PATH` for the model, `OPENAI_API_KEY` for API credentials, and `OPENAI_BASE_URL` for a compatible provider's endpoint. Adjust input paths and API model names in each script's configuration section.

For scripts requiring probes and vectors, extract features, train, and export with the unified entry point:

```bash
python run.py features --language en --layers 17 --output runs/en.npz
python run.py fit --features runs/en.npz --layer 17 --output runs/bundle.pt
python run.py export-legacy --bundle runs/bundle.pt --output-dir .
python legacy/safety_generate.py
```

Downstream scripts default to the English `probe_MFD_layer_17`. Adjust model, probe, and vector paths when using Chinese, another model, or another layer.

| File | Purpose |
|---|---|
| `dictionary_probe.py` | Layerwise ten-class probing on dictionary prompts; saves probes |
| `sentence_probe.py` | Layerwise ten-class probing on MVV sentences |
| `dictionary_transfer.py` | Bidirectional cross-lingual dictionary transfer |
| `dictionary_geometry.py` | Representation dispersion, directions, and cross-dataset similarity |
| `safety_generate.py` | AMF response generation on safety benchmarks |
| `utility_evaluate.py` | ARC / SciQ / MATH evaluation |
| `activation_observer.py` | Vice activation statistics at the final token of HarmBench prompts |
| `judge_harmbench.py`, `judge_xstest.py` | Safety evaluation |
| `judge_steering.py` | Moral polarity evaluation and plots for five steering coefficients |
| `1_...py` through `6_...py` | Two-round debate experiments |
| `generate_mvv.py` | Chinese sentence-pair generation |
| `annotate_cmfd.py` | Chinese dictionary polarity annotation |

Generated outputs are excluded by the root `.gitignore`.
