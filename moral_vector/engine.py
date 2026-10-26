"""Consistent decoder-block extraction, mean differences, and AMF injection.

The same zero-based decoder block is used for detection, extraction and injection.
"""
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import CLASSES, FOUNDATIONS


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(model_name, device='auto'):
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
    dtype = torch.float16 if str(device).startswith('cuda') else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = dict(torch_dtype=dtype)
    if str(device).startswith('cuda'):
        kwargs['device_map'] = {'': device}
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if not str(device).startswith('cuda'):
        model.to(device)
    model.eval()
    if not hasattr(model, 'model') or not hasattr(model.model, 'layers'):
        raise ValueError('Expected Llama/Qwen decoder blocks at model.model.layers')
    return model, tokenizer


def unpack(output):
    return output[0] if isinstance(output, tuple) else output


def replace(output, hidden):
    return (hidden,) + output[1:] if isinstance(output, tuple) else hidden


def input_device(model):
    return model.get_input_embeddings().weight.device


def validate_layer(model, layer):
    if not 0 <= layer < len(model.model.layers):
        raise ValueError(f'Invalid decoder block {layer}; model has {len(model.model.layers)} blocks')


def extract(model, tokenizer, texts, layers, batch_size=8, max_length=128):
    """Capture block outputs directly, including the un-normalized final block.

    Each result has shape (samples, hidden_dim), with last non-padding token pooling.
    """
    for layer in layers:
        validate_layer(model, layer)
    chunks = {layer: [] for layer in layers}
    for start in range(0, len(texts), batch_size):
        tokens = tokenizer(texts[start:start + batch_size], padding=True, truncation=True,
                           max_length=max_length, return_tensors='pt', return_token_type_ids=False).to(input_device(model))
        mask = tokens['attention_mask']
        # Also works for tokenizers that use left padding.
        positions = (torch.arange(mask.shape[1], device=mask.device)[None, :] * mask).max(1).values
        handles = []
        try:
            for layer in layers:
                def capture(module, inputs, output, layer=layer):
                    hidden = unpack(output)
                    ix = positions.to(hidden.device)
                    selected = hidden[torch.arange(len(ix), device=hidden.device), ix]
                    chunks[layer].append(selected.detach().float().cpu())
                handles.append(model.model.layers[layer].register_forward_hook(capture))
            with torch.inference_mode():
                model(**tokens, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
    return {layer: torch.cat(parts).numpy() for layer, parts in chunks.items()}


class LinearProbe(nn.Module):
    def __init__(self, dim, classes=10):
        super().__init__()
        self.linear = nn.Linear(dim, classes)

    def forward(self, x):
        return self.linear(x)


def train_probe(x, y, epochs=30, seed=42, device='cpu', batch_size=64, lr=1e-3,
                weight_decay=1e-4, validation=None, score_mode='final'):
    seed_everything(seed)
    probe = LinearProbe(x.shape[1]).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    dataset = torch.utils.data.TensorDataset(torch.as_tensor(x, dtype=torch.float32),
                                            torch.as_tensor(y, dtype=torch.long))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    best = 0.0
    for _ in range(epochs):
        probe.train()
        for bx, by in loader:
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(probe(bx.to(device)), by.to(device))
            loss.backward()
            optimizer.step()
        if validation is not None and score_mode == 'best':
            best = max(best, accuracy(probe, *validation))
    probe.eval()
    score = None if validation is None else (best if score_mode == 'best' else accuracy(probe, *validation))
    return probe, score


def accuracy(probe, x, y):
    with torch.inference_mode():
        pred = probe(torch.as_tensor(x, dtype=torch.float32, device=next(probe.parameters()).device)).argmax(-1)
    return float((pred.cpu().numpy() == np.asarray(y)).mean())


def mean_vectors(x, labels, train_indices):
    """Raw centroid difference, matching paper Eq. 3 (no unit normalization)."""
    labels = np.asarray(labels)
    result = {}
    for foundation in FOUNDATIONS:
        centroids = []
        for polarity in ('virtue', 'vice'):
            selected = [i for i in train_indices if labels[i] == CLASSES.index(f'{foundation}.{polarity}')]
            if not selected:
                raise ValueError(f'Missing training examples: {foundation}.{polarity}')
            centroids.append(x[selected].mean(0))
        result[foundation] = torch.as_tensor(centroids[0] - centroids[1], dtype=torch.float32)
    return result


def save_bundle(path, model_name, layer, probe, vectors, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(format_version=1, model=model_name, layer=layer, classes=list(CLASSES),
                    probe={k: v.detach().cpu() for k, v in probe.state_dict().items()},
                    vectors=vectors, metadata=metadata), path)


def load_bundle(path, model_name=None, layer=None):
    bundle = torch.load(path, map_location='cpu', weights_only=True)
    if bundle.get('format_version') != 1 or bundle['classes'] != list(CLASSES):
        raise ValueError('Unknown bundle format or class order')
    if model_name is not None and bundle['model'] != model_name:
        raise ValueError('Bundle and requested model differ; regenerate artifacts for this model')
    if layer is not None and bundle['layer'] != layer:
        raise ValueError('Bundle layer and requested layer differ')
    return bundle


class Steering:
    def __init__(self, model, bundle, mode='amf', coefficient=2.0, threshold=0.2, foundation=None):
        if mode not in ('baseline', 'mean', 'amf', 'single'):
            raise ValueError(mode)
        if mode == 'single' and foundation not in FOUNDATIONS:
            raise ValueError('single mode requires a valid foundation')
        self.model, self.bundle = model, bundle
        self.mode, self.coefficient, self.threshold, self.foundation = mode, coefficient, threshold, foundation
        validate_layer(model, bundle['layer'])
        self.probe = LinearProbe(model.config.hidden_size)
        self.probe.load_state_dict(bundle['probe'], strict=True)
        self.probe.eval()
        self.vectors = torch.stack([bundle['vectors'][f].reshape(-1) for f in FOUNDATIONS])
        if self.vectors.shape != (5, model.config.hidden_size):
            raise ValueError('Vector dimensions do not match model')
        self.vice_indices = [CLASSES.index(f'{f}.vice') for f in FOUNDATIONS]
        self.handle = None

    def hook(self, module, inputs, output):
        if self.mode == 'baseline':
            return output
        h = unpack(output)
        vectors = self.vectors.to(device=h.device, dtype=h.dtype)
        if self.mode == 'amf':
            self.probe.to(device=h.device, dtype=h.dtype)
            with torch.no_grad():
                probs = self.probe(h).float().softmax(-1)[..., self.vice_indices]
                alpha = (probs - self.threshold).relu() * self.coefficient
                delta = alpha.to(h.dtype) @ vectors
        elif self.mode == 'mean':
            delta = self.coefficient * vectors.mean(0)
        else:
            delta = self.coefficient * vectors[FOUNDATIONS.index(self.foundation)]
        return replace(output, h + delta)

    def __enter__(self):
        if self.handle is not None:
            raise RuntimeError('Steering hook already registered')
        self.handle = self.model.model.layers[self.bundle['layer']].register_forward_hook(self.hook)
        return self

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def generate(model, tokenizer, prompt, chat=False, max_new_tokens=150):
    if chat:
        prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                                               tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False)
    inputs = tokenizer(prompt, return_tensors='pt', return_token_type_ids=False).to(input_device(model))
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
