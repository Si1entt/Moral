"""Offline checks of data selection and moral vector operations."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM
try:
    from transformers import Qwen3Config, Qwen3ForCausalLM
except ImportError:
    Qwen3ForCausalLM = None

from moral_vector.data import CLASSES, FOUNDATIONS, ROOT, input_audit, read_pairs, sentences
from moral_vector.engine import (LinearProbe, Steering, extract, mean_vectors, load_bundle,
                                 save_bundle, train_probe)
from moral_vector.experiments import partition, balanced_binary
from moral_vector.utility import boxed_answer
from moral_vector.api import label_response


class Tokenizer:
    """Simple right-padding batch tokenizer for testing the actual model forward path."""
    def __call__(self, texts, **kwargs):
        class Batch(dict):
            def to(self, device):
                return Batch({k: v.to(device) for k, v in self.items()})
        rows = [[int(x) for x in text.split()] for text in texts]
        width = max(map(len, rows))
        return Batch(input_ids=torch.tensor([r + [0] * (width - len(r)) for r in rows]),
                     attention_mask=torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in rows]))


class ReleaseTests(unittest.TestCase):
    def model(self):
        torch.manual_seed(1)
        model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                                            num_hidden_layers=2, num_attention_heads=2,
                                            num_key_value_heads=2, pad_token_id=0))
        return model.eval()

    def bundle(self, model):
        probe = LinearProbe(model.config.hidden_size)
        with torch.no_grad():
            probe.linear.weight.zero_()
            probe.linear.bias.zero_()
        return dict(format_version=1, model='tiny', layer=0, classes=list(CLASSES),
                    probe=probe.state_dict(), vectors={f: torch.ones(model.config.hidden_size) * (i + 1)
                                                      for i, f in enumerate(FOUNDATIONS)}, metadata={})

    def test_exact_paper_data_and_first_pair(self):
        audit = input_audit()
        self.assertEqual(audit['en']['samples'], 4208)
        self.assertEqual(audit['zh']['samples'], 11014)
        expected = {'authority': 431, 'care': 470, 'fairness': 351, 'loyalty': 192, 'sanctity': 660}
        for foundation, count in expected.items():
            for polarity in ('virtue', 'vice'):
                self.assertEqual(audit['en']['per_class'][f'{foundation}.{polarity}'], count)
        row = read_pairs(ROOT / 'datasets/mfd_generated_results.jsonl')[0]
        texts, labels, ids = sentences([row])
        self.assertEqual(len(texts), 2)
        self.assertEqual(CLASSES[labels[1]], 'care.vice')
        self.assertEqual(ids[0], ids[1])

    def test_pair_group_partition_and_no_test_vector_leakage(self):
        labels, pair_ids = [], []
        for f in FOUNDATIONS:
            for i in range(20):
                pair_ids.extend([len(pair_ids) // 2] * 2)
                labels.extend([CLASSES.index(f'{f}.virtue'), CLASSES.index(f'{f}.vice')])
        data = dict(y=np.array(labels), pair_ids=np.array(pair_ids))
        train, test = partition(data)
        self.assertFalse(set(data['pair_ids'][train]) & set(data['pair_ids'][test]))
        self.assertEqual(len(train), 160)
        x = np.array([[3.0, 2.0] if CLASSES[y].endswith('virtue') else [1.0, 1.0] for y in labels])
        original = mean_vectors(x, labels, train)
        x[test] += 100000
        changed = mean_vectors(x, labels, train)
        for f in FOUNDATIONS:
            torch.testing.assert_close(original[f], torch.tensor([2., 1.], dtype=torch.float32))
            torch.testing.assert_close(original[f], changed[f])

    def test_amf_threshold_and_exact_weighted_fusion(self):
        model = self.model()
        bundle = self.bundle(model)
        h = torch.zeros(2, 3, model.config.hidden_size)
        # Uniform probabilities are 0.1: tau 0.2 -> exactly no intervention.
        manager = Steering(model, bundle, 'amf', threshold=0.2)
        torch.testing.assert_close(manager.hook(None, None, h), h)
        # tau .05: alpha=(.1-.05)*2=.1; vectors sum to 15 per coordinate.
        manager.threshold = 0.05
        torch.testing.assert_close(manager.hook(None, None, h), torch.full_like(h, 1.5))
        wrapped = manager.hook(None, None, (h, 'aux'))
        torch.testing.assert_close(wrapped[0], torch.full_like(h, 1.5))
        self.assertEqual(wrapped[1], 'aux')

    def test_real_decoder_padding_and_hook_cleanup(self):
        model = self.model()
        tok = Tokenizer()
        batch = extract(model, tok, ['1 2 3', '4 5'], [0, 1])
        single = extract(model, tok, ['4 5'], [0, 1])
        for layer in [0, 1]:
            np.testing.assert_allclose(batch[layer][1], single[layer][0], rtol=1e-4, atol=1e-5)
            self.assertEqual(len(model.model.layers[layer]._forward_hooks), 0)
        bundle = self.bundle(model)
        baseline = extract(model, tok, ['1 2 3'], [0])[0]
        with Steering(model, bundle, 'single', coefficient=2, foundation='care'):
            steered = extract(model, tok, ['1 2 3'], [0])[0]
        np.testing.assert_allclose(steered - baseline, np.full_like(baseline, 2), rtol=1e-4, atol=1e-5)
        try:
            with Steering(model, bundle):
                raise RuntimeError('test cleanup')
        except RuntimeError:
            pass
        self.assertEqual(len(model.model.layers[0]._forward_hooks), 0)

    def test_probe_training_bundle_roundtrip_and_model_guard(self):
        rng = np.random.default_rng(42)
        y = np.tile(np.arange(10), 5)
        x = np.eye(10, dtype=np.float32)[y] + rng.normal(0, 0.01, (50, 10)).astype(np.float32)
        probe, score = train_probe(x, y, epochs=3, validation=(x, y))
        self.assertTrue(np.isfinite(score))
        vectors = {f: torch.ones(10) for f in FOUNDATIONS}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bundle.pt'
            save_bundle(path, 'tiny', 0, probe, vectors, {'seed': 42})
            restored = load_bundle(path, 'tiny', 0)
            for key, value in probe.state_dict().items():
                torch.testing.assert_close(value, restored['probe'][key])
            with self.assertRaises(ValueError):
                load_bundle(path, 'other-model')

    @unittest.skipIf(Qwen3ForCausalLM is None, 'Qwen3 requires transformers >=4.51.3')
    def test_qwen3_real_blocks_and_steering(self):
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                                             num_hidden_layers=2, num_attention_heads=2,
                                             num_key_value_heads=2, head_dim=8)).eval()
        tokenizer = Tokenizer()
        baseline = extract(model, tokenizer, ['1 2 3', '4 5'], [0, 1])
        bundle = self.bundle(model)
        with Steering(model, bundle, 'single', coefficient=2, foundation='care'):
            steered = extract(model, tokenizer, ['1 2 3', '4 5'], [0])
        np.testing.assert_allclose(steered[0] - baseline[0], np.full_like(baseline[0], 2), atol=1e-5)
        self.assertEqual(len(model.model.layers[0]._forward_hooks), 0)

    def test_binary_negative_stratification(self):
        data = dict(x=np.arange(1000).reshape(100, 10), y=np.repeat(np.arange(10), 10))
        x, y = balanced_binary(data, 0, 42)
        self.assertEqual(y.sum(), 10)
        self.assertEqual(len(y), 20)
        self.assertEqual(len(set((x[y == 0, 0] // 100).tolist())), 9)

    def test_judge_failures_and_nested_math_answers(self):
        self.assertEqual(label_response('Reasoning: ok\nLabel: Virtue', ['Virtue', 'Vice']), 'Virtue')
        with self.assertRaises(ValueError):
            label_response('network error', ['Virtue', 'Vice', 'Neutral'])
        self.assertEqual(boxed_answer(r'earlier \boxed{1}, final \boxed{\frac{1}{2}}.'), r'\frac{1}{2}')


if __name__ == '__main__':
    unittest.main()
