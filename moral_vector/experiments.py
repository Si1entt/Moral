"""Experiment drivers for probing, moral vectors, and activation steering."""
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split

from .data import ROOT, CLASSES, FOUNDATIONS, read_pairs, read_dictionary, sentences, sha256
from .engine import (load_model, extract, train_probe, mean_vectors, save_bundle,
                     load_bundle, Steering, generate, seed_everything, LinearProbe)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def load_features(path, layer):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data['metadata']))
        return dict(x=data[f'x_{layer}'], y=data['labels'], pair_ids=data['pair_ids'], meta=meta)


def partition(features, seed=42):
    """Split pairs as groups: both polarities remain in the same 80/20 partition."""
    ids = features['pair_ids']
    unique_ids, first = np.unique(ids, return_index=True)
    categories = [CLASSES[int(features['y'][i])].split('.')[0] for i in first]
    train, test = train_test_split(unique_ids, test_size=0.2, stratify=categories, random_state=seed)
    return np.flatnonzero(np.isin(ids, train)), np.flatnonzero(np.isin(ids, test))


def features_command(args):
    name = 'mfd' if args.language == 'en' else 'cmfd'
    if args.input_kind == 'sentences':
        path = ROOT / 'datasets' / f'{name}_generated_results.jsonl'
        rows = read_pairs(path)
        if args.limit:
            rows = rows[:args.limit]
        texts, labels, pair_ids = sentences(rows)
    else:
        path = ROOT / 'datasets' / ('MFD.csv' if args.language == 'en' else 'C-MFD.csv')
        rows = read_dictionary(path)
        if args.limit:
            rows = rows[:args.limit]
        texts = [(f'Your first instinct for the word "{word}"is: ' if args.language == 'en'
                  else f'你对“{word}”这个词的第一直觉是：') for word, _ in rows]
        labels = [CLASSES.index(label) for _, label in rows]
        pair_ids = list(range(1, len(rows) + 1))
    seed_everything(args.seed)
    model, tokenizer = load_model(args.model, args.device)
    layers = list(range(len(model.model.layers))) if args.layers == 'all' else [int(x) for x in args.layers.split(',')]
    representations = extract(model, tokenizer, texts, layers, args.batch_size, args.max_length)
    metadata = dict(model=args.model, language=args.language, input_kind=args.input_kind,
                    input_sha256=sha256(path), layers=layers, layer_index='zero-based decoder block output',
                    max_length=args.max_length, seed=args.seed, limit=args.limit)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as stream:
        np.savez_compressed(stream, metadata=json.dumps(metadata), labels=np.asarray(labels),
                            pair_ids=np.asarray(pair_ids), **{f'x_{k}': v for k, v in representations.items()})
    print(f'Saved {len(texts)} representations at layers {layers} to {path}')


def probe_command(args):
    with np.load(args.features, allow_pickle=False) as cache:
        meta = json.loads(str(cache['metadata']))
    scores = []
    for layer in meta['layers']:
        data = load_features(args.features, layer)
        cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        accs = []
        for fold, (train, test) in enumerate(cv.split(data['x'], data['y'])):
            _, acc = train_probe(data['x'][train], data['y'][train], epochs=args.epochs,
                                 seed=args.seed + fold, device=args.probe_device,
                                 validation=(data['x'][test], data['y'][test]), score_mode=args.score_mode)
            accs.append(acc)
        scores.append(dict(layer=layer, fold_accuracy=accs, mean=float(np.mean(accs)), std=float(np.std(accs))))
    write_json(args.output, dict(metadata=meta, folds=args.folds, epochs=args.epochs,
                                score_mode=args.score_mode, seed=args.seed, scores=scores))


def fit_command(args):
    data = load_features(args.features, args.layer)
    train, test = partition(data, args.seed)
    probe, heldout = train_probe(data['x'][train], data['y'][train], args.epochs, args.seed,
                                 args.probe_device, validation=(data['x'][test], data['y'][test]))
    vectors = mean_vectors(data['x'], data['y'], train)
    metadata = dict(**data['meta'], train_sample_indices=train.tolist(), test_sample_indices=test.tolist(),
                    split_seed=args.seed, epochs=args.epochs, split='pair-grouped stratified 80/20',
                    vector_normalization='none', heldout_probe_accuracy=heldout)
    save_bundle(args.output, data['meta']['model'], args.layer, probe, vectors, metadata)
    print(f'Saved probe and five vectors to {args.output}; held-out accuracy={heldout:.4f}')


def export_legacy_command(args):
    bundle = load_bundle(args.bundle)
    root = Path(args.output_dir)
    language = bundle['metadata']['language']
    name = 'MFD' if language == 'en' else 'C-MFD'
    probe_dir = root / 'results/layerwise_probe_experiment/saved_probes'
    vector_dir = root / 'moral_vectors'
    layer = bundle['layer']
    targets = [probe_dir / f'probe_{name}_layer_{layer}.pt', probe_dir / 'label_classes.npy']
    targets += [vector_dir / f'layer_{layer}_{f.capitalize()}.pt' for f in FOUNDATIONS]
    if any(path.exists() for path in targets):
        raise FileExistsError('Legacy export would overwrite existing files; choose a fresh output directory')
    probe_dir.mkdir(parents=True, exist_ok=True)
    vector_dir.mkdir(parents=True, exist_ok=True)
    torch.save(bundle['probe'], targets[0])
    np.save(targets[1], np.array(CLASSES))
    for f, path in zip(FOUNDATIONS, targets[2:]):
        torch.save(bundle['vectors'][f], path)
    write_json(root / 'legacy_artifact_metadata.json', dict(model=bundle['model'], layer=layer, metadata=bundle['metadata']))
    print(f'Exported {len(targets)} generated artifacts to {root}')


def balanced_binary(data, class_idx, seed):
    labels = data['y']
    positive = np.flatnonzero(labels == class_idx)
    negative = np.flatnonzero(labels != class_idx)
    if len(negative) > len(positive):
        # Proportional stratification over the other nine moral classes.
        negative, _ = train_test_split(negative, train_size=len(positive),
                                       stratify=labels[negative], random_state=seed)
    count = min(len(positive), len(negative))
    selected = np.concatenate([positive[:count], negative[:count]])
    return data['x'][selected], np.r_[np.ones(count, dtype=int), np.zeros(count, dtype=int)]


def transfer_command(args):
    en, zh = load_features(args.en, args.layer), load_features(args.zh, args.layer)
    for key in ('model', 'input_kind', 'layer_index', 'max_length'):
        if en['meta'][key] != zh['meta'][key]:
            raise ValueError(f'Transfer feature mismatch: {key}')
    results = []
    for i, label in enumerate(CLASSES):
        x_en, y_en = balanced_binary(en, i, args.seed)
        x_zh, y_zh = balanced_binary(zh, i, args.seed)
        directions = {}
        for direction, x, y, target, target_y in [('en_to_zh', x_en, y_en, x_zh, y_zh),
                                                  ('zh_to_en', x_zh, y_zh, x_en, y_en)]:
            clf = LogisticRegression(C=0.1, solver='liblinear', class_weight='balanced', random_state=args.seed)
            cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
            scores = [float(clf.fit(x[tr], y[tr]).score(x[te], y[te])) for tr, te in cv.split(x, y)]
            clf.fit(x, y)
            directions[direction] = dict(source_cv_accuracy=float(np.mean(scores)),
                                         transfer_accuracy=float(clf.score(target, target_y)),
                                         source_samples=len(y), target_samples=len(target_y))
        directions['delta_en_minus_zh'] = (directions['en_to_zh']['transfer_accuracy'] -
                                             directions['zh_to_en']['transfer_accuracy'])
        results.append(dict(label=label, **directions))
    write_json(args.output, dict(layer=args.layer, en=en['meta'], zh=zh['meta'], seed=args.seed,
                                folds=args.folds, results=results))


def check_source(bundle):
    meta = bundle['metadata']
    if meta['input_kind'] != 'sentences' or meta['limit']:
        raise ValueError('This driver requires a bundle trained on the full MVV sentence input')
    name = 'mfd' if meta['language'] == 'en' else 'cmfd'
    path = ROOT / 'datasets' / f'{name}_generated_results.jsonl'
    if sha256(path) != meta['input_sha256']:
        raise ValueError('Input data changed since the bundle was created')
    rows = read_pairs(path)
    test_indices = set(meta['test_sample_indices'])
    return [row for i, row in enumerate(rows) if 2 * i in test_indices]


def steer_command(args):
    bundle = load_bundle(args.bundle, args.model)
    rows = check_source(bundle)
    if args.limit:
        rows = rows[:args.limit]
    model, tokenizer = load_model(args.model, args.device)
    results = []
    for row in rows:
        entry = dict(id=row['id'], category=row['foundation'].capitalize(), context=row['context'], generations={})
        for coeff in args.coefficients:
            with Steering(model, bundle, 'single', coefficient=coeff, foundation=row['foundation']):
                text = generate(model, tokenizer, row['context'], args.chat, args.max_new_tokens)
            entry['generations'][format(coeff, 'g')] = text
        results.append(entry)
    write_json(args.output, results)


def internal_command(args):
    bundle = load_bundle(args.bundle)
    data = load_features(args.features, bundle['layer'])
    if data['meta'] != {k: bundle['metadata'][k] for k in data['meta']}:
        raise ValueError('Features and bundle metadata differ')
    x = torch.as_tensor(data['x'][bundle['metadata']['test_sample_indices']], dtype=torch.float32)
    probe = LinearProbe(x.shape[1])
    probe.load_state_dict(bundle['probe'])
    with torch.inference_mode():
        baseline = probe(x).softmax(-1)
        effects = {f: (probe(x + args.coefficient * bundle['vectors'][f]).softmax(-1) - baseline).mean(0).tolist()
                   for f in FOUNDATIONS}
    write_json(args.output, dict(classes=list(CLASSES), coefficient=args.coefficient,
                                sample_count=len(x), mean_probability_change=effects))


def safety_command(args):
    bundle = load_bundle(args.bundle, args.model)
    model, tokenizer = load_model(args.model, args.device)
    for benchmark in args.benchmarks:
        name = 'harmbench_behaviors_text_all.csv' if benchmark == 'harmbench' else 'xstest_prompts.csv'
        with (ROOT / 'datasets' / name).open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        if benchmark == 'harmbench' and args.functional_category:
            rows = [r for r in rows if r['FunctionalCategory'] == args.functional_category]
        if args.limit:
            rows = rows[:args.limit]
        if not rows:
            raise ValueError(f'No {benchmark} inputs match the requested selection')
        for mode in args.conditions:
            for tau in (args.thresholds if mode == 'amf' else [None]):
                output = []
                with Steering(model, bundle, mode, args.coefficient, 0.2 if tau is None else tau):
                    for row in rows:
                        prompt = row['Behavior'] if benchmark == 'harmbench' else row['prompt']
                        if benchmark == 'harmbench' and args.include_context and row['ContextString']:
                            prompt += '\n\n' + row['ContextString']
                        response = generate(model, tokenizer, prompt, True, args.max_new_tokens)
                        output.append(dict(**row, generated_response=response))
                suffix = f'_tau_{tau:g}' if tau is not None else ''
                path = Path(args.output_dir) / f'{benchmark}_{mode}{suffix}.csv'
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('w', encoding='utf-8', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(output[0]) if output else list(rows[0]))
                    writer.writeheader()
                    writer.writerows(output)
                write_json(path.with_suffix('.metadata.json'), dict(model=args.model, layer=bundle['layer'],
                           condition=mode, tau=tau, coefficient=args.coefficient, count=len(output),
                           functional_category=args.functional_category, include_context=args.include_context))


def oracle_command(args):
    """Decode moral vector activations at a placeholder token."""
    bundle = load_bundle(args.bundle, args.model)
    model, tokenizer = load_model(args.model, args.device)
    from .engine import validate_layer, unpack, replace, input_device
    validate_layer(model, args.injection_layer)
    seed_everything(42)
    # Add a unique single placeholder token, so its exact position is unambiguous.
    if '<ACT>' not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({'additional_special_tokens': ['<ACT>']})
        model.resize_token_embeddings(len(tokenizer))
    act_id = tokenizer.convert_tokens_to_ids('<ACT>')
    results = {}
    for f in FOUNDATIONS:
        prompt = (f"Layer {bundle['layer']}: <ACT> Question: If this activation is a natural language concept, "
                  'the most relevant moral keyword or ethical principle is:')
        if args.chat:
            prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(prompt, return_tensors='pt', return_token_type_ids=False).to(input_device(model))
        mask = inputs['input_ids'].eq(act_id)
        def inject(module, inputs_, output):
            h = unpack(output)
            # With use_cache=True, subsequent decoding calls contain only the new token.
            if h.shape[1] != mask.shape[1]:
                return output
            delta = bundle['vectors'][f].to(device=h.device, dtype=h.dtype)
            return replace(output, h + mask.to(h.device).unsqueeze(-1) * (args.coefficient * delta))
        handle = model.model.layers[args.injection_layer].register_forward_hook(inject)
        try:
            with torch.inference_mode():
                tokens = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                        use_cache=True, pad_token_id=tokenizer.pad_token_id)
        finally:
            handle.remove()
        results[f] = tokenizer.decode(tokens[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    write_json(args.output, dict(implementation='placeholder activation decoding',
                                vector_layer=bundle['layer'], injection_layer=args.injection_layer, results=results))
