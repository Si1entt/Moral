"""Validated input readers; the paper uses the first minimal pair per seed record."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

FOUNDATIONS = ('care', 'fairness', 'loyalty', 'authority', 'sanctity')
CLASSES = tuple(sorted(f'{f}.{p}' for f in FOUNDATIONS for p in ('virtue', 'vice')))
ALIASES = dict(care='care', fair='fairness', loya='loyalty', auth='authority', sanc='sanctity')
ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_pairs(path):
    records = []
    with Path(path).open(encoding='utf-8-sig') as stream:
        for lineno, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            foundation, polarity = row['category'].split('.')
            foundation = ALIASES.get(foundation, foundation)
            if foundation not in FOUNDATIONS or polarity not in ('virtue', 'vice'):
                raise ValueError(f'{path}:{lineno}: invalid category')
            pair = row['pairs'][0]
            for key in ('context', 'virtue_suffix', 'vice_suffix'):
                if not isinstance(pair.get(key), str) or not pair[key].strip():
                    raise ValueError(f'{path}:{lineno}: missing {key}')
            records.append(dict(id=lineno, foundation=foundation, target_word=row['target_word'],
                                seed_category=row['category'], **pair))
    if not records:
        raise ValueError(f'Empty dataset: {path}')
    return records


def sentences(pairs):
    """Keep each pair adjacent. Labels come from suffix polarity, not seed polarity."""
    texts, labels, pair_ids = [], [], []
    for row in pairs:
        for polarity in ('virtue', 'vice'):
            texts.append(row['context'].strip() + ' ' + row[polarity + '_suffix'].strip())
            labels.append(CLASSES.index(f"{row['foundation']}.{polarity}"))
            pair_ids.append(row['id'])
    return texts, labels, pair_ids


def read_dictionary(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        result = []
        for row in reader:
            word = row.get('word', row.get('chinese'))
            label = row.get('foundation', row.get('foundation_labeled'))
            foundation, polarity = label.split('.')
            label = f'{ALIASES.get(foundation, foundation)}.{polarity}'
            if label not in CLASSES or not word:
                raise ValueError(f'Invalid dictionary record: {row}')
            result.append((word, label))
    return result


def input_audit():
    stats = {}
    for language, count in [('en', 4208), ('zh', 11014)]:
        name = 'mfd' if language == 'en' else 'cmfd'
        path = ROOT / 'datasets' / f'{name}_generated_results.jsonl'
        pairs = read_pairs(path)
        texts, labels, _ = sentences(pairs)
        if len(texts) != count:
            raise ValueError(f'{language}: paper expects {count} samples, got {len(texts)}')
        stats[language] = dict(records=len(pairs), samples=len(texts), sha256=sha256(path),
                               per_class=dict(Counter(CLASSES[i] for i in labels)))
    for name in ('MFD.csv', 'C-MFD.csv'):
        stats[name] = dict(records=len(read_dictionary(ROOT / 'datasets' / name)))
    for name, required in [('harmbench_behaviors_text_all.csv', {'Behavior', 'BehaviorID', 'FunctionalCategory'}),
                           ('xstest_prompts.csv', {'prompt', 'label', 'id'})]:
        with (ROOT / 'datasets' / name).open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            if not required <= set(reader.fieldnames):
                raise ValueError(f'{name}: missing columns')
            rows = list(reader)
        stats[name] = dict(records=len(rows), labels=dict(Counter(r.get('label', '') for r in rows)))
    utility = ROOT / 'datasets' / 'utility'
    sources = json.loads((utility / 'SOURCES.json').read_text(encoding='utf-8'))
    for name, expected in [('arc', 299), ('sciq', 1000), ('math', 500)]:
        path = utility / f'{name}.jsonl'
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        if len(rows) != expected or sha256(path) != sources[name]['sha256']:
            raise ValueError(f'{name}: utility input count or checksum mismatch')
        required = {'arc': {'question', 'answerKey'}, 'sciq': {'question', 'correct_answer'},
                    'math': {'problem', 'answer'}}[name]
        if not all(required <= row.keys() for row in rows):
            raise ValueError(f'{name}: utility input fields missing')
        stats[f'utility/{name}'] = dict(records=len(rows), source=sources[name]['source'])
    return stats
