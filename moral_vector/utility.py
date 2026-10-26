"""Utility data preparation and the historical short-answer scoring protocol."""
import io
import json
import urllib.request
import zipfile
from pathlib import Path

from .data import ROOT, sha256

UTILITY = ROOT / 'datasets' / 'utility'
SOURCES = {
    'sciq': 'https://ai2-public-datasets.s3.amazonaws.com/sciq/SciQ.zip',
    'arc': 'https://ai2-public-datasets.s3.amazonaws.com/arc/ARC-V1-Feb2018.zip',
    'math': 'https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl',
}
MATH_SIMPLE_EVALS = 'https://openaipublic.blob.core.windows.net/simple-evals/math_500_test.csv'


def download(url, direct=False):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if direct else urllib.request.build_opener()
    with opener.open(url, timeout=60) as response:
        return response.read()


class RemoteZip(io.RawIOBase):
    """Read a ZIP member via HTTP ranges without downloading the large ARC corpus."""
    def __init__(self, url, direct=False):
        self.url = url
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if direct else urllib.request.build_opener()
        with self.opener.open(urllib.request.Request(url, method='HEAD'), timeout=60) as response:
            self.length = int(response.headers['Content-Length'])
        self.position = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.length + offset
        return self.position

    def read(self, count=-1):
        count = self.length - self.position if count < 0 else min(count, self.length - self.position)
        if count <= 0:
            return b''
        request = urllib.request.Request(self.url, headers={'Range': f'bytes={self.position}-{self.position + count - 1}'})
        with self.opener.open(request, timeout=60) as response:
            if response.status != 206:
                raise RuntimeError('Remote server does not support partial ZIP downloads')
            content = response.read(count)
        self.position += len(content)
        return content


def boxed_answer(solution):
    """Extract the last balanced \\boxed{...} from the alternative MATH solution."""
    marker = '\\boxed{'
    start = solution.rfind(marker)
    if start < 0:
        raise ValueError('MATH solution has no boxed answer')
    start += len(marker)
    depth = 1
    for end in range(start, len(solution)):
        depth += (solution[end] == '{') - (solution[end] == '}')
        if depth == 0:
            return solution[start:end]
    raise ValueError('Unbalanced boxed answer')


def prepare_utility(args):
    UTILITY.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((UTILITY / 'SOURCES.json').read_text(encoding='utf-8')) if (UTILITY / 'SOURCES.json').exists() else {}
    failures = []
    for name in args.benchmarks:
        destination = UTILITY / f'{name}.jsonl'
        if destination.exists() and not args.force:
            print(f'Already present: {destination}')
            continue
        try:
            source = MATH_SIMPLE_EVALS if name == 'math' and args.math_source == 'simple-evals' else SOURCES[name]
            if name == 'math':
                payload = download(source, args.direct)
                if args.math_source == 'simple-evals':
                    import csv
                    originals = list(csv.DictReader(io.StringIO(payload.decode('utf-8-sig'))))
                    rows = [dict(problem=row['Question'], solution=row['Answer'], answer=boxed_answer(row['Answer'])) for row in originals]
                else:
                    rows = [json.loads(line) for line in payload.decode('utf-8').splitlines() if line.strip()]
                split = 'test'
            else:
                with zipfile.ZipFile(RemoteZip(source, args.direct)) as archive:
                    suffix = 'valid.json' if name == 'sciq' else 'ARC-Challenge-Dev.jsonl'
                    member = next(n for n in archive.namelist() if n.endswith(suffix))
                    content = archive.read(member).decode('utf-8')
                    rows = json.loads(content) if name == 'sciq' else [json.loads(line) for line in content.splitlines() if line.strip()]
                split = 'validation'
            expected = {'arc': 299, 'sciq': 1000, 'math': 500}[name]
            if len(rows) != expected:
                raise ValueError(f'{name}: expected {expected} records, got {len(rows)}')
            text = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows)
            destination.write_text(text, encoding='utf-8')
            metadata[name] = dict(source=source, split=split, records=len(rows), sha256=sha256(destination),
                                  transformation=('final answer extracted from last balanced boxed expression' if name == 'math' and args.math_source == 'simple-evals' else 'JSONL serialization of original records'))
            print(f'Prepared {name}: {len(rows)} input records')
        except Exception as exc:
            failures.append(name)
            print(f'{name}: {type(exc).__name__}: {exc}')
    (UTILITY / 'SOURCES.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    if failures:
        raise RuntimeError('Could not prepare: ' + ', '.join(failures))


def utility_command(args):
    from .engine import load_bundle, load_model, Steering, generate
    from .experiments import write_json
    bundle = load_bundle(args.bundle, args.model)
    datasets = {}
    # Check all inputs before loading the expensive model.
    for name in args.benchmarks:
        path = UTILITY / f'{name}.jsonl'
        if not path.exists():
            raise FileNotFoundError(f'{path}: run python run.py prepare-utility first')
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        datasets[name] = rows[:args.limit] if args.limit else rows
    model, tokenizer = load_model(args.model, args.device)
    results = []
    for condition in args.conditions:
        with Steering(model, bundle, condition, args.coefficient, args.threshold):
            for name, rows in datasets.items():
                correct = 0
                details = []
                for row in rows:
                    if name == 'arc':
                        # Official archive nests fields under question; HF flattens them.
                        question = row['question']
                        choices = question['choices']
                        labels = [choice['label'] for choice in choices]
                        options = '\n'.join(f"{c['label']}) {c['text']}" for c in choices)
                        prompt = f"Question: {question['stem']}\n{options}\nWhich is the correct option? Only output the option letter.\nAnswer:"
                        answer = row['answerKey']
                        limit = 5
                    elif name == 'sciq':
                        prompt = f"Question: {row['question']}\nProvide the exact correct answer:\nAnswer:"
                        answer = row['correct_answer']
                        limit = 10
                    else:
                        prompt = f"Solve the following math problem. Only output the final numerical answer.\nProblem: {row['problem']}\nFinal Answer:"
                        answer = str(row['answer']).strip()
                        limit = 25
                    pred = generate(model, tokenizer, prompt, True, limit)
                    if name == 'arc':
                        pred_label = next((c for c in pred if c in labels), '')
                        hit = pred_label == answer
                    else:
                        hit = answer.lower() in pred.lower()
                    correct += int(hit)
                    details.append(dict(answer=answer, prediction=pred, correct=hit))
                results.append(dict(condition=condition, benchmark=name, samples=len(rows),
                                    accuracy=correct / len(rows), predictions=details))
    write_json(args.output, dict(model=args.model, layer=bundle['layer'], coefficient=args.coefficient,
                                threshold=args.threshold, scoring='historical short-output substring/option heuristic', results=results))
