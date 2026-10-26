"""Run from the repository root. Outputs are created only when a command runs."""
import argparse
import json
import os
import sys

sys.dont_write_bytecode = True


def parser():
    p = argparse.ArgumentParser(description='Moral Vector experiments and dataset utilities')
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('check-data', help='Validate input schemas and the exact paper MVV counts')

    def output(cmd, default):
        cmd.add_argument('--output', default=default)

    def model(cmd):
        cmd.add_argument('--model', default=os.environ.get('MODEL_PATH', 'meta-llama/Llama-3.1-8B-Instruct'))
        cmd.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:0')

    def training(cmd):
        cmd.add_argument('--epochs', type=int, default=30)
        cmd.add_argument('--seed', type=int, default=42)
        cmd.add_argument('--probe-device', default='cpu')

    def steering(cmd):
        model(cmd)
        cmd.add_argument('--bundle', required=True)
        cmd.add_argument('--coefficient', type=float, default=2.0)
        cmd.add_argument('--max-new-tokens', type=int, default=150)
        cmd.add_argument('--limit', type=int, help='Smoke test limit; omitted means all selected inputs')

    f = sub.add_parser('features', help='Extract last-token decoder-block features')
    model(f)
    f.add_argument('--language', choices=['en', 'zh'], required=True)
    f.add_argument('--input-kind', choices=['sentences', 'dictionary'], default='sentences')
    f.add_argument('--layers', default='17', help='Comma-separated zero-based blocks, or all')
    f.add_argument('--batch-size', type=int, default=8)
    f.add_argument('--max-length', type=int, default=128)
    f.add_argument('--seed', type=int, default=42)
    f.add_argument('--limit', type=int, help='Limit seed records, for smoke tests only')
    f.add_argument('--output', required=True)

    cv = sub.add_parser('probe', help='Layerwise 5-fold 10-class probing (paper Appendix B)')
    cv.add_argument('--features', required=True)
    cv.add_argument('--folds', type=int, default=5)
    cv.add_argument('--score-mode', choices=['best', 'final'], default='best',
                    help='best matches historical maximum-across-epochs reporting; final avoids selecting on test folds')
    training(cv)
    output(cv, 'runs/probe.json')

    fit = sub.add_parser('fit', help='Train a deployable probe and extract five raw mean-difference vectors')
    fit.add_argument('--features', required=True)
    fit.add_argument('--layer', type=int, default=17)
    training(fit)
    output(fit, 'runs/bundle.pt')

    export = sub.add_parser('export-legacy', help='Export a generated bundle to the filenames expected by legacy scripts')
    export.add_argument('--bundle', required=True)
    export.add_argument('--output-dir', required=True, help='Use . to create legacy runtime paths in the repository root')

    transfer = sub.add_parser('transfer', help='Ten binary probes with bidirectional cross-language transfer')
    transfer.add_argument('--en', required=True)
    transfer.add_argument('--zh', required=True)
    transfer.add_argument('--layer', type=int, default=17)
    transfer.add_argument('--seed', type=int, default=42)
    transfer.add_argument('--folds', type=int, default=5)
    output(transfer, 'runs/transfer.json')

    gen = sub.add_parser('steer', help='Generate five coefficients on held-out MVV contexts')
    steering(gen)
    gen.add_argument('--coefficients', nargs='+', type=float, default=[-2, -1, 0, 1, 2])
    gen.add_argument('--chat', action='store_true', help='Default uses raw contexts, matching continuation experiments')
    output(gen, 'runs/steering_results.json')

    internal = sub.add_parser('internal', help='Five-by-ten probe probability changes on held-out features')
    internal.add_argument('--bundle', required=True)
    internal.add_argument('--features', required=True)
    internal.add_argument('--coefficient', type=float, default=2.0)
    output(internal, 'runs/internal.json')

    safety = sub.add_parser('safety', help='Baseline, mean and AMF generation; supports threshold sweeps')
    steering(safety)
    safety.add_argument('--benchmarks', nargs='+', choices=['harmbench', 'xstest'], default=['harmbench', 'xstest'])
    safety.add_argument('--conditions', nargs='+', choices=['baseline', 'mean', 'amf'], default=['baseline', 'mean', 'amf'])
    safety.add_argument('--thresholds', nargs='+', type=float, default=[0.2])
    safety.add_argument('--functional-category', help='Optional HarmBench FunctionalCategory filter')
    safety.add_argument('--include-context', action='store_true', help='Append ContextString; historical driver used Behavior alone')
    safety.add_argument('--output-dir', default='runs/safety')

    prep = sub.add_parser('prepare-utility', help='Download only the ARC validation, SciQ validation and MATH-500 test inputs')
    prep.add_argument('--benchmarks', nargs='+', choices=['arc', 'sciq', 'math'], default=['arc', 'sciq', 'math'])
    prep.add_argument('--direct', action='store_true', help='Use direct connections instead of environment proxy settings')
    prep.add_argument('--force', action='store_true')
    prep.add_argument('--math-source', choices=['huggingface', 'simple-evals'], default='huggingface',
                      help='Explicit alternative mirror requires documenting the changed source')

    utility = sub.add_parser('utility', help='Historical short-answer utility evaluation')
    steering(utility)
    utility.add_argument('--benchmarks', nargs='+', choices=['arc', 'sciq', 'math'], default=['arc', 'sciq', 'math'])
    utility.add_argument('--conditions', nargs='+', choices=['baseline', 'mean', 'amf'], default=['baseline', 'mean', 'amf'])
    utility.add_argument('--threshold', type=float, default=0.2)
    output(utility, 'runs/utility.json')

    oracle = sub.add_parser('oracle', help='Decode moral vector activations through placeholder injection')
    steering(oracle)
    oracle.add_argument('--injection-layer', type=int, default=1)
    oracle.add_argument('--chat', action='store_true')
    output(oracle, 'runs/oracle.json')

    judge = sub.add_parser('judge', help='API evaluation; model must be explicitly supplied')
    judge.add_argument('--input', required=True)
    judge.add_argument('--task', choices=['steering', 'harmbench', 'xstest'], required=True)
    judge.add_argument('--judge-model', required=True)
    output(judge, 'runs/judged.json')

    debate = sub.add_parser('debate', help='Two adversarial rounds for both baseline and positive steering')
    steering(debate)
    debate.add_argument('--input', required=True, help='Output of steer, including generations 0 and 2')
    debate.add_argument('--attacker-model', required=True)
    debate.add_argument('--judge-model', required=True)
    output(debate, 'runs/debate.json')

    data = sub.add_parser('generate-data', help='Regenerate minimal pairs via an OpenAI-compatible API')
    data.add_argument('--language', choices=['en', 'zh'], required=True)
    data.add_argument('--generator-model', required=True)
    data.add_argument('--limit', type=int)
    output(data, 'runs/generated_mvv.jsonl')
    return p


def main():
    args = parser().parse_args()
    # Basic input checks happen before expensive imports or model loading.
    for key in ('limit', 'epochs', 'folds', 'batch_size', 'max_length', 'max_new_tokens'):
        if hasattr(args, key) and getattr(args, key) is not None and getattr(args, key) <= 0:
            raise ValueError(f'--{key.replace("_", "-")} must be positive')
    if args.command == 'check-data':
        from moral_vector.data import input_audit
        print(json.dumps(input_audit(), ensure_ascii=False, indent=2))
    elif args.command in ('prepare-utility', 'utility'):
        from moral_vector.utility import prepare_utility, utility_command
        (prepare_utility if args.command == 'prepare-utility' else utility_command)(args)
    elif args.command in ('judge', 'debate', 'generate-data'):
        from moral_vector.api import judge_command, debate_command, generate_data_command
        {'judge': judge_command, 'debate': debate_command, 'generate-data': generate_data_command}[args.command](args)
    else:
        from moral_vector import experiments
        getattr(experiments, args.command.replace('-', '_') + '_command')(args)


if __name__ == '__main__':
    main()
