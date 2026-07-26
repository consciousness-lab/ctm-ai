"""Aggregate T2 StableToolBench baseline results into the method-comparison table.

Reads:
    <results_dir>/<method>/<test_set>/<qid>_<suffix>.json   (each with a `metrics` block)
and optional pass-rate JSONs from eval_ctm_toolbench.py:
    <eval_dir>/<method>/<test_set>_<model_name>.json

Prints a Markdown table: rows = metrics (Pass Rate, Δt, #Model Calls, #Tool Calls,
Total Tokens), columns = methods. Pass rate averaged over the given test sets.

Usage:
    python aggregate_baselines.py --results_dir ./results_t2 --eval_dir ./eval_t2 \
        --methods react ensemble moa debate orchestra metagpt autogen ctm \
        --test_sets G2_category G2_instruction G3_instruction
"""
import argparse
import glob
import json
import os
from collections import defaultdict

LABELS = {
    'react': 'Unified/ReAct',
    'ensemble': 'Ensemble',
    'orchestra': 'Orchestra',
    'debate': 'Debate',
    'metagpt': 'MetaGPT',
    'autogen': 'AutoGen',
    'moa': 'MoA',
    'ctm': 'CTM-AI',
}
METRIC_KEYS = ['num_model_calls', 'num_tool_calls', 'total_tokens', 'latency_seconds']


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return sum(xs) / len(xs) if xs else 0.0


def load_metrics(results_dir, method, test_set):
    files = glob.glob(os.path.join(results_dir, method, test_set, '*.json'))
    acc = defaultdict(list)
    n = 0
    for f in files:
        try:
            d = json.load(open(f))
        except (OSError, ValueError):
            continue
        m = d.get('metrics') or {}
        if not m:
            continue
        n += 1
        for k in METRIC_KEYS:
            acc[k].append(m.get(k))
    return n, {k: _mean(acc[k]) for k in METRIC_KEYS}


def load_pass_rate(eval_dir, method, test_set, model_name):
    if not eval_dir:
        return None
    path = os.path.join(eval_dir, method, f'{test_set}_{model_name}.json')
    if not os.path.exists(path):
        return None
    try:
        d = json.load(open(path))
    except (OSError, ValueError):
        return None
    sc = []
    for _qid, rec in d.items():
        for _run, st in (rec.get('is_solved') or {}).items():
            sc.append(
                1.0 if st.endswith('Solved') and 'Un' not in st
                else 0.5 if st.endswith('Unsure') else 0.0
            )
    return 100.0 * _mean(sc) if sc else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', required=True)
    ap.add_argument('--eval_dir', default=None)
    ap.add_argument('--model_name', default='baseline')
    ap.add_argument('--methods', nargs='+', default=list(LABELS))
    ap.add_argument(
        '--test_sets', nargs='+',
        default=['G2_category', 'G2_instruction', 'G3_instruction'],
    )
    args = ap.parse_args()

    data = {}
    for m in args.methods:
        per_set_pr, mets, ns = [], defaultdict(list), []
        for ts in args.test_sets:
            n, mm = load_metrics(args.results_dir, m, ts)
            pr = load_pass_rate(args.eval_dir, m, ts, args.model_name)
            ns.append(n)
            if pr is not None:
                per_set_pr.append(pr)
            for k in METRIC_KEYS:
                mets[k].append(mm[k])
        data[m] = {
            'pass': (sum(per_set_pr) / len(per_set_pr)) if per_set_pr else None,
            'mets': {k: _mean(mets[k]) for k in METRIC_KEYS},
            'n': sum(ns),
        }

    methods = [m for m in args.methods if m in data]
    hdr = '| Metric | ' + ' | '.join(LABELS.get(m, m) for m in methods) + ' |'
    print('\n### T2 — StableToolBench (avg over ' + ', '.join(args.test_sets) + ')\n')
    print(hdr)
    print('|' + '---|' * (len(methods) + 1))

    def row(name, fn):
        return '| ' + name + ' | ' + ' | '.join(fn(data[m]) for m in methods) + ' |'

    print(row('Pass Rate ↑', lambda d: f"{d['pass']:.1f}" if d['pass'] is not None else '·'))
    print(row('Δt (s) ↓', lambda d: f"{d['mets']['latency_seconds']:.1f}"))
    print(row('# Model Calls ↓', lambda d: f"{d['mets']['num_model_calls']:.1f}"))
    print(row('# Tool Calls ↓', lambda d: f"{d['mets']['num_tool_calls']:.1f}"))
    print(row('Total Tokens ↓', lambda d: f"{d['mets']['total_tokens']:.0f}"))
    print(row('n (queries)', lambda d: str(d['n'])))


if __name__ == '__main__':
    main()
