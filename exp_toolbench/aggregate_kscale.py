"""Aggregate processor-count (K) scaling results into the T1/T3 tables.

Reads a results tree laid out as:
    <results_dir>/K<K>/<test_set>/<qid>_ctm.json    (each with a `metrics` block)
and, optionally, per-(K,test_set) pass-rate JSONs produced by eval_ctm_toolbench.py:
    <eval_dir>/K<K>/<test_set>_<model_name>.json

Emits, per K, the averaged cost metrics and (if available) pass rate for each
test set, and prints Markdown tables in the T1 and T3 column layouts.

Usage:
    python aggregate_kscale.py --results_dir ./results_kscale \
        [--eval_dir ./eval_kscale --model_name qwen3_nonthinking_ctm] \
        [--test_sets G2_category G2_instruction G3_instruction]
"""
import argparse
import glob
import json
import os
from collections import defaultdict

# test_set -> short column label used in the paper tables
LABELS = {
    'G2_category': 'I2-Cat',
    'G2_instruction': 'I2-Inst',
    'G3_instruction': 'I3-Inst',
}

METRIC_KEYS = [
    'num_model_calls',
    'num_tool_calls',
    'total_tokens',
    'latency_seconds',
    'active_links',
    'num_links_added',
    'num_processors',
]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def load_metrics(results_dir, k, test_set):
    """Average the per-query metrics for one (K, test_set)."""
    files = glob.glob(os.path.join(results_dir, f'K{k}', test_set, '*_ctm.json'))
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
        for key in METRIC_KEYS:
            acc[key].append(m.get(key))
    return n, {key: _mean(acc[key]) for key in METRIC_KEYS}


def load_pass_rate(eval_dir, k, test_set, model_name):
    """Read solve/pass rate (%) from an eval_ctm_toolbench.py output, if present.

    That file is {qid: {is_solved: {run: 'AnswerStatus.X'}}}. Solved=1,
    Unsure=0.5, Unsolved=0; averaged over queries and runs.
    """
    if not eval_dir:
        return None
    path = os.path.join(eval_dir, f'K{k}', f'{test_set}_{model_name}.json')
    if not os.path.exists(path):
        return None
    try:
        label_cnt = json.load(open(path))
    except (OSError, ValueError):
        return None
    scores = []
    for qid, rec in label_cnt.items():
        solved = (rec or {}).get('is_solved', {})
        for _run, status in solved.items():
            if status == 'AnswerStatus.Solved':
                scores.append(1.0)
            elif status == 'AnswerStatus.Unsure':
                scores.append(0.5)
            else:
                scores.append(0.0)
    return 100.0 * _mean(scores) if scores else None


def _fmt(x, nd=2):
    return f'{x:.{nd}f}' if isinstance(x, (int, float)) else '-'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', required=True)
    ap.add_argument('--eval_dir', default=None, help='optional pass-rate dir')
    ap.add_argument('--model_name', default='qwen3_nonthinking_ctm')
    ap.add_argument('--ks', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32])
    ap.add_argument(
        '--test_sets',
        nargs='+',
        default=['G2_category', 'G2_instruction', 'G3_instruction'],
    )
    args = ap.parse_args()

    # collect
    rows = {}  # k -> {test_set -> (n, metrics, pass_rate)}
    for k in args.ks:
        rows[k] = {}
        for ts in args.test_sets:
            n, metrics = load_metrics(args.results_dir, k, ts)
            pr = load_pass_rate(args.eval_dir, k, ts, args.model_name)
            rows[k][ts] = (n, metrics, pr)

    def avg_over_sets(k, mkey):
        return _mean([rows[k][ts][1].get(mkey) for ts in args.test_sets])

    # ---- T1 layout: Pass rate per set + Avg API/Latency/Active Links ----
    print('\n### T1 — K ∈ {1,2,4,8,16} (Pass Rate per set, Avg API Calls, Latency, Active Links)\n')
    hdr = '| K | ' + ' | '.join(f'{LABELS[t]} Pass↑' for t in args.test_sets)
    hdr += ' | Avg API Calls↓ | Avg Latency↓ | Avg Active Links |'
    print(hdr)
    print('|' + '---|' * (len(args.test_sets) + 4))
    for k in args.ks:
        prs = []
        for ts in args.test_sets:
            pr = rows[k][ts][2]
            prs.append(f'{pr:.1f}' if pr is not None else '·')
        api = avg_over_sets(k, 'num_tool_calls')
        lat = avg_over_sets(k, 'latency_seconds')
        links = avg_over_sets(k, 'active_links')
        print(
            f'| {k} | '
            + ' | '.join(prs)
            + f' | {_fmt(api)} | {_fmt(lat)} | {_fmt(links)} |'
        )

    # ---- T3 layout: Pass rate per set + Latency, Model Calls, Active Links ----
    print('\n### T3 — K ∈ {2,4,8,16,32} (Pass Rate per set, Latency, Model Calls, Active Links)\n')
    hdr = '| K | ' + ' | '.join(f'{LABELS[t]}↑' for t in args.test_sets)
    hdr += ' | Latency↓ | # Model Calls↓ | # Active Links |'
    print(hdr)
    print('|' + '---|' * (len(args.test_sets) + 4))
    for k in args.ks:
        prs = []
        for ts in args.test_sets:
            pr = rows[k][ts][2]
            prs.append(f'{pr:.1f}' if pr is not None else '·')
        lat = avg_over_sets(k, 'latency_seconds')
        mc = avg_over_sets(k, 'num_model_calls')
        links = avg_over_sets(k, 'active_links')
        print(
            f'| {k} | '
            + ' | '.join(prs)
            + f' | {_fmt(lat)} | {_fmt(mc)} | {_fmt(links)} |'
        )

    # ---- coverage / per-set detail ----
    print('\n### Coverage (queries with metrics per K × test set)\n')
    print('| K | ' + ' | '.join(args.test_sets) + ' | avg #proc |')
    print('|' + '---|' * (len(args.test_sets) + 2))
    for k in args.ks:
        counts = ' | '.join(str(rows[k][ts][0]) for ts in args.test_sets)
        nproc = avg_over_sets(k, 'num_processors')
        print(f'| {k} | {counts} | {_fmt(nproc)} |')


if __name__ == '__main__':
    main()
