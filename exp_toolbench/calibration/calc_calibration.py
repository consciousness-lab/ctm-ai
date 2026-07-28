"""Step 3 — compute the confidence-calibration table.

Joins harvested chunks (estimator ① self_confidence) with the enriched
label/judge/logprob values, then per estimator reports:
AUROC ↑, AUPRC ↑, ECE ↓ (15 equal-mass bins), Brier ↓, Avg. rank ↓.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _load_val(path):
    d = {}
    if os.path.exists(path):
        for l in open(path):
            try:
                r = json.loads(l)
                if r.get('value') is not None:
                    d[(r['query_id'], r['processor_name'])] = float(r['value'])
            except Exception:  # noqa: BLE001
                pass
    return d


def ece(conf, y, n_bins=15):
    """Expected Calibration Error with equal-mass (adaptive) bins."""
    conf, y = np.asarray(conf), np.asarray(y)
    order = np.argsort(conf)
    conf, y = conf[order], y[order]
    e, N = 0.0, len(conf)
    for b in np.array_split(np.arange(N), n_bins):
        if len(b) == 0:
            continue
        e += abs(conf[b].mean() - y[b].mean()) * (len(b) / N)
    return e


def avg_rank(rows, key):
    """Per query, rank chunks by confidence desc; return mean normalized rank
    (rank/n, 1=top) of the CORRECT (y=1) chunks. Lower = better."""
    by_q = defaultdict(list)
    for r in rows:
        by_q[r['query_id']].append(r)
    ranks = []
    for q, rs in by_q.items():
        if not any(r['y'] == 1 for r in rs) or len(rs) < 2:
            continue
        srt = sorted(rs, key=lambda r: -r[key])
        n = len(srt)
        for i, r in enumerate(srt):
            if r['y'] == 1:
                ranks.append((i + 1) / n)
    return float(np.mean(ranks)) if ranks else float('nan')


def top1_acc(rows, key):
    """Per query with >=1 correct chunk, is the TOP-scored chunk correct?
    This is the signal CTM's up-tree competition actually uses. Higher = better."""
    by_q = defaultdict(list)
    for r in rows:
        by_q[r['query_id']].append(r)
    hit, tot = 0, 0
    for q, rs in by_q.items():
        if not any(r['y'] == 1 for r in rs) or len(rs) < 2:
            continue
        top = max(rs, key=lambda r: r[key])
        tot += 1
        hit += int(top['y'] == 1)
    return (hit / tot) if tot else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='calibration')
    ap.add_argument('--test_sets', nargs='+',
                    default=['G2_category', 'G2_instruction', 'G3_instruction'])
    ap.add_argument('--drop_unsure', action='store_true', default=True)
    args = ap.parse_args()

    rows = []
    for S in args.test_sets:
        chunks = f'{args.dir}/chunks/{S}.jsonl'
        if not os.path.exists(chunks):
            continue
        lab = _load_val(f'{args.dir}/enrich/{S}.label.jsonl')
        jud = _load_val(f'{args.dir}/enrich/{S}.judge.jsonl')
        lp = _load_val(f'{args.dir}/enrich/{S}.logprob.jsonl')
        sd = _load_val(f'{args.dir}/enrich/{S}.self_decoupled.jsonl')
        for l in open(chunks):
            r = json.loads(l)
            k = (r['query_id'], r['processor_name'])
            if k not in lab:
                continue
            y = lab[k]
            if args.drop_unsure and y == 0.5:
                continue
            sdv = sd.get(k)
            rows.append({
                'query_id': r['query_id'],
                'y': int(y >= 0.5),
                # CTM's actual selection signal = weight (r + c + 0.2s), max 2.2;
                # normalize to [0,1] (monotone → ranking metrics unaffected).
                'self': float(r.get('self_weight', 0.0)) / 2.2,
                'judge': jud.get(k),
                'logprob': lp.get(k),
                # self_decoupled weight is also r + c + 0.2s in [0,2.2] → /2.2.
                'self_decoupled': (sdv / 2.2) if sdv is not None else None,
            })

    ests = [('Self-reported score (CTM weight, coupled)', 'self'),
            ('Self-decoupled rubric (separate forward)', 'self_decoupled'),
            ('External LLM-as-a-judge (decoupled Qwen3-8B)', 'judge'),
            ('Self-prompt Yes/No logprobs', 'logprob')]
    npos = sum(r['y'] for r in rows)
    print(f'\nchunks with label: {len(rows)} | positive (Solved): {npos} '
          f'({100*npos/max(len(rows),1):.1f}%)\n')
    print(f"| {'Confidence estimator':44s} | {'AUROC↑':>6s} | {'AUPRC↑':>6s} | "
          f"{'ECE↓':>5s} | {'Brier↓':>6s} | {'AvgRank↓':>8s} | {'Top1↑':>6s} |")
    print('|' + '-' * 46 + '|' + '-' * 8 + '|' + '-' * 8 + '|' + '-' * 7 + '|'
          + '-' * 8 + '|' + '-' * 10 + '|' + '-' * 8 + '|')
    for name, key in ests:
        sub = [r for r in rows if r[key] is not None]
        if not sub:
            print(f'| {name:44s} |   -    |   -    |   -  |   -    |    -     |   -    |')
            continue
        y = np.array([r['y'] for r in sub])
        c = np.array([r[key] for r in sub])
        try:
            auroc = roc_auc_score(y, c) if len(set(y)) > 1 else float('nan')
            auprc = average_precision_score(y, c) if len(set(y)) > 1 else float('nan')
        except Exception:  # noqa: BLE001
            auroc = auprc = float('nan')
        e = ece(c, y)
        brier = float(np.mean((c - y) ** 2))
        ar = avg_rank(sub, key)
        t1 = top1_acc(sub, key)
        print(f'| {name:44s} | {auroc:6.3f} | {auprc:6.3f} | {e:5.3f} | '
              f'{brier:6.3f} | {ar:8.3f} | {t1:6.3f} |')
    print()


if __name__ == '__main__':
    main()
