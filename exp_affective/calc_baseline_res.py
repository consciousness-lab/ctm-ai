#!/usr/bin/env python3
"""F1 evaluator for multi-agent baseline jsonls.

Reads ``answer[0]`` (unified / debate / orchestra outputs) or re-derives a
majority vote from ``individual_answers`` (ensemble outputs) when available.
Yes/No extraction is robust — it handles free-form analytical responses that
don't start with Yes/No by scanning for verdict markers and the last line.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

_VERDICT_PATTERNS = [
    re.compile(r'\b(?:my\s+)?(?:final\s+)?answer\s*[:=]\s*["\']?(yes|no)\b', re.I),
    re.compile(r'\bverdict\s*[:=]\s*["\']?(yes|no)\b', re.I),
    re.compile(r'\bconclusion\s*[:=]\s*["\']?(yes|no)\b', re.I),
    re.compile(r'\bdetermination\s*[:=]\s*["\']?(yes|no)\b', re.I),
    re.compile(r'\b(?:is|it\s+is)\s+(humorous|sarcastic)\s*[.:!?]', re.I),
    re.compile(r'\bnot\s+(humorous|sarcastic)\s*[.:!?]', re.I),
]


def _starts_with_yes_no(text):
    if not text:
        return None
    t = text.lstrip('#*-_ \t"\'>').lstrip().lower()
    if t.startswith('yes'):
        return 'Yes'
    if t.startswith('no') and not t.startswith('no ' + 'matter'):
        return 'No'
    return None


def extract_verdict(answer):
    """Return 'Yes' / 'No' / None from a free-form answer string."""
    if answer is None:
        return None
    if isinstance(answer, list):
        answer = answer[0] if answer else None
    if not answer or not isinstance(answer, str):
        return None
    text = answer.strip()
    if not text:
        return None

    first = _starts_with_yes_no(text)
    if first is not None:
        return first

    lower = text.lower()
    # Structured verdict markers anywhere in the answer
    for pat in _VERDICT_PATTERNS:
        m = pat.search(text)
        if m:
            grp = m.group(1).lower()
            if grp == 'yes':
                return 'Yes'
            if grp == 'no':
                return 'No'
            # "is humorous/sarcastic" -> Yes; "not humorous/sarcastic" -> No
            prefix = text[max(0, m.start() - 4): m.start()].lower()
            if 'not' in prefix or pat.pattern.startswith(r'\bnot'):
                return 'No'
            return 'Yes'

    # Last non-empty line fallback (judge/controller verdicts usually end there)
    for line in reversed(text.splitlines()):
        ls = _starts_with_yes_no(line)
        if ls is not None:
            return ls
        ll = line.strip().lower()
        if ' yes' in f' {ll}' and ' no' not in f' {ll}':
            return 'Yes'
        if ' no' in f' {ll}' and ' yes' not in f' {ll}':
            return 'No'

    return None


def extract_pred(answer):
    """Yes/No -> 1/0 for the F1 eval loop."""
    v = extract_verdict(answer)
    if v == 'Yes':
        return 1
    if v == 'No':
        return 0
    return None


def ensemble_revote(item):
    """For ensemble-style outputs: re-derive the majority vote from
    ``individual_answers`` using the robust extractor. Returns a verdict
    string ('Yes' / 'No') or None.
    """
    indiv = item.get('individual_answers') or {}
    if not indiv:
        return None
    votes = []
    for _modality, ans in indiv.items():
        v = extract_verdict(ans)
        if v is not None:
            votes.append(v)
    if not votes:
        return None
    counts = Counter(votes)
    most_common = counts.most_common()
    if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
        # tie — fall back to first modality ordering (text > audio > video)
        for pref in ('text', 'audio', 'video'):
            if pref in indiv:
                v = extract_verdict(indiv[pref])
                if v is not None:
                    return v
    return most_common[0][0]


def label_to_binary(label, dataset):
    if dataset == 'mustard':
        if label is True:
            return 1
        if label is False:
            return 0
    elif dataset == 'urfunny':
        if label in (0, 1):
            return int(label)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('--dataset', '-d', choices=('mustard', 'urfunny'), default=None)
    args = parser.parse_args()

    path = Path(args.file)
    if args.dataset is None:
        name = path.name.lower()
        if 'mustard' in name:
            args.dataset = 'mustard'
        elif 'urfunny' in name:
            args.dataset = 'urfunny'
        else:
            raise SystemExit('Specify --dataset')

    y_true, y_pred = [], []
    skipped = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            for _k, item in data.items():
                tv = label_to_binary(item.get('label'), args.dataset)

                # Ensemble-style: re-derive majority vote from individual_answers
                pv = None
                if 'individual_answers' in item and item['individual_answers']:
                    revote = ensemble_revote(item)
                    if revote == 'Yes':
                        pv = 1
                    elif revote == 'No':
                        pv = 0

                # Fallback / non-ensemble: parse top-level answer field
                if pv is None:
                    pv = extract_pred(item.get('answer'))

                if tv is None or pv is None:
                    skipped += 1
                    continue
                y_true.append(tv)
                y_pred.append(pv)

    n = len(y_true)
    if n == 0:
        print(f'{path.name}: no valid samples, skipped={skipped}')
        return

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    pos = sum(y_true)
    neg = n - pos
    acc = (tp + tn) / n
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1_pos = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    neg_prec = tn / (tn + fn) if (tn + fn) else 0.0
    neg_rec = tn / (tn + fp) if (tn + fp) else 0.0
    f1_neg = 2 * neg_prec * neg_rec / (neg_prec + neg_rec) if (neg_prec + neg_rec) else 0.0
    f1_macro = (f1_pos + f1_neg) / 2
    f1_weighted = (f1_pos * pos + f1_neg * neg) / n if n else 0.0

    print(f'{path.name} | dataset={args.dataset} | n={n} skipped={skipped}')
    print(f'  pos={pos} neg={neg} | TP={tp} TN={tn} FP={fp} FN={fn}')
    print(f'  Acc={acc*100:.1f}%  P={prec:.3f}  R={rec:.3f}')
    print(f'  F1_pos={f1_pos:.4f}  F1_neg={f1_neg:.4f}')
    print(f'  F1_macro={f1_macro:.4f}  F1_weighted={f1_weighted:.4f}')


if __name__ == '__main__':
    main()
