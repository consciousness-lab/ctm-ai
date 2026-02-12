#!/usr/bin/env python3
"""
计算 CTM 输出的分类指标。

- mustard: label true/false → Yes→True(1), No→False(0)
- urfunny: label 0/1 → Yes→1, No→0

输出: accuracy, micro/macro/weighted F1, precision, recall.
"""

import argparse
import json
from pathlib import Path

try:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    from collections import defaultdict


def extract_prediction(parsed_answer) -> int | None:
    """从 parsed_answer 提取预测: Yes→1, No→0。"""
    if not parsed_answer or not isinstance(parsed_answer, list):
        return None
    text = (parsed_answer[0] or '').strip()
    if not text:
        return None
    text_lower = text.lower()
    if text_lower.startswith('yes'):
        return 1
    if text_lower.startswith('no'):
        return 0
    return None


def label_to_binary(label, dataset: str) -> int | None:
    """
    - mustard: True→1, False→0
    - urfunny: 0→0, 1→1
    """
    if dataset == 'mustard':
        if label is True:
            return 1
        if label is False:
            return 0
    elif dataset == 'urfunny':
        if label in (0, 1):
            return int(label)
    return None


def load_pairs(filepath: str, dataset: str) -> tuple[list[int], list[int], int]:
    """返回 (y_true, y_pred), 以及跳过的条数。"""
    y_true = []
    y_pred = []
    skipped = 0

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                for _key, item in data.items():
                    label = item.get('label')
                    parsed = item.get('parsed_answer')
                    true_val = label_to_binary(label, dataset)
                    pred_val = extract_prediction(parsed)
                    if true_val is None or pred_val is None:
                        skipped += 1
                        continue
                    y_true.append(true_val)
                    y_pred.append(pred_val)
            except (json.JSONDecodeError, TypeError):
                skipped += 1
    return y_true, y_pred, skipped


def calc_metrics_sklearn(y_true, y_pred) -> dict:
    """用 sklearn 计算指标。"""
    return {
        'acc': accuracy_score(y_true, y_pred),
        'precision_micro': precision_score(
            y_true, y_pred, average='micro', zero_division=0
        ),
        'precision_macro': precision_score(
            y_true, y_pred, average='macro', zero_division=0
        ),
        'precision_weighted': precision_score(
            y_true, y_pred, average='weighted', zero_division=0
        ),
        'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_weighted': recall_score(
            y_true, y_pred, average='weighted', zero_division=0
        ),
        'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }


def calc_metrics_manual(y_true, y_pred) -> dict:
    """不依赖 sklearn 的手动计算。"""
    n = len(y_true)
    if n == 0:
        return {
            'acc': 0.0,
            'precision_micro': 0.0,
            'precision_macro': 0.0,
            'precision_weighted': 0.0,
            'recall_micro': 0.0,
            'recall_macro': 0.0,
            'recall_weighted': 0.0,
            'f1_micro': 0.0,
            'f1_macro': 0.0,
            'f1_weighted': 0.0,
        }

    classes = sorted(set(y_true) | set(y_pred))
    cm = defaultdict(lambda: defaultdict(int))
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    tp = {c: cm[c][c] for c in classes}
    fp = {c: sum(cm[k][c] for k in classes if k != c) for c in classes}
    fn = {c: sum(cm[c][k] for k in classes if k != c) for c in classes}
    support = {c: sum(cm[c][k] for k in classes) for c in classes}

    precision_c = {}
    recall_c = {}
    f1_c = {}
    for c in classes:
        precision_c[c] = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        recall_c[c] = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        if precision_c[c] + recall_c[c] > 0:
            f1_c[c] = 2 * precision_c[c] * recall_c[c] / (precision_c[c] + recall_c[c])
        else:
            f1_c[c] = 0.0

    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())
    total_support = sum(support.values())

    p_micro = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    r_micro = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f_micro = (
        2 * p_micro * r_micro / (p_micro + r_micro) if (p_micro + r_micro) > 0 else 0.0
    )

    p_macro = sum(precision_c[c] for c in classes) / len(classes)
    r_macro = sum(recall_c[c] for c in classes) / len(classes)
    f_macro = sum(f1_c[c] for c in classes) / len(classes)

    p_weighted = (
        sum(precision_c[c] * support[c] for c in classes) / total_support
        if total_support > 0
        else 0.0
    )
    r_weighted = (
        sum(recall_c[c] * support[c] for c in classes) / total_support
        if total_support > 0
        else 0.0
    )
    f_weighted = (
        sum(f1_c[c] * support[c] for c in classes) / total_support
        if total_support > 0
        else 0.0
    )

    return {
        'acc': sum(tp.values()) / n,
        'precision_micro': p_micro,
        'precision_macro': p_macro,
        'precision_weighted': p_weighted,
        'recall_micro': r_micro,
        'recall_macro': r_macro,
        'recall_weighted': r_weighted,
        'f1_micro': f_micro,
        'f1_macro': f_macro,
        'f1_weighted': f_weighted,
    }


def main():
    parser = argparse.ArgumentParser(
        description='计算指定 CTM jsonl 的 accuracy, micro/macro/weighted F1, precision, recall.'
    )
    parser.add_argument(
        'file',
        help='CTM 输出 jsonl 路径，如 ctm_mustard.jsonl 或 ctm_urfunny.jsonl',
    )
    parser.add_argument(
        '--dataset',
        '-d',
        choices=('mustard', 'urfunny'),
        default=None,
        help='数据集类型：mustard (label true/false) 或 urfunny (label 0/1)。不指定则根据文件名推断',
    )
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        raise SystemExit(f'文件不存在: {path}')

    dataset = args.dataset
    if dataset is None:
        name_lower = path.name.lower()
        if 'mustard' in name_lower:
            dataset = 'mustard'
        elif 'urfunny' in name_lower:
            dataset = 'urfunny'
        else:
            raise SystemExit(
                '无法从文件名推断数据集类型，请用 -d mustard 或 -d urfunny 指定'
            )

    y_true, y_pred, skipped = load_pairs(str(path), dataset)
    if not y_true:
        print(f'没有有效样本（有效=0, 跳过={skipped}）')
        return

    if HAS_SKLEARN:
        m = calc_metrics_sklearn(y_true, y_pred)
    else:
        m = calc_metrics_manual(y_true, y_pred)

    print(
        f'File: {path.name}  |  dataset: {dataset}  |  n={len(y_true)}, skipped={skipped}'
    )
    print('-' * 56)
    print(f'  Accuracy:              {m["acc"]:.4f}')
    print(f'  Precision (micro):     {m["precision_micro"]:.4f}')
    print(f'  Precision (macro):     {m["precision_macro"]:.4f}')
    print(f'  Precision (weighted):  {m["precision_weighted"]:.4f}')
    print(f'  Recall (micro):        {m["recall_micro"]:.4f}')
    print(f'  Recall (macro):        {m["recall_macro"]:.4f}')
    print(f'  Recall (weighted):     {m["recall_weighted"]:.4f}')
    print(f'  F1 (micro):           {m["f1_micro"]:.4f}')
    print(f'  F1 (macro):           {m["f1_macro"]:.4f}')
    print(f'  F1 (weighted):         {m["f1_weighted"]:.4f}')


if __name__ == '__main__':
    main()
