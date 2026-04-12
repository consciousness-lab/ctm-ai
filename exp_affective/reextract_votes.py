"""Post-process JSONL output files to re-extract votes/answers from the stored
raw responses, using the updated extract_vote / extract_answer logic.

Use case: the running experiments stored valid `individual_answers` (or
`debate_history` / `conversation_history`) but some `final_vote` / `answer`
fields were computed with an older, too-strict extractor and show Unknown.

Re-extraction does NOT make new API calls — it operates only on the stored
response text.

Usage:
    python reextract_votes.py runs_mustard_fix/ensemble_mustard_gemini.jsonl
    python reextract_votes.py runs_mustard_fix/*.jsonl
"""

import argparse
import json
from collections import Counter

from llm_utils import normalize_label
from run_debate import extract_answer
from run_ensemble import extract_vote, majority_vote


def fix_ensemble_entry(entry):
    """Return (new_entry, changed) for an ensemble result."""
    individual = entry.get('individual_answers', {})
    agent_types = ['text', 'audio', 'video']
    new_votes = {t: extract_vote(individual.get(t)) for t in agent_types}
    votes_list = [new_votes[t] for t in agent_types]
    new_final = majority_vote(votes_list)
    new_dist = dict(Counter(votes_list))
    label_norm = normalize_label(entry.get('label'))
    new_correct = new_final == label_norm

    changed = (
        new_votes != entry.get('votes')
        or new_final != entry.get('final_vote')
        or new_dist != entry.get('vote_distribution')
        or label_norm != entry.get('label_normalized')
        or new_correct != entry.get('correct')
    )
    if changed:
        entry['votes'] = new_votes
        entry['final_vote'] = new_final
        entry['vote_distribution'] = new_dist
        entry['label_normalized'] = label_norm
        entry['correct'] = new_correct
        entry['answer'] = [new_final]
    return entry, changed


def fix_debate_entry(entry):
    """Return (new_entry, changed) for a debate result.

    Debate stores `answer` as [judge_verdict_text] and `final_votes` as the
    per-agent last-round votes extracted from debate_history.
    """
    changed = False
    # Re-extract final_votes from debate_history if stored there
    history = entry.get('debate_history', '')
    # The debate_history is a concatenated text of "[Agent Expert]: <response>"
    # Last round's votes are already-stored in final_votes; we leave those
    # alone unless the judge verdict itself is unparseable.
    judge = entry.get('answer', [None])[0]
    if judge:
        new_parsed = extract_answer(judge)
        old_parsed = entry.get('parsed_verdict')
        if new_parsed != old_parsed:
            entry['parsed_verdict'] = new_parsed
            changed = True
    # Also re-extract final_votes (best effort) — just pass through.
    return entry, changed


def fix_orchestra_entry(entry):
    """Return (new_entry, changed) for an orchestra result."""
    changed = False
    decision = entry.get('answer', [None])[0]
    if decision:
        new_parsed = extract_answer(decision)
        old_parsed = entry.get('parsed_verdict')
        if new_parsed != old_parsed:
            entry['parsed_verdict'] = new_parsed
            changed = True
    return entry, changed


def detect_method(entry_value):
    method = entry_value.get('method', '')
    if 'ensemble' in method:
        return 'ensemble'
    if 'debate' in method:
        return 'debate'
    if 'orchestra' in method:
        return 'orchestra'
    return None


def process_file(path, dry_run=False):
    method_counts = Counter()
    total = 0
    changed_count = 0
    out_lines = []

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            key = next(iter(obj.keys()))
            value = obj[key]
            method = detect_method(value)
            method_counts[method] += 1
            total += 1

            if method == 'ensemble':
                value, ch = fix_ensemble_entry(value)
            elif method == 'debate':
                value, ch = fix_debate_entry(value)
            elif method == 'orchestra':
                value, ch = fix_orchestra_entry(value)
            else:
                ch = False

            if ch:
                changed_count += 1
            obj[key] = value
            out_lines.append(json.dumps(obj, ensure_ascii=False) + '\n')

    print(f'{path}: {total} entries, methods={dict(method_counts)}, changed={changed_count}')

    if not dry_run and changed_count > 0:
        tmp = path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            f.writelines(out_lines)
        import os
        os.replace(tmp, path)
        print(f'  -> wrote {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Re-extract votes from JSONL output')
    parser.add_argument('files', nargs='+', help='JSONL files to fix in place')
    parser.add_argument('--dry-run', action='store_true', help='Do not write changes')
    args = parser.parse_args()
    for f in args.files:
        process_file(f, dry_run=args.dry_run)
