"""Step 2 — enrich harvested chunks with a label + estimator ②/③ confidences.

Modes (run once each):
  --mode label   : GPT-4o judges whether the chunk gist solves the query -> y in {0,0.5,1}
  --mode judge   : DECOUPLED Qwen3-8B (separate call, sees only query+gist) -> confidence in [0,1]  (estimator ②)
  --mode logprob : Qwen3-8B Yes/No logprobs -> softmax(Yes) in [0,1]                                (estimator ③)

Reads calibration/chunks/<set>.jsonl, writes calibration/enrich/<set>.<mode>.jsonl:
  {query_id, processor_name, value}
(estimator ① self_confidence is already in the harvested chunks.)
"""
import argparse
import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

VLLM_BASE = os.getenv('VLLM_API_BASE', 'http://localhost:8001/v1')
VLLM_MODEL = 'Qwen3-8B'
NOTHINK = {'chat_template_kwargs': {'enable_thinking': False}}


def _key(rec):
    return (rec['query_id'], rec['processor_name'])


def _parse_status(txt):
    t = (txt or '').lower()
    if 'unsure' in t:
        return 0.5
    if 'unsolved' in t:
        return 0.0
    if 'solved' in t:
        return 1.0
    return 0.5


def _parse_float(txt):
    m = re.search(r'[01]?\.\d+|[01](?:\.0+)?|\d{1,3}', txt or '')
    if not m:
        return 0.5
    v = float(m.group())
    if v > 1.0:  # 0-100 scale
        v = v / 100.0
    return max(0.0, min(1.0, v))


def _label(rec, client):
    p = (
        'Given a user query and a candidate answer, decide whether the answer '
        'actually solves the query.\nQuery: {q}\nAnswer: {a}\nReply with ONLY one '
        'word: Solved, Unsolved, or Unsure.'
    ).format(q=rec['query'][:1500], a=(rec['gist'] or '')[:2000])
    r = client.chat.completions.create(
        model='gpt-4o', messages=[{'role': 'user', 'content': p}], max_tokens=4, temperature=0.0
    )
    return _parse_status(r.choices[0].message.content)


def _judge(rec, client):
    p = (
        'Query: {q}\nResponse: {a}\nEstimate the probability from 0.00 to 1.00 that '
        'this response CORRECTLY and COMPLETELY solves the query. Reply with ONLY the number.'
    ).format(q=rec['query'][:1500], a=(rec['gist'] or '')[:2000])
    r = client.chat.completions.create(
        model=VLLM_MODEL, messages=[{'role': 'user', 'content': p}], max_tokens=8,
        temperature=0.0, extra_body=NOTHINK,
    )
    return _parse_float(r.choices[0].message.content)


def _logprob(rec, client):
    p = (
        'Query: {q}\nResponse: {a}\nDoes the response correctly and completely solve '
        'the query? Answer with exactly one word: Yes or No.'
    ).format(q=rec['query'][:1500], a=(rec['gist'] or '')[:2000])
    r = client.chat.completions.create(
        model=VLLM_MODEL, messages=[{'role': 'user', 'content': p}], max_tokens=1,
        temperature=0.0, logprobs=True, top_logprobs=20, extra_body=NOTHINK,
    )
    lz_yes, lz_no = None, None
    try:
        top = r.choices[0].logprobs.content[0].top_logprobs
        for t in top:
            tok = t.token.strip().lower()
            if tok == 'yes' and lz_yes is None:
                lz_yes = t.logprob
            elif tok == 'no' and lz_no is None:
                lz_no = t.logprob
    except Exception:  # noqa: BLE001
        pass
    if lz_yes is None and lz_no is None:
        return 0.5
    if lz_yes is None:
        lz_yes = lz_no - 10.0
    if lz_no is None:
        lz_no = lz_yes - 10.0
    m = max(lz_yes, lz_no)
    ey, en = math.exp(lz_yes - m), math.exp(lz_no - m)
    return ey / (ey + en)


_SELF_DECOUPLED_PROMPT = (
    'You wrote the response below to the task. Evaluate ONLY this response.\n'
    'Task: {q}\nResponse: {a}\n\n'
    'Score each from 0.0 to 1.0:\n'
    '- relevance: does the response provide specific, actionable data that answers '
    'the task? If it says "I cannot"/"please provide" or gives guidance instead of '
    'actual data, relevance MUST be 0.0-0.2; only responses with ACTUAL DATA from '
    'tool calls deserve 0.6+.\n'
    '- confidence: how certain that the information is correct? If no tool was '
    'successfully called or a tool errored, confidence MUST be 0.0-0.3.\n'
    '- surprise: does it bring novel information beyond the task context?\n\n'
    'Reply with ONLY a JSON object: '
    '{{"relevance": <0-1>, "confidence": <0-1>, "surprise": <0-1>}}'
)


def _self_decoupled(rec, client):
    """CTM self-rubric (relevance/confidence/surprise) in a SEPARATE forward pass
    (decoupled from answer generation). Returns weight r + c + 0.2s in [0, 2.2]."""
    r = client.chat.completions.create(
        model=VLLM_MODEL, messages=[{'role': 'user', 'content': _SELF_DECOUPLED_PROMPT.format(
            q=rec['query'][:1500], a=(rec['gist'] or '')[:2000])}],
        max_tokens=120, temperature=0.0, extra_body=NOTHINK,
    )
    txt = r.choices[0].message.content or ''

    def g(name):
        m = re.search(name + r'"?\s*:\s*([01]?\.\d+|[01](?:\.0+)?)', txt)
        return float(m.group(1)) if m else None

    rel, con, sur = g('relevance'), g('confidence'), g('surprise')
    if rel is None or con is None:
        return None
    return rel + con + 0.2 * (sur or 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True,
                    choices=['label', 'judge', 'logprob', 'self_decoupled'])
    ap.add_argument('--chunks', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--threads', type=int, default=8)
    args = ap.parse_args()

    if args.mode == 'label':
        client = OpenAI(api_key=os.environ['OPENAI_KEY'])
        fn = _label
    else:
        client = OpenAI(base_url=VLLM_BASE, api_key=os.getenv('VLLM_API_KEY', 'dummy'))
        fn = {'judge': _judge, 'logprob': _logprob,
              'self_decoupled': _self_decoupled}[args.mode]

    recs = [json.loads(l) for l in open(args.chunks) if l.strip()]
    done = set()
    if os.path.exists(args.out):
        for l in open(args.out):
            try:
                d = json.loads(l)
                done.add((d['query_id'], d['processor_name']))
            except Exception:  # noqa: BLE001
                pass
    todo = [r for r in recs if _key(r) not in done]
    print(f'{args.mode}: {len(todo)}/{len(recs)} to do (threads={args.threads})')

    def work(r):
        try:
            v = fn(r, client)
        except Exception as e:  # noqa: BLE001
            v = None
            print('  err', repr(e)[:100])
        return {'query_id': r['query_id'], 'processor_name': r['processor_name'], 'value': v}

    with open(args.out, 'a') as out, ThreadPoolExecutor(max_workers=args.threads) as pool:
        futs = [pool.submit(work, r) for r in todo]
        n = 0
        for f in as_completed(futs):
            out.write(json.dumps(f.result()) + '\n')
            out.flush()
            n += 1
            if n % 100 == 0:
                print(f'  {n}/{len(todo)}')
    print(f'{args.mode} done -> {args.out}')


if __name__ == '__main__':
    main()
