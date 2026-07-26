"""Multi-agent baselines for StableToolBench (T2 table), on the local Qwen3-8B.

All baselines share one primitive — a ReAct tool-calling agent (`ToolAgent`)
that drives the query's tools through the same `rapidapi_wrapper` env used by
CTM-AI (so tool execution goes to SERVICE_URL=/virtual and tool-calls are
counted identically). Baselines differ only in how many agents run and how
their answers are combined:

  react / unified   : 1 ReAct agent (the Unified/ReAct baseline)
  ensemble          : N independent agents -> LLM judge picks best (self-consistency)
  moa               : N proposers -> aggregator, repeated for L layers (Mixture-of-Agents)
  debate            : N agents answer, then critique each other for R rounds -> judge
  orchestra         : a planner splits the task, workers solve, aggregator merges
  metagpt           : role SOP (analyst -> solver -> reviewer) — faithful lightweight reimpl
  autogen           : round-robin group chat of N agents + judge — lightweight reimpl

Output per query: {query, query_id, final_answer, parsed_answer, metrics} — the
same schema run_ctm.py emits, so eval_ctm_toolbench.py / aggregate work directly.
"""
import argparse
import json
import multiprocessing
import os
import sys
import time

sys.path.append('..')

from litellm import completion  # noqa: E402

from ctm_ai.apis.api_manager import (  # noqa: E402
    contain,
    get_white_list,
    rapidapi_wrapper,
)
from ctm_ai.apis.api_server import standardize  # noqa: E402
from ctm_ai.utils.litellm_utils import get_completion_kwargs  # noqa: E402

# Real MoA framework: the paper's exact reference-synthesis prompt.
sys.path.insert(0, '/data/yiningz9/ctm-rebuttal/MoA')
try:
    from utils import inject_references_to_messages as _moa_inject  # noqa: E402
except ImportError:
    _moa_inject = None

FINISH_TOOL = {
    'type': 'function',
    'function': {
        'name': 'Finish',
        'description': (
            'Call this when you have the final answer, or to give up. You MUST '
            'call it at the end. Only final_answer is shown to the user, so it '
            'must contain enough information.'
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'return_type': {
                    'type': 'string',
                    'enum': ['give_answer', 'give_up_and_restart'],
                },
                'final_answer': {
                    'type': 'string',
                    'description': 'The final answer (required if give_answer).',
                },
            },
            'required': ['return_type'],
        },
    },
}

SYSTEM_PROMPT = (
    'You are AutoGPT, able to use many tools (functions) to handle the task. '
    'Use the subfunctions to gather real information; do not fabricate. '
    'Call each tool with correct arguments. Remember to ALWAYS call "Finish" at '
    'the end with a complete final_answer.\n'
)


class SimpleArgs:
    """Minimal args object the rapidapi_wrapper env needs."""

    def __init__(self, tool_root_dir, toolbench_key='', max_observation_length=1024):
        self.tool_root_dir = tool_root_dir
        self.toolbench_key = toolbench_key
        self.rapidapi_key = ''
        self.use_rapidapi_key = False
        self.api_customization = False
        self.max_observation_length = max_observation_length
        self.observ_compress_method = 'truncate'


def build_env(query, tool_root_dir, toolbench_key=''):
    """Create a rapidapi_wrapper tool env for one query (mirrors pipeline_runner)."""
    args = SimpleArgs(tool_root_dir, toolbench_key)
    white_list = get_white_list(tool_root_dir)
    origin = [standardize(c['tool_name']) for c in query.get('api_list', [])]
    td = contain(origin, white_list)
    if not td:
        return None
    tool_des = [[c['standard_tool_name'], c['description']] for c in td]
    env = rapidapi_wrapper(query, tool_des, args, process_id=0)
    return env


class Stats:
    def __init__(self):
        self.model_calls = 0
        self.total_tokens = 0

    def add(self, resp):
        self.model_calls += 1
        u = getattr(resp, 'usage', None)
        if u:
            self.total_tokens += getattr(u, 'total_tokens', 0) or 0


class ToolAgent:
    """A single ReAct tool-calling agent over one env."""

    def __init__(self, env, model='vllm/Qwen3-8B', max_steps=8, temperature=0.0, stats=None):
        self.env = env
        self.model = model
        self.kw = get_completion_kwargs(model)
        self.max_steps = max_steps
        self.temperature = temperature
        self.stats = stats or Stats()

    def _chat(self, messages, tools=None):
        call = dict(self.kw)
        call.update(messages=messages, max_tokens=1024, temperature=self.temperature)
        if tools is not None:
            call.update(tools=tools, tool_choice='auto')
        resp = completion(**call)
        self.stats.add(resp)
        return resp.choices[0].message

    def run(self, query, extra_context=''):
        tools = list(self.env.functions) + [FINISH_TOOL]
        sys_content = SYSTEM_PROMPT + self.env.task_description
        if extra_context:
            sys_content += '\n' + extra_context
        messages = [
            {'role': 'system', 'content': sys_content},
            {'role': 'user', 'content': query},
        ]
        final = ''
        for _ in range(self.max_steps):
            msg = self._chat(messages, tools=tools)
            tcs = getattr(msg, 'tool_calls', None)
            if not tcs:
                content = msg.content or ''
                messages.append({'role': 'assistant', 'content': content})
                messages.append(
                    {'role': 'user', 'content': 'Now call Finish with your final answer.'}
                )
                continue
            tc = tcs[0]
            name = tc.function.name or ''
            arguments = tc.function.arguments or '{}'
            if name.endswith('Finish') or name == 'Finish':
                try:
                    final = json.loads(arguments).get('final_answer', '') or ''
                except (ValueError, TypeError):
                    final = arguments
                break
            obs, code = self.env.step(name, arguments)
            messages.append(
                {
                    'role': 'assistant',
                    'content': None,
                    'tool_calls': [
                        {
                            'id': tc.id,
                            'type': 'function',
                            'function': {'name': name, 'arguments': arguments},
                        }
                    ],
                }
            )
            messages.append(
                {
                    'role': 'tool',
                    'tool_call_id': tc.id,
                    'name': name,
                    'content': str(obs)[:1024],
                }
            )
        return final


def _judge_pick(query, answers, model, stats):
    """LLM judge picks the best answer index from a list (1-based)."""
    listing = '\n'.join(f'[{i + 1}] {a[:800]}' for i, a in enumerate(answers))
    prompt = (
        f'Query: {query}\n\nCandidate answers:\n{listing}\n\n'
        'Pick the single best answer that most completely and correctly solves '
        'the query. Respond with ONLY the number.'
    )
    kw = get_completion_kwargs(model)
    resp = completion(**kw, messages=[{'role': 'user', 'content': prompt}], max_tokens=8, temperature=0.0)
    stats.add(resp)
    txt = (resp.choices[0].message.content or '1').strip()
    for tok in txt.replace('[', ' ').replace(']', ' ').split():
        if tok.isdigit() and 1 <= int(tok) <= len(answers):
            return answers[int(tok) - 1]
    return answers[0]


def _aggregate(query, answers, model, stats):
    """Aggregator LLM synthesizes one answer from several (MoA / orchestra)."""
    listing = '\n'.join(f'[{i + 1}] {a[:800]}' for i, a in enumerate(answers))
    prompt = (
        f'Query: {query}\n\nProposed answers from several agents:\n{listing}\n\n'
        'Synthesize a single, comprehensive, correct final answer that combines '
        'the accurate information above and directly answers the query.'
    )
    kw = get_completion_kwargs(model)
    resp = completion(**kw, messages=[{'role': 'user', 'content': prompt}], max_tokens=1024, temperature=0.0)
    stats.add(resp)
    return (resp.choices[0].message.content or '').strip()


# --------------------------------------------------------------------------
# Baseline orchestrations. Each returns (final_answer, stats).
# --------------------------------------------------------------------------


def run_react(env, query, model, n=1, rounds=1, layers=1):
    stats = Stats()
    ans = ToolAgent(env, model, temperature=0.0, stats=stats).run(query)
    return ans, stats


def run_ensemble(env, query, model, n=3, rounds=1, layers=1):
    stats = Stats()
    answers = [
        ToolAgent(env, model, temperature=0.7, stats=stats).run(query) for _ in range(n)
    ]
    answers = [a for a in answers if a] or ['']
    final = _judge_pick(query, answers, model, stats) if len(answers) > 1 else answers[0]
    return final, stats


def run_moa(env, query, model, n=3, rounds=1, layers=2):
    """Mixture-of-Agents (real framework). Layer-0 proposers are tool-calling
    agents; refinement/aggregation layers use MoA's own
    `inject_references_to_messages` (the paper's exact synthesis prompt)."""
    stats = Stats()
    kw = get_completion_kwargs(model)
    # Layer 0: n proposer tool-agents actually gather tool evidence.
    proposals = [
        ToolAgent(env, model, temperature=0.7, stats=stats).run(query) for _ in range(n)
    ]
    proposals = [p for p in proposals if p] or ['']

    def _moa_layer(refs, temp):
        msgs = [{'role': 'user', 'content': query}]
        msgs = _moa_inject(msgs, refs) if _moa_inject else msgs
        resp = completion(**kw, messages=msgs, max_tokens=1024, temperature=temp)
        stats.add(resp)
        return (resp.choices[0].message.content or '').strip()

    # Intermediate MoA layers: n aggregator-agents each synthesize the refs.
    for _ in range(max(0, layers - 1)):
        new = [_moa_layer(proposals, 0.7) for _ in range(n)]
        proposals = [p for p in new if p] or proposals
    # Final aggregation layer.
    final = _moa_layer(proposals, 0.0)
    return final, stats


def run_debate(env, query, model, n=3, rounds=2, layers=1):
    stats = Stats()
    answers = [
        ToolAgent(env, model, temperature=0.7, stats=stats).run(query) for _ in range(n)
    ]
    for _ in range(max(0, rounds - 1)):
        new = []
        for i in range(n):
            others = '\n'.join(
                f'Agent {j + 1}: {answers[j][:600]}' for j in range(n) if j != i
            )
            ctx = f'Other agents answered:\n{others}\nReconsider and improve your answer.'
            new.append(ToolAgent(env, model, temperature=0.5, stats=stats).run(query, extra_context=ctx))
        answers = [a for a in new if a] or answers
    final = _judge_pick(query, [a for a in answers if a] or [''], model, stats)
    return final, stats


def run_orchestra(env, query, model, n=3, rounds=1, layers=1):
    stats = Stats()
    # planner produces sub-questions
    kw = get_completion_kwargs(model)
    presp = completion(
        **kw,
        messages=[
            {
                'role': 'user',
                'content': f'Break this task into {n} concrete sub-questions, one per line:\n{query}',
            }
        ],
        max_tokens=256,
        temperature=0.0,
    )
    stats.add(presp)
    subs = [s.strip('- ').strip() for s in (presp.choices[0].message.content or '').splitlines() if s.strip()][:n]
    subs = subs or [query]
    workers = [ToolAgent(env, model, temperature=0.3, stats=stats).run(sq) for sq in subs]
    final = _aggregate(query, [w for w in workers if w] or [''], model, stats)
    return final, stats


def run_metagpt(env, query, model, n=1, rounds=1, layers=1):
    """Lightweight MetaGPT-style role SOP: analyst -> solver(ReAct) -> reviewer."""
    stats = Stats()
    kw = get_completion_kwargs(model)
    a = completion(**kw, messages=[{'role': 'user', 'content': f'As an analyst, list the key facts needed to answer: {query}'}], max_tokens=256, temperature=0.0)
    stats.add(a)
    plan = a.choices[0].message.content or ''
    draft = ToolAgent(env, model, temperature=0.2, stats=stats).run(query, extra_context=f'Analyst notes:\n{plan}')
    r = completion(**kw, messages=[{'role': 'user', 'content': f'As a reviewer, produce the final, corrected answer.\nQuery: {query}\nDraft: {draft}'}], max_tokens=1024, temperature=0.0)
    stats.add(r)
    return (r.choices[0].message.content or draft).strip(), stats


def run_autogen(env, query, model, n=3, rounds=2, layers=1):
    """Lightweight AutoGen-style round-robin group chat + judge."""
    return run_debate(env, query, model, n=n, rounds=rounds)


METHODS = {
    'react': run_react,
    'unified': run_react,
    'ensemble': run_ensemble,
    'moa': run_moa,
    'debate': run_debate,
    'orchestra': run_orchestra,
    'metagpt': run_metagpt,
    'autogen': run_autogen,
}


def run_one(task):
    method, query, args = task
    # Use the _ctm.json suffix so eval_ctm_toolbench.py (which globs *_ctm.json)
    # can score baseline outputs directly; the method is encoded in the dir path.
    out_path = os.path.join(args.output_answer_file, f'{query["query_id"]}_ctm.json')
    if os.path.exists(out_path):
        return
    env = build_env(query, args.tool_root_dir, args.toolbench_key)
    if env is None or not env.functions:
        return
    t0 = time.time()
    try:
        final, stats = METHODS[method](
            env, query['query'], args.model, n=args.n_agents, rounds=args.rounds, layers=args.layers
        )
    except Exception as e:  # noqa: BLE001
        final, stats = f'[ERROR] {e!r}', Stats()
    latency = time.time() - t0
    os.makedirs(args.output_answer_file, exist_ok=True)
    data = {
        'query': query['query'],
        'query_id': query['query_id'],
        'final_answer': final,
        'weight_score': 0.0,
        'parsed_answer': final,
        'metrics': {
            'latency_seconds': latency,
            'num_model_calls': stats.model_calls,
            'num_tool_calls': getattr(env, 'tool_call_count', 0),
            'total_tokens': stats.total_tokens,
            'num_processors': len(env.functions),
        },
    }
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True, choices=list(METHODS))
    ap.add_argument('--input_query_file', required=True)
    ap.add_argument('--output_answer_file', required=True)
    ap.add_argument('--tool_root_dir', required=True)
    ap.add_argument('--toolbench_key', default='')
    ap.add_argument('--model', default='vllm/Qwen3-8B')
    ap.add_argument('--n_agents', type=int, default=3)
    ap.add_argument('--rounds', type=int, default=2)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--num_processes', type=int, default=1)
    ap.add_argument('--query_id', type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.output_answer_file, exist_ok=True)
    queries = json.load(open(args.input_query_file))
    if args.query_id is not None:
        queries = [q for q in queries if q.get('query_id') == args.query_id]
    tasks = [(args.method, q, args) for q in queries]
    print(f'{args.method}: {len(tasks)} queries, {args.num_processes} procs')
    if args.num_processes > 1:
        with multiprocessing.Pool(args.num_processes) as pool:
            pool.map(run_one, tasks)
    else:
        for t in tasks:
            run_one(t)
    print('done')


if __name__ == '__main__':
    main()
