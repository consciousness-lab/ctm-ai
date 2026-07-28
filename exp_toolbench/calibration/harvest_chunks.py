"""Step 1 of the confidence-calibration study: harvest per-(query,chunk) data.

For each StableToolBench query we build the tool env + ToolCTM and run ONLY the
initial phase (`ask_processors`), which makes every tool-processor synthesize a
chunk (gist) and self-report relevance/confidence/surprise — jointly, exactly as
CTM does. We dump one record per chunk. This is estimator ① (self-reported,
joint) plus the raw material (query + gist) for estimators ②/③ and the label.

Output: <out>/<test_set>.jsonl, one JSON per chunk:
  {query_id, query, processor_name, gist, self_relevance, self_confidence,
   self_surprise, self_weight}
"""
import argparse
import json
import multiprocessing
import os
import sys

sys.path.append('..')

from ctm_ai.apis.api_manager import contain, get_white_list, rapidapi_wrapper  # noqa: E402
from ctm_ai.apis.api_server import standardize  # noqa: E402


class _Args:
    def __init__(self, tool_root_dir):
        self.tool_root_dir = tool_root_dir
        self.toolbench_key = ''
        self.rapidapi_key = ''
        self.use_rapidapi_key = False
        self.api_customization = False
        self.max_observation_length = 1024
        self.observ_compress_method = 'truncate'


def _build_env(query, tool_root_dir):
    args = _Args(tool_root_dir)
    wl = get_white_list(tool_root_dir)
    origin = [standardize(c['tool_name']) for c in query.get('api_list', [])]
    td = contain(origin, wl)
    if not td:
        return None
    tool_des = [[c['standard_tool_name'], c['description']] for c in td]
    return rapidapi_wrapper(query, tool_des, args, process_id=0)


def _one(task):
    query, ctm_name, tool_root_dir, out_dir, test_set = task
    qid = query['query_id']
    out_path = os.path.join(out_dir, f'{test_set}__{qid}.jsonl')
    if os.path.exists(out_path):
        return
    try:
        env = _build_env(query, tool_root_dir)
        if env is None or not env.functions:
            return
        from ctm_ai.ctms import ToolCTM

        ctm = ToolCTM(api_manager=env, ctm_name=ctm_name)
        chunks = ctm.ask_processors(query['query'], api_manager=env, phase='initial')
    except Exception as e:  # noqa: BLE001
        with open(out_path, 'w') as f:
            f.write(json.dumps({'query_id': qid, 'error': repr(e)}) + '\n')
        return
    recs = []
    for c in chunks or []:
        recs.append(
            {
                'query_id': qid,
                'query': query['query'],
                'processor_name': getattr(c, 'processor_name', ''),
                'gist': getattr(c, 'gist', '') or '',
                'self_relevance': getattr(c, 'relevance', 0.0),
                'self_confidence': getattr(c, 'confidence', 0.0),
                'self_surprise': getattr(c, 'surprise', 0.0),
                'self_weight': getattr(c, 'weight', 0.0),
            }
        )
    with open(out_path, 'w') as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_query_file', required=True)
    ap.add_argument('--test_set', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--tool_root_dir', required=True)
    ap.add_argument('--ctm_name', default='tooluse_ctm_qwen3')
    ap.add_argument('--num_processes', type=int, default=16)
    ap.add_argument('--query_id', type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    queries = json.load(open(args.input_query_file))
    if args.query_id is not None:
        queries = [q for q in queries if q.get('query_id') == args.query_id]
    tasks = [
        (q, args.ctm_name, args.tool_root_dir, args.out_dir, args.test_set)
        for q in queries
    ]
    print(f'{args.test_set}: harvesting {len(tasks)} queries')
    if args.num_processes > 1:
        with multiprocessing.Pool(args.num_processes) as pool:
            pool.map(_one, tasks)
    else:
        for t in tasks:
            _one(t)
    # merge per-query files into one jsonl
    merged = os.path.join(args.out_dir, f'{args.test_set}.jsonl')
    n = 0
    with open(merged, 'w') as out:
        for q in queries:
            p = os.path.join(args.out_dir, f'{args.test_set}__{q["query_id"]}.jsonl')
            if os.path.exists(p):
                for ln in open(p):
                    if ln.strip() and 'error' not in ln[:20]:
                        out.write(ln)
                        n += 1
    print(f'merged {n} chunk records -> {merged}')


if __name__ == '__main__':
    main()
