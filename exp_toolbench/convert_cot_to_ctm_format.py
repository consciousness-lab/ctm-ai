"""Convert StableToolBench CoT outputs to CTM-compatible format for evaluation."""
import json
import glob
import os
import argparse


def convert_cot_file(cot_path: str, query_meta: dict) -> dict:
    """Convert a single CoT output file to CTM-compatible format."""
    cot = json.load(open(cot_path))
    ag = cot.get('answer_generation', {})
    final_answer = ag.get('final_answer', '')

    # Parse final_answer if it's JSON with return_type
    if final_answer and final_answer.strip().startswith('{'):
        try:
            parsed = json.loads(final_answer)
            if isinstance(parsed, dict) and 'final_answer' in parsed:
                final_answer = parsed['final_answer']
        except json.JSONDecodeError:
            pass

    return {
        'query': query_meta.get('query', ''),
        'query_id': query_meta.get('query_id'),
        'final_answer': final_answer,
        'weight_score': 0.0,
        'parsed_answer': final_answer,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cot_dir', required=True, help='Directory with CoT output JSON files')
    parser.add_argument('--query_file', required=True, help='Original query file')
    parser.add_argument('--output_dir', required=True, help='Output directory for CTM-format files')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load query metadata
    queries = json.load(open(args.query_file))
    query_index = {q['query_id']: q for q in queries}

    files = sorted(glob.glob(os.path.join(args.cot_dir, '*.json')))
    converted = 0
    for f in files:
        basename = os.path.basename(f)
        # Extract query_id from filename like "12345_CoT@1.json"
        qid_str = basename.split('_')[0]
        try:
            qid = int(qid_str)
        except ValueError:
            continue

        query_meta = query_index.get(qid, {'query': '', 'query_id': qid})
        ctm_format = convert_cot_file(f, query_meta)

        out_path = os.path.join(args.output_dir, f'{qid}_ctm.json')
        with open(out_path, 'w') as wf:
            json.dump(ctm_format, wf, indent=2)
        converted += 1

    print(f'Converted {converted} files to {args.output_dir}')


if __name__ == '__main__':
    main()
