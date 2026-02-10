"""
Multimodal debate: 3 agents (Video, Audio, Text) debate over multiple rounds.

Each round, 3 agents analyze in parallel, then refine based on others' answers.
After all rounds, a judge (text-only) makes the final decision.

Examples:
python run_debate.py --dataset_name urfunny --provider qwen --rounds 3
python run_debate.py --dataset_name mustard --provider gemini --rounds 3
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import litellm
from dataset_configs import get_dataset_config
from llm_utils import (
    StatsTracker,
    check_api_key,
    create_agent,
    load_data,
    load_processed_keys,
    load_sample_inputs,
    save_result_to_jsonl,
)

sys.path.append('..')

ROUNDS = 3
COST_INPUT_PER_1M = 0.075
COST_OUTPUT_PER_1M = 0.30


def extract_answer(response):
    """Extract Yes/No from agent response."""
    if response is None:
        return 'Unknown'
    lower = response.lower()
    if 'my answer: yes' in lower:
        return 'Yes'
    elif 'my answer: no' in lower:
        return 'No'
    if lower.strip().endswith('yes'):
        return 'Yes'
    elif lower.strip().endswith('no'):
        return 'No'
    return 'Unknown'


def run_instance(
    test_file, dataset, dataset_name,
    video_agent, audio_agent, text_agent,
    tracker, output_file,
):
    """Process one sample: load inputs -> multi-round debate -> judge -> save."""
    start_time = time.time()
    total_prompt_tokens = 0
    total_completion_tokens = 0
    num_api_calls = 0

    # Step 1: Load inputs
    inputs = load_sample_inputs(test_file, dataset, dataset_name)
    target_sentence = inputs['target_sentence']
    label = inputs['label']
    muted_video_path = inputs['muted_video_path']
    audio_path = inputs['audio_path']
    config = inputs['config']

    prompts = config.get_debate_prompts()

    print(f'--- Debate for {test_file} ({ROUNDS} rounds) ---')

    # Step 2: Run logic — multi-round debate
    debate_history = ''
    video_answer = ''
    audio_answer = ''
    text_answer = ''

    def run_agent(agent_type, query):
        if agent_type == 'video':
            return agent_type, *video_agent.call(query, video_path=muted_video_path)
        elif agent_type == 'audio':
            return agent_type, *audio_agent.call(query, audio_path=audio_path)
        elif agent_type == 'text':
            return agent_type, *text_agent.call(query)

    for round_num in range(1, ROUNDS + 1):
        print(f'  Round {round_num}:')
        round_history = f'=== Round {round_num} ===\n\n'

        if round_num == 1:
            video_query = prompts['video_init']
            audio_query = prompts['audio_init']
            text_query = f"{prompts['text_init']}\n\ntarget text: '{target_sentence}'"
        else:
            video_query = prompts['video_refine'].format(
                own_answer=video_answer, audio_answer=audio_answer, text_answer=text_answer,
            )
            audio_query = prompts['audio_refine'].format(
                own_answer=audio_answer, video_answer=video_answer, text_answer=text_answer,
            )
            text_query = (
                f"{prompts['text_refine'].format(own_answer=text_answer, video_answer=video_answer, audio_answer=audio_answer)}"
                f"\n\ntarget text: '{target_sentence}'"
            )

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(run_agent, 'video', video_query): 'video',
                executor.submit(run_agent, 'audio', audio_query): 'audio',
                executor.submit(run_agent, 'text', text_query): 'text',
            }
            results = {}
            for future in as_completed(futures):
                agent_type, response, usage = future.result()
                results[agent_type] = (response, usage)
                num_api_calls += 1
                total_prompt_tokens += usage.get('prompt_tokens', 0)
                total_completion_tokens += usage.get('completion_tokens', 0)

        video_answer = results['video'][0] or ''
        audio_answer = results['audio'][0] or ''
        text_answer = results['text'][0] or ''

        round_history += f'[Video Expert]: {video_answer}\n\n'
        round_history += f'[Audio Expert]: {audio_answer}\n\n'
        round_history += f'[Text Expert]: {text_answer}\n\n'
        debate_history += round_history

        print(f'    Video: {extract_answer(video_answer)} | Audio: {extract_answer(audio_answer)} | Text: {extract_answer(text_answer)}')

    # Judge makes final decision
    judge_query = f"{prompts['judge'].format(debate_history=debate_history)}\n\ntarget text: '{target_sentence}'"
    final_verdict, usage = text_agent.call(judge_query)
    num_api_calls += 1
    total_prompt_tokens += usage.get('prompt_tokens', 0)
    total_completion_tokens += usage.get('completion_tokens', 0)

    end_time = time.time()
    duration = end_time - start_time
    tracker.add(duration, total_prompt_tokens, total_completion_tokens, num_api_calls)

    print(f'  Judge: {final_verdict[:80] if final_verdict else "None"}... ({duration:.1f}s)')

    # Step 3: Save result
    result = {
        test_file: {
            'answer': [final_verdict],
            'debate_history': debate_history,
            'final_votes': {
                'video': extract_answer(video_answer),
                'audio': extract_answer(audio_answer),
                'text': extract_answer(text_answer),
            },
            'label': label,
            'method': f'debate_{ROUNDS}rounds',
            'usage': {
                'prompt_tokens': total_prompt_tokens,
                'completion_tokens': total_completion_tokens,
                'api_calls': num_api_calls,
            },
            'latency': duration,
        }
    }
    save_result_to_jsonl(result, output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3-Agent Multimodal Debate')
    parser.add_argument(
        '--dataset_name', type=str, default='urfunny',
        choices=['urfunny', 'mustard'], help='Dataset name (default: urfunny)',
    )
    parser.add_argument(
        '--provider', type=str, default='gemini',
        choices=['gemini', 'qwen'], help='LLM provider (default: gemini)',
    )
    parser.add_argument(
        '--model', type=str, default=None,
        help='Model name for litellm (default: auto based on provider)',
    )
    parser.add_argument(
        '--dataset', type=str, default=None,
        help='Path to dataset JSON file (default: auto based on dataset_name)',
    )
    parser.add_argument(
        '--output', type=str, default=None, help='Output JSONL file path',
    )
    parser.add_argument(
        '--temperature', type=float, default=1.0,
        help='Sampling temperature (default: 1.0)',
    )
    parser.add_argument(
        '--rounds', type=int, default=3, help='Number of debate rounds (default: 3)',
    )
    args = parser.parse_args()

    ROUNDS = args.rounds

    config = get_dataset_config(args.dataset_name)
    if args.dataset is None:
        args.dataset = config.get_default_dataset_path()
    output_file = args.output or f'debate_{args.dataset_name}_{args.provider}.jsonl'

    check_api_key(args.provider)
    litellm.set_verbose = False

    tracker = StatsTracker(
        cost_input_per_1m=COST_INPUT_PER_1M, cost_output_per_1m=COST_OUTPUT_PER_1M
    )

    video_agent = create_agent(
        'video', provider=args.provider, model=args.model, temperature=args.temperature
    )
    audio_agent = create_agent(
        'audio', provider=args.provider, model=args.model, temperature=args.temperature
    )
    text_agent = create_agent(
        'text', provider=args.provider, model=args.model, temperature=args.temperature
    )
    print(f'Dataset: {args.dataset_name} | Provider: {args.provider} | Model: {text_agent.model}')

    dataset = load_data(args.dataset)
    test_list = list(dataset.keys())
    processed_keys = load_processed_keys(output_file)
    if processed_keys:
        print(f'Resuming: {len(processed_keys)} done, {len(test_list) - len(processed_keys)} remaining')

    try:
        for test_file in test_list:
            if test_file in processed_keys:
                continue
            try:
                run_instance(
                    test_file, dataset, args.dataset_name,
                    video_agent, audio_agent, text_agent,
                    tracker, output_file,
                )
            except Exception as e:
                print(f'[ERROR] {test_file}: {e}')
                continue
            time.sleep(2)
    finally:
        tracker.print_summary(f'Debate - {ROUNDS} Rounds')
