"""
Controller-agent query augmentation (orchestra) with 3 agents (Video, Audio, Text).

A controller generates targeted questions for 3 modality experts.
Over multiple rounds, experts answer and the controller refines questions.
Finally, the controller decides.

Examples:
python run_orchestra.py --dataset_name urfunny --provider qwen --rounds 3
python run_orchestra.py --dataset_name mustard --provider gemini --rounds 3
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


def parse_controller_questions(response):
    """Parse controller response to extract questions for each agent."""
    questions = {'video': '', 'audio': '', 'text': ''}
    if response is None:
        return questions

    for line in response.split('\n'):
        lower = line.lower()
        if 'video_question:' in lower:
            questions['video'] = line.split(':', 1)[1].strip() if ':' in line else ''
        elif 'audio_question:' in lower:
            questions['audio'] = line.split(':', 1)[1].strip() if ':' in line else ''
        elif 'text_question:' in lower:
            questions['text'] = line.split(':', 1)[1].strip() if ':' in line else ''

    # Fallback if parsing failed
    if not questions['video']:
        questions['video'] = 'What visual cues do you observe that might be relevant?'
    if not questions['audio']:
        questions['audio'] = 'What audio cues do you observe that might be relevant?'
    if not questions['text']:
        questions['text'] = 'What textual cues do you observe that might be relevant?'

    return questions


def run_instance(
    test_file,
    dataset,
    dataset_name,
    video_agent,
    audio_agent,
    text_agent,
    tracker,
    output_file,
):
    """Process one sample: load inputs -> controller-agent rounds -> decision -> save."""
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

    prompts = config.get_query_aug_prompts()

    print(f'--- Orchestra for {test_file} ({ROUNDS} rounds) ---')

    # Step 2: Run logic — controller-agent query augmentation
    conversation_history = ''
    video_response = ''
    audio_response = ''
    text_response = ''

    def run_agent(agent_type, query):
        if agent_type == 'video':
            return agent_type, *video_agent.call(query, video_path=muted_video_path)
        elif agent_type == 'audio':
            return agent_type, *audio_agent.call(query, audio_path=audio_path)
        elif agent_type == 'text':
            return agent_type, *text_agent.call(query)

    for round_num in range(1, ROUNDS + 1):
        print(f'  Round {round_num}:')
        round_history = f'\n=== Round {round_num} ===\n'

        # Controller generates questions
        if round_num == 1:
            controller_query = prompts['controller_init']
        else:
            controller_query = prompts['controller_followup'].format(
                prev_round=round_num - 1,
                video_response=video_response[:500] if video_response else 'None',
                audio_response=audio_response[:500] if audio_response else 'None',
                text_response=text_response[:500] if text_response else 'None',
                round_num=round_num,
            )

        controller_response, usage = text_agent.call(controller_query)
        num_api_calls += 1
        total_prompt_tokens += usage.get('prompt_tokens', 0)
        total_completion_tokens += usage.get('completion_tokens', 0)

        questions = parse_controller_questions(controller_response)
        round_history += f'[Controller Questions]:\n{controller_response}\n\n'
        print(f'    Controller generated questions')

        # Prepare and run agents in parallel
        video_query = prompts['video_agent'].format(question=questions['video'])
        audio_query = prompts['audio_agent'].format(question=questions['audio'])
        text_query = prompts['text_agent'].format(
            question=questions['text'], text=target_sentence
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

        video_response = results['video'][0] or ''
        audio_response = results['audio'][0] or ''
        text_response = results['text'][0] or ''

        round_history += f'[Video Expert]: {video_response}\n\n'
        round_history += f'[Audio Expert]: {audio_response}\n\n'
        round_history += f'[Text Expert]: {text_response}\n\n'
        conversation_history += round_history

        print(f'    Video: {video_response[:60] if video_response else "None"}...')
        print(f'    Audio: {audio_response[:60] if audio_response else "None"}...')
        print(f'    Text:  {text_response[:60] if text_response else "None"}...')

    # Controller final decision
    decision_query = prompts['controller_decision'].format(
        num_rounds=ROUNDS, conversation_history=conversation_history
    )
    final_verdict, usage = text_agent.call(decision_query)
    num_api_calls += 1
    total_prompt_tokens += usage.get('prompt_tokens', 0)
    total_completion_tokens += usage.get('completion_tokens', 0)

    end_time = time.time()
    duration = end_time - start_time
    tracker.add(duration, total_prompt_tokens, total_completion_tokens, num_api_calls)

    print(
        f'  Decision: {final_verdict[:80] if final_verdict else "None"}... ({duration:.1f}s)'
    )

    # Step 3: Save result
    result = {
        test_file: {
            'answer': [final_verdict],
            'conversation_history': conversation_history,
            'label': label,
            'method': f'orchestra_{ROUNDS}rounds',
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
    parser = argparse.ArgumentParser(
        description='Controller-Agent Query Augmentation (Orchestra)'
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='urfunny',
        choices=['urfunny', 'mustard'],
        help='Dataset name (default: urfunny)',
    )
    parser.add_argument(
        '--provider',
        type=str,
        default='gemini',
        choices=['gemini', 'qwen'],
        help='LLM provider (default: gemini)',
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Model name for litellm (default: auto based on provider)',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Path to dataset JSON file (default: auto based on dataset_name)',
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSONL file path',
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Sampling temperature (default: 1.0)',
    )
    parser.add_argument(
        '--rounds',
        type=int,
        default=3,
        help='Number of questioning rounds (default: 3)',
    )
    args = parser.parse_args()

    ROUNDS = args.rounds

    config = get_dataset_config(args.dataset_name)
    if args.dataset is None:
        args.dataset = config.get_default_dataset_path()
    output_file = args.output or f'orchestra_{args.dataset_name}_{args.provider}.jsonl'

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
    print(
        f'Dataset: {args.dataset_name} | Provider: {args.provider} | Model: {text_agent.model}'
    )

    dataset = load_data(args.dataset)
    test_list = list(dataset.keys())
    processed_keys = load_processed_keys(output_file)
    if processed_keys:
        print(
            f'Resuming: {len(processed_keys)} done, {len(test_list) - len(processed_keys)} remaining'
        )

    try:
        for test_file in test_list:
            if test_file in processed_keys:
                continue
            try:
                run_instance(
                    test_file,
                    dataset,
                    args.dataset_name,
                    video_agent,
                    audio_agent,
                    text_agent,
                    tracker,
                    output_file,
                )
            except Exception as e:
                print(f'[ERROR] {test_file}: {e}')
                continue
            time.sleep(2)
    finally:
        tracker.print_summary(f'Orchestra - {ROUNDS} Rounds')
