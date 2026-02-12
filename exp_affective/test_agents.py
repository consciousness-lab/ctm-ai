"""
Quick sanity check: verify each agent can actually perceive its modality.

Tests:
1. TextAgent   — can it read the text we give it?
2. AudioAgent  — can it hear what's said in the audio?
3. VideoAgent  — can it see what's shown in the video?
4. MultimodalAgent — can it both see and hear from the full video?

Usage:
    python test_agents.py --provider qwen --dataset_name urfunny
    python test_agents.py --provider gemini --dataset_name mustard
"""

import argparse
import json
import sys

import litellm
from dataset_configs import get_dataset_config
from llm_utils import (
    check_api_key,
    create_agent,
    load_data,
    load_sample_inputs,
)

sys.path.append('..')


def test_text_agent(agent, inputs):
    """Test: can TextAgent read the text?"""
    text = inputs['target_sentence']
    query = f'Repeat back the following sentence exactly, then describe its tone in one line:\n\n"{text}"'
    answer, usage = agent.call(query)
    return answer


def test_audio_agent(agent, inputs):
    """Test: can AudioAgent hear the audio?"""
    audio_path = inputs['audio_path']
    query = (
        'Focus ONLY on what the person is SAYING in the audio. The video is just a black screen.'
        'What is the person saying? What is their tone of voice?'
    )
    answer, usage = agent.call(query, audio_path=audio_path)
    return answer


def test_video_agent(agent, inputs):
    """Test: can VideoAgent see the video frames?"""
    video_path = inputs['muted_video_path']
    query = (
        'Describe what you see in this video. '
        'What does the person look like? What are they doing? '
        'Describe any facial expressions or gestures.'
    )
    answer, usage = agent.call(query, video_path=video_path)
    return answer


def test_multimodal_agent(agent, inputs):
    """Test: can MultimodalAgent both see and hear?"""
    video_path = inputs['full_video_path']
    query = (
        'This video has both visual and audio content.\n'
        "1. Describe what you SEE (the person's appearance, expressions, gestures).\n"
        '2. Describe what you HEAR (what the person is saying, their tone).\n'
        'Please clearly separate the visual and audio descriptions.'
    )
    answer, usage = agent.call(query, video_path=video_path)
    return answer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test agent modality perception')
    parser.add_argument(
        '--provider',
        type=str,
        default='qwen',
        choices=['gemini', 'qwen'],
        help='LLM provider (default: qwen)',
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Model name (default: auto based on provider)',
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='urfunny',
        choices=['urfunny', 'mustard'],
        help='Dataset to pick a sample from',
    )
    parser.add_argument(
        '--sample_key',
        type=str,
        default=None,
        help='Specific sample key to test (default: first sample in dataset)',
    )
    args = parser.parse_args()

    check_api_key(args.provider)
    litellm.set_verbose = False

    # Load dataset and pick a sample
    config = get_dataset_config(args.dataset_name)
    dataset = load_data(config.get_default_dataset_path())
    sample_key = args.sample_key or list(dataset.keys())[0]

    inputs = load_sample_inputs(sample_key, dataset, args.dataset_name)

    print(f'Provider: {args.provider}')
    print(f'Dataset:  {args.dataset_name}')
    print(f'Sample:   {sample_key}')
    print(f'Text:     {inputs["target_sentence"][:100]}...')
    print(f'Audio:    {inputs["audio_path"]}')
    print(f'Video:    {inputs["muted_video_path"]}')
    print(f'Full:     {inputs["full_video_path"]}')
    print()

    tests = [
        ('TextAgent', 'text', test_text_agent),
        ('AudioAgent', 'audio', test_audio_agent),
        ('VideoAgent', 'video', test_video_agent),
        ('MultimodalAgent', 'multimodal', test_multimodal_agent),
    ]

    for name, agent_type, test_fn in tests:
        print('=' * 60)
        print(f'TEST: {name}')
        print('=' * 60)

        agent = create_agent(agent_type, provider=args.provider, model=args.model)
        print(f'Model: {agent.model}')

        answer = test_fn(agent, inputs)

        if answer:
            print(f'\nResponse:\n{answer}\n')
        else:
            print(f'\n[FAILED] No response from {name}\n')
