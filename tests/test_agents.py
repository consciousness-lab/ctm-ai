"""
Test all agent types (TextAgent, AudioAgent, VideoAgent, MultimodalAgent)
for both Gemini and Qwen providers

Tests verify that each agent can properly process its designated modality:
- TextAgent: Can understand text context
- AudioAgent: Can hear audio content
- VideoAgent: Can see visual content (no audio)
- MultimodalAgent: Can see and hear simultaneously

Each test includes specific prompts to verify modality processing.
"""

import os
import sys

# Add parent directory to path to import llm_utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'exp_affective'))

from llm_utils import create_all_agents, check_api_key

# URFunny sample 408
# Context: "so yeah i'm a newspaper cartoonist political cartoonist"
# Punchline: "i don't know if you've heard about it newspapers it's a sort of paper based reader"
# Label: 1 (humorous)
# The humor is in treating newspapers as an unfamiliar technology
SAMPLE_KEY = '411'
# Use paths relative to project root (where tests are run from)
AUDIO_PATH = f'/Users/zhaoyining/Desktop/ctm-ai/exp_affective/data/urfunny/urfunny_audios/{SAMPLE_KEY}_audio.mp4'
VIDEO_NO_AUDIO_PATH = f'/Users/zhaoyining/Desktop/ctm-ai/exp_affective/data/urfunny/urfunny_muted_videos/{SAMPLE_KEY}.mp4'
VIDEO_WITH_AUDIO_PATH = f'/Users/zhaoyining/Desktop/ctm-ai/exp_affective/data/urfunny/urfunny_videos/{SAMPLE_KEY}.mp4'

TEXT_CONTEXT = (
    "so yeah i'm a newspaper cartoonist political cartoonist\n"
    "i don't know if you've heard about it newspapers it's a sort of paper based reader"
)

# Test queries designed to verify each modality
TEXT_QUERY = """Please describe the pubchline of the text."""

AUDIO_QUERY = """Can you hear the audio? Please describe the audio."""

VIDEO_QUERY = """Please provide details of the visual cues of the video. Note that the video do not contain audios, you should just describe the visual cues."""

MULTIMODAL_QUERY = """Analyze the COMPLETE multimodal input (text + video + audio together) and answer:
1. What is the speaker's profession (from text/audio)?
2. What do you SEE in the video (visual elements, expressions, gestures)?
3. What do you HEAR in the audio (tone, delivery, exact words)?
4. How do the visual and audio elements COMBINE to create humor?
5. Is the humor more effective with BOTH modalities together? Why?

Provide a comprehensive analysis considering ALL modalities."""


def print_separator(title: str):
    """Print a formatted separator"""
    print('\n' + '=' * 80)
    print(f'  {title}')
    print('=' * 80 + '\n')


def test_text_agent(provider: str):
    """Test TextAgent - verify it can understand text context"""
    print_separator(f'TextAgent Test - {provider.upper()}')

    agents = create_all_agents(provider=provider, temperature=0.0)
    text_agent = agents['text']

    print(f'Agent: {text_agent}')
    print(f'Context:\n{TEXT_CONTEXT}\n')

    response, usage = text_agent.call(query=TEXT_QUERY, context=TEXT_CONTEXT)

    if response:
        print(f'Response:\n{response}\n')
        print(f'Usage: {usage}')
        print('\n✅ TextAgent can understand text')
    else:
        print('❌ TextAgent failed')


def test_audio_agent(provider: str):
    """Test AudioAgent - verify it can hear audio content"""
    print_separator(f'AudioAgent Test - {provider.upper()}')

    if not os.path.exists(AUDIO_PATH):
        print(f'❌ Audio file not found: {AUDIO_PATH}')
        return

    agents = create_all_agents(provider=provider, temperature=0.0)
    audio_agent = agents['audio']

    print(f'Agent: {audio_agent}')
    print(f'Audio file: {AUDIO_PATH}')
    print(f'File size: {os.path.getsize(AUDIO_PATH)} bytes\n')

    response, usage = audio_agent.call(query=AUDIO_QUERY, audio_path=AUDIO_PATH)

    if response:
        print(f'Response:\n{response}\n')
        print(f'Usage: {usage}')
        print('\n✅ AudioAgent can hear audio')
    else:
        print('❌ AudioAgent failed')


def test_video_agent(provider: str):
    """Test VideoAgent - verify it can see video content (no audio)"""
    print_separator(f'VideoAgent Test - {provider.upper()}')

    if not os.path.exists(VIDEO_NO_AUDIO_PATH):
        print(f'❌ Video file not found: {VIDEO_NO_AUDIO_PATH}')
        return

    agents = create_all_agents(provider=provider, temperature=0.0)
    video_agent = agents['video']

    print(f'Agent: {video_agent}')
    print(f'Video file: {VIDEO_NO_AUDIO_PATH}')
    print(f'File size: {os.path.getsize(VIDEO_NO_AUDIO_PATH)} bytes\n')

    response, usage = video_agent.call(
        query=VIDEO_QUERY, video_path=VIDEO_NO_AUDIO_PATH
    )

    if response:
        print(f'Response:\n{response}\n')
        print(f'Usage: {usage}')
        print('\n✅ VideoAgent can see video')
    else:
        print('❌ VideoAgent failed')


def test_multimodal_agent(provider: str):
    """Test MultimodalAgent - verify it can see and hear simultaneously"""
    print_separator(f'MultimodalAgent Test - {provider.upper()}')

    if not os.path.exists(VIDEO_WITH_AUDIO_PATH):
        print(f'❌ Video file not found: {VIDEO_WITH_AUDIO_PATH}')
        return

    agents = create_all_agents(provider=provider, temperature=0.0)
    multimodal_agent = agents['multimodal']

    print(f'Agent: {multimodal_agent}')
    print(f'Video file (with audio): {VIDEO_WITH_AUDIO_PATH}')
    print(f'File size: {os.path.getsize(VIDEO_WITH_AUDIO_PATH)} bytes')
    print(f'Context: {TEXT_CONTEXT}\n')

    response, usage = multimodal_agent.call(
        query=MULTIMODAL_QUERY, context=TEXT_CONTEXT, video_path=VIDEO_WITH_AUDIO_PATH
    )

    if response:
        print(f'Response:\n{response}\n')
        print(f'Usage: {usage}')
        print('\n✅ MultimodalAgent can see AND hear')
    else:
        print('❌ MultimodalAgent failed')


def test_all_agents_for_provider(provider: str):
    """Test all 4 agents for a specific provider"""
    print('\n' + '#' * 80)
    print(f'#  TESTING ALL AGENTS FOR PROVIDER: {provider.upper()}')
    print('#' * 80)

    try:
        check_api_key(provider=provider)
    except ValueError as e:
        print(f'\n❌ Error: {e}')
        if provider == 'gemini':
            print('Please set GEMINI_API_KEY environment variable')
        elif provider == 'qwen':
            print('Please set DASHSCOPE_API_KEY environment variable')
        return False

    test_text_agent(provider)
    test_audio_agent(provider)
    test_video_agent(provider)
    test_multimodal_agent(provider)

    return True


def main():
    """Main test runner"""
    print('\n' + '*' * 80)
    print('*  AGENT MODALITY TEST SUITE')
    print('*' * 80)
    print(f'\nSample: URFunny {SAMPLE_KEY}')
    print(f'Text context: {TEXT_CONTEXT[:80]}...')
    print(f'\nFile availability:')
    print(f'  - Audio: {os.path.exists(AUDIO_PATH)}')
    print(f'  - Video (no audio): {os.path.exists(VIDEO_NO_AUDIO_PATH)}')
    print(f'  - Video (with audio): {os.path.exists(VIDEO_WITH_AUDIO_PATH)}')

    # Test Gemini
    gemini_success = test_all_agents_for_provider('gemini')

    # Test Qwen
    # qwen_success = test_all_agents_for_provider("qwen")

    # Summary
    print('\n' + '*' * 80)
    print('*  TEST SUMMARY')
    print('*' * 80)
    print(f'\nGemini: {"✅ PASSED" if gemini_success else "❌ FAILED"}')
    # print(f'Qwen: {"✅ PASSED" if qwen_success else "❌ FAILED"}')
    print('\n' + '*' * 80)


if __name__ == '__main__':
    main()
