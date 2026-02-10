"""Test AudioProcessor with decomposed scoring for both Gemini and Qwen providers."""

import os
import sys

from ctm_ai.processors import BaseProcessor


def test_audio_processor(provider='gemini'):
    """Test AudioProcessor with an audio file.

    Args:
        provider: Either "gemini" or "qwen"
    """
    # Select model based on provider
    if provider == 'qwen':
        # Use qwen/ prefix - will be auto-converted to openai/ by get_completion_kwargs
        model = 'qwen/qwen3-omni-flash'
        print('\n' + '=' * 60)
        print(f'Testing AudioProcessor with QWEN')
        print('=' * 60)
    else:
        model = 'gemini/gemini-2.5-flash-lite'
        print('\n' + '=' * 60)
        print(f'Testing AudioProcessor with GEMINI')
        print('=' * 60)

    processor = BaseProcessor('audio_processor', model=model)
    audio_path = '/Users/zhaoyining/Desktop/ctm-ai/exp_affective/data/urfunny/urfunny_audios/408_audio.mp4'

    if not os.path.exists(audio_path):
        print(f'Test skipped: Audio file not found at {audio_path}')
        return

    print(f'Model: {processor.model}')
    print(f'Provider: {processor.provider}')
    print(f'Audio file: {audio_path}')
    print(f'File size: {os.path.getsize(audio_path)} bytes\n')

    query = 'Analyze the tone and emotion in this audio. Describe the tone of the audio and tell me what the person is talking.'

    chunk = processor.ask(
        query=query,
        audio_path=audio_path,
    )

    print('\nResponse:', chunk.gist)
    print('\nScores:')
    print(f'  Relevance:  {chunk.relevance:.2f}')
    print(f'  Confidence: {chunk.confidence:.2f}')
    print(f'  Surprise:   {chunk.surprise:.2f}')
    print(f'  Weight:     {chunk.weight:.2f}')

    if chunk.additional_question:
        print(f'\nAdditional Question: {chunk.additional_question}')

    # Assertions
    assert chunk is not None
    assert chunk.gist
    assert 0 <= chunk.relevance <= 1
    assert 0 <= chunk.confidence <= 1
    assert 0 <= chunk.surprise <= 1
    assert chunk.weight > 0

    print(f'\n✅ {provider.upper()} test passed!')
    return chunk


if __name__ == '__main__':
    # Check command line arguments
    if len(sys.argv) > 1:
        provider = sys.argv[1].lower()
        if provider not in ['gemini', 'qwen']:
            print('Usage: python test_audio_processor.py [gemini|qwen]')
            print('Default: gemini')
            sys.exit(1)
        test_audio_processor(provider)
    else:
        # Test both providers by default
        print('\n' + '*' * 60)
        print('Testing AudioProcessor with BOTH providers')
        print('*' * 60)

        try:
            test_audio_processor('gemini')
        except Exception as e:
            print(f'\n❌ Gemini test failed: {e}')

        try:
            test_audio_processor('qwen')
        except Exception as e:
            print(f'\n❌ Qwen test failed: {e}')

        print('\n' + '*' * 60)
        print('All tests completed!')
        print('*' * 60)
