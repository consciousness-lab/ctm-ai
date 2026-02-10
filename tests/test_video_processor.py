"""Test VideoProcessor with decomposed scoring for both Gemini and Qwen providers."""

import os
import sys

from ctm_ai.processors import BaseProcessor


def test_video_processor(provider='gemini'):
    """Test VideoProcessor with a video file.

    Args:
        provider: Either "gemini" or "qwen"
    """
    # Select model based on provider
    if provider == 'qwen':
        # Use qwen/ prefix - will be auto-converted to openai/ by get_completion_kwargs
        model = 'qwen/qwen3-omni-flash'
        print('\n' + '=' * 60)
        print('Testing VideoProcessor with QWEN')
        print('=' * 60)
    else:
        model = 'gemini/gemini-2.5-flash'
        print('\n' + '=' * 60)
        print('Testing VideoProcessor with GEMINI')
        print('=' * 60)

    processor = BaseProcessor('video_processor', model=model)
    video_path = '/Users/zhaoyining/Desktop/ctm-ai/exp_affective/data/urfunny/urfunny_muted_videos/408.mp4'

    if not os.path.exists(video_path):
        print(f'Test skipped: Video file not found at {video_path}')
        return

    print(f'Model: {processor.model}')
    print(f'Provider: {processor.provider}')

    # Check file size
    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
    print(f'Video file: {video_path}')
    print(f'File size: {file_size_mb:.2f}MB\n')

    if file_size_mb > 20:
        print(f'Error: Video file exceeds 20MB limit (size: {file_size_mb:.2f}MB)')
        return

    query = 'Analyze this video. Describe the scene and the characters in detail.'

    chunk = processor.ask(
        query=query,
        video_path=video_path,
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
            print('Usage: python test_video_processor.py [gemini|qwen]')
            print('Default: gemini')
            sys.exit(1)
        test_video_processor(provider)
    else:
        # Test both providers by default
        print('\n' + '*' * 60)
        print('Testing VideoProcessor with BOTH providers')
        print('*' * 60)

        try:
            test_video_processor('gemini')
        except Exception as e:
            print(f'\n❌ Gemini test failed: {e}')

        try:
            test_video_processor('qwen')
        except Exception as e:
            print(f'\n❌ Qwen test failed: {e}')

        print('\n' + '*' * 60)
        print('All tests completed!')
        print('*' * 60)
