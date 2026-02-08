"""Test AudioProcessor with decomposed scoring."""

import os

from ctm_ai.processors import BaseProcessor


def test_audio_processor():
    """Test AudioProcessor with an audio file."""
    processor = BaseProcessor('audio_processor')
    audio_path = (
        '/Users/zhaoyining/Desktop/ctm-ai/exp_mustard/mustard_audios/2_1_audio.mp4'
    )

    if not os.path.exists(audio_path):
        print(f'Test skipped: Audio file not found at {audio_path}')
        return

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

    print('\nTest passed!')
    return chunk


if __name__ == '__main__':
    test_audio_processor()
