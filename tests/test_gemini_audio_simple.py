"""
Simple test to verify Gemini audio support with different model versions
Uses official litellm format from documentation
"""

import base64
import os
from pathlib import Path

import litellm

# Set API key
os.environ['GEMINI_API_KEY'] = os.getenv('GEMINI_API_KEY', '')

# Enable verbose mode to see raw API calls
litellm.set_verbose = True

# Test audio file
AUDIO_PATH = '../exp_affective/data/urfunny/urfunny_audios/408_audio.mp4'


def test_gemini_audio(model_name):
    """Test a specific Gemini model with audio input"""
    print('\n' + '=' * 80)
    print(f'Testing: {model_name}')
    print('=' * 80)

    if not os.path.exists(AUDIO_PATH):
        print(f'❌ Audio file not found: {AUDIO_PATH}')
        return

    # Read and encode audio
    with open(AUDIO_PATH, 'rb') as f:
        audio_bytes = f.read()
    encoded_data = base64.b64encode(audio_bytes).decode('utf-8')

    print(f'Audio file: {AUDIO_PATH}')
    print(f'File size: {len(audio_bytes)} bytes')
    print(f'Encoded size: {len(encoded_data)} chars\n')

    # Build request using official litellm format
    try:
        response = litellm.completion(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'text',
                            'text': "Please listen to the audio and tell me: 1) What is the speaker's profession? 2) What are they talking about? 3) Describe the tone of voice.",
                        },
                        {
                            'type': 'file',
                            'file': {
                                'file_data': f'data:audio/mp4;base64,{encoded_data}',
                            },
                        },
                    ],
                }
            ],
            temperature=0.0,
        )

        print('✅ Response received:')
        print('-' * 80)
        print(response.choices[0].message.content)
        print('-' * 80)
        print(f'\nUsage:')
        print(f'  Prompt tokens: {response.usage.prompt_tokens}')
        print(f'  Completion tokens: {response.usage.completion_tokens}')
        print(f'  Total tokens: {response.usage.total_tokens}')

        # Check if response indicates it can hear audio
        response_text = response.choices[0].message.content.lower()
        if (
            'cannot hear' in response_text
            or "can't hear" in response_text
            or 'no audio' in response_text
        ):
            print('\n❌ Model says it CANNOT hear audio!')
            return False
        else:
            print('\n✅ Model CAN hear audio!')
            return True

    except Exception as e:
        print(f'\n❌ Error: {e}')
        return False


def main():
    """Test multiple Gemini model versions"""
    print('\n' + '*' * 80)
    print('GEMINI AUDIO SUPPORT TEST')
    print('*' * 80)

    models_to_test = [
        'gemini/gemini-2.5-flash-lite',
    ]

    results = {}

    for model in models_to_test:
        try:
            result = test_gemini_audio(model)
            results[model] = result
        except KeyboardInterrupt:
            print('\n\nTest interrupted by user')
            break
        except Exception as e:
            print(f'\n❌ Test failed with exception: {e}')
            results[model] = False

    # Summary
    print('\n\n' + '*' * 80)
    print('SUMMARY')
    print('*' * 80)

    for model, result in results.items():
        status = '✅ SUPPORTS AUDIO' if result else '❌ NO AUDIO SUPPORT'
        print(f'{model:<40} {status}')

    print('*' * 80)


if __name__ == '__main__':
    if not os.getenv('GEMINI_API_KEY'):
        print('❌ Error: GEMINI_API_KEY environment variable not set')
        print("Please set it with: export GEMINI_API_KEY='your-key'")
        exit(1)

    main()
