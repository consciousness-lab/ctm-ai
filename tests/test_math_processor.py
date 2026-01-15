"""
Test file for Wolfram|Alpha LLM API integration with MathProcessor.

Usage:
    python -m tests.test_math_processor

Make sure to set the environment variable:
    export WOLFRAM_APPID=your_appid_here
"""

import os
import sys
import urllib.parse

import requests

# Wolfram|Alpha LLM API base URL
WOLFRAM_API_BASE_URL = 'https://www.wolframalpha.com/api/v1/llm-api'


def test_wolfram_api_connection(appid: str) -> bool:
    """Test if the Wolfram|Alpha API is accessible with the given AppID."""
    print('=' * 60)
    print('Testing Wolfram|Alpha LLM API Connection')
    print('=' * 60)

    # Simple test query
    test_query = '2 + 2'

    params = {
        'input': test_query,
        'appid': appid,
        'maxchars': 500,
    }

    try:
        print(f'\nTest Query: "{test_query}"')
        print(f'API URL: {WOLFRAM_API_BASE_URL}')
        print('Sending request...\n')

        response = requests.get(WOLFRAM_API_BASE_URL, params=params, timeout=30)

        print(f'Status Code: {response.status_code}')

        if response.status_code == 200:
            print('\n✅ API Connection Successful!')
            print('\nResponse:')
            print('-' * 40)
            print(response.text[:500] if len(response.text) > 500 else response.text)
            print('-' * 40)
            return True
        else:
            print(f'\n❌ API Error: {response.status_code}')
            print(f'Response: {response.text}')
            return False

    except requests.exceptions.RequestException as e:
        print(f'\n❌ Connection Error: {str(e)}')
        return False


def test_math_queries(appid: str) -> None:
    """Test various mathematical queries."""
    print('\n' + '=' * 60)
    print('Testing Mathematical Queries')
    print('=' * 60)

    test_queries = [
        'derivative of x^2',
        'solve x^2 - 4 = 0',
        'integrate sin(x) dx',
        '10 densest elemental metals',
        'distance from earth to moon',
    ]

    for query in test_queries:
        print(f'\n📝 Query: "{query}"')
        print('-' * 40)

        params = {
            'input': query,
            'appid': appid,
            'maxchars': 1000,
        }

        try:
            response = requests.get(WOLFRAM_API_BASE_URL, params=params, timeout=30)

            if response.status_code == 200:
                # Show first 500 chars of response
                result = response.text
                if len(result) > 500:
                    print(result[:500] + '\n... (truncated)')
                else:
                    print(result)
                print('✅ Success')
            else:
                print(f'❌ Error: {response.status_code}')
                print(response.text[:200])

        except requests.exceptions.RequestException as e:
            print(f'❌ Error: {str(e)}')


def test_math_processor_integration(appid: str) -> None:
    """Test the MathProcessor class integration."""
    print('\n' + '=' * 60)
    print('Testing MathProcessor Integration')
    print('=' * 60)

    # Set environment variable for the processor
    os.environ['WOLFRAM_APPID'] = appid
    os.environ['GEMINI_API_KEY'] = os.environ.get('GEMINI_API_KEY', 'dummy_key')

    try:
        from ctm_ai.processors import BaseProcessor

        # Create MathProcessor instance
        processor = BaseProcessor(name='math_processor')

        print(f'\n✅ MathProcessor created successfully')
        print(f'   - Name: {processor.name}')
        print(f'   - Wolfram AppID: {processor.wolfram_appid[:8]}...')
        print(f'   - Max chars: {processor.max_chars}')

        # Test the internal API call method
        print('\n📝 Testing internal _call_wolfram_api method...')
        result = processor._call_wolfram_api('square root of 144')
        print(f'Query: "square root of 144"')
        print(f'Result (first 300 chars):')
        print(result[:300] if len(result) > 300 else result)
        print('✅ Internal API call successful')

    except ImportError as e:
        print(f'\n⚠️ Could not import MathProcessor: {e}')
        print('   Make sure you are running from the project root directory.')
    except Exception as e:
        print(f'\n❌ Error testing MathProcessor: {e}')


def main():
    """Main test function."""
    print('\n🔬 Wolfram|Alpha LLM API Test Suite')
    print('=' * 60)

    # Get AppID from environment or use the provided one
    appid = os.environ.get('WOLFRAM_APPID', 'mathapi')

    # Allow override from command line
    if len(sys.argv) > 1:
        appid = sys.argv[1]

    print(f'\nUsing AppID: {appid}')

    # Test 1: Basic API connection
    if not test_wolfram_api_connection(appid):
        print('\n⚠️ Basic API test failed. Please check your AppID.')
        print('   Make sure WOLFRAM_APPID is set correctly.')
        return

    # Test 2: Various math queries
    test_math_queries(appid)

    # Test 3: MathProcessor integration
    test_math_processor_integration(appid)

    print('\n' + '=' * 60)
    print('🎉 All tests completed!')
    print('=' * 60 + '\n')


if __name__ == '__main__':
    main()
