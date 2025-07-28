"""
Utilities module for Tiny BFCL
Contains helper functions and utilities
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json_file(data: Dict[str, Any], file_path: str, indent: int = 2):
    """Save data to JSON file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def create_directory(path: str) -> Path:
    """Create directory if it doesn't exist"""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def format_model_response(response: Any) -> str:
    """Format model response for display"""
    if isinstance(response, list):
        return '\n'.join([str(item) for item in response])
    elif isinstance(response, dict):
        return json.dumps(response, indent=2, ensure_ascii=False)
    else:
        return str(response)


def calculate_cost(
    input_tokens: int, output_tokens: int, input_price: float, output_price: float
) -> float:
    """Calculate API cost based on token usage"""
    input_cost = (input_tokens / 1_000_000) * input_price
    output_cost = (output_tokens / 1_000_000) * output_price
    return input_cost + output_cost


def format_duration(seconds: float) -> str:
    """Format duration in human readable format"""
    if seconds < 60:
        return f'{seconds:.2f}s'
    elif seconds < 3600:
        minutes = seconds / 60
        return f'{minutes:.1f}m'
    else:
        hours = seconds / 3600
        return f'{hours:.1f}h'


def validate_test_case(test_case: Dict[str, Any]) -> bool:
    """Validate test case structure"""
    required_fields = ['id', 'function', 'question']

    for field in required_fields:
        if field not in test_case:
            print(f"Error: Missing required field '{field}' in test case")
            return False

    if not isinstance(test_case['function'], list):
        print("Error: 'function' field must be a list")
        return False

    if not isinstance(test_case['question'], list):
        print("Error: 'question' field must be a list")
        return False

    return True


def merge_results(
    existing_results: List[Dict[str, Any]], new_results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Merge new results with existing results"""
    result_dict = {}

    # Add existing results
    for result in existing_results:
        result_dict[result['id']] = result

    # Add/update new results
    for result in new_results:
        result_dict[result['id']] = result

    return list(result_dict.values())


def get_model_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate statistics from results"""
    if not results:
        return {}

    total_input_tokens = sum(result.get('input_token_count', 0) for result in results)
    total_output_tokens = sum(result.get('output_token_count', 0) for result in results)
    total_latency = sum(result.get('latency', 0) for result in results)

    error_count = sum(
        1 for result in results if 'Error' in str(result.get('result', ''))
    )
    success_count = len(results) - error_count

    return {
        'total_test_cases': len(results),
        'successful_cases': success_count,
        'error_cases': error_count,
        'success_rate': success_count / len(results) if results else 0,
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens,
        'total_latency': total_latency,
        'average_latency': total_latency / len(results) if results else 0,
    }


def print_results_summary(results: List[Dict[str, Any]], model_name: str):
    """Print summary of results"""
    stats = get_model_stats(results)

    print(f'\n📊 Results Summary for {model_name}')
    print('-' * 50)
    print(f'Total test cases: {stats["total_test_cases"]}')
    print(f'Successful cases: {stats["successful_cases"]}')
    print(f'Error cases: {stats["error_cases"]}')
    print(f'Success rate: {stats["success_rate"]:.2%}')
    print(f'Total input tokens: {stats["total_input_tokens"]:,}')
    print(f'Total output tokens: {stats["total_output_tokens"]:,}')
    print(f'Total latency: {format_duration(stats["total_latency"])}')
    print(f'Average latency: {format_duration(stats["average_latency"])}')


def check_api_keys() -> Dict[str, bool]:
    """Check if API keys are set"""
    return {
        'openai': bool(os.getenv('OPENAI_API_KEY')),
        'anthropic': bool(os.getenv('ANTHROPIC_API_KEY')),
        'google': bool(os.getenv('GOOGLE_API_KEY')),
    }


def print_api_key_status():
    """Print API key status"""
    api_keys = check_api_keys()

    print('🔑 API Key Status:')
    print('-' * 30)
    for provider, has_key in api_keys.items():
        status = '✅ Set' if has_key else '❌ Not set'
        print(f'{provider.capitalize()}: {status}')
    print()
